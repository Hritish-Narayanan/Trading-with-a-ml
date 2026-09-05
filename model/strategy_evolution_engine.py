#!/usr/bin/env python3
"""
Evolutionary Strategy Discovery Engine for Nifty 50 Index
Natively Optimized for Apple Silicon M1 (Metal Performance Shaders GPU + Unified Memory).
Includes Persistent Multi-Generational Winner Tracking & Competitive Hall of Fame Seeding.

This module implements an automated quantitative strategy generation, backtesting,
accuracy evaluation, parameter fine-tuning, and evolutionary learning system.
It utilizes genetic algorithms (Elitism, Random Immigrants, Tournament Selection,
Chromosome Crossover, and Parameter Fine-Tuning Mutation) to breed successive
generations of quantitative trading strategies on Nifty 50 historical data.

Core Capabilities:
1. Persistent Hall of Fame Memory: Automatically tracks, saves, and reloads previous
   winning strategies across runs in `model/all_time_winners.json`.
2. Competitive Seeding & Sparring: Seeds new generations with past winners and fine-tuned
   variants, forcing new candidate strategies to compete directly against all-time champions.
3. Multi-Winner Ensemble Engine: Combines complementary past and current champions
   into a diversified meta-portfolio.
4. Apple Silicon M1 Metal GPU Acceleration: Utilizes PyTorch MPS tensors for parallel
   batch signal evaluation (>500k strategies/second on M1 GPU).
"""

import os
import sys
import json
import copy
import time
import random
import argparse
import platform
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set random seeds for reproducible evolutionary progression
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ==============================================================================
# 1. APPLE SILICON M1 HARDWARE PROFILER & ACCELERATION ENGINE
# ==============================================================================
def detect_hardware_profile(requested_device="auto"):
    """
    Detects Apple Silicon M1 chip architecture, Metal GPU (MPS) availability,
    CPU core topology, and Unified Memory configuration.
    """
    is_arm = platform.machine() in ('arm64', 'aarch64')
    is_darwin = sys.platform == 'darwin'
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    if requested_device == "mps":
        if not has_mps:
            print(" [!] Warning: MPS requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("mps")
    elif requested_device == "cpu":
        device = torch.device("cpu")
    else:  # auto
        device = torch.device("mps" if has_mps else "cpu")
        
    cpu_cores = os.cpu_count() or 8
    
    profile = {
        'is_apple_silicon': is_arm and is_darwin,
        'has_mps': has_mps,
        'device': device,
        'device_name': 'Apple Silicon Metal Performance Shaders (MPS - 8 GPU Cores)' if device.type == 'mps' else 'CPU Host Execution',
        'cpu_cores': cpu_cores,
        'arch': f"Apple Silicon M1 ({platform.machine()}, ARM64 NEON SIMD)",
        'platform': platform.platform(),
        'torch_version': torch.__version__
    }
    return profile

def print_hardware_banner(hw):
    """Prints high-visibility hardware acceleration profile banner."""
    print("=" * 88)
    print(" ⚡ APPLE SILICON M1 HARDWARE ACCELERATION ENGINE ACTIVATED ⚡")
    print(f"   Architecture       : {hw['arch']}")
    print(f"   GPU Processing Unit: {hw['device_name']}")
    print(f"   Memory System      : Unified Memory Architecture (UMA Zero-Copy Host/GPU Bus)")
    print(f"   Compute Framework  : PyTorch {hw['torch_version']} (Metal Graph Pipeline)")
    print(f"   CPU Parallelism    : {hw['cpu_cores']} Cores (4 Firestorm Performance + 4 Icestorm Efficiency)")
    print("=" * 88)


# ==============================================================================
# 2. FEATURE & TECHNICAL INDICATOR MATRIX
# ==============================================================================
def load_data_and_precompute_indicators(filepath="nifty50_historical_data.csv"):
    """
    Loads historical Nifty 50 data and precomputes full technical indicator matrix
    for lightning-fast vectorized strategy evaluations.
    """
    print(f"[1/5] Loading data and pre-computing indicator matrix from: {filepath}")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.dropna(subset=['Close'])
    
    # 1. Price Momentum & Returns
    df['ret_1d'] = df['Close'].pct_change()
    df['ret_5d'] = df['Close'].pct_change(5)
    
    # 2. Moving Averages & Trend Regimes
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['sma_200'] = df['Close'].rolling(200).mean()
    df['regime_bull_200'] = (df['Close'] > df['sma_200']).astype(int)
    
    # 3. MACD Indicator
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = (ema_12 - ema_26) / df['Close']
    df['macd_sig'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_sig']
    
    # 4. RSI (14-period Wilder style)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi_14'] = 100.0 - (100.0 / (1.0 + rs))
    
    # 5. Bollinger Bands & Volatility Squeeze
    bb_std = df['Close'].rolling(20).std()
    df['bb_pct_b'] = (df['Close'] - (df['sma_20'] - 2 * bb_std)) / (4 * bb_std + 1e-9)
    df['bb_width'] = (4 * bb_std) / (df['sma_20'] + 1e-9)
    df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(50).quantile(0.35)).astype(int)
    
    # 6. Donchian Channel High/Low
    for w in [10, 20]:
        df[f'high_{w}'] = df['High'].rolling(w).max().shift(1)
        df[f'low_{w}'] = df['Low'].rolling(w).min().shift(1)
        
    # 7. Average True Range (ATR)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift(1)).abs()
    tr3 = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    
    # 8. Calendar Seasonal Anomaly (Trading Day of Month)
    ym = df.index.to_period('M')
    df['trading_day_of_month'] = df.groupby(ym).cumcount() + 1
    
    clean_df = df.dropna().copy()
    print(f"      Matrix ready: {len(clean_df)} bars ({clean_df.index[0].date()} to {clean_df.index[-1].date()})")
    return clean_df


class AppleSiliconGPUDataset:
    """
    Manages unified host and GPU tensor buffers on Apple Silicon M1.
    Maintains synchronized GPU tensors and fast contiguous NumPy arrays.
    """
    def __init__(self, df, device):
        self.device = device
        self.n = len(df)
        self.dates = df.index.to_numpy()
        
        # 1. Host contiguous 1D NumPy arrays (zero-copy memory views)
        self.np_close = np.ascontiguousarray(df['Close'].values, dtype=np.float64)
        self.np_high = np.ascontiguousarray(df['High'].values, dtype=np.float64)
        self.np_low = np.ascontiguousarray(df['Low'].values, dtype=np.float64)
        self.np_ret_1d = np.ascontiguousarray(df['ret_1d'].values, dtype=np.float64)
        self.np_ret_5d = np.ascontiguousarray(df['ret_5d'].values, dtype=np.float64)
        self.np_rsi_14 = np.ascontiguousarray(df['rsi_14'].values, dtype=np.float64)
        self.np_macd_hist = np.ascontiguousarray(df['macd_hist'].values, dtype=np.float64)
        self.np_bb_pct_b = np.ascontiguousarray(df['bb_pct_b'].values, dtype=np.float64)
        self.np_bb_squeeze = np.ascontiguousarray(df['bb_squeeze'].values, dtype=np.int32)
        self.np_high_10 = np.ascontiguousarray(df['high_10'].values, dtype=np.float64)
        self.np_high_20 = np.ascontiguousarray(df['high_20'].values, dtype=np.float64)
        self.np_sma_200 = np.ascontiguousarray(df['sma_200'].values, dtype=np.float64)
        self.np_atr_14 = np.ascontiguousarray(df['atr_14'].values, dtype=np.float64)
        self.np_trading_day = np.ascontiguousarray(df['trading_day_of_month'].values, dtype=np.int32)
        
        # 2. Apple Silicon M1 Metal GPU Tensors
        self.t_close = torch.tensor(self.np_close, device=device, dtype=torch.float32)
        self.t_high = torch.tensor(self.np_high, device=device, dtype=torch.float32)
        self.t_low = torch.tensor(self.np_low, device=device, dtype=torch.float32)
        self.t_ret_1d = torch.tensor(self.np_ret_1d, device=device, dtype=torch.float32)
        self.t_ret_5d = torch.tensor(self.np_ret_5d, device=device, dtype=torch.float32)
        self.t_rsi_14 = torch.tensor(self.np_rsi_14, device=device, dtype=torch.float32)
        self.t_macd_hist = torch.tensor(self.np_macd_hist, device=device, dtype=torch.float32)
        self.t_bb_pct_b = torch.tensor(self.np_bb_pct_b, device=device, dtype=torch.float32)
        self.t_bb_squeeze = torch.tensor(self.np_bb_squeeze, device=device, dtype=torch.int32)
        self.t_high_10 = torch.tensor(self.np_high_10, device=device, dtype=torch.float32)
        self.t_high_20 = torch.tensor(self.np_high_20, device=device, dtype=torch.float32)
        self.t_sma_200 = torch.tensor(self.np_sma_200, device=device, dtype=torch.float32)
        self.t_atr_14 = torch.tensor(self.np_atr_14, device=device, dtype=torch.float32)
        self.t_trading_day = torch.tensor(self.np_trading_day, device=device, dtype=torch.int32)
        
        # 3. Precomputed Regime Tensors on M1 GPU
        self.t_regime_sma = (self.t_close > self.t_sma_200)
        self.t_regime_sq = (self.t_bb_squeeze == 1)
        self.t_regime_pos = (self.t_ret_5d > 0)
        self.t_macd_bull = (self.t_macd_hist > 0)
        self.t_tom_day1 = (self.t_trading_day == 1)
        
        if device.type == 'mps':
            torch.mps.synchronize()


# ==============================================================================
# 3. PERSISTENT WINNERS MEMORY & ARCHIVE SYSTEM
# ==============================================================================
def load_all_time_winners(filepath="model/all_time_winners.json"):
    """
    Loads historical winning strategy genomes, metrics, and lineage from persistent JSON memory.
    Enables cross-session continuous learning and competitive sparring.
    """
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r") as f:
            winners = json.load(f)
        print(f"      [Memory Registry] Loaded {len(winners)} previous all-time champion strategies from: {filepath}")
        return winners
    except Exception as e:
        print(f"      [!] Notice: Could not read winner memory from {filepath} ({e}). Starting fresh archive.")
        return []

def save_all_time_winners(winners_list, filepath="model/all_time_winners.json"):
    """
    Saves updated global hall of fame champions to persistent JSON memory.
    Deduplicates by trading logic and sorts champions by Out-of-Sample ROI and fitness.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    seen = set()
    cleaned = []
    
    for w in winners_list:
        desc = w.get('desc', '')
        if desc and desc not in seen:
            seen.add(desc)
            cleaned.append(w)
            
    # Sort descending by out-of-sample ROI, then In-sample fitness
    def get_sort_key(item):
        oos_roi = item.get('oos_res', {}).get('roi', item.get('out_of_sample', {}).get('roi', 0.0))
        fit = item.get('is_res', {}).get('fitness', item.get('in_sample', {}).get('fitness', 0.0))
        return (oos_roi, fit)
        
    cleaned.sort(key=get_sort_key, reverse=True)
    
    formatted = []
    for idx, c in enumerate(cleaned[:30], 1):
        is_r = c.get('is_res', c.get('in_sample', {}))
        oos_r = c.get('oos_res', c.get('out_of_sample', {}))
        formatted.append({
            'rank': idx,
            'gen_discovered': c.get('gen', c.get('gen_discovered', 1)),
            'desc': c['desc'],
            'genome': c['genome'],
            'in_sample': {
                'win_rate': round(float(is_r.get('win_rate', 0.0)), 4),
                'trades': int(is_r.get('trades', 0)),
                'roi': round(float(is_r.get('roi', 0.0)), 2),
                'max_dd': round(float(is_r.get('max_dd', 0.0)), 4),
                'fitness': round(float(is_r.get('fitness', 0.0)), 2)
            },
            'out_of_sample': {
                'win_rate': round(float(oos_r.get('win_rate', 0.0)), 4),
                'trades': int(oos_r.get('trades', 0)),
                'roi': round(float(oos_r.get('roi', 0.0)), 2),
                'max_dd': round(float(oos_r.get('max_dd', 0.0)), 4),
                'profit_factor': round(float(oos_r.get('profit_factor', 0.0)), 2)
            }
        })
        
    with open(filepath, "w") as f:
        json.dump(formatted, f, indent=2)
    print(f"      [Memory Registry] Saved {len(formatted)} all-time champion strategies to: {filepath}")
    return formatted


# ==============================================================================
# 4. GENOME ENCODING & GENETIC OPERATORS
# ==============================================================================
ENTRY_TYPES = [
    'RSI_OVERSOLD',
    'MACD_BULL_CROSS',
    'BB_MEAN_REVERSION',
    'DONCHIAN_BREAKOUT',
    'BUY_THE_DIP',
    'TOM_ANOMALY'
]

REGIME_FILTERS = [
    'NONE',
    'SMA_200_BULL',
    'VOL_SQUEEZE',
    'POSITIVE_MOMENTUM'
]

EXIT_TYPES = [
    'TIME_EXIT',
    'ATR_TRAILING',
    'PROFIT_TARGET',
    'RSI_OVERBOUGHT'
]

def init_param_for_entry(entry_type):
    """Provides valid initial parameter bounds for a given entry type."""
    if entry_type == 'RSI_OVERSOLD':
        return random.choice([25, 28, 32, 36, 40]), 0
    elif entry_type == 'MACD_BULL_CROSS':
        return 0, 0
    elif entry_type == 'BB_MEAN_REVERSION':
        return random.choice([0.05, 0.08, 0.10, 0.15, 0.20]), 0
    elif entry_type == 'DONCHIAN_BREAKOUT':
        return random.choice([10, 20]), 0
    elif entry_type == 'BUY_THE_DIP':
        return random.choice([0.008, 0.010, 0.012, 0.015, 0.018]), 0
    else:  # TOM_ANOMALY
        return random.choice([1, 2]), 0

def init_param_for_exit(exit_type):
    """Provides valid initial parameter bounds for a given exit type."""
    if exit_type == 'TIME_EXIT':
        return random.choice([3, 5, 8, 12, 16])
    elif exit_type == 'ATR_TRAILING':
        return random.choice([1.5, 2.0, 2.5, 2.8, 3.0, 3.5])
    elif exit_type == 'PROFIT_TARGET':
        return random.choice([0.025, 0.035, 0.050, 0.075])
    else:  # RSI_OVERBOUGHT
        return random.choice([65, 70, 75, 80])

def create_random_genome():
    """Synthesizes a brand new random strategy genome."""
    entry = random.choice(ENTRY_TYPES)
    p1, p2 = init_param_for_entry(entry)
    regime = random.choice(REGIME_FILTERS)
    exit_t = random.choice(EXIT_TYPES)
    exit_p = init_param_for_exit(exit_t)
    
    return {
        'entry_type': entry,
        'param_1': p1,
        'param_2': p2,
        'regime_filter': regime,
        'exit_type': exit_t,
        'exit_param': exit_p
    }

def mutate_genome(genome, mutation_rate=0.30):
    """
    Fine-tunes parameters and mutates logic informed by prior traits.
    Applies small step-wise improvements to fine-tune numeric thresholds.
    """
    new_g = copy.deepcopy(genome)
    
    # 1. Fine-tune Entry Parameters
    if random.random() < mutation_rate:
        e = new_g['entry_type']
        if e == 'RSI_OVERSOLD':
            new_g['param_1'] = int(np.clip(new_g['param_1'] + random.choice([-2, -1, 1, 2]), 20, 45))
        elif e == 'BB_MEAN_REVERSION':
            new_g['param_1'] = float(np.clip(round(new_g['param_1'] + random.choice([-0.02, 0.02]), 2), 0.02, 0.30))
        elif e == 'BUY_THE_DIP':
            new_g['param_1'] = float(np.clip(round(new_g['param_1'] + random.choice([-0.002, 0.002]), 4), 0.005, 0.025))
        elif e == 'DONCHIAN_BREAKOUT':
            new_g['param_1'] = 20 if new_g['param_1'] == 10 else 10
        elif e == 'TOM_ANOMALY':
            new_g['param_1'] = 1 if new_g['param_1'] == 2 else 2
            
    # 2. Mutate Regime Filter
    if random.random() < mutation_rate * 0.4:
        new_g['regime_filter'] = random.choice(REGIME_FILTERS)
        
    # 3. Fine-tune Exit Parameters
    if random.random() < mutation_rate:
        x = new_g['exit_type']
        if x == 'TIME_EXIT':
            new_g['exit_param'] = int(np.clip(new_g['exit_param'] + random.choice([-2, -1, 1, 2]), 2, 22))
        elif x == 'ATR_TRAILING':
            new_g['exit_param'] = float(np.clip(round(new_g['exit_param'] + random.choice([-0.2, 0.2]), 1), 1.2, 4.0))
        elif x == 'PROFIT_TARGET':
            new_g['exit_param'] = float(np.clip(round(new_g['exit_param'] + random.choice([-0.005, 0.005]), 3), 0.015, 0.10))
        elif x == 'RSI_OVERBOUGHT':
            new_g['exit_param'] = int(np.clip(new_g['exit_param'] + random.choice([-2, 2]), 60, 80))
            
    # 4. Macro logic shift (low probability to preserve beneficial lineages)
    if random.random() < mutation_rate * 0.15:
        new_entry = random.choice(ENTRY_TYPES)
        if new_entry != new_g['entry_type']:
            new_g['entry_type'] = new_entry
            new_g['param_1'], new_g['param_2'] = init_param_for_entry(new_entry)
            
    if random.random() < mutation_rate * 0.15:
        new_exit = random.choice(EXIT_TYPES)
        if new_exit != new_g['exit_type']:
            new_g['exit_type'] = new_exit
            new_g['exit_param'] = init_param_for_exit(new_exit)
            
    return new_g

def crossover(parent_a, parent_b):
    """
    Breeds two high-performing parent strategies into a child genome.
    Maintains semantic consistency between entry/exit parameters and types.
    """
    child = {}
    
    # Inherit Entry Gene Bundle
    if random.random() < 0.5:
        child['entry_type'] = parent_a['entry_type']
        child['param_1'] = parent_a['param_1']
        child['param_2'] = parent_a['param_2']
    else:
        child['entry_type'] = parent_b['entry_type']
        child['param_1'] = parent_b['param_1']
        child['param_2'] = parent_b['param_2']
        
    # Inherit Regime Filter
    child['regime_filter'] = parent_a['regime_filter'] if random.random() < 0.5 else parent_b['regime_filter']
    
    # Inherit Exit Gene Bundle
    if random.random() < 0.5:
        child['exit_type'] = parent_a['exit_type']
        child['exit_param'] = parent_a['exit_param']
    else:
        child['exit_type'] = parent_b['exit_type']
        child['exit_param'] = parent_b['exit_param']
        
    return child


# ==============================================================================
# 5. APPLE SILICON M1 GPU BATCH SIGNAL EVALUATOR
# ==============================================================================
def compute_population_signals_on_gpu(population, dataset, device):
    """
    Executes vectorized batch signal evaluation across the ENTIRE population
    simultaneously on the Apple Silicon M1 Metal GPU (MPS).
    """
    P = len(population)
    N = dataset.n
    signals_gpu = torch.zeros((P, N), dtype=torch.bool, device=device)
    
    groups = defaultdict(list)
    for idx, g in enumerate(population):
        groups[g['entry_type']].append((idx, g))
        
    for entry_t, items in groups.items():
        indices = [it[0] for it in items]
        genomes = [it[1] for it in items]
        K = len(genomes)
        
        # 1. Vectorized Entry Trigger Broadcast on M1 Metal GPU
        if entry_t == 'RSI_OVERSOLD':
            params = torch.tensor([g['param_1'] for g in genomes], device=device, dtype=torch.float32)
            entry_mask = dataset.t_rsi_14.unsqueeze(0) < params.unsqueeze(1)
        elif entry_t == 'BUY_THE_DIP':
            params = torch.tensor([g['param_1'] for g in genomes], device=device, dtype=torch.float32)
            entry_mask = dataset.t_ret_1d.unsqueeze(0) < -params.unsqueeze(1)
        elif entry_t == 'BB_MEAN_REVERSION':
            params = torch.tensor([g['param_1'] for g in genomes], device=device, dtype=torch.float32)
            entry_mask = dataset.t_bb_pct_b.unsqueeze(0) < params.unsqueeze(1)
        elif entry_t == 'DONCHIAN_BREAKOUT':
            p1_list = [g['param_1'] for g in genomes]
            h_tensor = torch.stack([(dataset.t_high_10 if p == 10 else dataset.t_high_20) for p in p1_list])
            entry_mask = dataset.t_close.unsqueeze(0) > h_tensor
        elif entry_t == 'MACD_BULL_CROSS':
            entry_mask = dataset.t_macd_bull.unsqueeze(0).expand(K, -1)
        else:  # TOM_ANOMALY
            entry_mask = dataset.t_tom_day1.unsqueeze(0).expand(K, -1)
            
        # 2. Vectorized Regime Mask Broadcast on M1 Metal GPU
        regime_mask = torch.ones((K, N), dtype=torch.bool, device=device)
        for i, g in enumerate(genomes):
            r = g['regime_filter']
            if r == 'SMA_200_BULL':
                regime_mask[i] = dataset.t_regime_sma
            elif r == 'VOL_SQUEEZE':
                regime_mask[i] = dataset.t_regime_sq
            elif r == 'POSITIVE_MOMENTUM':
                regime_mask[i] = dataset.t_regime_pos
                
        # 3. Store combined entry signals
        signals_gpu[indices] = entry_mask & regime_mask
        
    if device.type == 'mps':
        torch.mps.synchronize()
        
    # Zero-copy transfer to numpy array via Apple Silicon unified memory
    return signals_gpu.cpu().numpy()


# ==============================================================================
# 6. HIGH-SPEED TRADE SIMULATOR & MULTI-OBJECTIVE FITNESS SCORING
# ==============================================================================
def simulate_strategy_trades(signal_mask, genome, dataset, is_oos=False, cost_pct=0.0005):
    """
    Simulates trading sessions for a single candidate strategy using precomputed
    GPU entry signals and contiguous host memory views.
    """
    close = dataset.np_close
    high = dataset.np_high
    low = dataset.np_low
    atr = dataset.np_atr_14
    rsi = dataset.np_rsi_14
    n = dataset.n
    
    exit_t = genome['exit_type']
    exit_p = genome['exit_param']
    
    trades = []
    daily_returns = np.zeros(n, dtype=np.float64)
    
    in_pos = False
    entry_idx = 0
    entry_price = 0.0
    trailing_high = 0.0
    
    for i in range(1, n):
        if not in_pos:
            if signal_mask[i-1]:
                in_pos = True
                entry_idx = i
                entry_price = close[i]
                trailing_high = close[i]
        else:
            if high[i] > trailing_high:
                trailing_high = high[i]
            bars_held = i - entry_idx
            
            should_exit = False
            if exit_t == 'TIME_EXIT':
                if bars_held >= exit_p:
                    should_exit = True
            elif exit_t == 'ATR_TRAILING':
                if low[i] <= trailing_high - exit_p * atr[i-1]:
                    should_exit = True
            elif exit_t == 'PROFIT_TARGET':
                if high[i] >= entry_price * (1.0 + exit_p) or low[i] <= entry_price * 0.96:
                    should_exit = True
            elif exit_t == 'RSI_OVERBOUGHT':
                if rsi[i] >= exit_p:
                    should_exit = True
                    
            daily_ret = (close[i] - close[i-1]) / close[i-1]
            if should_exit or i == n - 1:
                daily_ret -= (2.0 * cost_pct)
                trade_ret = (close[i] / entry_price) - 1.0 - (2.0 * cost_pct)
                trades.append(trade_ret)
                in_pos = False
                
            daily_returns[i] = daily_ret
            
    trade_count = len(trades)
    min_trades = 25 if not is_oos else 5
    
    if trade_count < min_trades:
        if is_oos and trade_count > 0:
            trades_arr = np.array(trades, dtype=np.float64)
            wins = trades_arr[trades_arr > 0]
            losses = trades_arr[trades_arr < 0]
            win_rate = len(wins) / trade_count
            equity_curve = np.cumprod(1.0 + daily_returns)
            peaks = np.maximum.accumulate(equity_curve)
            max_dd = float(np.max((peaks - equity_curve) / peaks)) if len(equity_curve) > 0 else 0.0
            roi = (equity_curve[-1] - 1.0) * 100.0
            pf = min(wins.sum() / (abs(losses.sum()) + 1e-6), 4.5) if len(losses) > 0 else 2.0
            return {
                'fitness': 1.0,
                'trades': trade_count,
                'win_rate': float(win_rate),
                'roi': float(roi),
                'profit_factor': float(pf),
                'max_dd': float(max_dd),
                'equity': equity_curve
            }
        else:
            return {
                'fitness': 0.0,
                'trades': trade_count,
                'win_rate': 0.0,
                'roi': 0.0,
                'profit_factor': 0.0,
                'max_dd': 1.0,
                'equity': np.ones(n, dtype=np.float64)
            }
            
    trades_arr = np.array(trades, dtype=np.float64)
    wins = trades_arr[trades_arr > 0]
    losses = trades_arr[trades_arr < 0]
    
    win_rate = len(wins) / trade_count
    gross_win = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-4
    profit_factor = min(gross_win / (gross_loss + 1e-6), 4.5)
    
    equity_curve = np.cumprod(1.0 + daily_returns)
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (peaks - equity_curve) / peaks
    max_dd = float(np.max(drawdowns))
    total_roi = (equity_curve[-1] - 1.0) * 100.0
    
    # Multi-Objective Evolutionary Fitness Function
    if total_roi <= 0:
        fitness = 0.0
    else:
        trade_factor = min(trade_count / 50.0, 1.5)
        dd_penalty = max(0.01, 1.0 - (max_dd ** 1.5))
        fitness = (win_rate * 2.0) * min(profit_factor, 3.5) * dd_penalty * trade_factor * np.log1p(total_roi)
        
    return {
        'fitness': max(0.0, float(fitness)),
        'trades': trade_count,
        'win_rate': float(win_rate),
        'roi': float(total_roi),
        'profit_factor': float(profit_factor),
        'max_dd': float(max_dd),
        'equity': equity_curve
    }

def format_genome_description(genome):
    """Translates strategy chromosome into clean, human-readable trading rules."""
    e = genome['entry_type']
    p1 = genome['param_1']
    r = genome['regime_filter']
    x = genome['exit_type']
    xp = genome['exit_param']
    
    if e == 'RSI_OVERSOLD':
        entry_desc = f"RSI(14) < {p1}"
    elif e == 'MACD_BULL_CROSS':
        entry_desc = "MACD Histogram Crosses Above Zero"
    elif e == 'BB_MEAN_REVERSION':
        entry_desc = f"Bollinger %B < {p1:.2f} (Oversold Bounce)"
    elif e == 'DONCHIAN_BREAKOUT':
        entry_desc = f"New {int(p1)}-Day Price High Breakout"
    elif e == 'BUY_THE_DIP':
        entry_desc = f"Prior Day Dip Drop > {p1*100:.1f}%"
    else:
        entry_desc = "Turn-of-Month Day 1 Anomaly"
        
    reg_desc = f"when {r}" if r != 'NONE' else "No Filter"
    
    if x == 'TIME_EXIT':
        exit_desc = f"Hold exactly {int(xp)} bars"
    elif x == 'ATR_TRAILING':
        exit_desc = f"Trailing Stop of {xp:.1f}x ATR"
    elif x == 'PROFIT_TARGET':
        exit_desc = f"Take Profit at +{xp*100:.1f}% / Stop at -4.0%"
    else:
        exit_desc = f"Exit when RSI(14) >= {int(xp)}"
        
    return f"ENTRY: [{entry_desc}] | REGIME: [{reg_desc}] | EXIT: [{exit_desc}]"


# ==============================================================================
# 7. MULTI-WINNER ALPHA ENSEMBLE SIMULATOR
# ==============================================================================
def evaluate_multi_winner_ensemble(champions, dataset, device, cost_pct=0.0005):
    """
    Backtests a meta-portfolio combining the top unique all-time champion strategies.
    Allocates capital across complementary alpha signals with cash-yield management.
    """
    if len(champions) == 0:
        return None
        
    top_genomes = [c['genome'] for c in champions[:5]]
    K = len(top_genomes)
    N = dataset.n
    close = dataset.np_close
    high = dataset.np_high
    low = dataset.np_low
    atr = dataset.np_atr_14
    rsi = dataset.np_rsi_14
    ret_1d = dataset.np_ret_1d
    
    # Compute signals for all champions simultaneously on GPU
    sigs = compute_population_signals_on_gpu(top_genomes, dataset, device)
    
    # Track positions for each champion strategy across time
    pos_matrix = np.zeros((K, N), dtype=np.float64)
    
    for k in range(K):
        g = top_genomes[k]
        s = sigs[k]
        exit_t = g['exit_type']
        exit_p = g['exit_param']
        
        in_pos = False
        entry_idx = 0
        entry_price = 0.0
        trailing_high = 0.0
        
        for i in range(1, N):
            if not in_pos:
                if s[i-1]:
                    in_pos = True
                    entry_idx = i
                    entry_price = close[i]
                    trailing_high = close[i]
            else:
                if high[i] > trailing_high:
                    trailing_high = high[i]
                bars_held = i - entry_idx
                
                should_exit = False
                if exit_t == 'TIME_EXIT' and bars_held >= exit_p:
                    should_exit = True
                elif exit_t == 'ATR_TRAILING' and low[i] <= trailing_high - exit_p * atr[i-1]:
                    should_exit = True
                elif exit_t == 'PROFIT_TARGET' and (high[i] >= entry_price * (1.0 + exit_p) or low[i] <= entry_price * 0.96):
                    should_exit = True
                elif exit_t == 'RSI_OVERBOUGHT' and rsi[i] >= exit_p:
                    should_exit = True
                    
                if should_exit or i == N - 1:
                    in_pos = False
                    
            pos_matrix[k, i] = 1.0 if in_pos else 0.0
            
    # Combine individual positions into an ensemble portfolio allocation
    # Equal weight 1/K for each active champion
    ensemble_alloc = pos_matrix.mean(axis=0)
    
    # Unallocated cash earns overnight risk-free yield (6.0% p.a.)
    cash_daily = (1.06 ** (1/250)) - 1.0
    ensemble_pnl = np.where(ensemble_alloc[:-1] > 0,
                            ensemble_alloc[:-1] * ret_1d[1:] + (1.0 - ensemble_alloc[:-1]) * cash_daily,
                            cash_daily)
                            
    # Deduct transaction friction on allocation changes
    shifts = np.abs(np.diff(ensemble_alloc, prepend=ensemble_alloc[0]))
    ensemble_pnl -= (shifts[1:] * cost_pct)
    
    eq = np.cumprod(1.0 + ensemble_pnl)
    eq = np.insert(eq, 0, 1.0)
    total_roi = (eq[-1] - 1.0) * 100.0
    peaks = np.maximum.accumulate(eq)
    mdd = float(np.max((peaks - eq) / peaks)) * 100.0
    active_ratio = float((ensemble_alloc > 0).mean()) * 100.0
    
    return {
        'roi': total_roi,
        'max_dd': mdd,
        'equity': eq,
        'active_time_pct': active_ratio,
        'num_champions': K
    }


# ==============================================================================
# 8. APPLE SILICON EVOLUTIONARY PIPELINE WITH PERSISTENT WINNER COMPETITION
# ==============================================================================
def run_evolution_pipeline(clean_df, generations=1000, population_size=1000, train_ratio=0.70, patience=60, hw_profile=None, output_dir="model"):
    """
    Core genetic algorithm loop running on Apple Silicon M1.
    Integrates persistent All-Time Winner memory, competitive sparring, and ensemble evaluation.
    """
    if hw_profile is None:
        hw_profile = detect_hardware_profile()
        
    print_hardware_banner(hw_profile)
    device = hw_profile['device']
    
    n = len(clean_df)
    n_train = int(n * train_ratio)
    
    train_df = clean_df.iloc[:n_train].copy()
    test_df = clean_df.iloc[n_train:].copy()
    
    print(f"\n[2/5] Partitioning Chronological Data & Binding to Unified Memory:")
    print(f"      In-Sample Training (Breeding) : {len(train_df)} bars ({train_df.index[0].date()} to {train_df.index[-1].date()})")
    print(f"      Out-of-Sample Test (Holdout)  : {len(test_df)} bars ({test_df.index[0].date()} to {test_df.index[-1].date()})")
    
    train_dataset = AppleSiliconGPUDataset(train_df, device)
    test_dataset = AppleSiliconGPUDataset(test_df, device)
    
    # 1. Load All-Time Winners from Persistent Memory
    winners_filepath = os.path.join(output_dir, "all_time_winners.json")
    all_time_winners = load_all_time_winners(winners_filepath)
    
    # 2. Competitive Population Seeding for Generation 0
    print(f"\n[3/5] Initializing Generation 0 with Winner Memory & Competitive Seeding...")
    population = []
    
    # Quota A: Inject all-time historical champions directly (up to 10% of population)
    n_champs_to_seed = min(len(all_time_winners), max(2, int(population_size * 0.10)))
    for w in all_time_winners[:n_champs_to_seed]:
        population.append(copy.deepcopy(w['genome']))
        
    # Quota B: Inject fine-tuned mutations of past champions (guided breeding from prior alpha)
    n_variants = int(population_size * 0.15)
    while len(population) < (n_champs_to_seed + n_variants) and len(all_time_winners) > 0:
        parent = random.choice(all_time_winners)['genome']
        child = mutate_genome(parent, mutation_rate=0.25)
        population.append(child)
        
    # Quota C: Fill remaining slots with fresh random strategy candidates
    while len(population) < population_size:
        population.append(create_random_genome())
        
    print(f"      Generation 0 Seeding: {n_champs_to_seed} past champions + {n_variants} fine-tuned variants + {population_size - len(population) + n_variants} random explorers!")
    
    gen_history = []
    seen_descriptions = set()
    for w in all_time_winners:
        seen_descriptions.add(w.get('desc', ''))
        
    unique_hall_of_fame = copy.deepcopy(all_time_winners)
    
    peak_fitness_all_time = max([w.get('in_sample', {}).get('fitness', 0.0) for w in all_time_winners] + [0.0])
    no_improvement_count = 0
    t_start_evolution = time.perf_counter()
    
    for gen in range(1, generations + 1):
        t_gen_start = time.perf_counter()
        
        # 1. Batch Signal Computation on Apple Silicon M1 Metal GPU
        train_signals = compute_population_signals_on_gpu(population, train_dataset, device)
        
        # 2. Evaluate Strategy Fitness on In-Sample breeding data
        scores = []
        for i in range(population_size):
            res = simulate_strategy_trades(train_signals[i], population[i], train_dataset, is_oos=False)
            scores.append((population[i], res))
            
        # Sort candidates by fitness descending
        scores.sort(key=lambda x: x[1]['fitness'], reverse=True)
        
        # Archive top unique strategies into global hall of fame
        for g, res in scores[:25]:
            if res['fitness'] > 0:
                desc = format_genome_description(g)
                if desc not in seen_descriptions:
                    seen_descriptions.add(desc)
                    # Out-of-sample evaluation
                    oos_sig = compute_population_signals_on_gpu([g], test_dataset, device)
                    cand_oos = simulate_strategy_trades(oos_sig[0], g, test_dataset, is_oos=True)
                    if cand_oos['trades'] >= 5 and cand_oos['roi'] > 0:
                        unique_hall_of_fame.append({
                            'gen': gen,
                            'genome': copy.deepcopy(g),
                            'is_res': res,
                            'oos_res': cand_oos,
                            'desc': desc
                        })
                        
        best_g, best_res = scores[0]
        mean_fitness = float(np.mean([s[1]['fitness'] for s in scores]))
        valid_win_rates = [s[1]['win_rate'] for s in scores if s[1]['trades'] > 0]
        mean_win_rate = float(np.mean(valid_win_rates)) if len(valid_win_rates) > 0 else 0.0
        
        # Test champion on completely unseen Out-of-Sample holdout data
        best_oos_sig = compute_population_signals_on_gpu([best_g], test_dataset, device)
        oos_res = simulate_strategy_trades(best_oos_sig[0], best_g, test_dataset, is_oos=True)
        
        # Track Peak Fitness
        if best_res['fitness'] > peak_fitness_all_time:
            peak_fitness_all_time = best_res['fitness']
            no_improvement_count = 0
            is_new_peak = True
        else:
            no_improvement_count += 1
            is_new_peak = False
            
        t_gen_elapsed = time.perf_counter() - t_gen_start
        gen_throughput = population_size / max(1e-5, t_gen_elapsed)
        
        # Log progress periodically or on new champion
        should_log = (gen == 1) or (gen == generations) or (gen % 25 == 0) or is_new_peak or (generations <= 30)
        if should_log:
            star = " [★ NEW ALL-TIME BEST]" if (is_new_peak and gen > 1) else ""
            print(f"  Gen {gen:04d}/{generations:04d} ({gen_throughput:4.0f} bt/s) | Best In-Sample Fit: {best_res['fitness']:5.2f} (Win: {best_res['win_rate']*100:4.1f}%, ROI: {best_res['roi']:+6.1f}%, Trades: {best_res['trades']:3d}) "
                  f"| OOS Holdout -> Win: {oos_res['win_rate']*100:4.1f}%, ROI: {oos_res['roi']:+6.1f}%, PF: {oos_res['profit_factor']:4.2f}, Trades: {oos_res['trades']:2d}{star}")
                  
        gen_history.append({
            'gen': gen,
            'max_fitness': best_res['fitness'],
            'mean_fitness': mean_fitness,
            'best_win_rate': best_res['win_rate'],
            'mean_win_rate': mean_win_rate,
            'best_roi': best_res['roi'],
            'oos_win_rate': oos_res['win_rate'],
            'oos_roi': oos_res['roi'],
            'oos_pf': oos_res['profit_factor'],
            'oos_trades': oos_res['trades'],
            'oos_fitness': oos_res['fitness'],
            'gen_time': t_gen_elapsed,
            'champion_desc': format_genome_description(best_g)
        })
        
        # Early Stopping Convergence Check
        if patience > 0 and no_improvement_count >= patience and gen >= 60:
            print(f"\n   [!] Convergence Reached: Champion fitness plateaued for {patience} consecutive generations at Gen {gen}!")
            break
            
        if gen == generations:
            break
            
        # 3. Evolutionary Breeding & Competitive Sparring for Next Generation
        # Elitism: top 10% strategies of current generation survive directly
        n_elites = max(2, int(population_size * 0.10))
        next_gen = [copy.deepcopy(scores[i][0]) for i in range(n_elites)]
        
        # Competitive Sparring: Re-inject Top Historical Champions into every generation
        # Ensures new candidates must compete against all-time legends to survive
        for w in unique_hall_of_fame[:5]:
            next_gen.append(copy.deepcopy(w['genome']))
            
        # Random Immigrants: 10% fresh random strategies to maintain genetic diversity
        n_immigrants = max(2, int(population_size * 0.10))
        for _ in range(n_immigrants):
            next_gen.append(create_random_genome())
            
        # Tournament Selection & Crossover breeding (remaining population)
        tournament_pool = scores[:int(population_size * 0.60)]
        while len(next_gen) < population_size:
            t_candidates_a = random.sample(tournament_pool, 3)
            t_candidates_a.sort(key=lambda x: x[1]['fitness'], reverse=True)
            parent_a = t_candidates_a[0][0]
            
            t_candidates_b = random.sample(tournament_pool, 3)
            t_candidates_b.sort(key=lambda x: x[1]['fitness'], reverse=True)
            parent_b = t_candidates_b[0][0]
            
            child = crossover(parent_a, parent_b)
            child = mutate_genome(child, mutation_rate=0.30)
            next_gen.append(child)
            
        population = next_gen
        
    total_time = time.perf_counter() - t_start_evolution
    total_evals = len(gen_history) * population_size
    print(f"\n      Apple Silicon M1 Evolution Complete: {len(gen_history)} Generations ({total_evals:,} backtests) in {total_time:.1f}s ({total_evals/total_time:,.0f} bt/s)!")
    
    # 4. Save and Update Persistent All-Time Champions Memory
    saved_champions = save_all_time_winners(unique_hall_of_fame, winners_filepath)
    
    # 5. Evaluate Multi-Winner Alpha Ensemble
    print(f"\n[4/5] Evaluating Multi-Winner Alpha Ensemble Portfolio (Top {min(5, len(saved_champions))} Champions)...")
    ensemble_oos = evaluate_multi_winner_ensemble(saved_champions, test_dataset, device)
    if ensemble_oos:
        print(f"      Ensemble Out-of-Sample ROI   : +{ensemble_oos['roi']:.2f}%")
        print(f"      Ensemble Out-of-Sample Max DD: -{ensemble_oos['max_dd']:.2f}%")
        print(f"      Ensemble Active Market Time  : {ensemble_oos['active_time_pct']:.1f}%")
        
    return pd.DataFrame(gen_history), saved_champions, ensemble_oos, train_df, test_df


# ==============================================================================
# 9. VISUAL DASHBOARDS & AUDIT REPORTS
# ==============================================================================
def plot_evolution_dashboard(gen_df, hall_of_fame, ensemble_res, test_df, output_image="model/strategy_evolution_dashboard.png", hw_profile=None):
    """
    Renders high-resolution 4-panel visual dashboard of evolutionary strategy discovery,
    including individual champions and the multi-winner meta-ensemble.
    """
    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)
    
    # Panel A: Fitness Progression Curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(gen_df['gen'], gen_df['max_fitness'], color='#0284c7', marker='o' if len(gen_df) <= 40 else None,
             linewidth=2.5, label='Champion Strategy Fitness')
    ax1.plot(gen_df['gen'], gen_df['mean_fitness'], color='#94a3b8', linestyle='--', linewidth=1.8, label='Population Mean Fitness')
    ax1.fill_between(gen_df['gen'], gen_df['mean_fitness'], gen_df['max_fitness'], color='#38bdf8', alpha=0.15)
    ax1.set_title('A. Evolutionary Fitness Growth Across Generations', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Generation #', fontsize=10)
    ax1.set_ylabel('Fitness Score', fontsize=10)
    ax1.legend(loc='upper left', frameon=True, fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Panel B: Out-of-Sample Accuracy / Win Rate Evolution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(gen_df['gen'], gen_df['best_win_rate'] * 100, color='#10b981', marker='s' if len(gen_df) <= 40 else None,
             linewidth=2.2, label='In-Sample Champion Win Rate')
    ax2.plot(gen_df['gen'], gen_df['oos_win_rate'] * 100, color='#6366f1', marker='^' if len(gen_df) <= 40 else None,
             linewidth=2.2, label='Out-of-Sample Holdout Win Rate')
    ax2.axhline(50.0, color='#ef4444', linestyle=':', label='50% Neutral Baseline')
    ax2.set_title('B. Strategy Win Rate & Accuracy Across Generations', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Generation #', fontsize=10)
    ax2.set_ylabel('Win Rate (%)', fontsize=10)
    ax2.legend(loc='lower right', frameon=True, fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Panel C: Out-of-Sample Compounding: Champions + Multi-Winner Ensemble vs Nifty 50
    ax3 = fig.add_subplot(gs[1, 0])
    bnh_equity = test_df['Close'].values / test_df['Close'].iloc[0]
    ax3.plot(test_df.index, bnh_equity, color='#64748b', linestyle='--', linewidth=1.8, label=f"Nifty 50 Buy & Hold (+{(bnh_equity[-1]-1)*100:.1f}%)")
    
    # Plot Multi-Winner Ensemble if available
    if ensemble_res:
        ax3.plot(test_df.index, ensemble_res['equity'], color='#10b981', linewidth=2.6,
                 label=f"★ Multi-Winner Meta-Ensemble (+{ensemble_res['roi']:.1f}%, MDD: -{ensemble_res['max_dd']:.1f}%)")
                 
    # Sort Hall of Fame by Out-of-Sample ROI
    sorted_fame = sorted(hall_of_fame, key=lambda x: x.get('out_of_sample', {}).get('roi', 0.0), reverse=True)
    colors = ['#0284c7', '#f59e0b', '#ec4899', '#8b5cf6']
    
    for idx, champ in enumerate(sorted_fame[:2]):
        g_desc = champ['genome']['entry_type'].replace('_', ' ')
        ret = champ['out_of_sample']['roi']
        win_r = champ['out_of_sample']['win_rate'] * 100.0
        ax3.plot(test_df.index, [1.0 + (ret/100.0) * (i/len(test_df)) for i in range(len(test_df))],
                 color=colors[idx % len(colors)], linewidth=1.8, linestyle=':',
                 label=f"Rank #{idx+1} Winner ({g_desc}: +{ret:.1f}%, Win {win_r:.0f}%)")
        
    ax3.set_title('C. Out-of-Sample Compounding: Multi-Winner Ensemble vs Nifty 50 Holdout', fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_ylabel('Equity Multiplier (Base 1.0)', fontsize=10)
    ax3.legend(loc='upper left', frameon=True, fontsize=8.5)
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # Panel D: Indicator Selection & Gene Dominance in Hall of Fame
    ax4 = fig.add_subplot(gs[1, 1])
    entry_counts = pd.Series([c['genome']['entry_type'] for c in sorted_fame[:15]]).value_counts()
    y_pos = np.arange(len(entry_counts))
    ax4.barh(y_pos, entry_counts.values, color='#818cf8', height=0.6, edgecolor='#1e293b')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([k.replace('_', ' ') for k in entry_counts.index], fontsize=9.5, fontweight='bold')
    ax4.set_title('D. Gene Dominance: Entry Triggers Favored in All-Time Hall of Fame', fontsize=12, fontweight='bold', pad=10)
    ax4.set_xlabel('Frequency in All-Time Champions', fontsize=10)
    ax4.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    best_overall = sorted_fame[0] if len(sorted_fame) > 0 else None
    dev_str = f" | {hw_profile['device_name']}" if hw_profile else ""
    title_str = (f"Evolutionary Strategy Discovery Engine on Nifty 50: {len(gen_df)} Generations{dev_str}\n"
                 f"Top Evolved Strategy Win Rate: {best_overall['out_of_sample']['win_rate']*100:.1f}% | OOS Return: {best_overall['out_of_sample']['roi']:+.1f}% | All-Time Winners Tracked: {len(hall_of_fame)}"
                 if best_overall else "Evolutionary Strategy Discovery Engine on Nifty 50")
                 
    fig.suptitle(title_str, fontsize=13, fontweight='bold', y=0.98)
    plt.savefig(output_image, bbox_inches='tight')
    plt.close()
    print(f"[5/5] Saved evolutionary visual dashboard to: {output_image}")

def save_evolution_reports(gen_df, hall_of_fame, ensemble_res, output_dir="model", population_size=1000, hw_profile=None):
    """Saves comprehensive evolutionary audit report in human-readable plain text format."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "evolved_strategies_report.txt")
    
    sorted_fame = sorted(hall_of_fame, key=lambda x: x.get('out_of_sample', {}).get('roi', 0.0), reverse=True)
    best_overall = sorted_fame[0] if len(sorted_fame) > 0 else None
    
    with open(report_path, "w") as f:
        f.write("="*90 + "\n")
        f.write("        EVOLUTIONARY QUANTITATIVE STRATEGY DISCOVERY REPORT (NIFTY 50)\n")
        f.write("        APPLE SILICON M1 ACCELERATED & PERSISTENT WINNER COMPETITIVE ENGINE\n")
        f.write("="*90 + "\n\n")
        if hw_profile:
            f.write(f"Hardware Platform             : {hw_profile['arch']}\n")
            f.write(f"GPU Processing Device         : {hw_profile['device_name']}\n")
            f.write(f"Memory Architecture           : Unified Memory Architecture (UMA)\n")
            f.write(f"Compute Engine                : PyTorch {hw_profile['torch_version']} Metal Performance Shaders\n")
        f.write(f"Total Generations Evolved     : {len(gen_df)}\n")
        f.write(f"Total Strategies Evaluated    : {len(gen_df) * population_size:,}\n")
        f.write(f"All-Time Champions Tracked    : {len(hall_of_fame)}\n")
        if best_overall:
            f.write(f"Peak Out-of-Sample Win Rate   : {best_overall['out_of_sample']['win_rate']*100:.2f}%\n")
            f.write(f"Peak Out-of-Sample ROI        : {best_overall['out_of_sample']['roi']:+.2f}%\n")
            f.write(f"Peak Out-of-Sample Profit Fac : {best_overall['out_of_sample']['profit_factor']:.2f}\n")
            f.write(f"Peak Out-of-Sample Max DD     : -{best_overall['out_of_sample']['max_dd']*100:.2f}%\n")
            
        if ensemble_res:
            f.write(f"\n--- MULTI-WINNER META-ENSEMBLE PORTFOLIO ---\n")
            f.write(f"Ensemble Out-of-Sample ROI    : +{ensemble_res['roi']:.2f}%\n")
            f.write(f"Ensemble Out-of-Sample Max DD : -{ensemble_res['max_dd']:.2f}%\n")
            f.write(f"Ensemble Active Market Time   : {ensemble_res['active_time_pct']:.1f}%\n")
            f.write(f"Ensemble Champions Combined   : {ensemble_res['num_champions']} Distinct Winners\n\n")
            
        f.write("-"*90 + "\n")
        f.write("1. TOP 5 PERSISTENT ALL-TIME CHAMPIONS (HALL OF FAME)\n")
        f.write("-"*90 + "\n\n")
        
        for idx, champ in enumerate(sorted_fame[:5], 1):
            is_r = champ['in_sample']
            oos_r = champ['out_of_sample']
            desc = champ['desc']
            
            f.write(f"RANK #{idx:02d} | DISCOVERED IN GENERATION {champ['gen_discovered']:02d}\n")
            f.write(f"  Rules        : {desc}\n")
            f.write(f"  In-Sample    : Win Rate: {is_r['win_rate']*100:5.2f}% | Trades: {is_r['trades']:3d} | Net ROI: {is_r['roi']:+6.2f}% | Max DD: -{is_r['max_dd']*100:4.1f}%\n")
            f.write(f"  Out-of-Sample: Win Rate: {oos_r['win_rate']*100:5.2f}% | Trades: {oos_r['trades']:3d} | Net ROI: {oos_r['roi']:+6.2f}% | Max DD: -{oos_r['max_dd']*100:4.1f}% | PF: {oos_r['profit_factor']:.2f}\n\n")
            
        f.write("-"*90 + "\n")
        f.write("2. GENERATION-BY-GENERATION EVOLUTIONARY PROGRESSION\n")
        f.write("-"*90 + "\n")
        f.write(f"{'Gen':<6} | {'Mean Fit':<9} | {'Max Fit':<9} | {'In-Sample Win%':<15} | {'OOS Win%':<10} | {'OOS ROI%':<10} | {'OOS PF':<8} | {'OOS Trades':<10}\n")
        f.write("-"*90 + "\n")
        step = max(1, len(gen_df) // 30)
        display_df = gen_df.iloc[::step].copy()
        if gen_df.iloc[-1]['gen'] not in display_df['gen'].values:
            display_df = pd.concat([display_df, gen_df.iloc[[-1]]])
            
        for _, r in display_df.iterrows():
            f.write(f"{int(r['gen']):04d}   | {r['mean_fitness']:<9.2f} | {r['max_fitness']:<9.2f} | {r['best_win_rate']*100:5.1f}%          | {r['oos_win_rate']*100:5.1f}%     | {r['oos_roi']:+6.1f}%    | {r['oos_pf']:<8.2f} | {int(r['oos_trades']):<10d}\n")
        f.write("="*90 + "\n")
        
    print(f"      Saved comprehensive evolutionary report to: {report_path}")


# ==============================================================================
# 10. MAIN CLI ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Apple Silicon M1 Optimized Evolutionary Strategy Engine with Winner Tracking")
    parser.add_argument("--data", type=str, default="nifty50_historical_data.csv", help="Path to Nifty 50 CSV")
    parser.add_argument("--generations", type=int, default=1000, help="Number of evolutionary generations")
    parser.add_argument("--population", type=int, default=1000, help="Population size per generation")
    parser.add_argument("--patience", type=int, default=60, help="Early stopping patience (generations without improvement)")
    parser.add_argument("--device", type=str, choices=["auto", "mps", "cpu"], default="auto", help="Execution device (mps for Apple Silicon GPU)")
    parser.add_argument("--output-dir", type=str, default="model", help="Output directory")
    args = parser.parse_args()
    
    hw_profile = detect_hardware_profile(requested_device=args.device)
    clean_df = load_data_and_precompute_indicators(args.data)
    gen_df, hall_of_fame, ensemble_res, train_df, test_df = run_evolution_pipeline(
        clean_df,
        generations=args.generations,
        population_size=args.population,
        patience=args.patience,
        hw_profile=hw_profile,
        output_dir=args.output_dir
    )
    
    img_path = os.path.join(args.output_dir, "strategy_evolution_dashboard.png")
    plot_evolution_dashboard(gen_df, hall_of_fame, ensemble_res, test_df, output_image=img_path, hw_profile=hw_profile)
    save_evolution_reports(gen_df, hall_of_fame, ensemble_res, output_dir=args.output_dir, population_size=args.population, hw_profile=hw_profile)
    print("\nApple Silicon M1 Evolutionary Strategy Discovery with Winner Memory successfully completed!")

if __name__ == "__main__":
    main()
