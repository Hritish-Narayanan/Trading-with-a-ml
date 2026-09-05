#!/usr/bin/env python3
"""
Nifty 50 Systematic 5-Year Rolling Walk-Forward & Self-Correcting ML Pipeline
=============================================================================
Methodology:
1. Rolls a systematic 5-Year training window across the entire 19-year dataset
   (e.g., Train on [T - 5yr, T] -> Predict unseen forward period [T, T + Step]).
2. In each cycle, the model fine-tunes hyperparameters, adapts to changing market
   regimes (bull, bear, high volatility), and optimizes decision thresholds.
3. Tests on strictly unseen forward market sessions.
4. Self-Corrects: Incorporates the newly observed market outcomes, adjusts its
   feature weights and model selection, and rolls forward to the next cycle.
5. Generates cumulative out-of-sample equity curves, rolling accuracy tracking,
   comprehensive multi-panel visual dashboard, and predictions CSV.
"""

import os
import sys
import time
import argparse
import random
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# Configure plotting styles
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.9

def load_and_engineer_full_features(filepath="nifty50_historical_data.csv", horizon=1):
    if not os.path.exists(filepath):
        parent_filepath = os.path.join("..", filepath)
        if os.path.exists(parent_filepath):
            filepath = parent_filepath
        else:
            raise FileNotFoundError(f"Dataset not found at {filepath}")
            
    print(f"[1/4] Loading master dataset: {filepath}")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.dropna(subset=['Close'])
    
    print(f"      Total records: {len(df)} sessions ({df.index[0].date()} to {df.index[-1].date()})")
    print(f"[2/4] Engineering multi-regime technical and market structure features (Horizon: {horizon}d)...")
    
    data = df.copy()
    
    # 1. Price Momentum & Lagged Returns
    for lag in [1, 2, 3, 5, 10, 21]:
        data[f'ret_{lag}d'] = data['Close'].pct_change(lag)
        
    # 2. Intraday Dynamics
    data['normalized_spread'] = (data['High'] - data['Low']) / data['Close']
    data['close_to_open'] = (data['Close'] - data['Open']) / data['Open']
    data['overnight_gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
    
    # 3. Moving Average Divergences
    for ma in [5, 10, 20, 50, 200]:
        sma = data['Close'].rolling(ma).mean()
        data[f'dist_sma_{ma}'] = (data['Close'] - sma) / sma
        
    # MACD
    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['macd'] = (ema_12 - ema_26) / data['Close']
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_hist'] = data['macd'] - data['macd_signal']
    
    # 4. RSI(14)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    data['rsi_14'] = 100 - (100 / (1 + rs))
    
    # 5. Volatility & Bands
    data['vol_5d'] = data['ret_1d'].rolling(5).std() * np.sqrt(252)
    data['vol_20d'] = data['ret_1d'].rolling(20).std() * np.sqrt(252)
    data['vol_ratio'] = data['vol_5d'] / (data['vol_20d'] + 1e-9)
    
    bb_mid = data['Close'].rolling(20).mean()
    bb_std = data['Close'].rolling(20).std()
    data['bb_pct_b'] = (data['Close'] - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9)
    data['bb_width'] = (4 * bb_std) / (bb_mid + 1e-9)
    
    tr1 = data['High'] - data['Low']
    tr2 = (data['High'] - data['Close'].shift(1)).abs()
    tr3 = (data['Low'] - data['Close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data['atr_14_norm'] = tr.rolling(14).mean() / data['Close']
    
    # 6. Regime & Mean-Reversion Streaks
    data['regime_bull_200'] = (data['Close'] > data['Close'].rolling(200).mean()).astype(int)
    ret_sign = np.sign(data['ret_1d'].fillna(0))
    data['streak'] = ret_sign.groupby((ret_sign != ret_sign.shift()).cumsum()).cumsum()
    data['williams_r'] = (data['High'].rolling(14).max() - data['Close']) / (data['High'].rolling(14).max() - data['Low'].rolling(14).min() + 1e-9)
    
    # 7. Volume Dynamics
    vol_sma20 = data['Volume'].rolling(20).mean()
    data['volume_ratio'] = data['Volume'] / (vol_sma20 + 1.0)
    data['volume_ret_interaction'] = data['ret_1d'] * data['volume_ratio']
    
    # 8. Calendar
    data['day_of_week'] = data.index.dayofweek
    data['day_of_month'] = data.index.day
    ym = data.index.to_period('M')
    data['trading_day_of_month'] = data.groupby(ym).cumcount() + 1
    data['is_tom_window'] = (data['trading_day_of_month'] <= 5).astype(int)
    
    # 9. Target: Forward Horizon Direction (1 = Up, 0 = Down)
    data['target'] = (data['Close'].shift(-horizon) > data['Close']).astype(int)
    data['next_day_ret'] = data['Close'].pct_change().shift(-1).fillna(0.0)
    data['fwd_horizon_ret'] = data['Close'].pct_change(horizon).shift(-horizon).fillna(0.0)
    
    clean_data = data.dropna().copy()
    meta_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'target', 'next_day_ret', 'fwd_horizon_ret']
    feature_cols = [c for c in clean_data.columns if c not in meta_cols]
    
    print(f"      Clean feature matrix: {len(clean_data)} bars with {len(feature_cols)} predictors.")
    return clean_data, feature_cols

class SoftVotingEnsemble:
    """
    Blends probabilities from diverse model families (Tree + Boosting + Linear)
    to minimize prediction variance and maximize out-of-sample generalization.
    """
    def __init__(self, models_dict):
        self.models = models_dict
        
    def fit(self, X_unscaled, X_scaled, y):
        for name, item in self.models.items():
            X = X_scaled if item['needs_scale'] else X_unscaled
            item['model'].fit(X, y)
            
    def predict_proba(self, X_unscaled, X_scaled):
        probs = []
        for name, item in self.models.items():
            X = X_scaled if item['needs_scale'] else X_unscaled
            probs.append(item['model'].predict_proba(X)[:, 1])
        return np.mean(probs, axis=0)

def tune_single_window(X_tr, y_tr, X_val, y_val, n_trials=16):
    """
    Rapid, effective hyperparameter tuning and ensembling on validation fold for 5-year lookback.
    """
    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    
    best_acc = 0.0
    best_clf = None
    best_thresh = 0.50
    best_name = ""
    best_needs_scale = False
    
    candidates = [
        # 1. Gradient Boosting (Tree Boosting)
        ("HistGradientBoosting", HistGradientBoostingClassifier(learning_rate=0.025, max_iter=70, max_depth=3, min_samples_leaf=25, l2_regularization=2.5, random_state=42), False),
        ("GradientBoosting_Shallow", GradientBoostingClassifier(n_estimators=70, learning_rate=0.03, max_depth=2, subsample=0.8, random_state=43), False),
        # 2. Random Forests & Extra Trees
        ("ExtraTrees_Regularized", ExtraTreesClassifier(n_estimators=120, max_depth=4, min_samples_leaf=4, max_features='sqrt', random_state=44, n_jobs=-1), False),
        ("RandomForest_Balanced", RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, class_weight='balanced', random_state=45, n_jobs=-1), False),
        # 3. Regularized Linear Classifiers
        ("LogisticRegression_L2", LogisticRegression(C=0.02, max_iter=500, random_state=46), True),
        ("LogisticRegression_StrongL2", LogisticRegression(C=0.005, max_iter=500, random_state=47), True),
        # 4. Additional Variants
        ("HistGradientBoosting_Slow", HistGradientBoostingClassifier(learning_rate=0.015, max_iter=90, max_depth=2, min_samples_leaf=35, l2_regularization=3.0, random_state=48), False),
        ("ExtraTrees_Deep", ExtraTreesClassifier(n_estimators=140, max_depth=5, min_samples_leaf=3, max_features=0.4, random_state=49, n_jobs=-1), False),
    ]
    
    fitted_candidates = {}
    
    for name, clf, needs_scale in candidates[:n_trials]:
        X_t = X_tr_s if needs_scale else X_tr
        X_v = X_val_s if needs_scale else X_val
        
        clf.fit(X_t, y_tr)
        
        if hasattr(clf, 'predict_proba'):
            val_proba = clf.predict_proba(X_v)[:, 1]
            thresholds = np.linspace(0.44, 0.56, 13)
            accs = [accuracy_score(y_val, (val_proba >= t).astype(int)) for t in thresholds]
            idx = int(np.argmax(accs))
            thresh = float(thresholds[idx])
            acc = accs[idx]
        else:
            y_pred = clf.predict(X_v)
            val_proba = np.full(len(y_val), 0.50)
            acc = accuracy_score(y_val, y_pred)
            thresh = 0.50
            
        fitted_candidates[name] = {
            'model': clf,
            'needs_scale': needs_scale,
            'acc': acc,
            'thresh': thresh,
            'val_proba': val_proba
        }
        
        if acc > best_acc:
            best_acc = acc
            best_clf = clf
            best_thresh = thresh
            best_name = name
            best_needs_scale = needs_scale
            
    # 5. Build Soft Voting Ensemble of Top 3 Diverse Models (Boosting + Tree + Linear)
    ens_models = {
        'HGB': fitted_candidates['HistGradientBoosting'],
        'ET': fitted_candidates['ExtraTrees_Regularized'],
        'LR': fitted_candidates['LogisticRegression_L2']
    }
    ens_proba = np.mean([item['val_proba'] for item in ens_models.values()], axis=0)
    thresholds = np.linspace(0.45, 0.55, 11)
    ens_accs = [accuracy_score(y_val, (ens_proba >= t).astype(int)) for t in thresholds]
    best_ens_idx = int(np.argmax(ens_accs))
    ens_acc = ens_accs[best_ens_idx]
    ens_thresh = float(thresholds[best_ens_idx])
    
    # If Ensemble matches or beats best individual model, select Ensemble for superior out-of-sample stability
    if ens_acc >= best_acc - 0.01:
        best_clf = SoftVotingEnsemble(ens_models)
        best_name = "VotingEnsemble(HGB+ET+LR)"
        best_thresh = ens_thresh
        best_acc = ens_acc
        best_needs_scale = False
            
    return best_clf, best_name, best_thresh, best_needs_scale, best_acc

def execute_self_correcting_walk_forward(df, feature_cols, window_years=5, step_months=6, n_trials=12):
    """
    Executes systematic rolling 5-year Walk-Forward Optimization across the entire history.
    """
    print(f"\n[3/4] Launching Systematic Self-Correcting Walk-Forward Engine...")
    print(f"      Lookback Window : {window_years} Years (Fixed Rolling Baseline)")
    print(f"      Forward Step    : {step_months} Months (Self-Correction Re-tuning Frequency)")
    
    min_date = df.index.min()
    max_date = df.index.max()
    first_train_end = min_date + pd.DateOffset(years=window_years)
    
    current_train_end = first_train_end
    cycle_idx = 1
    
    all_predictions = []
    cycle_summary = []
    
    while current_train_end < max_date:
        train_start = current_train_end - pd.DateOffset(years=window_years)
        test_end = min(current_train_end + pd.DateOffset(months=step_months), max_date)
        
        # Slices
        train_mask = (df.index >= train_start) & (df.index < current_train_end)
        test_mask = (df.index >= current_train_end) & (df.index < test_end)
        
        train_df = df.loc[train_mask]
        test_df = df.loc[test_mask]
        
        if len(test_df) < 10:
            break
            
        # Chronological split of 5-year lookback: First 80% train, last 20% validation
        n_train_total = len(train_df)
        n_sub_train = int(n_train_total * 0.80)
        
        sub_train_df = train_df.iloc[:n_sub_train]
        val_df = train_df.iloc[n_sub_train:]
        
        X_sub_tr, y_sub_tr = sub_train_df[feature_cols], sub_train_df['target']
        X_val, y_val = val_df[feature_cols], val_df['target']
        
        # 1. Hyperparameter Tuning & Threshold Optimization on lookback window
        champion_clf, champion_name, opt_thresh, needs_scale, val_acc = tune_single_window(
            X_sub_tr, y_sub_tr, X_val, y_val, n_trials=n_trials
        )
        
        # 2. Self-Correction & Refitting on full 5-year window
        X_full_train = train_df[feature_cols]
        y_full_train = train_df['target']
        
        scaler = RobustScaler()
        if needs_scale:
            X_full_train_eval = scaler.fit_transform(X_full_train)
            X_test_eval = scaler.transform(test_df[feature_cols])
        else:
            X_full_train_eval = X_full_train
            X_test_eval = test_df[feature_cols]
            
        if isinstance(champion_clf, SoftVotingEnsemble):
            champion_clf.fit(X_full_train, X_full_train_eval, y_full_train)
            test_probs = champion_clf.predict_proba(test_df[feature_cols], X_test_eval)
        else:
            champion_clf.fit(X_full_train_eval, y_full_train)
            if hasattr(champion_clf, 'predict_proba'):
                test_probs = champion_clf.predict_proba(X_test_eval)[:, 1]
            else:
                test_probs = np.full(len(test_df), 0.50)
                
        test_preds = (test_probs >= opt_thresh).astype(int)
        test_acc = accuracy_score(test_df['target'], test_preds)
        majority_baseline = max(test_df['target'].mean(), 1.0 - test_df['target'].mean())
        
        # High confidence subset
        hc_mask = (test_probs >= 0.53) | (test_probs <= 0.47)
        hc_acc = accuracy_score(test_df['target'][hc_mask], test_preds[hc_mask]) if hc_mask.sum() > 5 else test_acc
        
        # Log Cycle Details
        print(f"  Cycle {cycle_idx:02d} | Train: {train_start.strftime('%Y-%m')} to {current_train_end.strftime('%Y-%m')} "
              f"-> Test: {current_train_end.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')} ({len(test_df):3d} bars) | "
              f"Model: {champion_name:<25} | OOS Acc: {test_acc*100:5.2f}% (High-Conf: {hc_acc*100:5.1f}%) | Base: {majority_baseline*100:5.1f}%")
              
        cycle_summary.append({
            'cycle': cycle_idx,
            'train_start': train_start.date(),
            'train_end': current_train_end.date(),
            'test_start': test_df.index[0].date(),
            'test_end': test_df.index[-1].date(),
            'model_name': champion_name,
            'val_acc': val_acc,
            'opt_threshold': opt_thresh,
            'test_acc': test_acc,
            'hc_acc': hc_acc,
            'majority_baseline': majority_baseline,
            'num_test_bars': len(test_df)
        })
        
        # Record session-by-session predictions
        cycle_preds = pd.DataFrame({
            'Date': test_df.index,
            'Close': test_df['Close'],
            'Actual_Target': test_df['target'],
            'Pred_Direction': test_preds,
            'Pred_Prob_Up': test_probs,
            'Next_Day_Return': test_df['next_day_ret'],
            'Cycle': cycle_idx,
            'Model': champion_name
        }).set_index('Date')
        
        all_predictions.append(cycle_preds)
        
        # 4. Self-Correcting Roll: Advance current_train_end by step_months
        current_train_end = current_train_end + pd.DateOffset(months=step_months)
        cycle_idx += 1
        
    full_preds_df = pd.concat(all_predictions)
    summary_df = pd.DataFrame(cycle_summary)
    
    # Calculate overall Walk-Forward Out-of-Sample Metrics
    total_oos_acc = accuracy_score(full_preds_df['Actual_Target'], full_preds_df['Pred_Direction'])
    overall_baseline = max(full_preds_df['Actual_Target'].mean(), 1.0 - full_preds_df['Actual_Target'].mean())
    total_auc = roc_auc_score(full_preds_df['Actual_Target'], full_preds_df['Pred_Prob_Up'])
    cm = confusion_matrix(full_preds_df['Actual_Target'], full_preds_df['Pred_Direction'])
    
    # Confidence tier progression
    conf_tiers = []
    for delta in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
        mask = (full_preds_df['Pred_Prob_Up'] >= 0.50 + delta) | (full_preds_df['Pred_Prob_Up'] <= 0.50 - delta)
        if mask.sum() > 0:
            acc_c = accuracy_score(full_preds_df.loc[mask, 'Actual_Target'], full_preds_df.loc[mask, 'Pred_Direction'])
            conf_tiers.append({
                'delta': delta,
                'min_prob': 0.50 + delta,
                'accuracy': acc_c,
                'count': int(mask.sum()),
                'pct_days': float(mask.mean() * 100.0)
            })
    conf_df = pd.DataFrame(conf_tiers)
    
    # Simulate Strategy Return
    full_preds_df['Strategy_Return'] = np.where(full_preds_df['Pred_Direction'] == 1, full_preds_df['Next_Day_Return'], 0.0)
    full_preds_df['Cum_Strategy'] = (1.0 + full_preds_df['Strategy_Return']).cumprod()
    full_preds_df['Cum_Benchmark'] = (1.0 + full_preds_df['Next_Day_Return']).cumprod()
    
    total_strat_ret = (full_preds_df['Cum_Strategy'].iloc[-1] - 1.0) * 100.0
    total_bnh_ret = (full_preds_df['Cum_Benchmark'].iloc[-1] - 1.0) * 100.0
    
    print(f"\n" + "="*85)
    print(f"      SYSTEMATIC WALK-FORWARD SELF-CORRECTING RESULTS SUMMARY")
    print(f"="*85)
    print(f"      Total Out-of-Sample Cycles     : {len(summary_df)} adaptive cycles")
    print(f"      Total Out-of-Sample Period     : {full_preds_df.index[0].date()} to {full_preds_df.index[-1].date()} (~{(full_preds_df.index[-1]-full_preds_df.index[0]).days/365.25:.1f} years)")
    print(f"      Total Evaluated Trading Days   : {len(full_preds_df):,} sessions")
    print(f"      Overall Out-of-Sample Accuracy : {total_oos_acc*100:6.2f}%")
    print(f"      Overall ROC-AUC Score          : {total_auc:6.4f}")
    print(f"      Walk-Forward Strategy Return   : {total_strat_ret:+.2f}%")
    print(f"      Buy-and-Hold Benchmark Return  : {total_bnh_ret:+.2f}%\n")
    print(f"      FINE-TUNING ACCURACY BY CONVICTION FILTER:")
    print(f"      ------------------------------------------------------------")
    for _, r in conf_df.iterrows():
        print(f"      Confidence >= {r['min_prob']*100:.0f}% (|P-0.5| >= {r['delta']:.2f}) -> Accuracy: {r['accuracy']*100:5.2f}% on {int(r['count']):,} sessions ({r['pct_days']:.1f}% of days)")
    print(f"="*85 + "\n")
    
    return full_preds_df, summary_df, conf_df, {
        'total_oos_acc': total_oos_acc,
        'overall_baseline': overall_baseline,
        'total_auc': total_auc,
        'cm': cm,
        'total_strat_ret': total_strat_ret,
        'total_bnh_ret': total_bnh_ret
    }

def plot_walk_forward_dashboard(full_preds_df, summary_df, conf_df, metrics, output_image="model/self_correcting_walk_forward.png"):
    """
    Renders high-resolution visual dashboard of the walk-forward self-correcting model.
    """
    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)
    
    # 1. Panel A: Cycle-by-Cycle Out-of-Sample Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    x_ticks = np.arange(len(summary_df))
    bars = ax1.bar(x_ticks, summary_df['test_acc'] * 100, color='#0284c7', width=0.65, label='Cycle OOS Accuracy', edgecolor='#0f172a', alpha=0.85)
    ax1.plot(x_ticks, summary_df['majority_baseline'] * 100, color='#ef4444', linestyle='--', linewidth=2, label='Majority Class Baseline')
    ax1.axhline(metrics['total_oos_acc'] * 100, color='#16a34a', linestyle='-', linewidth=2, label=f"Overall Mean Acc ({metrics['total_oos_acc']*100:.1f}%)")
    
    for bar, acc, base in zip(bars, summary_df['test_acc'], summary_df['majority_baseline']):
        if acc > base:
            bar.set_color('#10b981')
            
    ax1.set_title('A. Period-by-Period Out-of-Sample Accuracy Across Self-Correction Cycles', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Self-Correction Cycle #', fontsize=10)
    ax1.set_ylabel('Accuracy (%)', fontsize=10)
    ax1.set_xticks(x_ticks[::2])
    ax1.set_xticklabels([f"C{c}\n{str(d)[:7]}" for c, d in zip(summary_df['cycle'][::2], summary_df['test_start'][::2])], fontsize=8)
    ax1.legend(loc='lower left', frameon=True, fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 2. Panel B: Accuracy vs Prediction Conviction (Fine-Tuning Curve)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(conf_df['min_prob'] * 100, conf_df['accuracy'] * 100, marker='o', markersize=8, color='#0284c7', linewidth=2.8, label='Out-of-Sample Accuracy')
    ax2.axhline(metrics['overall_baseline'] * 100, color='#ef4444', linestyle='--', linewidth=1.8, label=f"Market Baseline ({metrics['overall_baseline']*100:.1f}%)")
    for _, r in conf_df.iterrows():
        ax2.annotate(f"{r['accuracy']*100:.1f}%\n({r['pct_days']:.0f}% days)",
                     (r['min_prob'] * 100, r['accuracy'] * 100),
                     textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8.5, fontweight='bold')
    ax2.set_title('B. Fine-Tuning Accuracy: Impact of Signal Conviction Thresholding', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Minimum Probability Threshold (%) [Conviction Level]', fontsize=10)
    ax2.set_ylabel('Realized Accuracy (%)', fontsize=10)
    ax2.legend(loc='lower right', frameon=True, fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # 3. Panel C: Multi-Year Walk-Forward Strategy Return vs Buy-and-Hold
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(full_preds_df.index, full_preds_df['Cum_Strategy'], color='#0284c7', linewidth=2.2, label=f"Self-Correcting ML ({metrics['total_strat_ret']:+.1f}%)")
    ax3.plot(full_preds_df.index, full_preds_df['Cum_Benchmark'], color='#64748b', linewidth=1.8, linestyle='--', label=f"Nifty 50 Buy-&-Hold ({metrics['total_bnh_ret']:+.1f}%)")
    ax3.set_title('C. Multi-Year Walk-Forward Out-of-Sample Strategy Compounding', fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_ylabel('Cumulative Multiplier (Base 1.0)', fontsize=10)
    ax3.legend(loc='upper left', frameon=True, fontsize=9)
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # 4. Panel D: Multi-Year Confusion Matrix Heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    cm = metrics['cm']
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    annot = np.array([[f"{cm[0,0]:,}\n({cm_norm[0,0]:.1f}%)", f"{cm[0,1]:,}\n({cm_norm[0,1]:.1f}%)"],
                      [f"{cm[1,0]:,}\n({cm_norm[1,0]:.1f}%)", f"{cm[1,1]:,}\n({cm_norm[1,1]:.1f}%)"]])
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False, ax=ax4,
                xticklabels=['Pred Down', 'Pred Up'], yticklabels=['Actual Down', 'Actual Up'],
                annot_kws={"size": 12, "weight": "bold"})
    ax4.set_title(f"D. Overall Confusion Matrix Across {len(full_preds_df):,} Out-of-Sample Days", fontsize=12, fontweight='bold', pad=10)
    
    fig.suptitle(f"Systematic 5-Year Rolling Walk-Forward & Adaptive Self-Correcting ML Model on Nifty 50\n"
                 f"Evaluated Across {len(summary_df)} Consecutive Regimes (2013 – 2026) | High-Conviction Accuracy: {conf_df['accuracy'].max()*100:.1f}%",
                 fontsize=14, fontweight='bold', y=0.98)
                 
    plt.savefig(output_image, bbox_inches='tight')
    plt.close()
    print(f"      Saved comprehensive visual dashboard to: {output_image}")

def save_walk_forward_reports(full_preds_df, summary_df, conf_df, metrics, output_dir="model"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Predictions CSV
    preds_csv = os.path.join(output_dir, "walk_forward_predictions.csv")
    full_preds_df.to_csv(preds_csv)
    print(f"      Saved all {len(full_preds_df):,} out-of-sample predictions to: {preds_csv}")
    
    # 2. Text Report
    report_path = os.path.join(output_dir, "self_correcting_report.txt")
    with open(report_path, "w") as f:
        f.write("="*85 + "\n")
        f.write("     NIFTY 50 SYSTEMATIC 5-YEAR ROLLING WALK-FORWARD & SELF-CORRECTING REPORT\n")
        f.write("="*85 + "\n\n")
        f.write(f"Total Out-of-Sample Cycles      : {len(summary_df)} consecutive regimes\n")
        f.write(f"Total Out-of-Sample Period      : {full_preds_df.index[0].date()} to {full_preds_df.index[-1].date()}\n")
        f.write(f"Total Unseen Sessions Evaluated : {len(full_preds_df):,} trading sessions\n")
        f.write(f"Overall Out-of-Sample Accuracy  : {metrics['total_oos_acc']*100:.2f}%\n")
        f.write(f"Overall Majority Class Baseline : {metrics['overall_baseline']*100:.2f}%\n")
        f.write(f"Overall ROC-AUC Score           : {metrics['total_auc']:.4f}\n")
        f.write(f"Walk-Forward Cumulative Return  : {metrics['total_strat_ret']:+.2f}%\n")
        f.write(f"Nifty 50 Buy-and-Hold Return    : {metrics['total_bnh_ret']:+.2f}%\n\n")
        f.write("-"*85 + "\n")
        f.write("ACCURACY FINE-TUNING VIA SIGNAL CONVICTION THRESHOLDING\n")
        f.write("-"*85 + "\n")
        f.write(f"{'Min Confidence':<18} | {'Prob Filter':<15} | {'Realized OOS Acc':<18} | {'Covered Days':<18}\n")
        f.write("-"*85 + "\n")
        for _, r in conf_df.iterrows():
            f.write(f"Conviction >= {r['min_prob']*100:4.1f}%  | |P-0.5| >= {r['delta']:.2f} | {r['accuracy']*100:6.2f}%            | {int(r['count']):,} days ({r['pct_days']:4.1f}%)\n")
        f.write("\n" + "-"*85 + "\n")
        f.write("PERIOD-BY-PERIOD ADAPTIVE SELF-CORRECTION CYCLE BREAKDOWN\n")
        f.write("-"*85 + "\n")
        f.write(f"{'Cycle':<6} | {'Test Horizon':<23} | {'Champion Architecture':<27} | {'Val Acc':<8} | {'OOS Acc':<8} | {'Base Acc':<8} | {'Edge':<6}\n")
        f.write("-"*85 + "\n")
        for _, row in summary_df.iterrows():
            edge = (row['test_acc'] - row['majority_baseline']) * 100
            edge_str = f"{edge:+5.1f}%"
            f.write(f"{row['cycle']:02d}     | {str(row['test_start'])} to {str(row['test_end'])} | {row['model_name']:<27} | {row['val_acc']*100:5.1f}%  | {row['test_acc']*100:5.1f}%  | {row['majority_baseline']*100:5.1f}%  | {edge_str}\n")
        f.write("="*85 + "\n")
    print(f"      Saved detailed textual report to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Systematic 5-Year Rolling Walk-Forward Self-Correcting Model on Nifty 50")
    parser.add_argument("--data", type=str, default="nifty50_historical_data.csv", help="Path to Nifty 50 CSV")
    parser.add_argument("--window-years", type=int, default=5, help="Rolling Lookback Window in Years")
    parser.add_argument("--step-months", type=int, default=6, help="Re-tuning and Forward Step Frequency in Months")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon in days (e.g. 1 for daily, 3 for swing, 5 for weekly)")
    parser.add_argument("--output-dir", type=str, default="model", help="Output directory")
    args = parser.parse_args()
    
    # 1. Load & Engineer
    clean_df, feat_cols = load_and_engineer_full_features(args.data, horizon=args.horizon)
    
    # 2. Run Systematic Walk-Forward Self-Correction
    full_preds, summary_df, conf_df, metrics = execute_self_correcting_walk_forward(
        clean_df, feat_cols, window_years=args.window_years, step_months=args.step_months, n_trials=args.trials
    )
    
    # 3. Generate Visuals and Reports
    img_path = os.path.join(args.output_dir, "self_correcting_walk_forward.png")
    plot_walk_forward_dashboard(full_preds, summary_df, conf_df, metrics, output_image=img_path)
    save_walk_forward_reports(full_preds, summary_df, conf_df, metrics, output_dir=args.output_dir)
    print("\nWalk-forward self-correcting pipeline completed successfully!")

if __name__ == "__main__":
    main()
