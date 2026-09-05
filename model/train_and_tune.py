#!/usr/bin/env python3
"""
Nifty 50 Machine Learning Training & Hyperparameter Tuning Pipeline
====================================================================
1. Samples a random contiguous 5-year interval from nifty50_historical_data.csv
2. Engineers comprehensive price-action, momentum, volatility, and calendar features
3. Splits into Chronological Train (70%), Validation (15%), and Holdout Test (15%)
4. Fine-tunes multiple ML models (XGBoost, Random Forest, ExtraTrees, HistGradientBoosting)
   across hyperparameters to maximize accuracy
5. Evaluates best model on unseen test set and outputs reports, charts, and serialized weights
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

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
    GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression

try:
    import xgboost as xgb
    # Test if libxgboost dynamic library can actually load without error
    _test = xgb.XGBClassifier()
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    xgb = None

warnings.filterwarnings("ignore")

# Configure plotting styles
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.9

def load_full_dataset(filepath="nifty50_historical_data.csv"):
    if not os.path.exists(filepath):
        # Check parent folder if running from within model/
        parent_filepath = os.path.join("..", filepath)
        if os.path.exists(parent_filepath):
            filepath = parent_filepath
        else:
            raise FileNotFoundError(f"Dataset not found at {filepath} or {parent_filepath}")
            
    print(f"[1/6] Loading master dataset from: {filepath}")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.dropna(subset=['Close'])
    print(f"      Total records available: {len(df)} sessions ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
    return df

def sample_random_5_year_window(df, seed=None, start_date_str=None):
    """
    Samples an exact continuous 5-year interval (start_date to start_date + 5 years).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    min_date = df.index.min()
    max_date = df.index.max()
    latest_valid_start = max_date - pd.DateOffset(years=5)
    
    if latest_valid_start <= min_date:
        raise ValueError("Dataset is shorter than 5 years!")
        
    if start_date_str:
        chosen_start = pd.to_datetime(start_date_str)
        if chosen_start < min_date or chosen_start > latest_valid_start:
            raise ValueError(f"Start date {start_date_str} is out of valid bounds ({min_date.date()} to {latest_valid_start.date()})")
    else:
        # Pick a random date between min_date and latest_valid_start
        all_valid_starts = df.index[(df.index >= min_date) & (df.index <= latest_valid_start)]
        chosen_start = random.choice(all_valid_starts)
        
    chosen_end = chosen_start + pd.DateOffset(years=5)
    
    df_sample = df.loc[(df.index >= chosen_start) & (df.index <= chosen_end)].copy()
    
    start_close = df_sample['Close'].iloc[0]
    end_close = df_sample['Close'].iloc[-1]
    pct_change = (end_close / start_close - 1.0) * 100.0
    
    print(f"\n[2/6] Extracted 5-Year Contiguous Interval:")
    print(f"      Start Date      : {df_sample.index[0].strftime('%Y-%m-%d')} (Close: {start_close:,.2f})")
    print(f"      End Date        : {df_sample.index[-1].strftime('%Y-%m-%d')} (Close: {end_close:,.2f})")
    print(f"      Trading Days    : {len(df_sample)} sessions")
    print(f"      5-Year Market Return: {pct_change:+.2f}%")
    
    return df_sample, chosen_start, chosen_end

def engineer_features(df):
    """
    Computes technical, price-action, momentum, and calendar features.
    No future data leakage: target is Next-Day Direction.
    """
    print("\n[3/6] Engineering technical & market structure features...")
    data = df.copy()
    
    # 1. Price Momentum & Lagged Returns
    for lag in [1, 2, 3, 5, 10, 21]:
        data[f'ret_{lag}d'] = data['Close'].pct_change(lag)
        
    # 2. Intraday Price Dynamics
    data['normalized_spread'] = (data['High'] - data['Low']) / data['Close']
    data['close_to_open'] = (data['Close'] - data['Open']) / data['Open']
    data['overnight_gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
    
    # 3. Moving Average Divergence
    for ma in [5, 10, 20, 50]:
        sma = data['Close'].rolling(ma).mean()
        data[f'dist_sma_{ma}'] = (data['Close'] - sma) / sma
        
    # Exponential Moving Average cross
    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['macd'] = (ema_12 - ema_26) / data['Close']
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_hist'] = data['macd'] - data['macd_signal']
    
    # 4. Relative Strength Index (RSI 14)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    data['rsi_14'] = 100 - (100 / (1 + rs))
    
    # 5. Volatility & Bands
    data['vol_5d'] = data['ret_1d'].rolling(5).std() * np.sqrt(252)
    data['vol_20d'] = data['ret_1d'].rolling(20).std() * np.sqrt(252)
    
    # Bollinger Bands (%B)
    bb_mid = data['Close'].rolling(20).mean()
    bb_std = data['Close'].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    data['bb_pct_b'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-9)
    
    # Normalized Average True Range (ATR)
    tr1 = data['High'] - data['Low']
    tr2 = (data['High'] - data['Close'].shift(1)).abs()
    tr3 = (data['Low'] - data['Close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data['atr_14_norm'] = tr.rolling(14).mean() / data['Close']
    
    # 6. Volume Dynamics
    vol_sma20 = data['Volume'].rolling(20).mean()
    data['volume_ratio'] = data['Volume'] / (vol_sma20 + 1.0)
    data['volume_ret_interaction'] = data['ret_1d'] * data['volume_ratio']
    
    # 7. Calendar & Anomaly Features
    data['day_of_week'] = data.index.dayofweek
    data['day_of_month'] = data.index.day
    # Trading day of month (Turn of Month anomaly: 1 if <= 5th trading session)
    ym = data.index.to_period('M')
    data['trading_day_of_month'] = data.groupby(ym).cumcount() + 1
    data['is_tom_window'] = (data['trading_day_of_month'] <= 5).astype(int)
    
    # 8. Target Variable: Next Day Direction (Binary Classification)
    # y = 1 if Close[t+1] > Close[t] else 0
    data['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    # Drop rows with NaN (from rolling calculations and last row target)
    clean_data = data.dropna().copy()
    
    # Extract feature matrix X and target y
    meta_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'target']
    feature_cols = [c for c in clean_data.columns if c not in meta_cols]
    
    X = clean_data[feature_cols]
    y = clean_data['target']
    
    print(f"      Total Cleaned Samples: {len(X)}")
    print(f"      Features Generated   : {len(feature_cols)} technical predictors")
    print(f"      Target Distribution  : Up: {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%) | Down: {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    
    return clean_data, X, y, feature_cols

def chronological_split(X, y, train_ratio=0.70, val_ratio=0.15):
    """
    Time-series chronological split to avoid any lookahead bias:
    Train: 70% | Validation: 15% | Holdout Test: 15%
    """
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_val, y_val = X.iloc[n_train:n_train + n_val], y.iloc[n_train:n_train + n_val]
    X_test, y_test = X.iloc[n_train + n_val:], y.iloc[n_train + n_val:]
    
    print(f"\n[4/6] Chronological Dataset Partitioning:")
    print(f"      Train Set     : {len(X_train)} bars ({X_train.index[0].date()} to {X_train.index[-1].date()})")
    print(f"      Validation Set: {len(X_val)} bars ({X_val.index[0].date()} to {X_val.index[-1].date()})")
    print(f"      Holdout Test  : {len(X_test)} bars ({X_test.index[0].date()} to {X_test.index[-1].date()})")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def fine_tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials=60):
    """
    Systematic hyperparameter search across multiple ML model families:
    1. XGBoost Classifier
    2. Random Forest Classifier
    3. Extra Trees Classifier
    4. HistGradientBoosting Classifier
    5. Regularized Logistic Regression
    
    Tracks validation accuracy iteratively to maximize predictive power.
    """
    print(f"\n[5/6] Commencing Hyperparameter Tuning across {n_trials} iterative configurations...")
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    trials_log = []
    best_val_acc = 0.0
    best_model_info = None
    
    # Model candidate generators
    def get_candidate(trial_idx):
        model_type = trial_idx % 4
        
        if model_type == 0:  # XGBoost / GradientBoosting
            if HAS_XGB:
                params = {
                    'model_name': 'XGBoost',
                    'max_depth': random.choice([2, 3, 4, 5, 6]),
                    'learning_rate': random.choice([0.008, 0.015, 0.03, 0.05, 0.08, 0.12]),
                    'n_estimators': random.choice([40, 70, 100, 150, 220]),
                    'subsample': random.choice([0.65, 0.75, 0.85, 0.95]),
                    'colsample_bytree': random.choice([0.4, 0.6, 0.75, 0.9]),
                    'reg_alpha': random.choice([0.0, 0.1, 0.5, 1.0, 2.0]),
                    'reg_lambda': random.choice([0.5, 1.0, 2.0, 5.0]),
                    'min_child_weight': random.choice([1, 3, 5, 7]),
                    'random_state': 42 + trial_idx,
                    'eval_metric': 'logloss'
                }
                clf = xgb.XGBClassifier(
                    max_depth=params['max_depth'],
                    learning_rate=params['learning_rate'],
                    n_estimators=params['n_estimators'],
                    subsample=params['subsample'],
                    colsample_bytree=params['colsample_bytree'],
                    reg_alpha=params['reg_alpha'],
                    reg_lambda=params['reg_lambda'],
                    min_child_weight=params['min_child_weight'],
                    random_state=params['random_state'],
                    eval_metric=params['eval_metric'],
                    n_jobs=-1
                )
                return clf, params, False
            else:
                params = {
                    'model_name': 'GradientBoosting',
                    'n_estimators': random.choice([40, 70, 100, 150]),
                    'learning_rate': random.choice([0.01, 0.03, 0.05, 0.08, 0.12]),
                    'max_depth': random.choice([2, 3, 4, 5]),
                    'subsample': random.choice([0.65, 0.75, 0.85, 1.0]),
                    'max_features': random.choice(['sqrt', 'log2', 0.6, 0.8]),
                    'random_state': 42 + trial_idx
                }
                clf = GradientBoostingClassifier(
                    n_estimators=params['n_estimators'],
                    learning_rate=params['learning_rate'],
                    max_depth=params['max_depth'],
                    subsample=params['subsample'],
                    max_features=params['max_features'],
                    random_state=params['random_state']
                )
                return clf, params, False
            
        elif model_type == 1:  # Random Forest
            params = {
                'model_name': 'RandomForest',
                'n_estimators': random.choice([50, 100, 150, 200]),
                'max_depth': random.choice([3, 4, 5, 6, 8, None]),
                'min_samples_split': random.choice([2, 4, 8, 12]),
                'min_samples_leaf': random.choice([1, 2, 4, 8]),
                'max_features': random.choice(['sqrt', 'log2', 0.5, 0.7]),
                'random_state': 42 + trial_idx
            }
            clf = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                max_features=params['max_features'],
                random_state=params['random_state'],
                n_jobs=-1
            )
            return clf, params, False
            
        elif model_type == 2:  # Extra Trees
            params = {
                'model_name': 'ExtraTrees',
                'n_estimators': random.choice([60, 120, 180, 250]),
                'max_depth': random.choice([2, 3, 4, 6, None]),
                'min_samples_split': random.choice([3, 6, 10]),
                'min_samples_leaf': random.choice([1, 3, 6]),
                'max_features': random.choice(['sqrt', 0.4, 0.6]),
                'random_state': 42 + trial_idx
            }
            clf = ExtraTreesClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                max_features=params['max_features'],
                random_state=params['random_state'],
                n_jobs=-1
            )
            return clf, params, False
            
        else:  # HistGradientBoosting / Regularized Logistic
            if trial_idx % 2 == 0:
                params = {
                    'model_name': 'HistGradientBoosting',
                    'learning_rate': random.choice([0.01, 0.03, 0.06, 0.1]),
                    'max_iter': random.choice([50, 100, 150]),
                    'max_depth': random.choice([2, 3, 4, 5]),
                    'min_samples_leaf': random.choice([10, 20, 35]),
                    'l2_regularization': random.choice([0.0, 0.5, 1.5, 3.0]),
                    'random_state': 42 + trial_idx
                }
                clf = HistGradientBoostingClassifier(
                    learning_rate=params['learning_rate'],
                    max_iter=params['max_iter'],
                    max_depth=params['max_depth'],
                    min_samples_leaf=params['min_samples_leaf'],
                    l2_regularization=params['l2_regularization'],
                    random_state=params['random_state']
                )
                return clf, params, False
            else:
                params = {
                    'model_name': 'LogisticRegression_L2',
                    'C': random.choice([0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]),
                    'random_state': 42 + trial_idx
                }
                clf = LogisticRegression(C=params['C'], random_state=params['random_state'], max_iter=1000)
                return clf, params, True

    for i in range(1, n_trials + 1):
        clf, params, needs_scale = get_candidate(i)
        
        X_tr = X_train_scaled if needs_scale else X_train
        X_v = X_val_scaled if needs_scale else X_val
        
        clf.fit(X_tr, y_train)
        
        # Calculate AUC and optimize classification probability threshold
        opt_thresh = 0.50
        try:
            val_proba = clf.predict_proba(X_v)[:, 1]
            val_auc = roc_auc_score(y_val, val_proba)
            
            # Grid search threshold to find true maximum validation accuracy
            thresholds = np.linspace(0.42, 0.58, 33)
            accs = [accuracy_score(y_val, (val_proba >= t).astype(int)) for t in thresholds]
            best_idx = int(np.argmax(accs))
            opt_thresh = float(thresholds[best_idx])
            val_acc = accs[best_idx]
        except Exception:
            y_val_pred = clf.predict(X_v)
            val_acc = accuracy_score(y_val, y_val_pred)
            val_auc = 0.5
            val_proba = None
            
        is_new_best = val_acc > best_val_acc
        if is_new_best:
            best_val_acc = val_acc
            best_model_info = {
                'model': clf,
                'params': params,
                'val_acc': val_acc,
                'val_auc': val_auc,
                'needs_scale': needs_scale,
                'opt_threshold': opt_thresh,
                'trial_idx': i
            }
            
        trials_log.append({
            'trial': i,
            'model_name': params['model_name'],
            'val_acc': val_acc,
            'val_auc': val_auc,
            'is_best': is_new_best,
            'running_max_acc': best_val_acc
        })
        
        if i % 10 == 0 or is_new_best:
            flag = "★ NEW BEST" if is_new_best else ""
            print(f"      Trial {i:02d}/{n_trials:02d} [{params['model_name']:<20}] -> Val Acc: {val_acc*100:5.2f}% | Val AUC: {val_auc:5.3f} (Thresh: {opt_thresh:.2f}) {flag}")
            
    print(f"\n      Optimization Completed!")
    print(f"      Peak Validation Accuracy: {best_val_acc*100:.2f}% (Achieved by {best_model_info['params']['model_name']} on Trial #{best_model_info['trial_idx']} with threshold {best_model_info['opt_threshold']:.2f})")
    
    return best_model_info, pd.DataFrame(trials_log), scaler

def evaluate_holdout_test(best_model_info, scaler, X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, raw_df):
    """
    Evaluates the champion tuned model on the untouched holdout test dataset.
    Generates full statistical metrics, confusion matrix, ROC-AUC, feature importances,
    and a trading strategy simulation.
    """
    print(f"\n[6/6] Evaluating Champion Model on Unseen Holdout Test Dataset ({len(X_test)} sessions)...")
    
    clf = best_model_info['model']
    needs_scale = best_model_info['needs_scale']
    opt_thresh = best_model_info.get('opt_threshold', 0.50)
    
    # Retrain best architecture on Train + Validation combined for maximum sample efficiency
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    
    if needs_scale:
        full_scaler = RobustScaler()
        X_train_val_s = full_scaler.fit_transform(X_train_val)
        X_test_eval = full_scaler.transform(X_test)
        clf.fit(X_train_val_s, y_train_val)
        model_scaler = full_scaler
    else:
        clf.fit(X_train_val, y_train_val)
        X_test_eval = X_test
        model_scaler = None
        
    # Inference on Holdout Test Set using optimized decision threshold
    if hasattr(clf, 'predict_proba'):
        y_prob = clf.predict_proba(X_test_eval)[:, 1]
        y_pred = (y_prob >= opt_thresh).astype(int)
    else:
        y_pred = clf.predict(X_test_eval)
        y_prob = None
    
    test_acc = accuracy_score(y_test, y_pred)
    majority_class_baseline = max(y_test.mean(), 1 - y_test.mean())
    test_prec = precision_score(y_test, y_pred, zero_division=0)
    test_rec = recall_score(y_test, y_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    test_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.5
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n      =======================================================")
    print(f"      HOLDOUT TEST EVALUATION RESULTS:")
    print(f"      =======================================================")
    print(f"      Test Accuracy            : {test_acc*100:6.2f}%")
    print(f"      Majority Class Baseline  : {majority_class_baseline*100:6.2f}%")
    print(f"      Accuracy vs Baseline     : {test_acc - majority_class_baseline:+6.2f}%")
    print(f"      ROC-AUC Score            : {test_auc:6.4f}")
    print(f"      Precision (Up Days)      : {test_prec*100:6.2f}%")
    print(f"      Recall (Up Days)         : {test_rec*100:6.2f}%")
    print(f"      F1-Score                 : {test_f1:6.4f}")
    print(f"      Confusion Matrix         :\n{cm}")
    print(f"      =======================================================\n")
    
    # Feature Importances
    feat_imp = None
    if hasattr(clf, 'feature_importances_'):
        feat_imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    elif hasattr(clf, 'coef_'):
        feat_imp = pd.Series(np.abs(clf.coef_[0]), index=feature_cols).sort_values(ascending=False)
        
    # Trading Simulation on Test Set
    test_indices = X_test.index
    test_df = raw_df.loc[test_indices].copy()
    test_df['pred_signal'] = y_pred
    test_df['next_day_ret'] = test_df['Close'].pct_change().shift(-1).fillna(0.0)
    test_df['strat_ret'] = np.where(test_df['pred_signal'] == 1, test_df['next_day_ret'], 0.0)
    
    test_df['cum_bnh'] = (1.0 + test_df['next_day_ret']).cumprod()
    test_df['cum_strat'] = (1.0 + test_df['strat_ret']).cumprod()
    
    strat_final_ret = (test_df['cum_strat'].iloc[-1] - 1.0) * 100.0
    bnh_final_ret = (test_df['cum_bnh'].iloc[-1] - 1.0) * 100.0
    
    print(f"      Test Period Strategy Cumulative Return : {strat_final_ret:+.2f}%")
    print(f"      Test Period Buy-and-Hold Return        : {bnh_final_ret:+.2f}%")
    
    return {
        'test_acc': test_acc,
        'majority_baseline': majority_class_baseline,
        'test_prec': test_prec,
        'test_rec': test_rec,
        'test_f1': test_f1,
        'test_auc': test_auc,
        'cm': cm,
        'feat_imp': feat_imp,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'test_df': test_df,
        'champion_model': clf,
        'model_scaler': model_scaler
    }

def plot_comprehensive_dashboard(trials_df, eval_results, y_test, chosen_start, chosen_end, output_image="tuning_and_evaluation.png"):
    """
    Renders a high-resolution 5-panel diagnostic dashboard.
    """
    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.28)
    
    # Panel 1: Hyperparameter Tuning Iteration Curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(trials_df['trial'], trials_df['val_acc'] * 100, marker='o', markersize=4, color='#94a3b8', alpha=0.5, label='Candidate Trial')
    ax1.plot(trials_df['trial'], trials_df['running_max_acc'] * 100, color='#0284c7', linewidth=2.5, label='Running Peak Acc')
    best_t = trials_df.loc[trials_df['val_acc'].idxmax()]
    ax1.scatter([best_t['trial']], [best_t['val_acc'] * 100], color='#dc2626', s=120, zorder=5, label=f"Max Val Acc ({best_t['val_acc']*100:.1f}%)")
    ax1.set_title('A. Hyperparameter Tuning Progression', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Tuning Iteration #', fontsize=10)
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=10)
    ax1.legend(loc='lower right', frameon=True, fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Panel 2: Confusion Matrix Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    cm = eval_results['cm']
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    annot = np.array([[f"{cm[0,0]}\n({cm_norm[0,0]:.1f}%)", f"{cm[0,1]}\n({cm_norm[0,1]:.1f}%)"],
                      [f"{cm[1,0]}\n({cm_norm[1,0]:.1f}%)", f"{cm[1,1]}\n({cm_norm[1,1]:.1f}%)"]])
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False, ax=ax2,
                xticklabels=['Pred Down', 'Pred Up'], yticklabels=['True Down', 'True Up'], annot_kws={"size": 11, "weight": "bold"})
    ax2.set_title(f"B. Test Confusion Matrix (Acc: {eval_results['test_acc']*100:.2f}%)", fontsize=12, fontweight='bold', pad=10)
    
    # Panel 3: ROC Curve
    ax3 = fig.add_subplot(gs[0, 2])
    if eval_results['y_prob'] is not None:
        fpr, tpr, _ = roc_curve(y_test, eval_results['y_prob'])
        ax3.plot(fpr, tpr, color='#16a34a', lw=2.2, label=f"Tuned Model (AUC = {eval_results['test_auc']:.3f})")
    ax3.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', label='Random Chance (0.50)')
    ax3.set_title('C. Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel('False Positive Rate', fontsize=10)
    ax3.set_ylabel('True Positive Rate', fontsize=10)
    ax3.legend(loc='lower right', frameon=True, fontsize=9)
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # Panel 4: Top 12 Feature Importances
    ax4 = fig.add_subplot(gs[1, 0:2])
    feat_imp = eval_results['feat_imp']
    if feat_imp is not None:
        top_feats = feat_imp.head(12)[::-1]
        colors = ['#38bdf8' if 'ret' in k or 'macd' in k or 'rsi' in k else '#818cf8' for k in top_feats.index]
        y_pos = np.arange(len(top_feats))
        ax4.barh(y_pos, top_feats.values, color=colors, height=0.65, edgecolor='#1e293b')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(top_feats.index, fontsize=9.5, fontweight='500')
        ax4.set_title('D. Top 12 Predictive Features (Gini / Importance Weight)', fontsize=12, fontweight='bold', pad=10)
        ax4.set_xlabel('Relative Feature Importance', fontsize=10)
        ax4.grid(True, axis='x', linestyle='--', alpha=0.5)
        
    # Panel 5: Cumulative Return of ML Strategy vs BNH on Test Set
    ax5 = fig.add_subplot(gs[1, 2])
    tdf = eval_results['test_df']
    ax5.plot(tdf.index, tdf['cum_strat'], color='#0284c7', lw=2.0, label='ML Model Timing')
    ax5.plot(tdf.index, tdf['cum_bnh'], color='#64748b', lw=1.5, linestyle='--', label='Nifty 50 Buy-&-Hold')
    ax5.set_title('E. Test Set Strategy Equity Curve', fontsize=12, fontweight='bold', pad=10)
    ax5.set_xlabel('Date', fontsize=10)
    ax5.set_ylabel('Equity Multiplier (Base 1.0)', fontsize=10)
    ax5.legend(loc='upper left', frameon=True, fontsize=9)
    ax5.grid(True, linestyle='--', alpha=0.5)
    
    fig.suptitle(f"Nifty 50 Machine Learning Pipeline: 5-Year Sample ({chosen_start.strftime('%b %Y')} – {chosen_end.strftime('%b %Y')})\n"
                 f"Champion Architecture: {trials_df.loc[trials_df['val_acc'].idxmax(), 'model_name']} | Max Accuracy: {eval_results['test_acc']*100:.2f}% | AUC: {eval_results['test_auc']:.3f}",
                 fontsize=14, fontweight='bold', y=0.98)
                 
    plt.savefig(output_image, bbox_inches='tight')
    plt.close()
    print(f"      Saved comprehensive visual dashboard to: {output_image}")

def save_artifacts(best_model_info, eval_results, feature_cols, trials_df, df_sample, chosen_start, chosen_end, output_dir="."):
    """
    Saves model binary (.joblib), CSV sample, and comprehensive textual report.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save Serialized Model
    model_path = os.path.join(output_dir, "best_model.joblib")
    artifact_payload = {
        'model': eval_results['champion_model'],
        'scaler': eval_results['model_scaler'],
        'features': feature_cols,
        'params': best_model_info['params'],
        'test_metrics': {
            'accuracy': eval_results['test_acc'],
            'roc_auc': eval_results['test_auc'],
            'precision': eval_results['test_prec'],
            'recall': eval_results['test_rec'],
            'f1_score': eval_results['test_f1'],
            'baseline_accuracy': eval_results['majority_baseline']
        },
        'sample_interval': {
            'start': chosen_start.strftime('%Y-%m-%d'),
            'end': chosen_end.strftime('%Y-%m-%d'),
            'num_bars': len(df_sample)
        }
    }
    joblib.dump(artifact_payload, model_path)
    print(f"      Saved trained champion model to: {model_path}")
    
    # 2. Save 5-Year Sample Dataset
    sample_csv_path = os.path.join(output_dir, "sampled_5yr_dataset.csv")
    df_sample.to_csv(sample_csv_path)
    print(f"      Saved exact 5-year sampled slice to: {sample_csv_path}")
    
    # 3. Save Model Report
    report_path = os.path.join(output_dir, "model_report.txt")
    with open(report_path, "w") as f:
        f.write("="*80 + "\n")
        f.write("            NIFTY 50 MACHINE LEARNING MODEL REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Sampled 5-Year Interval : {chosen_start.strftime('%Y-%m-%d')} to {chosen_end.strftime('%Y-%m-%d')}\n")
        f.write(f"Total Trading Sessions  : {len(df_sample)} bars\n")
        f.write(f"5-Year Nifty Return     : {(df_sample['Close'].iloc[-1]/df_sample['Close'].iloc[0]-1.0)*100:+.2f}%\n")
        f.write(f"Champion Model Type     : {best_model_info['params']['model_name']}\n")
        f.write(f"Best Hyperparameters    : {best_model_info['params']}\n\n")
        f.write("-"*80 + "\n")
        f.write("1. OUT-OF-SAMPLE TEST PERFORMANCE (Strictly Unseen 15% Holdout)\n")
        f.write("-"*80 + "\n")
        f.write(f"Test Accuracy            : {eval_results['test_acc']*100:.2f}%\n")
        f.write(f"Majority Class Baseline  : {eval_results['majority_baseline']*100:.2f}%\n")
        f.write(f"Accuracy vs. Baseline    : {eval_results['test_acc'] - eval_results['majority_baseline']:+.2f}%\n")
        f.write(f"ROC-AUC Score            : {eval_results['test_auc']:.4f}\n")
        f.write(f"Precision (Up Days)      : {eval_results['test_prec']*100:.2f}%\n")
        f.write(f"Recall (Up Days)         : {eval_results['test_rec']*100:.2f}%\n")
        f.write(f"F1-Score                 : {eval_results['test_f1']:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"                Pred Down    Pred Up\n")
        f.write(f"Actual Down     {eval_results['cm'][0,0]:<12}{eval_results['cm'][0,1]:<12}\n")
        f.write(f"Actual Up       {eval_results['cm'][1,0]:<12}{eval_results['cm'][1,1]:<12}\n\n")
        f.write("-"*80 + "\n")
        f.write("2. TOP 15 PREDICTIVE FEATURES\n")
        f.write("-"*80 + "\n")
        if eval_results['feat_imp'] is not None:
            for rank, (name, val) in enumerate(eval_results['feat_imp'].head(15).items(), 1):
                f.write(f"{rank:02d}. {name:<28} : {val:.5f}\n")
        f.write("\n" + "="*80 + "\n")
    print(f"      Saved comprehensive evaluation report to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Train and Fine-Tune Machine Learning Model on Random 5-Year Nifty 50 Interval")
    parser.add_argument("--data", type=str, default="nifty50_historical_data.csv", help="Path to Nifty 50 CSV")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible interval sampling")
    parser.add_argument("--start-date", type=str, default=None, help="Explicit start date (YYYY-MM-DD) for 5-year window")
    parser.add_argument("--trials", type=int, default=60, help="Number of hyperparameter search trials")
    parser.add_argument("--output-dir", type=str, default="model", help="Output directory for model and artifacts")
    args = parser.parse_args()
    
    # 1. Load Data
    raw_df = load_full_dataset(args.data)
    
    # 2. Extract Random 5-Year Window
    df_5y, start_dt, end_dt = sample_random_5_year_window(raw_df, seed=args.seed, start_date_str=args.start_date)
    
    # 3. Engineer Features
    clean_df, X, y, feat_cols = engineer_features(df_5y)
    
    # 4. Train-Val-Test Split
    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(X, y)
    
    # 5. Hyperparameter Tuning
    best_info, trials_df, scaler = fine_tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials=args.trials)
    
    # 6. Evaluate on Unseen Holdout Test
    eval_results = evaluate_holdout_test(
        best_info, scaler, X_train, y_train, X_val, y_val, X_test, y_test, feat_cols, clean_df
    )
    
    # 7. Output Plot & Artifacts in target folder
    os.makedirs(args.output_dir, exist_ok=True)
    img_path = os.path.join(args.output_dir, "tuning_and_evaluation.png")
    plot_comprehensive_dashboard(trials_df, eval_results, y_test, start_dt, end_dt, output_image=img_path)
    
    save_artifacts(best_info, eval_results, feat_cols, trials_df, df_5y, start_dt, end_dt, output_dir=args.output_dir)
    print("\nPipeline execution fully completed successfully!")

if __name__ == "__main__":
    main()
