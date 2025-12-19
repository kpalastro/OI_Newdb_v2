"""
Train Swing Trading Ensemble from Scratch

This script trains the SwingTradingEnsemble (XGBoost + LightGBM) for 1-3 day swing trading.
It uses a longer look-forward horizon compared to intraday models.

Usage:
    python train_swing_ensemble.py --exchange NSE --days 90
    python train_swing_ensemble.py --exchange BSE --days 90
"""
import argparse
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from typing import List, Dict, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, f1_score, precision_score

import database_new as db
from time_utils import today_ist
from feature_engineering import (
    REQUIRED_FEATURE_COLUMNS,
    prepare_training_features,
)
from models.swing_ensemble import SwingTradingEnsemble

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def define_swing_triple_barrier_target(
    df: pd.DataFrame,
    look_forward_minutes: int = 2880,  # ~2 days (assuming 5-min bars: 2880/5 = 576 bars)
    pt: float = 1.5,  # Profit target: 1.5x volatility
    sl: float = 1.0   # Stop loss: 1.0x volatility
) -> pd.DataFrame:
    """
    Triple Barrier Method for Swing Trading (1-3 day horizon).
    
    Labels:
      1: Hit Upper Barrier (Profit Take) first
     -1: Hit Lower Barrier (Stop Loss) first
      0: Hit Vertical Barrier (Time Limit) first
    
    Args:
        look_forward_minutes: Number of minutes to look forward (default ~2 days)
        pt: Profit target multiplier (relative to volatility)
        sl: Stop loss multiplier (relative to volatility)
    """
    LOGGER.info(f"Defining Swing Triple Barrier targets (look_forward={look_forward_minutes} minutes)...")
    data = df.copy()
    
    # Calculate dynamic volatility (longer window for swing: 240 minutes = 4 hours)
    returns = data['underlying_price'].pct_change()
    vol = returns.rolling(240).std()  # 4-hour rolling volatility
    
    # Floor volatility to avoid near-zero barriers
    vol = np.maximum(vol, 0.0005)
    
    # Convert look_forward_minutes to number of bars
    # Assuming 5-minute bars (adjust if different)
    bars_per_minute = 1.0 / 5.0  # 1 bar per 5 minutes
    look_forward_bars = int(look_forward_minutes * bars_per_minute)
    
    # Barriers are relative to the Close at time t
    upper_barrier = data['underlying_price'] * (1 + vol * pt)
    lower_barrier = data['underlying_price'] * (1 - vol * sl)
    
    # Initialize hit times with infinity
    hit_upper_time = pd.Series(np.inf, index=data.index)
    hit_lower_time = pd.Series(np.inf, index=data.index)
    
    # Vectorized check for each step in the look_forward window
    for k in range(1, look_forward_bars + 1):
        future_price = data['underlying_price'].shift(-k)
        
        # Check Upper Breach
        mask_u = (future_price > upper_barrier) & (hit_upper_time == np.inf)
        hit_upper_time[mask_u] = k
        
        # Check Lower Breach
        mask_l = (future_price < lower_barrier) & (hit_lower_time == np.inf)
        hit_lower_time[mask_l] = k
    
    # Assign labels based on which barrier was hit first
    # 1 if Upper < Lower (Profit first)
    # -1 if Lower < Upper (Stop first)
    # 0 if both are inf (Time limit reached)
    target = np.zeros(len(data))
    target = np.where(hit_upper_time < hit_lower_time, 1, target)
    target = np.where(hit_lower_time < hit_upper_time, -1, target)
    
    # Convert to classification labels: -1 -> 0 (Sell), 0 -> 1 (Hold), 1 -> 2 (Buy)
    # This matches the multiclass format expected by XGBoost/LightGBM (3 classes: 0, 1, 2)
    target_class = np.zeros(len(target), dtype=int)
    target_class[target == -1] = 0  # Sell -> 0
    target_class[target == 0] = 1   # Hold -> 1
    target_class[target == 1] = 2    # Buy -> 2
    
    data['target'] = target_class
    
    # Remove last rows where we can't look forward
    data = data.iloc[:-look_forward_bars]
    
    # Drop rows with NaN targets or prices
    data = data.dropna(subset=['target', 'underlying_price'])
    
    target_dist = pd.Series(target_class).value_counts(normalize=True).sort_index()
    LOGGER.info(f"Target Distribution:\n{target_dist}")
    LOGGER.info(f"Target counts: {pd.Series(target_class).value_counts().sort_index()}")
    
    return data


def train_swing_ensemble(
    exchange: str,
    days: int = 90,
    n_splits: int = 5
) -> Dict[str, Any]:
    """
    Train Swing Trading Ensemble from scratch.
    
    Args:
        exchange: Exchange name (NSE, BSE, etc.)
        days: Number of days of historical data to use
        n_splits: Number of cross-validation splits
    """
    # 1. Load Data
    end_date = today_ist() + timedelta(days=1)
    start_date = today_ist() - timedelta(days=days)
    
    LOGGER.info(f"Loading data from {start_date} to {end_date} for {exchange}...")
    raw_data = db.load_historical_data_for_ml(exchange, start_date, end_date)
    
    if raw_data.empty:
        LOGGER.error("No data found.")
        return {}
    
    LOGGER.info(f"Loaded {len(raw_data)} rows of raw data.")
    
    # 2. Prepare Features
    LOGGER.info("Preparing features...")
    df = prepare_training_features(raw_data, REQUIRED_FEATURE_COLUMNS)
    
    if df.empty:
        LOGGER.error("No data after feature preparation.")
        return {}
    
    LOGGER.info(f"Features prepared: {len(df)} rows, {len(df.columns)} columns.")
    
    # 3. Define Swing Trading Targets (1-3 day horizon)
    df = define_swing_triple_barrier_target(df, look_forward_minutes=2880)  # ~2 days
    
    if df.empty or 'target' not in df.columns:
        LOGGER.error("Failed to create targets.")
        return {}
    
    LOGGER.info(f"After target creation: {len(df)} rows.")
    
    # 4. Get feature columns
    feature_cols = [c for c in REQUIRED_FEATURE_COLUMNS if c in df.columns]
    LOGGER.info(f"Using {len(feature_cols)} features for training.")
    
    if len(feature_cols) == 0:
        LOGGER.error("No valid features found.")
        return {}
    
    # 5. Cross-Validation
    LOGGER.info(f"Starting Time-Series Cross-Validation with {n_splits} splits...")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    X = df[feature_cols].reset_index(drop=True)
    y = df['target'].values
    
    fold_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        LOGGER.info(f"\n--- Fold {fold + 1}/{n_splits} ---")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train ensemble
        ensemble = SwingTradingEnsemble()
        ensemble.fit(X_train.values, y_train)
        
        # Predict
        probs = ensemble.predict_proba(X_test.values)
        preds = probs.argmax(axis=1)
        
        # Metrics
        precision = precision_score(y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        
        LOGGER.info(f"Fold {fold + 1}: Precision={precision:.3f}, F1={f1:.3f}")
        fold_metrics.append({
            'fold': fold + 1,
            'precision': precision,
            'f1': f1,
            'train_size': len(y_train),
            'test_size': len(y_test)
        })
    
    avg_precision = np.mean([m['precision'] for m in fold_metrics])
    avg_f1 = np.mean([m['f1'] for m in fold_metrics])
    
    LOGGER.info(f"\n=== Cross-Validation Results ===")
    LOGGER.info(f"Average Precision: {avg_precision:.3f}")
    LOGGER.info(f"Average F1: {avg_f1:.3f}")
    
    # 6. Final Training on All Data
    LOGGER.info("\n=== Training Final Model on All Data ===")
    final_ensemble = SwingTradingEnsemble()
    final_ensemble.fit(X.values, y)
    
    # 7. Save Model
    model_dir = Path('models') / exchange
    model_dir.mkdir(parents=True, exist_ok=True)
    
    swing_path = model_dir / 'swing_ensemble.pkl'
    joblib.dump(final_ensemble, swing_path)
    
    LOGGER.info(f"✓ Swing ensemble saved to {swing_path}")
    
    # Save training metadata
    metadata = {
        'exchange': exchange,
        'training_date': str(today_ist()),
        'days_of_data': days,
        'n_features': len(feature_cols),
        'feature_columns': feature_cols,
        'cv_results': fold_metrics,
        'avg_precision': float(avg_precision),
        'avg_f1': float(avg_f1),
        'target_distribution': pd.Series(y).value_counts().to_dict()
    }
    
    metadata_path = model_dir / 'swing_ensemble_metadata.json'
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    LOGGER.info(f"✓ Metadata saved to {metadata_path}")
    
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Swing Trading Ensemble from scratch'
    )
    parser.add_argument(
        '--exchange',
        type=str,
        required=True,
        choices=['NSE', 'BSE', 'NSE_MONTHLY', 'BANKNIFTY_MONTHLY'],
        help='Exchange to train for'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=90,
        help='Number of days of historical data to use (default: 90)'
    )
    parser.add_argument(
        '--n-splits',
        type=int,
        default=5,
        help='Number of cross-validation splits (default: 5)'
    )
    
    args = parser.parse_args()
    
    try:
        metadata = train_swing_ensemble(
            exchange=args.exchange,
            days=args.days,
            n_splits=args.n_splits
        )
        
        if metadata:
            LOGGER.info("\n✓ Training completed successfully!")
            LOGGER.info(f"Model saved to: models/{args.exchange}/swing_ensemble.pkl")
        else:
            LOGGER.error("Training failed. Check logs above for details.")
            exit(1)
            
    except Exception as e:
        LOGGER.exception(f"Training failed with error: {e}")
        exit(1)

