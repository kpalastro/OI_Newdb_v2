#!/usr/bin/env python3
"""
Train Option Return Prediction Models

This script trains separate models to predict CE and PE option price returns
directly, using ITM OI and Volume features as inputs. This aligns training
with actual option trading outcomes.

Key Features:
- Separate models for CE and PE option returns
- Multiple time horizons (3m, 5m, 10m, 15m)
- Quality filters for tradeable samples
- Turning point detection model
- Walk-forward validation
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import joblib
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import database_new as db
from feature_engineering import prepare_training_features, REQUIRED_FEATURE_COLUMNS
from utils.itm_feature_evaluator import create_itm_optimal_range_features
from config import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def load_option_price_data(
    exchange: str,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """
    Load option price data from option_chain_snapshots and join with ml_features.
    Uses SQL JOIN for efficient matching.
    """
    LOGGER.info(f"Loading option price data from {start_date} to {end_date}...")
    
    # Load ml_features
    ml_data = db.load_historical_data_for_ml(exchange, start_date, end_date)
    if ml_data.empty:
        LOGGER.error("No ml_features data found.")
        return pd.DataFrame()
    
    LOGGER.info(f"Loaded {len(ml_data)} ml_features records")
    
    # Query option prices using SQL JOIN
    conn = db.get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Use SQL to efficiently join and get ATM option prices
        # Note: We select specific columns to avoid ambiguity with option_chain_snapshots.underlying_price
        query = """
            WITH ml_with_rounded_ts AS (
                SELECT 
                    DATE_TRUNC('minute', timestamp) as ts_minute,
                    timestamp,
                    underlying_price as ml_underlying_price
                FROM ml_features
                WHERE timestamp >= %s 
                  AND timestamp <= %s
                  AND exchange = %s
            ),
            atm_strikes AS (
                SELECT DISTINCT ON (DATE_TRUNC('minute', ocs.timestamp))
                    DATE_TRUNC('minute', ocs.timestamp) as ts_minute,
                    ocs.timestamp as ocs_timestamp,
                    ocs.strike,
                    ml.ml_underlying_price,
                    ABS(ocs.strike - ml.ml_underlying_price) as distance
                FROM option_chain_snapshots ocs
                INNER JOIN ml_with_rounded_ts ml ON 
                    DATE_TRUNC('minute', ocs.timestamp) = ml.ts_minute
                    AND ocs.exchange = %s
                WHERE ocs.timestamp >= %s 
                  AND ocs.timestamp <= %s
                  AND ocs.exchange = %s
                ORDER BY DATE_TRUNC('minute', ocs.timestamp), ABS(ocs.strike - ml.ml_underlying_price)
            )
            SELECT 
                ml.timestamp,
                ml.ml_underlying_price as underlying_price,
                atm.strike as strike_price,
                ocs.option_type,
                ocs.ltp as price,
                ocs.pct_change_3m,
                ocs.pct_change_5m,
                ocs.pct_change_10m,
                ocs.pct_change_15m,
                ocs.spread,
                ocs.order_book_imbalance
            FROM ml_with_rounded_ts ml
            INNER JOIN atm_strikes atm ON ml.ts_minute = atm.ts_minute
            INNER JOIN option_chain_snapshots ocs ON 
                ocs.timestamp = atm.ocs_timestamp
                AND ocs.strike = atm.strike
                AND ocs.exchange = %s
                AND ocs.option_type IN ('CE', 'PE')
            ORDER BY ml.timestamp, ocs.option_type
        """
        
        cursor.execute(query, (
            start_date, end_date, exchange,  # ml_features WHERE
            exchange,  # INNER JOIN exchange
            start_date, end_date, exchange,  # ocs WHERE
            exchange  # final INNER JOIN exchange
        ))
        
        rows = cursor.fetchall()
        LOGGER.info(f"Fetched {len(rows)} option price records")
        
        if not rows:
            LOGGER.warning("No option price data found. Returning ml_features only.")
            return ml_data
        
        # Build option data dictionary
        option_dict = {}
        for row in rows:
            ts, underlying, strike, opt_type, price, pct_3m, pct_5m, pct_10m, pct_15m, spread, imbalance = row
            
            if ts not in option_dict:
                option_dict[ts] = {
                    'timestamp': ts,
                    'strike_price': strike,
                    'ce_price': None,
                    'pe_price': None,
                    'ce_price_change_3m': None,
                    'pe_price_change_3m': None,
                    'ce_price_change_5m': None,
                    'pe_price_change_5m': None,
                    'ce_price_change_10m': None,
                    'pe_price_change_10m': None,
                    'ce_price_change_15m': None,
                    'pe_price_change_15m': None,
                    'ce_spread': None,
                    'pe_spread': None,
                    'ce_order_book_imbalance': None,
                    'pe_order_book_imbalance': None,
                }
            
            if opt_type == 'CE':
                option_dict[ts]['ce_price'] = price
                option_dict[ts]['ce_price_change_3m'] = pct_3m
                option_dict[ts]['ce_price_change_5m'] = pct_5m
                option_dict[ts]['ce_price_change_10m'] = pct_10m
                option_dict[ts]['ce_price_change_15m'] = pct_15m
                option_dict[ts]['ce_spread'] = spread
                option_dict[ts]['ce_order_book_imbalance'] = imbalance
            elif opt_type == 'PE':
                option_dict[ts]['pe_price'] = price
                option_dict[ts]['pe_price_change_3m'] = pct_3m
                option_dict[ts]['pe_price_change_5m'] = pct_5m
                option_dict[ts]['pe_price_change_10m'] = pct_10m
                option_dict[ts]['pe_price_change_15m'] = pct_15m
                option_dict[ts]['pe_spread'] = spread
                option_dict[ts]['pe_order_book_imbalance'] = imbalance
        
        option_df = pd.DataFrame(list(option_dict.values()))
        LOGGER.info(f"Created option DataFrame with {len(option_df)} records")
        
        # Merge with ml_features
        ml_data['timestamp'] = pd.to_datetime(ml_data['timestamp'])
        option_df['timestamp'] = pd.to_datetime(option_df['timestamp'])
        
        ml_data = ml_data.merge(option_df, on='timestamp', how='left')
        LOGGER.info(f"Merged data: {len(ml_data)} records, {ml_data['ce_price'].notna().sum()} with option prices")
        
        return ml_data
        
    except Exception as e:
        LOGGER.error(f"Error loading option price data: {e}", exc_info=True)
        return ml_data
        
    finally:
        cursor.close()
        db.release_db_connection(conn)


def apply_quality_filters(df: pd.DataFrame, min_option_price: float = 1.0) -> pd.DataFrame:
    """
    Apply quality filters to ensure tradeable samples.
    
    Filters:
    - Remove rows with missing option prices
    - Remove rows with very low option prices (illiquid)
    - Remove rows with high spreads (illiquid)
    - Remove extreme price changes (likely data errors)
    """
    LOGGER.info(f"Applying quality filters. Initial rows: {len(df)}")
    
    # Check if option price columns exist
    if 'ce_price' not in df.columns or 'pe_price' not in df.columns:
        LOGGER.warning("Option price columns not found. Skipping quality filters.")
        return df
    
    filtered = df.copy()
    
    # Filter 1: Must have both CE and PE prices
    filtered = filtered[
        filtered['ce_price'].notna() & 
        filtered['pe_price'].notna()
    ]
    LOGGER.info(f"After price filter: {len(filtered)} rows")
    
    # Filter 2: Minimum option price (illiquid filter)
    filtered = filtered[
        (filtered['ce_price'] >= min_option_price) &
        (filtered['pe_price'] >= min_option_price)
    ]
    LOGGER.info(f"After min price filter: {len(filtered)} rows")
    
    # Filter 3: Reasonable price changes (remove extreme outliers)
    # Cap at ±200% to remove data errors
    for col in ['ce_price_change_3m', 'pe_price_change_3m', 
                'ce_price_change_5m', 'pe_price_change_5m',
                'ce_price_change_10m', 'pe_price_change_10m',
                'ce_price_change_15m', 'pe_price_change_15m']:
        if col in filtered.columns:
            filtered = filtered[
                (filtered[col].isna()) | 
                ((filtered[col] >= -200) & (filtered[col] <= 200))
            ]
    
    LOGGER.info(f"After price change filter: {len(filtered)} rows")
    
    # Filter 4: Spread filter (if spread data available)
    if 'ce_spread' in filtered.columns and 'pe_spread' in filtered.columns:
        # Remove rows where spread > 10% of price (illiquid)
        filtered = filtered[
            (filtered['ce_spread'].isna()) | 
            ((filtered['ce_spread'] / filtered['ce_price'].clip(lower=1.0)) < 0.10)
        ]
        filtered = filtered[
            (filtered['pe_spread'].isna()) | 
            ((filtered['pe_spread'] / filtered['pe_price'].clip(lower=1.0)) < 0.10)
        ]
        LOGGER.info(f"After spread filter: {len(filtered)} rows")
    
    LOGGER.info(f"Final filtered rows: {len(filtered)} ({len(filtered)/len(df)*100:.1f}% of original)")
    
    return filtered


def create_option_return_targets(
    df: pd.DataFrame,
    horizon: str = '3m'
) -> Tuple[pd.Series, pd.Series]:
    """
    Create regression targets for CE and PE option returns.
    
    Args:
        df: DataFrame with option price change columns
        horizon: '3m', '5m', '10m', or '15m'
    
    Returns:
        (ce_target, pe_target): Series of option price changes
    """
    ce_col = f'ce_price_change_{horizon}'
    pe_col = f'pe_price_change_{horizon}'
    
    if ce_col not in df.columns or pe_col not in df.columns:
        raise ValueError(f"Missing price change columns for horizon {horizon}")
    
    ce_target = df[ce_col].copy()
    pe_target = df[pe_col].copy()
    
    # Fill NaN with 0 (no change) for rows that passed quality filters
    ce_target = ce_target.fillna(0.0)
    pe_target = pe_target.fillna(0.0)
    
    return ce_target, pe_target


def train_option_return_model(
    X: pd.DataFrame,
    y: pd.Series,
    option_type: str,
    horizon: str,
    exchange: str,
    n_splits: int = 5
) -> Dict[str, Any]:
    """
    Train a LightGBM regression model to predict option returns.
    
    Args:
        X: Feature matrix
        y: Target (option price change %)
        option_type: 'CE' or 'PE'
        horizon: '3m', '5m', '10m', or '15m'
        exchange: Exchange name
        n_splits: Number of CV splits
    
    Returns:
        Dictionary with model, metrics, and feature importance
    """
    LOGGER.info(f"Training {option_type} {horizon} return model...")
    LOGGER.info(f"  Samples: {len(X)}, Features: {len(X.columns)}")
    
    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fold_metrics = []
    models = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        LOGGER.info(f"  Fold {fold + 1}/{n_splits}")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train LightGBM
        model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Directional accuracy (sign prediction)
        direction_accuracy = np.mean(np.sign(y_test) == np.sign(y_pred))
        
        # Average return when model predicts positive
        positive_pred_mask = y_pred > 0
        avg_return_on_long = y_test[positive_pred_mask].mean() if positive_pred_mask.sum() > 0 else 0.0
        
        fold_metrics.append({
            'fold': fold + 1,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'avg_return_on_long': avg_return_on_long,
            'train_size': len(y_train),
            'test_size': len(y_test)
        })
        
        models.append(model)
        
        LOGGER.info(f"    R²: {r2:.4f}, MAE: {mae:.4f}, Direction Acc: {direction_accuracy:.4f}")
    
    # Average metrics
    avg_metrics = {
        'avg_r2': np.mean([m['r2'] for m in fold_metrics]),
        'avg_mae': np.mean([m['mae'] for m in fold_metrics]),
        'avg_direction_accuracy': np.mean([m['direction_accuracy'] for m in fold_metrics]),
        'avg_return_on_long': np.mean([m['avg_return_on_long'] for m in fold_metrics]),
    }
    
    # Train final model on all data
    LOGGER.info("  Training final model on all data...")
    final_model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    final_model.fit(X, y)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'model': final_model,
        'fold_metrics': fold_metrics,
        'avg_metrics': avg_metrics,
        'feature_importance': feature_importance,
        'option_type': option_type,
        'horizon': horizon
    }


def train_turning_point_model(
    X: pd.DataFrame,
    df: pd.DataFrame,
    exchange: str,
    n_splits: int = 5
) -> Dict[str, Any]:
    """
    Train a model to detect turning points (reversal risk).
    
    Target: Binary classification (1 = turning point / reversal risk, 0 = continuation)
    Features: Volume spikes, divergences, rapid changes
    """
    LOGGER.info("Training turning point detection model...")
    
    # Create turning point labels
    # A turning point occurs when:
    # 1. Price change reverses sign in next period
    # 2. Volume spike occurs (exhaustion)
    # 3. Strong divergence that reverses
    
    # Simple heuristic: If price change is large and volume spikes, likely turning point
    # Use correct column names from feature engineering
    itm_ce_vol_col = 'itm_volume_ce_pct_change_3m_wavg'
    itm_pe_vol_col = 'itm_volume_pe_pct_change_3m_wavg'
    
    # Check if columns exist, use 0 if not
    ce_vol = df[itm_ce_vol_col] if itm_ce_vol_col in df.columns else pd.Series(0.0, index=df.index)
    pe_vol = df[itm_pe_vol_col] if itm_pe_vol_col in df.columns else pd.Series(0.0, index=df.index)
    
    # Calculate divergence if not present
    if 'divergence' not in df.columns:
        ce_delta = df.get('itm_oi_ce_pct_change_3m_wavg', pd.Series(0.0, index=df.index))
        pe_delta = df.get('itm_oi_pe_pct_change_3m_wavg', pd.Series(0.0, index=df.index))
        divergence = ce_delta - pe_delta
    else:
        divergence = df['divergence']
    
    turning_point = (
        (ce_vol > 16.0) |  # CE volume spike
        (pe_vol < -3.0) |  # PE volume decreasing (bullish exhaustion)
        (divergence.abs() > 3.0)  # Strong divergence
    ).astype(int)
    
    y = turning_point
    
    min_turning_point_samples = 10
    if y.sum() < min_turning_point_samples:
        LOGGER.warning(f"Insufficient turning point samples: {y.sum()} (need >= {min_turning_point_samples}). Skipping turning point model.")
        return {}
    
    LOGGER.info(f"  Turning point samples: {y.sum()}/{len(y)} ({y.sum()/len(y)*100:.1f}%)")
    
    # Time-series CV
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fold_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train classifier
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=16,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            class_weight='balanced'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
        
        fold_metrics.append({
            'fold': fold + 1,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        })
    
    # Final model
    final_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=16,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        class_weight='balanced'
    )
    
    final_model.fit(X, y)
    
    return {
        'model': final_model,
        'fold_metrics': fold_metrics,
        'avg_metrics': {
            'avg_precision': np.mean([m['precision'] for m in fold_metrics]),
            'avg_recall': np.mean([m['recall'] for m in fold_metrics]),
            'avg_f1': np.mean([m['f1'] for m in fold_metrics]),
            'avg_auc': np.mean([m['auc'] for m in fold_metrics]),
        }
    }


def train_all_option_models(
    exchange: str = 'NSE',
    days: int = 90,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    horizons: List[str] = ['3m', '5m', '10m', '15m'],
    min_option_price: float = 1.0
) -> Dict[str, Any]:
    """
    Main training function for all option return models.
    
    Args:
        exchange: Exchange name
        days: Number of days of historical data (used if start_date/end_date not provided)
        start_date: Start date for training data (overrides days parameter)
        end_date: End date for training data (overrides days parameter)
        horizons: List of time horizons to train
        min_option_price: Minimum option price for quality filter
    
    Returns:
        Dictionary with all trained models and metrics
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TRAINING OPTION RETURN PREDICTION MODELS")
    LOGGER.info("=" * 80)
    
    # 1. Load data - use provided dates or calculate from days
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=days)
    
    # Validate dates
    if start_date >= end_date:
        raise ValueError(f"start_date ({start_date}) must be before end_date ({end_date})")
    
    LOGGER.info(f"Training period: {start_date.date()} to {end_date.date()}")
    LOGGER.info(f"Training duration: {(end_date - start_date).days} days")
    
    # Load option price data
    df = load_option_price_data(exchange, start_date, end_date)
    if df.empty:
        LOGGER.error("No data loaded. Exiting.")
        return {}
    
    # 2. Prepare features
    LOGGER.info("Preparing features...")
    df = prepare_training_features(df, REQUIRED_FEATURE_COLUMNS)
    
    # Add ITM optimal range features (apply row-wise since function expects dict)
    LOGGER.info("Adding ITM optimal range features...")
    try:
        # Apply to each row
        itm_features_list = []
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            itm_features = create_itm_optimal_range_features(row_dict)
            itm_features_list.append(itm_features)
        
        # Convert to DataFrame and merge
        itm_features_df = pd.DataFrame(itm_features_list, index=df.index)
        df = pd.concat([df, itm_features_df], axis=1)
        LOGGER.info(f"Added {len(itm_features_df.columns)} ITM optimal range features")
    except Exception as e:
        LOGGER.warning(f"Failed to add ITM optimal range features: {e}")
    
    # 3. Apply quality filters (only if option price data exists)
    if 'ce_price' in df.columns and 'pe_price' in df.columns:
        LOGGER.info("Applying quality filters...")
        df = apply_quality_filters(df, min_option_price=min_option_price)
        
        if df.empty:
            LOGGER.error("No data after quality filters. Exiting.")
            return {}
    else:
        LOGGER.warning("No option price data found. Cannot train option return models.")
        LOGGER.warning("Please ensure option_chain_snapshots table has data for the specified period.")
        return {}
    
    # 4. Prepare feature matrix
    feature_cols = [c for c in REQUIRED_FEATURE_COLUMNS if c in df.columns]
    # Add ITM optimal range features (avoid duplicates)
    itm_features = [c for c in df.columns if c.startswith('itm_') and c not in feature_cols]
    feature_cols.extend(itm_features)
    
    # Remove duplicate columns
    feature_cols = list(dict.fromkeys(feature_cols))  # Preserves order, removes duplicates
    
    X = df[feature_cols].copy()
    
    # Remove duplicate columns from X (in case DataFrame has duplicates)
    X = X.loc[:, ~X.columns.duplicated()]
    
    # Fill NaN features with 0 instead of removing rows (more lenient)
    # This preserves more training samples
    LOGGER.info(f"Before NaN handling: {len(X)} samples")
    nan_counts = X.isna().sum()
    if nan_counts.sum() > 0:
        LOGGER.info(f"Features with NaN: {nan_counts[nan_counts > 0].to_dict()}")
        # Fill NaN with 0 for numeric features
        X = X.fillna(0.0)
        LOGGER.info(f"After filling NaN with 0: {len(X)} samples")
    
    # Only remove rows where ALL features are NaN (shouldn't happen after fillna, but safety check)
    valid_mask = ~X.isna().all(axis=1)
    X = X[valid_mask]
    df = df[valid_mask]
    
    LOGGER.info(f"Final training set: {len(X)} samples, {len(X.columns)} features")
    
    # 5. Train models for each horizon and option type
    results = {
        'exchange': exchange,
        'training_date': datetime.now().isoformat(),
        'training_start_date': start_date.isoformat(),
        'training_end_date': end_date.isoformat(),
        'n_samples': len(X),
        'models': {}
    }
    
    for horizon in horizons:
        LOGGER.info(f"\n{'=' * 80}")
        LOGGER.info(f"Training models for {horizon} horizon")
        LOGGER.info(f"{'=' * 80}")
        
        # Create targets
        ce_target, pe_target = create_option_return_targets(df, horizon)
        
        # Diagnostic: Show target statistics
        LOGGER.info(f"Target statistics for {horizon}:")
        LOGGER.info(f"  CE: Total={len(ce_target)}, Non-null={ce_target.notna().sum()}, "
                   f"Non-zero={((ce_target != 0) & ce_target.notna()).sum()}, "
                   f"Mean={ce_target[ce_target.notna()].mean():.2f}%, "
                   f"Std={ce_target[ce_target.notna()].std():.2f}%")
        LOGGER.info(f"  PE: Total={len(pe_target)}, Non-null={pe_target.notna().sum()}, "
                   f"Non-zero={((pe_target != 0) & pe_target.notna()).sum()}, "
                   f"Mean={pe_target[pe_target.notna()].mean():.2f}%, "
                   f"Std={pe_target[pe_target.notna()].std():.2f}%")
        
        # Filter to valid targets (non-null, allow zero changes for now)
        # Note: We allow zero changes because they're still valid training samples
        # The model should learn to predict when there's no movement
        valid_ce = ce_target.notna()  # Allow zero changes
        valid_pe = pe_target.notna()  # Allow zero changes
        
        # Alternative: If we want non-zero only, use:
        # valid_ce = ce_target.notna() & (ce_target != 0)
        # valid_pe = pe_target.notna() & (pe_target != 0)
        
        LOGGER.info(f"  Valid CE samples: {valid_ce.sum()}, Valid PE samples: {valid_pe.sum()}")
        
        # Train CE model
        if valid_ce.sum() > 100:
            ce_result = train_option_return_model(
                X[valid_ce], ce_target[valid_ce],
                'CE', horizon, exchange
            )
            results['models'][f'ce_{horizon}'] = ce_result
        else:
            LOGGER.warning(f"Insufficient CE samples for {horizon}. Skipping.")
        
        # Train PE model
        if valid_pe.sum() > 100:
            pe_result = train_option_return_model(
                X[valid_pe], pe_target[valid_pe],
                'PE', horizon, exchange
            )
            results['models'][f'pe_{horizon}'] = pe_result
        else:
            LOGGER.warning(f"Insufficient PE samples for {horizon}. Skipping.")
    
    # 6. Train turning point model
    LOGGER.info(f"\n{'=' * 80}")
    LOGGER.info("Training Turning Point Detection Model")
    LOGGER.info(f"{'=' * 80}")
    
    turning_point_result = train_turning_point_model(X, df, exchange)
    if turning_point_result:
        results['models']['turning_point'] = turning_point_result
    
    # 7. Save models and feature columns
    model_dir = Path(f"models/option_returns/{exchange}")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info(f"\nSaving models to {model_dir}...")
    
    # Save feature columns used for training (all models use the same features)
    feature_cols_path = model_dir / "feature_columns.pkl"
    joblib.dump(feature_cols, feature_cols_path)
    LOGGER.info(f"  Saved feature columns: {feature_cols_path} ({len(feature_cols)} features)")
    
    for model_name, model_result in results['models'].items():
        if 'model' in model_result:
            model_path = model_dir / f"{model_name}_model.pkl"
            joblib.dump(model_result['model'], model_path)
            LOGGER.info(f"  Saved: {model_path}")
            
            # Save feature importance
            if 'feature_importance' in model_result:
                importance_path = model_dir / f"{model_name}_importance.csv"
                model_result['feature_importance'].to_csv(importance_path, index=False)
    
    # Save training results summary
    summary_path = model_dir / "training_summary.json"
    summary = {
        'exchange': results['exchange'],
        'training_date': results['training_date'],
        'training_start_date': start_date.isoformat(),
        'training_end_date': end_date.isoformat(),
        'training_duration_days': (end_date - start_date).days,
        'n_samples': results['n_samples'],
        'model_metrics': {}
    }
    
    for model_name, model_result in results['models'].items():
        if 'avg_metrics' in model_result:
            summary['model_metrics'][model_name] = model_result['avg_metrics']
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    LOGGER.info(f"  Saved: {summary_path}")
    
    # 8. Print summary
    LOGGER.info(f"\n{'=' * 80}")
    LOGGER.info("TRAINING SUMMARY")
    LOGGER.info(f"{'=' * 80}")
    
    for model_name, model_result in results['models'].items():
        if 'avg_metrics' in model_result:
            LOGGER.info(f"\n{model_name}:")
            for metric, value in model_result['avg_metrics'].items():
                LOGGER.info(f"  {metric}: {value:.4f}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train option return prediction models')
    parser.add_argument('--exchange', '-e', default='NSE', help='Exchange name')
    parser.add_argument('--days', '-d', type=int, default=90, 
                       help='Days of historical data (used if --start-date not provided)')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date for training data (YYYY-MM-DD). Overrides --days.')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for training data (YYYY-MM-DD). Defaults to today if not provided.')
    parser.add_argument('--horizons', nargs='+', default=['3m', '5m', '10m', '15m'],
                       help='Time horizons to train')
    parser.add_argument('--min-price', type=float, default=1.0,
                       help='Minimum option price for quality filter')
    
    args = parser.parse_args()
    
    # Parse dates if provided
    start_date = None
    end_date = None
    
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Invalid start-date format: {args.start_date}. Use YYYY-MM-DD")
    
    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Invalid end-date format: {args.end_date}. Use YYYY-MM-DD")
    
    train_all_option_models(
        exchange=args.exchange,
        days=args.days,
        start_date=start_date,
        end_date=end_date,
        horizons=args.horizons,
        min_option_price=args.min_price
    )
