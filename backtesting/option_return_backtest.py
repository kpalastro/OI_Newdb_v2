#!/usr/bin/env python3
"""
Backtest Option Return Prediction Models

This script backtests the trained option return models by:
1. Loading historical data with option prices
2. Using models to predict option returns
3. Simulating trades based on predictions
4. Calculating actual PnL from option price changes
5. Comparing predicted vs actual performance

Usage:
    python3 -m backtesting.option_return_backtest --exchange NSE --start 2026-01-19 --end 2026-01-23
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import argparse
from dataclasses import dataclass, field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import database_new as db
from feature_engineering import prepare_training_features, REQUIRED_FEATURE_COLUMNS
from utils.itm_feature_evaluator import create_itm_optimal_range_features
from models.option_return_predictor import OptionReturnPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


@dataclass
class OptionTradeRecord:
    """Record of an option trade in backtest."""
    timestamp: datetime
    option_type: str  # 'CE' or 'PE'
    horizon: str  # '3m', '5m', '10m', '15m'
    predicted_return: float  # Predicted price change %
    actual_return: float  # Actual price change %
    entry_price: float  # Option price at entry
    exit_price: float  # Option price at exit
    quantity_lots: int
    gross_pnl: float
    net_pnl: float
    transaction_cost: float
    confidence: float
    turning_point_prob: float
    recommendation: str


@dataclass
class OptionBacktestResult:
    """Results of option return backtest."""
    config: Dict[str, Any]
    trades: List[OptionTradeRecord]
    metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'config': self.config,
            'num_trades': len(self.trades),
            'metrics': self.metrics,
            'trades': [trade.__dict__ for trade in self.trades]
        }


def load_backtest_data_with_options(
    exchange: str,
    start_date: date,
    end_date: date
) -> pd.DataFrame:
    """
    Load historical data with option prices for backtesting.
    Uses the same SQL approach as training.
    """
    LOGGER.info(f"Loading backtest data from {start_date} to {end_date}...")
    
    # Convert dates to datetime (start of start_date, end of end_date)
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    
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
                ocs.symbol,
                ocs.ltp as price,
                ocs.pct_change_3m,
                ocs.pct_change_5m,
                ocs.pct_change_10m,
                ocs.pct_change_15m
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
            start_datetime, end_datetime, exchange,
            exchange,
            start_datetime, end_datetime, exchange,
            exchange
        ))
        
        rows = cursor.fetchall()
        LOGGER.info(f"Fetched {len(rows)} option price records")
        
        if not rows:
            LOGGER.warning("No option price data found.")
            # Ensure option price columns exist even if empty
            ml_data['ce_price'] = None
            ml_data['pe_price'] = None
            ml_data['strike_price'] = None
            ml_data['ce_price_change_3m'] = None
            ml_data['pe_price_change_3m'] = None
            ml_data['ce_price_change_5m'] = None
            ml_data['pe_price_change_5m'] = None
            ml_data['ce_price_change_10m'] = None
            ml_data['pe_price_change_10m'] = None
            ml_data['ce_price_change_15m'] = None
            ml_data['pe_price_change_15m'] = None
            return ml_data
        
        # Build option data dictionary
        option_dict = {}
        for row in rows:
            ts, underlying, strike, opt_type, symbol, price, pct_3m, pct_5m, pct_10m, pct_15m = row
            
            if ts not in option_dict:
                option_dict[ts] = {
                    'timestamp': ts,
                    'strike_price': strike,
                    'ce_price': None,
                    'pe_price': None,
                    'ce_symbol': None,
                    'pe_symbol': None,
                    'ce_price_change_3m': None,
                    'pe_price_change_3m': None,
                    'ce_price_change_5m': None,
                    'pe_price_change_5m': None,
                    'ce_price_change_10m': None,
                    'pe_price_change_10m': None,
                    'ce_price_change_15m': None,
                    'pe_price_change_15m': None,
                }
            
            if opt_type == 'CE':
                option_dict[ts]['ce_price'] = price
                option_dict[ts]['ce_symbol'] = symbol
                option_dict[ts]['ce_price_change_3m'] = pct_3m
                option_dict[ts]['ce_price_change_5m'] = pct_5m
                option_dict[ts]['ce_price_change_10m'] = pct_10m
                option_dict[ts]['ce_price_change_15m'] = pct_15m
            elif opt_type == 'PE':
                option_dict[ts]['pe_price'] = price
                option_dict[ts]['pe_symbol'] = symbol
                option_dict[ts]['pe_price_change_3m'] = pct_3m
                option_dict[ts]['pe_price_change_5m'] = pct_5m
                option_dict[ts]['pe_price_change_10m'] = pct_10m
                option_dict[ts]['pe_price_change_15m'] = pct_15m
        
        option_df = pd.DataFrame(list(option_dict.values()))
        LOGGER.info(f"Created option DataFrame with {len(option_df)} records")
        
        # Merge with ml_features
        ml_data['timestamp'] = pd.to_datetime(ml_data['timestamp'])
        option_df['timestamp'] = pd.to_datetime(option_df['timestamp'])
        
        ml_data = ml_data.merge(option_df, on='timestamp', how='left')
        
        # Ensure all option price columns exist (fill missing with None)
        required_cols = [
            'ce_price', 'pe_price', 'strike_price',
            'ce_symbol', 'pe_symbol',
            'ce_price_change_3m', 'pe_price_change_3m',
            'ce_price_change_5m', 'pe_price_change_5m',
            'ce_price_change_10m', 'pe_price_change_10m',
            'ce_price_change_15m', 'pe_price_change_15m'
        ]
        for col in required_cols:
            if col not in ml_data.columns:
                ml_data[col] = None
        
        num_with_prices = ml_data['ce_price'].notna().sum() if 'ce_price' in ml_data.columns else 0
        LOGGER.info(f"Merged data: {len(ml_data)} records, {num_with_prices} with option prices")
        
        return ml_data
        
    except Exception as e:
        LOGGER.error(f"Error loading option price data: {e}", exc_info=True)
        return ml_data
        
    finally:
        cursor.close()
        db.release_db_connection(conn)


def backtest_option_returns(
    exchange: str = 'NSE',
    start_date: date = None,
    end_date: date = None,
    min_confidence: float = 0.5,
    min_predicted_return: float = 0.5,
    transaction_cost_bps: float = 2.0,
    slippage_bps: float = 1.0,
    account_size: float = 1_000_000.0,
    margin_per_lot: float = 75_000.0,
    max_risk_per_trade: float = 0.02,
    use_turning_point_filter: bool = True,
    turning_point_threshold: float = 0.7
) -> OptionBacktestResult:
    """
    Backtest option return prediction models.
    
    Args:
        exchange: Exchange name
        start_date: Start date for backtest
        end_date: End date for backtest
        min_confidence: Minimum confidence to take trade
        min_predicted_return: Minimum predicted return % to take trade
        transaction_cost_bps: Transaction cost in basis points
        slippage_bps: Slippage in basis points
        account_size: Account size for position sizing
        margin_per_lot: Margin required per lot
        max_risk_per_trade: Maximum risk per trade (fraction)
        use_turning_point_filter: Whether to filter trades when turning point detected
        turning_point_threshold: Probability threshold for turning point filter
    
    Returns:
        OptionBacktestResult with trades and metrics
    """
    LOGGER.info("=" * 80)
    LOGGER.info("OPTION RETURN BACKTEST")
    LOGGER.info("=" * 80)
    
    # Default dates (last 5 days if not specified)
    if end_date is None:
        end_date = datetime.now().date()
    if start_date is None:
        start_date = end_date - timedelta(days=5)
    
    LOGGER.info(f"Backtest period: {start_date} to {end_date}")
    
    # Load data
    df = load_backtest_data_with_options(exchange, start_date, end_date)
    if df.empty:
        LOGGER.error("No data loaded. Exiting.")
        return OptionBacktestResult(
            config={},
            trades=[],
            metrics={'error': 'No data loaded'}
        )
    
    # Filter to rows with option prices (check if columns exist first)
    if 'ce_price' not in df.columns or 'pe_price' not in df.columns:
        LOGGER.error("Option price columns missing. Cannot backtest.")
        return OptionBacktestResult(
            config={},
            trades=[],
            metrics={'error': 'Option price columns missing'}
        )
    
    df = df[df['ce_price'].notna() & df['pe_price'].notna()].copy()
    if df.empty:
        LOGGER.error("No data with option prices. Exiting.")
        return OptionBacktestResult(
            config={},
            trades=[],
            metrics={'error': 'No option price data'}
        )
    
    LOGGER.info(f"Backtest data: {len(df)} records with option prices")
    
    # Prepare features
    LOGGER.info("Preparing features...")
    # Save timestamp before feature preparation
    if 'timestamp' in df.columns:
        timestamps = df['timestamp'].copy()
    else:
        LOGGER.error("'timestamp' column missing before feature preparation")
        return OptionBacktestResult(
            config={},
            trades=[],
            metrics={'error': 'timestamp missing'}
        )
    
    df = prepare_training_features(df, REQUIRED_FEATURE_COLUMNS)
    
    # Restore timestamp if it was removed
    if 'timestamp' not in df.columns:
        df['timestamp'] = timestamps.values[:len(df)]
    
    # Add ITM optimal range features
    try:
        itm_features_list = []
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            itm_features = create_itm_optimal_range_features(row_dict)
            itm_features_list.append(itm_features)
        
        itm_features_df = pd.DataFrame(itm_features_list, index=df.index)
        # Convert ITM features to numeric
        for col in itm_features_df.columns:
            itm_features_df[col] = pd.to_numeric(itm_features_df[col], errors='coerce').fillna(0.0)
        
        df = pd.concat([df, itm_features_df], axis=1)
    except Exception as e:
        LOGGER.warning(f"Failed to add ITM optimal range features: {e}")
    
    # Load option return predictor
    predictor = OptionReturnPredictor(exchange)
    if not predictor.models_loaded:
        LOGGER.error("Option return models not loaded. Cannot backtest.")
        LOGGER.error(f"Models found: {list(predictor.models.keys())}")
        LOGGER.error(f"Feature columns loaded: {predictor.feature_columns is not None}")
        return OptionBacktestResult(
            config={},
            trades=[],
            metrics={'error': 'Models not loaded'}
        )
    
    LOGGER.info(f"Models loaded: {len(predictor.models)} models")
    LOGGER.info(f"Feature columns: {len(predictor.feature_columns) if predictor.feature_columns else 0} features")
    
    # Prepare feature matrix
    # Use feature columns from predictor if available, otherwise build from data
    if predictor.feature_columns:
        feature_cols = predictor.feature_columns.copy()
        # Add missing columns with 0.0
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
    else:
        feature_cols = [c for c in REQUIRED_FEATURE_COLUMNS if c in df.columns]
        itm_features = [c for c in df.columns if c.startswith('itm_') and c not in feature_cols]
        feature_cols.extend(itm_features)
        feature_cols = list(dict.fromkeys(feature_cols))
    
    LOGGER.info(f"Feature columns: {len(feature_cols)}")
    LOGGER.info(f"Samples before NaN filter: {len(df)}")
    
    # Remove rows with NaN features (but be lenient - fill NaN with 0 instead of removing)
    # Only remove rows where critical features are missing
    critical_features = [c for c in feature_cols if c in df.columns]
    if critical_features:
        # Fill NaN with 0 for numeric features
        for col in critical_features:
            try:
                col_dtype = str(df[col].dtype)
                if 'float' in col_dtype or 'int' in col_dtype:
                    df[col] = df[col].fillna(0.0)
            except Exception:
                pass
        
        # Only remove rows where all features are NaN
        valid_mask = ~df[critical_features].isna().all(axis=1)
        df = df[valid_mask]
    
    LOGGER.info(f"Valid samples for backtest: {len(df)}")
    
    # Run backtest
    trades: List[OptionTradeRecord] = []
    total_cost_rate = (transaction_cost_bps + slippage_bps) / 10000.0
    
    # Diagnostic counters
    no_prediction_count = 0
    turning_point_filtered = 0
    low_confidence_filtered = 0
    low_return_filtered = 0
    missing_price_count = 0
    successful_predictions = 0
    
    LOGGER.info(f"Starting backtest loop with {len(df)} rows...")
    
    # Ensure timestamp column exists
    if 'timestamp' not in df.columns:
        LOGGER.error("'timestamp' column missing from DataFrame")
        return OptionBacktestResult(
            config={},
            trades=[],
            metrics={'error': 'timestamp column missing'}
        )
    
    for idx, row in df.iterrows():
        timestamp = row.get('timestamp', None)
        if timestamp is None:
            continue
        
        # Check if we have option prices
        if pd.isna(row.get('ce_price')) or pd.isna(row.get('pe_price')):
            missing_price_count += 1
            continue
        
        # Create feature row - ensure all feature columns exist and are numeric
        feature_dict = {}
        for col in feature_cols:
            if col in row.index:
                val = row[col]
                # Convert to numeric if needed
                try:
                    feature_dict[col] = float(val) if pd.notna(val) else 0.0
                except (ValueError, TypeError):
                    feature_dict[col] = 0.0
            else:
                feature_dict[col] = 0.0
        
        feature_row = pd.DataFrame([feature_dict], columns=feature_cols)
        # Ensure all columns are numeric
        for col in feature_row.columns:
            feature_row[col] = pd.to_numeric(feature_row[col], errors='coerce').fillna(0.0)
        
        # Get predictions for all horizons
        horizons = ['3m', '5m', '10m', '15m']
        best_horizon = '3m'
        best_prediction = None
        best_expected_return = 0.0
        
        for horizon in horizons:
            try:
                pred = predictor.predict_returns(feature_row, horizon)
                
                if pred and pred.get('ce_return') is not None:
                    successful_predictions += 1
                    # Choose option with higher expected return
                    ce_ret = pred.get('ce_return', 0)
                    pe_ret = pred.get('pe_return', 0)
                    max_return = max(ce_ret, pe_ret)
                    max_conf = max(pred.get('ce_confidence', 0), pred.get('pe_confidence', 0))
                    
                    if max_return > best_expected_return:
                        best_expected_return = max_return
                        best_horizon = horizon
                        best_prediction = pred
            except Exception as e:
                LOGGER.debug(f"Prediction failed for {horizon}: {e}")
                continue
        
        if best_prediction is None:
            no_prediction_count += 1
            continue
        
        # Apply filters
        max_conf = max(best_prediction.get('ce_confidence', 0), best_prediction.get('pe_confidence', 0))
        tp_prob = best_prediction.get('turning_point_prob', 0)
        
        if tp_prob > turning_point_threshold and use_turning_point_filter:
            turning_point_filtered += 1
            continue  # Skip trade if turning point detected
        
        if max_conf < min_confidence:
            low_confidence_filtered += 1
            continue  # Skip if confidence too low
        
        if best_expected_return < min_predicted_return:
            low_return_filtered += 1
            continue  # Skip if predicted return too low
        
        # Determine which option to trade
        option_type = None
        predicted_return = 0.0
        entry_price = 0.0
        actual_return = 0.0
        option_symbol = None
        
        if best_prediction['ce_return'] > best_prediction['pe_return']:
            option_type = 'CE'
            predicted_return = best_prediction['ce_return']
            entry_price = row['ce_price']
            option_symbol = row.get('ce_symbol')
            actual_return_col = f'ce_price_change_{best_horizon}'
            if actual_return_col in row and pd.notna(row[actual_return_col]):
                actual_return = row[actual_return_col]
        else:
            option_type = 'PE'
            predicted_return = best_prediction['pe_return']
            entry_price = row['pe_price']
            option_symbol = row.get('pe_symbol')
            actual_return_col = f'pe_price_change_{best_horizon}'
            if actual_return_col in row and pd.notna(row[actual_return_col]):
                actual_return = row[actual_return_col]
        
        if entry_price <= 0 or pd.isna(entry_price):
            missing_price_count += 1
            continue
        
        # Calculate exit timestamp (entry time + horizon duration)
        horizon_minutes = int(best_horizon.replace('m', ''))
        exit_timestamp = timestamp + timedelta(minutes=horizon_minutes)
        
        # Calculate position size
        confidence = max(best_prediction['ce_confidence'], best_prediction['pe_confidence'])
        risk_amount = account_size * max_risk_per_trade
        position_value = risk_amount / (abs(predicted_return) / 100.0) if predicted_return != 0 else 0
        quantity_lots = max(1, int(position_value / margin_per_lot))
        
        # Calculate PnL
        exit_price = entry_price * (1 + actual_return / 100.0)
        gross_pnl = (exit_price - entry_price) * quantity_lots * 50  # 50 is lot size
        transaction_cost = entry_price * quantity_lots * 50 * total_cost_rate
        net_pnl = gross_pnl - transaction_cost
        
            # Create trade record
        trade = OptionTradeRecord(
            timestamp=timestamp,
            option_type=option_type,
            horizon=best_horizon,
            predicted_return=predicted_return,
            actual_return=actual_return,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity_lots=quantity_lots,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            transaction_cost=transaction_cost,
            confidence=confidence,
            turning_point_prob=best_prediction['turning_point_prob'],
            recommendation=best_prediction['recommendation']
        )
        
        trades.append(trade)
        
        # Record trade to database
        try:
            # Prepare feature payload (sample of features for analysis)
            feature_payload = {col: float(feature_dict.get(col, 0.0)) for col in feature_cols[:50]}  # Limit to first 50 features
            
            # Prepare model metadata
            model_metadata = {
                'horizon': best_horizon,
                'predicted_ce_return': best_prediction.get('ce_return'),
                'predicted_pe_return': best_prediction.get('pe_return'),
                'ce_confidence': best_prediction.get('ce_confidence'),
                'pe_confidence': best_prediction.get('pe_confidence'),
                'turning_point_prob': best_prediction.get('turning_point_prob'),
                'recommendation': best_prediction.get('recommendation'),
                'backtest_config': {
                    'min_confidence': min_confidence,
                    'min_predicted_return': min_predicted_return,
                    'use_turning_point_filter': use_turning_point_filter,
                    'turning_point_threshold': turning_point_threshold
                }
            }
            
            db.record_option_return_backtest_trade(
                exchange=exchange,
                timestamp=timestamp,
                option_type=option_type,
                horizon=best_horizon,
                predicted_return=predicted_return,
                actual_return=actual_return,
                predicted_ce_return=best_prediction.get('ce_return', 0.0),
                predicted_pe_return=best_prediction.get('pe_return', 0.0),
                ce_confidence=best_prediction.get('ce_confidence', 0.0),
                pe_confidence=best_prediction.get('pe_confidence', 0.0),
                turning_point_prob=best_prediction.get('turning_point_prob', 0.0),
                recommendation=best_prediction.get('recommendation', ''),
                entry_price=entry_price,
                exit_price=exit_price,
                quantity_lots=quantity_lots,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                transaction_cost=transaction_cost,
                confidence=confidence,
                exit_timestamp=exit_timestamp,
                option_symbol=option_symbol,
                feature_payload=feature_payload,
                model_metadata=model_metadata
            )
        except Exception as e:
            LOGGER.warning(f"Failed to record trade to database: {e}")
    
    # Print diagnostics
    print(f"\n{'=' * 80}")
    print("BACKTEST DIAGNOSTICS")
    print(f"{'=' * 80}")
    print(f"Total rows processed: {len(df)}")
    print(f"Successful predictions: {successful_predictions}")
    print(f"Trades generated: {len(trades)}")
    print(f"\nFiltering breakdown:")
    print(f"  Missing price: {missing_price_count}")
    print(f"  No prediction: {no_prediction_count}")
    print(f"  Turning point filtered: {turning_point_filtered}")
    print(f"  Low confidence filtered: {low_confidence_filtered}")
    print(f"  Low return filtered: {low_return_filtered}")
    
    # Show sample predictions if available
    if successful_predictions > 0 and len(trades) == 0:
        print(f"\n⚠️  Predictions were made but no trades passed filters.")
        print(f"   Consider lowering --min-confidence or --min-return thresholds.")
        print(f"   Current thresholds: confidence >= {min_confidence}, return >= {min_predicted_return}%")
    
    LOGGER.info(f"Generated {len(trades)} trades")
    LOGGER.info(f"Successful predictions: {successful_predictions}")
    LOGGER.info(f"Filtered: {no_prediction_count} no prediction, {turning_point_filtered} turning point, "
                f"{low_confidence_filtered} low confidence, {low_return_filtered} low return, "
                f"{missing_price_count} missing price")
    
    # Calculate metrics
    if trades:
        total_pnl = sum(t.net_pnl for t in trades)
        winning_trades = [t for t in trades if t.net_pnl > 0]
        losing_trades = [t for t in trades if t.net_pnl < 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0.0
        avg_win = np.mean([t.net_pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.net_pnl for t in losing_trades]) if losing_trades else 0.0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf') if avg_win > 0 else 0.0
        
        # Prediction accuracy
        direction_correct = sum(1 for t in trades if np.sign(t.predicted_return) == np.sign(t.actual_return))
        direction_accuracy = direction_correct / len(trades) if trades else 0.0
        
        # Return prediction error
        prediction_errors = [abs(t.predicted_return - t.actual_return) for t in trades]
        mae = np.mean(prediction_errors) if prediction_errors else 0.0
        
        # By option type
        ce_trades = [t for t in trades if t.option_type == 'CE']
        pe_trades = [t for t in trades if t.option_type == 'PE']
        
        metrics = {
            'total_trades': len(trades),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'direction_accuracy': direction_accuracy,
            'mae': mae,
            'ce_trades': len(ce_trades),
            'pe_trades': len(pe_trades),
            'ce_total_pnl': sum(t.net_pnl for t in ce_trades),
            'pe_total_pnl': sum(t.net_pnl for t in pe_trades),
            'by_horizon': {}
        }
        
        # Metrics by horizon
        for horizon in ['3m', '5m', '10m', '15m']:
            horizon_trades = [t for t in trades if t.horizon == horizon]
            if horizon_trades:
                horizon_pnl = sum(t.net_pnl for t in horizon_trades)
                horizon_win_rate = len([t for t in horizon_trades if t.net_pnl > 0]) / len(horizon_trades)
                metrics['by_horizon'][horizon] = {
                    'num_trades': len(horizon_trades),
                    'total_pnl': horizon_pnl,
                    'win_rate': horizon_win_rate
                }
    else:
        metrics = {
            'total_trades': 0,
            'error': 'No trades generated'
        }
    
    config = {
        'exchange': exchange,
        'start_date': str(start_date),
        'end_date': str(end_date),
        'min_confidence': min_confidence,
        'min_predicted_return': min_predicted_return,
        'transaction_cost_bps': transaction_cost_bps,
        'slippage_bps': slippage_bps,
        'account_size': account_size
    }
    
    result = OptionBacktestResult(
        config=config,
        trades=trades,
        metrics=metrics
    )
    
    return result


def print_backtest_results(result: OptionBacktestResult):
    """Print formatted backtest results."""
    print("\n" + "=" * 80)
    print("OPTION RETURN BACKTEST RESULTS")
    print("=" * 80)
    
    print(f"\nConfig:")
    for key, value in result.config.items():
        print(f"  {key}: {value}")
    
    if not result.trades:
        print("\nNo trades generated.")
        return
    
    print(f"\n{'=' * 80}")
    print("OVERALL METRICS")
    print(f"{'=' * 80}")
    print(f"Total Trades: {result.metrics['total_trades']}")
    print(f"Total PnL: ₹{result.metrics['total_pnl']:,.2f}")
    print(f"Win Rate: {result.metrics['win_rate']:.1%}")
    print(f"Average Win: ₹{result.metrics['avg_win']:,.2f}")
    print(f"Average Loss: ₹{result.metrics['avg_loss']:,.2f}")
    print(f"Profit Factor: {result.metrics['profit_factor']:.2f}")
    print(f"Direction Accuracy: {result.metrics['direction_accuracy']:.1%}")
    print(f"Mean Absolute Error: {result.metrics['mae']:.2f}%")
    
    print(f"\n{'=' * 80}")
    print("BY OPTION TYPE")
    print(f"{'=' * 80}")
    print(f"CE Trades: {result.metrics['ce_trades']}, PnL: ₹{result.metrics['ce_total_pnl']:,.2f}")
    print(f"PE Trades: {result.metrics['pe_trades']}, PnL: ₹{result.metrics['pe_total_pnl']:,.2f}")
    
    print(f"\n{'=' * 80}")
    print("BY HORIZON")
    print(f"{'=' * 80}")
    for horizon, horizon_metrics in result.metrics['by_horizon'].items():
        print(f"{horizon}:")
        print(f"  Trades: {horizon_metrics['num_trades']}")
        print(f"  PnL: ₹{horizon_metrics['total_pnl']:,.2f}")
        print(f"  Win Rate: {horizon_metrics['win_rate']:.1%}")
    
    print(f"\n{'=' * 80}")
    print("SAMPLE TRADES (First 10)")
    print(f"{'=' * 80}")
    print(f"{'Timestamp':<20} {'Type':<4} {'Horizon':<6} {'Pred %':<8} {'Actual %':<9} {'PnL':<12} {'Conf':<6}")
    print("-" * 80)
    for trade in result.trades[:10]:
        print(f"{str(trade.timestamp):<20} {trade.option_type:<4} {trade.horizon:<6} "
              f"{trade.predicted_return:>7.2f}% {trade.actual_return:>8.2f}% "
              f"₹{trade.net_pnl:>10,.2f} {trade.confidence:>5.2f}")


def main():
    parser = argparse.ArgumentParser(description='Backtest option return prediction models')
    parser.add_argument('--exchange', '-e', default='NSE', help='Exchange name')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--min-confidence', type=float, default=0.5, help='Minimum confidence')
    parser.add_argument('--min-return', type=float, default=0.5, help='Minimum predicted return %')
    parser.add_argument('--output', '-o', help='Output JSON file for results')
    
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start, '%Y-%m-%d').date() if args.start else None
    end_date = datetime.strptime(args.end, '%Y-%m-%d').date() if args.end else None
    
    result = backtest_option_returns(
        exchange=args.exchange,
        start_date=start_date,
        end_date=end_date,
        min_confidence=args.min_confidence,
        min_predicted_return=args.min_return
    )
    
    print_backtest_results(result)
    
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        print(f"\n✓ Results saved to {args.output}")


if __name__ == '__main__':
    main()
