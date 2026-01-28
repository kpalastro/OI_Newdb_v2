#!/usr/bin/env python3
"""
Backfill metadata for paper_trading_metrics table.

This script attempts to reconstruct metadata for trades that were recorded
before the metadata fix was implemented.

Sources:
1. Recommendations log (logs/recommendations/YYYY-MM-DD.jsonl)
2. ml_features table (for option return predictions)
3. Basic reconstruction from existing fields
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from dateutil import parser as date_parser

import database_new as db
from time_utils import now_ist


def convert_to_json_serializable(obj: Any) -> Any:
    """Convert numpy/pandas types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def load_recommendations_log(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Load recommendations from JSONL files."""
    LOGGER.info(f"Loading recommendations from {start_date.date()} to {end_date.date()}...")
    
    recommendations = []
    current_date = start_date.date()
    end_date_only = end_date.date()
    
    while current_date <= end_date_only:
        log_file = Path(f"logs/recommendations/{current_date.strftime('%Y-%m-%d')}.jsonl")
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            rec['log_date'] = current_date
                            recommendations.append(rec)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                LOGGER.warning(f"Error reading {log_file}: {e}")
        
        current_date += timedelta(days=1)
    
    if not recommendations:
        LOGGER.warning("No recommendations found in log files")
        return pd.DataFrame()
    
    df = pd.DataFrame(recommendations)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    LOGGER.info(f"Loaded {len(df)} recommendations")
    return df


def match_recommendation(
    timestamp: datetime,
    exchange: str,
    signal: str,
    confidence: float,
    recommendations_df: pd.DataFrame,
    time_window_seconds: int = 30
) -> Optional[Dict[str, Any]]:
    """Match a paper trading metric with a recommendation log entry."""
    if recommendations_df.empty:
        return None
    
    # Filter by exchange
    exchange_recs = recommendations_df[recommendations_df['exchange'] == exchange].copy()
    if exchange_recs.empty:
        return None
    
    # Find closest match by timestamp
    exchange_recs['time_diff'] = (exchange_recs['timestamp'] - timestamp).abs()
    exchange_recs = exchange_recs[exchange_recs['time_diff'] <= timedelta(seconds=time_window_seconds)]
    
    if exchange_recs.empty:
        return None
    
    # Try to match by signal and confidence (fuzzy match)
    if signal and 'signal' in exchange_recs.columns:
        signal_match = exchange_recs[exchange_recs['signal'] == signal]
        if not signal_match.empty:
            exchange_recs = signal_match
    
    # Get closest match
    closest = exchange_recs.loc[exchange_recs['time_diff'].idxmin()]
    
    # Build metadata from recommendation
    metadata = {}
    if 'regime' in closest:
        metadata['regime'] = convert_to_json_serializable(closest['regime'])
    if 'kelly_fraction' in closest:
        metadata['kelly_fraction'] = convert_to_json_serializable(closest['kelly_fraction'])
    if 'recommended_lots' in closest:
        metadata['recommended_lots'] = convert_to_json_serializable(closest['recommended_lots'])
    if 'model_version' in closest:
        metadata['model_version'] = convert_to_json_serializable(closest['model_version'])
    
    return metadata if metadata else None


def get_ml_features_metadata(
    timestamp: datetime,
    exchange: str
) -> Optional[Dict[str, Any]]:
    """Try to get option return prediction metadata from ml_features table."""
    try:
        conn = db.get_db_connection()
        cursor = conn.cursor()
        
        # Find closest ml_features record
        # Check if option_return_prediction column exists first
        check_col_query = """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'ml_features' 
              AND column_name = 'option_return_prediction'
        """
        cursor.execute(check_col_query)
        has_opt_pred_col = cursor.fetchone() is not None
        
        if has_opt_pred_col:
            query = """
                SELECT 
                    option_return_prediction,
                    ce_return, pe_return,
                    ce_confidence, pe_confidence,
                    turning_point_prob
                FROM ml_features
                WHERE exchange = %s
                  AND timestamp >= %s - INTERVAL '5 minutes'
                  AND timestamp <= %s + INTERVAL '5 minutes'
                ORDER BY ABS(EXTRACT(EPOCH FROM (timestamp - %s)))
                LIMIT 1
            """
        else:
            # Fallback: try to get from other columns if option_return_prediction doesn't exist
            query = """
                SELECT 
                    NULL as option_return_prediction,
                    ce_return, pe_return,
                    ce_confidence, pe_confidence,
                    turning_point_prob
                FROM ml_features
                WHERE exchange = %s
                  AND timestamp >= %s - INTERVAL '5 minutes'
                  AND timestamp <= %s + INTERVAL '5 minutes'
                ORDER BY ABS(EXTRACT(EPOCH FROM (timestamp - %s)))
                LIMIT 1
            """
        
        cursor.execute(query, (exchange, timestamp, timestamp, timestamp))
        row = cursor.fetchone()
        
        db.release_db_connection(conn)
        
        if row and row[0]:  # option_return_prediction exists
            metadata = {}
            
            # Parse option_return_prediction JSONB if it's a string
            opt_pred = row[0]
            if isinstance(opt_pred, str):
                try:
                    opt_pred = json.loads(opt_pred)
                except:
                    pass
            
            if isinstance(opt_pred, dict):
                # Extract best prediction
                best_horizon = opt_pred.get('best_horizon')
                if best_horizon:
                    pred_data = opt_pred.get(best_horizon, {})
                    if pred_data:
                        metadata['strategy_name'] = 'OptimizedOptionReturn'
                        metadata['horizon'] = convert_to_json_serializable(best_horizon)
                        metadata['predicted_return'] = convert_to_json_serializable(pred_data.get('max_return', 0.0))
                        metadata['option_type'] = convert_to_json_serializable(pred_data.get('recommended_type', 'CE'))
                        metadata['ce_return'] = convert_to_json_serializable(pred_data.get('ce_return', 0.0))
                        metadata['pe_return'] = convert_to_json_serializable(pred_data.get('pe_return', 0.0))
                        metadata['ce_confidence'] = convert_to_json_serializable(pred_data.get('ce_confidence', 0.0))
                        metadata['pe_confidence'] = convert_to_json_serializable(pred_data.get('pe_confidence', 0.0))
                        metadata['option_return_prediction'] = convert_to_json_serializable(opt_pred)
            
            if row[1] is not None:  # ce_return
                metadata['ce_return'] = convert_to_json_serializable(row[1])
            if row[2] is not None:  # pe_return
                metadata['pe_return'] = convert_to_json_serializable(row[2])
            if row[3] is not None:  # ce_confidence
                metadata['ce_confidence'] = convert_to_json_serializable(row[3])
            if row[4] is not None:  # pe_confidence
                metadata['pe_confidence'] = convert_to_json_serializable(row[4])
            if row[5] is not None:  # turning_point_prob
                metadata['turning_point_prob'] = convert_to_json_serializable(row[5])
            
            return metadata if metadata else None
            
    except Exception as e:
        LOGGER.debug(f"Error getting ml_features metadata: {e}")
        return None


def reconstruct_metadata(
    timestamp: datetime,
    exchange: str,
    signal: str,
    confidence: float,
    recommendations_df: pd.DataFrame
) -> Optional[Dict[str, Any]]:
    """Reconstruct metadata from available sources."""
    metadata = {}
    
    # 1. Try to match with recommendations log
    rec_metadata = match_recommendation(timestamp, exchange, signal, confidence, recommendations_df)
    if rec_metadata:
        metadata.update(rec_metadata)
    
    # 2. Try to get option return prediction from ml_features
    ml_metadata = get_ml_features_metadata(timestamp, exchange)
    if ml_metadata:
        metadata.update(ml_metadata)
    
    # 3. Add basic metadata if we have signal info
    if signal and signal != 'HOLD':
        if not metadata.get('strategy_name'):
            # Infer strategy from signal characteristics
            if metadata.get('option_return_prediction'):
                metadata['strategy_name'] = 'OptimizedOptionReturn'
            else:
                metadata['strategy_name'] = 'ML_Base'
    
    return metadata if metadata else None


def backfill_metadata(
    exchange: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    dry_run: bool = True
) -> Dict[str, Any]:
    """
    Backfill metadata for paper_trading_metrics entries with NULL metadata.
    
    Args:
        exchange: Exchange to backfill (None for all)
        start_date: Start date for backfilling (None for all)
        end_date: End date for backfilling (None for all)
        dry_run: If True, only show what would be updated without making changes
    """
    LOGGER.info("=" * 80)
    LOGGER.info("BACKFILLING PAPER TRADING METADATA")
    LOGGER.info("=" * 80)
    
    if dry_run:
        LOGGER.info("🔍 DRY RUN MODE - No changes will be made")
    
    # Load recommendations
    if start_date and end_date:
        recommendations_df = load_recommendations_log(start_date, end_date)
    else:
        # Load last 30 days
        end_date = now_ist()
        start_date = end_date - timedelta(days=30)
        recommendations_df = load_recommendations_log(start_date, end_date)
    
    # Query paper_trading_metrics with NULL metadata
    conn = db.get_db_connection()
    cursor = conn.cursor()
    
    query = """
        SELECT 
            id, timestamp, exchange, executed, reason, signal,
            confidence, quantity_lots, pnl, constraint_violation
        FROM paper_trading_metrics
        WHERE metadata IS NULL
    """
    params = []
    
    if exchange:
        query += " AND exchange = %s"
        params.append(exchange)
    
    if start_date:
        query += " AND timestamp >= %s"
        params.append(start_date)
    
    if end_date:
        query += " AND timestamp <= %s"
        params.append(end_date)
    
    query += " ORDER BY timestamp DESC"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    LOGGER.info(f"Found {len(rows)} records with NULL metadata")
    
    if not rows:
        LOGGER.info("No records to backfill")
        db.release_db_connection(conn)
        return {'updated': 0, 'failed': 0, 'skipped': 0}
    
    # Process each record
    updated = 0
    failed = 0
    skipped = 0
    
    for row in rows:
        (record_id, timestamp, exchange_val, executed, reason, signal,
         confidence, quantity_lots, pnl, constraint_violation) = row
        
        try:
            # Reconstruct metadata
            metadata = reconstruct_metadata(
                timestamp, exchange_val, signal, confidence, recommendations_df
            )
            
            if not metadata:
                skipped += 1
                LOGGER.debug(f"Could not reconstruct metadata for record {record_id}")
                continue
            
            # Convert all values to JSON-serializable types
            metadata = convert_to_json_serializable(metadata)
            
            if dry_run:
                LOGGER.info(f"[DRY RUN] Would update record {record_id} ({exchange_val}, {timestamp}): {metadata}")
                updated += 1
            else:
                # Update database
                try:
                    metadata_json = json.dumps(metadata)
                    update_query = """
                        UPDATE paper_trading_metrics
                        SET metadata = %s::jsonb
                        WHERE id = %s
                    """
                    cursor.execute(update_query, (metadata_json, record_id))
                    updated += 1
                    if updated % 100 == 0:
                        conn.commit()
                        LOGGER.info(f"Updated {updated} records...")
                except (TypeError, ValueError) as json_err:
                    failed += 1
                    LOGGER.error(f"JSON serialization error for record {record_id}: {json_err}")
                    LOGGER.debug(f"Problematic metadata: {metadata}")
                    continue
        
        except Exception as e:
            failed += 1
            LOGGER.error(f"Error processing record {record_id}: {e}")
    
    if not dry_run:
        conn.commit()
        LOGGER.info(f"Committed {updated} updates")
    
    db.release_db_connection(conn)
    
    result = {
        'updated': updated,
        'failed': failed,
        'skipped': skipped,
        'total': len(rows)
    }
    
    LOGGER.info("=" * 80)
    LOGGER.info("BACKFILL SUMMARY")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Total records: {result['total']}")
    LOGGER.info(f"Updated: {result['updated']}")
    LOGGER.info(f"Skipped (no metadata found): {result['skipped']}")
    LOGGER.info(f"Failed: {result['failed']}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Backfill metadata for paper_trading_metrics")
    parser.add_argument('--exchange', help='Exchange to backfill (NSE, BSE, etc.)')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--execute', action='store_true', 
                       help='Actually update database (default is dry-run)')
    
    args = parser.parse_args()
    
    start_date = None
    end_date = None
    
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    result = backfill_metadata(
        exchange=args.exchange,
        start_date=start_date,
        end_date=end_date,
        dry_run=not args.execute
    )
    
    if not args.execute:
        print("\n💡 To actually update the database, run with --execute flag")
