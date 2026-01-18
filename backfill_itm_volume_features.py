#!/usr/bin/env python3
"""
Backfill script for ITM Volume Percentage Change Features and PCRV.

This script backfills:
- itm_volume_ce_pct_change_3m_wavg (ITM Call Volume % Change)
- itm_volume_pe_pct_change_3m_wavg (ITM Put Volume % Change)
- pcr_total_volume (PCRV - Put Call Ratio of Volume)

It recalculates these features from historical option chain snapshots and updates
the ml_features table's feature_payload JSONB column and pcr_total_volume column.

Usage:
    python backfill_itm_volume_features.py --exchange NSE --days 90
    python backfill_itm_volume_features.py --exchange BSE --days 90
"""

import argparse
import logging
import json
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
import sys

from database_new import (
    get_db_connection, release_db_connection, _get_placeholder, 
    _coerce_iso_timestamp, db_lock
)
from feature_engineering import (
    _calculate_option_aggregates,
    _is_itm,
    _safe_div
)
from time_utils import now_ist, to_ist, today_ist

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_option_snapshots_for_timestamp(
    exchange: str, 
    timestamp: datetime
) -> tuple[List[Dict], List[Dict], Optional[float], Optional[float]]:
    """
    Load option chain snapshots for a specific timestamp and reconstruct call/put option lists.
    
    Returns:
        (calls, puts, underlying_price, atm_strike)
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        ph = _get_placeholder()
        timestamp_iso = _coerce_iso_timestamp(timestamp)
        
        # Load all option snapshots for this timestamp
        cursor.execute(f"""
            SELECT strike, option_type, oi, ltp, token, underlying_price,
                   pct_change_3m, pct_change_5m, pct_change_10m, pct_change_15m, pct_change_30m,
                   iv, volume, best_bid, best_ask, bid_quantity, ask_quantity,
                   spread, order_book_imbalance, moneyness, time_to_expiry_seconds
            FROM option_chain_snapshots
            WHERE exchange = {ph} AND timestamp = {ph}
            ORDER BY strike, option_type
        """, (exchange, timestamp_iso))
        
        rows = cursor.fetchall()
        if not rows:
            return [], [], None, None
        
        calls = []
        puts = []
        underlying_price = None
        strikes = []
        
        for row in rows:
            (strike, opt_type, oi, ltp, token, underlying_price_val,
             pct_change_3m, pct_change_5m, pct_change_10m, pct_change_15m, pct_change_30m,
             iv, volume, best_bid, best_ask, bid_quantity, ask_quantity,
             spread, order_book_imbalance, moneyness, time_to_expiry_seconds) = row
            
            if underlying_price_val and underlying_price is None:
                underlying_price = float(underlying_price_val)
            
            strikes.append(float(strike))
            
            # Calculate position (distance from ATM)
            # We'll calculate ATM after we have underlying_price
            opt_dict = {
                'strike': float(strike),
                'latest_oi': int(oi) if oi else 0,
                'ltp': float(ltp) if ltp else 0.0,
                'instrument_token': int(token) if token else 0,
                'volume': int(volume) if volume else 0,
                'iv': float(iv) if iv else None,
                'best_bid': float(best_bid) if best_bid else None,
                'best_ask': float(best_ask) if best_ask else None,
                'bid_quantity': float(bid_quantity) if bid_quantity else None,
                'ask_quantity': float(ask_quantity) if ask_quantity else None,
                'spread': float(spread) if spread else None,
                'order_book_imbalance': float(order_book_imbalance) if order_book_imbalance else None,
                'pct_changes': {
                    '3m': float(pct_change_3m) if pct_change_3m is not None else None,
                    '5m': float(pct_change_5m) if pct_change_5m is not None else None,
                    '10m': float(pct_change_10m) if pct_change_10m is not None else None,
                    '15m': float(pct_change_15m) if pct_change_15m is not None else None,
                    '30m': float(pct_change_30m) if pct_change_30m is not None else None,
                }
            }
            
            if opt_type.upper() == 'CE':
                calls.append(opt_dict)
            elif opt_type.upper() == 'PE':
                puts.append(opt_dict)
        
        # Calculate ATM strike (nearest strike to underlying price)
        atm_strike = None
        if underlying_price and strikes:
            # Find nearest strike
            strikes_sorted = sorted(set(strikes))
            atm_strike = min(strikes_sorted, key=lambda s: abs(s - underlying_price))
            
            # Calculate position for each option (distance from ATM in strikes)
            strike_step = 50.0 if exchange == 'NSE' else 100.0  # Approximate
            for opt in calls + puts:
                position = (opt['strike'] - atm_strike) / strike_step
                opt['position'] = round(position, 1)
        
        return calls, puts, underlying_price, atm_strike
        
    except Exception as e:
        logger.error(f"Error loading option snapshots: {e}", exc_info=True)
        return [], [], None, None
    finally:
        release_db_connection(conn)


def calculate_new_features(
    calls: List[Dict],
    puts: List[Dict],
    atm_strike: float,
    underlying_price: float,
    time_to_expiry_days: float = 1.0
) -> Dict[str, float]:
    """
    Calculate the new ITM volume percentage change features and PCRV.
    
    Returns:
        Dictionary with:
        - itm_volume_ce_pct_change_3m_wavg
        - itm_volume_pe_pct_change_3m_wavg
        - pcr_total_volume (PCRV)
    """
    if not calls or not puts or not atm_strike:
        return {
            'itm_volume_ce_pct_change_3m_wavg': 0.0,
            'itm_volume_pe_pct_change_3m_wavg': 0.0,
            'pcr_total_volume': 0.0
        }
    
    try:
        # Use the existing _calculate_option_aggregates function
        option_aggs = _calculate_option_aggregates(
            call_options=calls,
            put_options=puts,
            atm_strike=atm_strike,
            time_to_expiry_days=time_to_expiry_days,
            spot_price=underlying_price,
            risk_free_rate=0.10
        )
        
        # Calculate PCRV (Put Call Ratio of Volume)
        pcrv = _safe_div(option_aggs.total_pe_volume, option_aggs.total_ce_volume)
        
        return {
            'itm_volume_ce_pct_change_3m_wavg': option_aggs.itm_volume_ce_pct_change_3m_wavg,
            'itm_volume_pe_pct_change_3m_wavg': option_aggs.itm_volume_pe_pct_change_3m_wavg,
            'pcr_total_volume': pcrv
        }
    except Exception as e:
        logger.error(f"Error calculating new features: {e}", exc_info=True)
        return {
            'itm_volume_ce_pct_change_3m_wavg': 0.0,
            'itm_volume_pe_pct_change_3m_wavg': 0.0,
            'pcr_total_volume': 0.0
        }


def update_ml_features_payload(
    exchange: str,
    timestamp: datetime,
    new_features: Dict[str, float]
) -> bool:
    """
    Update the feature_payload JSONB column and pcr_total_volume column in ml_features table.
    
    Updates:
    - feature_payload JSONB with itm_volume_ce_pct_change_3m_wavg and itm_volume_pe_pct_change_3m_wavg
    - pcr_total_volume column directly (PCRV)
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        ph = _get_placeholder()
        timestamp_iso = _coerce_iso_timestamp(timestamp)
        
        # Get existing feature_payload
        cursor.execute(f"""
            SELECT feature_payload, pcr_total_volume
            FROM ml_features
            WHERE exchange = {ph} 
              AND DATE_TRUNC('minute', timestamp) = DATE_TRUNC('minute', {ph}::timestamp)
            LIMIT 1
        """, (exchange, timestamp_iso))
        
        row = cursor.fetchone()
        if not row:
            logger.debug(f"No ml_features record found for {exchange} at {timestamp_iso}")
            return False
        
        existing_payload = row[0]
        existing_pcrv = row[1]
        
        # Parse existing payload
        if existing_payload:
            try:
                if isinstance(existing_payload, str):
                    payload_dict = json.loads(existing_payload)
                else:
                    payload_dict = existing_payload
            except (json.JSONDecodeError, TypeError):
                payload_dict = {}
        else:
            payload_dict = {}
        
        # Update with new features (ITM volume percentage changes)
        payload_dict.update({
            'itm_volume_ce_pct_change_3m_wavg': new_features.get('itm_volume_ce_pct_change_3m_wavg', 0.0),
            'itm_volume_pe_pct_change_3m_wavg': new_features.get('itm_volume_pe_pct_change_3m_wavg', 0.0)
        })
        
        # Get PCRV value
        pcrv_value = new_features.get('pcr_total_volume', existing_pcrv if existing_pcrv is not None else 0.0)
        
        # Update both feature_payload and pcr_total_volume column
        updated_payload = json.dumps(payload_dict)
        cursor.execute(f"""
            UPDATE ml_features
            SET feature_payload = {ph},
                pcr_total_volume = {ph}
            WHERE exchange = {ph}
              AND DATE_TRUNC('minute', timestamp) = DATE_TRUNC('minute', {ph}::timestamp)
        """, (updated_payload, pcrv_value, exchange, timestamp_iso))
        
        conn.commit()
        return cursor.rowcount > 0
        
    except Exception as e:
        logger.error(f"Error updating ml_features: {e}", exc_info=True)
        if 'conn' in locals():
            conn.rollback()
        return False
    finally:
        release_db_connection(conn)


def get_timestamps_to_backfill(exchange: str, days: int) -> List[datetime]:
    """
    Get list of unique timestamps from ml_features table that need backfilling.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        ph = _get_placeholder()
        end_date = today_ist()
        start_date = end_date - timedelta(days=days)
        
        cursor.execute(f"""
            SELECT DISTINCT timestamp
            FROM ml_features
            WHERE exchange = {ph}
              AND timestamp >= {ph}
              AND timestamp <= {ph}
            ORDER BY timestamp ASC
        """, (exchange, start_date.isoformat(), end_date.isoformat()))
        
        timestamps = [row[0] for row in cursor.fetchall()]
        return timestamps
        
    except Exception as e:
        logger.error(f"Error getting timestamps: {e}", exc_info=True)
        return []
    finally:
        release_db_connection(conn)


def backfill_features(exchange: str, days: int, batch_size: int = 100):
    """
    Main backfill function.
    """
    logger.info(f"Starting backfill for {exchange} (last {days} days)")
    
    # Get timestamps to process
    timestamps = get_timestamps_to_backfill(exchange, days)
    logger.info(f"Found {len(timestamps)} timestamps to process")
    
    if not timestamps:
        logger.warning("No timestamps found to backfill")
        return
    
    updated_count = 0
    skipped_count = 0
    error_count = 0
    
    for i, timestamp in enumerate(timestamps, 1):
        try:
            # Load option snapshots
            calls, puts, underlying_price, atm_strike = load_option_snapshots_for_timestamp(
                exchange, timestamp
            )
            
            if not calls or not puts:
                skipped_count += 1
                if i % 100 == 0:
                    logger.debug(f"Progress: {i}/{len(timestamps)} - Skipped (no option data)")
                continue
            
            if not underlying_price or not atm_strike:
                skipped_count += 1
                if i % 100 == 0:
                    logger.debug(f"Progress: {i}/{len(timestamps)} - Skipped (no price/ATM)")
                continue
            
            # Calculate time to expiry (approximate - use 1 day default)
            time_to_expiry_days = 1.0
            
            # Calculate new features
            new_features = calculate_new_features(
                calls, puts, atm_strike, underlying_price, time_to_expiry_days
            )
            
            # Update ml_features table
            if update_ml_features_payload(exchange, timestamp, new_features):
                updated_count += 1
                if updated_count % 50 == 0:
                    logger.info(
                        f"Progress: {i}/{len(timestamps)} - "
                        f"Updated: {updated_count}, Skipped: {skipped_count}, Errors: {error_count}"
                    )
            else:
                skipped_count += 1
                
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing timestamp {timestamp}: {e}", exc_info=True)
            if error_count > 10:
                logger.error("Too many errors, stopping backfill")
                break
    
    logger.info(
        f"Backfill complete! Updated: {updated_count}, Skipped: {skipped_count}, Errors: {error_count}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Backfill ITM volume percentage change features and PCRV (Put Call Ratio of Volume)"
    )
    parser.add_argument(
        '--exchange',
        required=True,
        choices=['NSE', 'BSE'],
        help='Exchange to backfill'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=90,
        help='Number of days to backfill (default: 90)'
    )
    
    args = parser.parse_args()
    
    backfill_features(args.exchange, args.days)


if __name__ == '__main__':
    main()
