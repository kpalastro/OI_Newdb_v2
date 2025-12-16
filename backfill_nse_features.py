#!/usr/bin/env python3
"""
Backfill script for NSE Option Chain Features.

This script backfills the 8 new NSE option chain features for the last N days
using historical data from the database and Kite API.

Usage:
    python backfill_nse_features.py --days 10 --exchange NSE
"""

import argparse
import logging
from datetime import datetime, date, timedelta
from typing import Optional, Dict
import sys

from database_new import get_db_connection, release_db_connection, _get_placeholder, _sanitize_feature_value
from config import get_config
from feature_engineering import (
    _calculate_nearest_atm_strike,
    _fetch_nse_option_chain_data,
    _parse_nse_option_chain_data,
    _get_cached_nse_data,
    _cache_nse_data
)
from time_utils import now_ist, to_ist
from kite_trade import KiteApp

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_underlying_token_for_exchange(exchange: str, kite_obj: KiteApp) -> Optional[int]:
    """Get underlying token for an exchange."""
    try:
        # Get config to find the correct exchange segment and symbol
        config = get_config()
        exchange_config = config.exchange_configs.get(exchange, {})
        
        # For NSE, underlying_symbol is 'NIFTY 50' and ltp_exchange is 'NSE'
        underlying_symbol = exchange_config.get('underlying_symbol', 'NIFTY 50')
        ltp_exchange = exchange_config.get('ltp_exchange', 'NSE')
        
        # Fetch instruments for the ltp_exchange (e.g., 'NSE' for NSE)
        instruments = kite_obj.instruments(ltp_exchange)
        logger.info(f"Fetched {len(instruments)} instruments from {ltp_exchange} exchange")
        
        # Search for the underlying symbol in the instruments
        for inst in instruments:
            if (inst.get('exchange') == ltp_exchange and 
                inst.get('tradingsymbol') == underlying_symbol):
                token = inst.get('instrument_token')
                logger.info(f"Found {underlying_symbol} token: {token} in {ltp_exchange}")
                return token
        
        # If not found, try alternative approaches
        logger.warning(f"Could not find {underlying_symbol} in {ltp_exchange} exchange")
        logger.info("Trying alternative search patterns...")
        
        # Try searching by name field (sometimes name matches instead of tradingsymbol)
        for inst in instruments:
            if inst.get('exchange') == ltp_exchange:
                if (inst.get('name') == underlying_symbol or 
                    inst.get('name') == underlying_symbol.replace(' ', '')):
                    token = inst.get('instrument_token')
                    logger.info(f"Found {underlying_symbol} by name, token: {token}")
                    return token
        
        # Try searching with space variations
        variations = [underlying_symbol, underlying_symbol.replace(' ', '-'), underlying_symbol.replace(' ', '')]
        for variation in variations:
            for inst in instruments:
                if (inst.get('exchange') == ltp_exchange and 
                    inst.get('tradingsymbol') == variation):
                    token = inst.get('instrument_token')
                    logger.info(f"Found {variation} token: {token}")
                    return token
        
        # Log some examples to help debug
        sample_symbols = [inst.get('tradingsymbol') for inst in instruments[:20] if inst.get('tradingsymbol')]
        logger.debug(f"Sample symbols in {ltp_exchange}: {sample_symbols[:10]}")
        
        # Log all unique names to help identify the correct field
        unique_names = set(inst.get('name') for inst in instruments if inst.get('name'))
        nifty_names = [name for name in unique_names if 'NIFTY' in name.upper() or '50' in name]
        if nifty_names:
            logger.debug(f"Found NIFTY-related names in {ltp_exchange}: {nifty_names[:5]}")
        
        return None
    except Exception as e:
        logger.error(f"Error getting underlying token: {e}", exc_info=True)
        return None


def get_open_price_from_kite(kite_obj: KiteApp, underlying_token: int, 
                             target_date: date) -> Optional[float]:
    """
    Get open price for a specific date from Kite historical data.
    
    Args:
        kite_obj: KiteApp instance
        underlying_token: Instrument token for the underlying
        target_date: Date to get open price for
    
    Returns:
        Open price for the date or None if not available
    """
    try:
        # Market opens at 9:15 AM IST
        from_dt = datetime.combine(target_date, datetime.min.time().replace(hour=9, minute=15))
        to_dt = datetime.combine(target_date, datetime.min.time().replace(hour=16, minute=0))
        
        # Ensure IST timezone
        from_dt = to_ist(from_dt)
        to_dt = to_ist(to_dt)
        
        # Try to fetch historical data
        candles = None
        try:
            candles = kite_obj.historical_data(
                underlying_token,
                from_dt,
                to_dt,
                'minute',
                continuous=False,
                oi=False
            )
        except (KeyError, TypeError, AttributeError) as api_error:
            # API response might not have expected structure
            logger.debug(f"Kite API error for {target_date}: {api_error}")
            return None
        except Exception as api_error:
            # Other API errors (network, auth, etc.)
            logger.debug(f"Kite API request failed for {target_date}: {api_error}")
            return None
        
        # Validate candles data
        if candles is None:
            logger.debug(f"No candles data returned from Kite for {target_date}")
            return None
        
        if not isinstance(candles, list):
            logger.debug(f"Kite returned non-list data for {target_date}: {type(candles)}")
            return None
        
        if len(candles) == 0:
            logger.debug(f"Empty candles list from Kite for {target_date} (possibly holiday/weekend)")
            return None
        
        # Get the first candle of the day (9:15 AM)
        first_candle = candles[0]
        if not isinstance(first_candle, dict):
            logger.debug(f"First candle is not a dict for {target_date}: {type(first_candle)}")
            return None
        
        open_price = first_candle.get('open')
        if open_price is not None:
            try:
                return float(open_price)
            except (ValueError, TypeError):
                logger.debug(f"Could not convert open price to float for {target_date}: {open_price}")
                return None
        
        logger.debug(f"No open price in first candle for {target_date}")
        return None
        
    except Exception as e:
        # Kite may not have historical data for weekends/holidays/old dates - this is expected
        logger.debug(f"Kite API unavailable for {target_date} (will use database price): {type(e).__name__}")
        return None


def get_historical_records(exchange: str, days: int) -> list:
    """Get historical records from ml_features table for the last N days."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        ph = _get_placeholder()
        
        # Calculate date range (start of day to end of day in IST)
        end_date = now_ist()
        start_date = end_date - timedelta(days=days)
        
        # Set to start and end of day
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        query = f"""
            SELECT timestamp, exchange, underlying_price, feature_payload
            FROM ml_features
            WHERE exchange = {ph}
              AND timestamp >= {ph}
              AND timestamp <= {ph}
              AND underlying_price IS NOT NULL
            ORDER BY timestamp ASC
        """
        
        cursor.execute(query, (exchange, start_date, end_date))
        rows = cursor.fetchall()
        
        # Convert to list of dicts
        columns = [desc[0] for desc in cursor.description]
        records = []
        for row in rows:
            record = dict(zip(columns, row))
            # Ensure timestamp is datetime object
            if isinstance(record['timestamp'], str):
                record['timestamp'] = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
            records.append(record)
        
        release_db_connection(conn)
        
        logger.info(f"Found {len(records)} records for {exchange} from {start_date.date()} to {end_date.date()}")
        return records
        
    except Exception as e:
        logger.error(f"Error loading historical records: {e}", exc_info=True)
        if 'conn' in locals():
            release_db_connection(conn)
        return []


def update_record_with_nse_features(exchange: str, timestamp: datetime, 
                                   open_price: float, strike_difference: int,
                                   nse_metrics: Dict[str, float]) -> bool:
    """
    Update a single record in ml_features table with NSE option chain features.
    
    Args:
        exchange: Exchange name
        timestamp: Timestamp of the record
        open_price: Open price used to calculate ATM strike
        strike_difference: Strike difference (50 for NIFTY, 100 for BANKNIFTY)
        nse_metrics: Dictionary with NSE feature values
    
    Returns:
        True if update successful, False otherwise
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        ph = _get_placeholder()
        
        # Ensure timestamp is timezone-aware for comparison
        if timestamp.tzinfo is None:
            timestamp = to_ist(timestamp)
        else:
            timestamp = to_ist(timestamp)
        
        # Prepare update values
        update_vals = [
            _sanitize_feature_value(nse_metrics.get('total_oi_call', 0.0)),
            _sanitize_feature_value(nse_metrics.get('total_oi_put', 0.0)),
            _sanitize_feature_value(nse_metrics.get('total_oi_change_call', 0.0)),
            _sanitize_feature_value(nse_metrics.get('total_oi_change_put', 0.0)),
            _sanitize_feature_value(nse_metrics.get('total_volume_call', 0.0)),
            _sanitize_feature_value(nse_metrics.get('total_volume_put', 0.0)),
            _sanitize_feature_value(nse_metrics.get('oi_change_diff_put_call', 0.0)),
            _sanitize_feature_value(nse_metrics.get('oi_change_diff_put_call', 0.0)),  # oi_next_sentiment
        ]
        
        # Update query for PostgreSQL
        if get_config().db_type == 'postgres':
            update_query = f"""
                UPDATE ml_features
                SET nse_next_oi_call_total = {ph},
                    nse_next_oi_put_total = {ph},
                    nse_next_oi_change_call_total = {ph},
                    nse_next_oi_change_put_total = {ph},
                    nse_next_volume_call_total = {ph},
                    nse_next_volume_put_total = {ph},
                    nse_next_oi_change_diff_put_call = {ph},
                    oi_next_sentiment = {ph}
                WHERE exchange = {ph} AND timestamp = {ph}
            """
            cursor.execute(update_query, (*update_vals, exchange, timestamp))
        else:
            # SQLite
            update_query = f"""
                UPDATE ml_features
                SET nse_next_oi_call_total = {ph},
                    nse_next_oi_put_total = {ph},
                    nse_next_oi_change_call_total = {ph},
                    nse_next_oi_change_put_total = {ph},
                    nse_next_volume_call_total = {ph},
                    nse_next_volume_put_total = {ph},
                    nse_next_oi_change_diff_put_call = {ph},
                    oi_next_sentiment = {ph}
                WHERE exchange = {ph} AND timestamp = {ph}
            """
            cursor.execute(update_query, (*update_vals, exchange, timestamp))
        
        rows_updated = cursor.rowcount
        conn.commit()
        release_db_connection(conn)
        
        if rows_updated > 0:
            logger.debug(f"Updated record for {exchange} at {timestamp}")
            return True
        else:
            logger.warning(f"No record updated for {exchange} at {timestamp} (record may not exist)")
            return False
            
    except Exception as e:
        logger.error(f"Error updating record for {exchange} at {timestamp}: {e}", exc_info=True)
        if 'conn' in locals():
            release_db_connection(conn)
        return False


def backfill_nse_features(exchange: str = 'NSE', days: int = 10, 
                          use_kite: bool = False, use_nse_api: bool = False):
    """
    Backfill NSE option chain features for historical data.
    
    IMPORTANT NOTES:
    - NSE API only provides CURRENT option chain data, not historical
    - Historical option chain data is not available from public APIs
    - This script will set features to 0.0 for historical dates
    - For future dates (today forward), features will be calculated live
    
    Args:
        exchange: Exchange name (default: 'NSE')
        days: Number of days to backfill (default: 10)
        use_kite: Whether to use Kite API for open prices (default: False)
                  If False, uses underlying_price from database (first record of day)
        use_nse_api: Whether to try fetching from NSE API (default: False)
                     Note: NSE API only supports CURRENT data, not historical
    """
    logger.info(f"Starting backfill for {exchange} for last {days} days")
    logger.warning("="*60)
    logger.warning("IMPORTANT: NSE API only provides CURRENT data, not historical.")
    logger.warning("Historical records will have NSE features set to 0.0")
    logger.warning("Only TODAY's data (if market is open) can get real NSE features.")
    logger.warning("="*60)
    
    config = get_config()
    strike_difference = config.exchange_configs.get(exchange, {}).get('strike_difference', 50)
    
    # Initialize Kite if needed (requires 2FA and credentials)
    kite_obj = None
    underlying_token = None
    if use_kite:
        try:
            from connector import Connector
            config = get_config()
            if config.user_id and config.password:
                logger.info("Initializing Kite connection (this will prompt for 2FA)...")
                connector = Connector(
                    user_id=config.user_id,
                    password=config.password,
                    twofa_callback=lambda: input("Enter 2FA code: ").strip()
                )
                kite_obj = connector.initialize_kite()
                if kite_obj:
                    logger.info("Kite initialized successfully. Looking for underlying token...")
                    underlying_token = get_underlying_token_for_exchange(exchange, kite_obj)
                    if not underlying_token:
                        logger.warning(f"Could not get underlying token for {exchange} from Kite")
                        logger.info("Will use underlying_price from database records instead")
                        use_kite = False
                    else:
                        logger.info(f"Successfully found underlying token: {underlying_token}")
                else:
                    logger.warning("Could not initialize Kite. Using database prices.")
                    use_kite = False
            else:
                logger.warning("Kite credentials not configured in environment/config.")
                logger.info("Using underlying_price from database records")
                use_kite = False
        except Exception as e:
            logger.error(f"Error initializing Kite: {e}")
            logger.info("Falling back to using underlying_price from database records")
            use_kite = False
    else:
        logger.info("Using underlying_price from database records (Kite not requested)")
    
    # Get historical records
    records = get_historical_records(exchange, days)
    
    if not records:
        logger.error("No historical records found. Exiting.")
        return
    
    # Group records by date to get open price once per day
    records_by_date = {}
    for record in records:
        record_date = record['timestamp'].date() if isinstance(record['timestamp'], datetime) else record['timestamp']
        if record_date not in records_by_date:
            records_by_date[record_date] = []
        records_by_date[record_date].append(record)
    
    updated_count = 0
    skipped_count = 0
    error_count = 0
    
    # Process each day
    for day_date, day_records in sorted(records_by_date.items()):
        logger.info(f"Processing {day_date}: {len(day_records)} records")
        
        # Get open price for this day
        open_price = None
        if use_kite and underlying_token:
            open_price = get_open_price_from_kite(kite_obj, underlying_token, day_date)
            if open_price is None:
                logger.debug(f"Kite API did not return open price for {day_date}, will use database price")
        
        # Fallback: use first record's underlying_price as open price
        if open_price is None and day_records:
            first_record = day_records[0]
            open_price = first_record.get('underlying_price')
            if open_price:
                if use_kite:
                    logger.info(f"Using underlying_price from database as open price for {day_date}: {open_price} (Kite unavailable)")
                else:
                    logger.debug(f"Using underlying_price from first record as open price: {open_price}")
        
        if open_price is None:
            logger.warning(f"Could not get open price for {day_date}. Skipping day.")
            skipped_count += len(day_records)
            continue
        
        # Calculate ATM strike
        atm_strike = _calculate_nearest_atm_strike(open_price, strike_difference)
        logger.info(f"Day {day_date}: Open price = {open_price}, ATM strike = {atm_strike}")
        
        # Try to fetch NSE option chain data
        # NOTE: NSE API only supports CURRENT/live data, not historical
        # For historical dates, we'll set all features to 0.0
        nse_metrics = None
        is_today = (day_date == now_ist().date())
        
        if use_nse_api and is_today:
            logger.info(f"Attempting to fetch CURRENT NSE data for strike {atm_strike}")
            option_chain_data = _fetch_nse_option_chain_data(atm_strike)
            if option_chain_data:
                nse_metrics = _parse_nse_option_chain_data(option_chain_data)
                logger.info(f"Successfully fetched NSE data for {day_date}")
            else:
                logger.warning(f"Could not fetch NSE API data for {day_date}")
        else:
            if not is_today:
                logger.info(f"Skipping NSE API for {day_date} (historical date - API only supports current data)")
            else:
                logger.info(f"Not using NSE API (use_nse_api=False)")
        
        # If NSE API failed or not using it, set features to 0
        if nse_metrics is None:
            nse_metrics = {
                'total_oi_call': 0.0,
                'total_oi_put': 0.0,
                'total_oi_change_call': 0.0,
                'total_oi_change_put': 0.0,
                'total_volume_call': 0.0,
                'total_volume_put': 0.0,
                'oi_change_diff_put_call': 0.0
            }
            if not is_today:
                logger.info(f"Using zero values for NSE features for historical date {day_date}")
        
        # Update all records for this day
        for record in day_records:
            timestamp = record['timestamp']
            success = update_record_with_nse_features(
                exchange, timestamp, open_price, strike_difference, nse_metrics
            )
            if success:
                updated_count += 1
            else:
                error_count += 1
        
        logger.info(f"Completed {day_date}: Updated {len(day_records)} records")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Backfill Summary:")
    logger.info(f"  Total records processed: {len(records)}")
    logger.info(f"  Successfully updated: {updated_count}")
    logger.info(f"  Skipped: {skipped_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Backfill NSE Option Chain Features',
        epilog="""
IMPORTANT NOTES:
- Kite is NOT required - script uses database prices by default
- NSE API only provides CURRENT data, not historical
- Historical dates will have features set to 0.0 (this is expected)
- Only TODAY's data can get real NSE features (if market is open)

Recommended usage (no Kite needed):
  python backfill_nse_features.py --exchange NSE --days 10
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--exchange', type=str, default='NSE',
                       help='Exchange name (default: NSE)')
    parser.add_argument('--days', type=int, default=10,
                       help='Number of days to backfill (default: 10)')
    parser.add_argument('--use-kite', action='store_true',
                       help='[OPTIONAL] Use Kite API for open prices (requires credentials and 2FA). Default: use database prices')
    parser.add_argument('--use-nse-api', action='store_true',
                       help='[OPTIONAL] Try NSE API for TODAY only (historical dates will be 0). Default: set all to 0')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("NSE Option Chain Features Backfill Script")
    logger.info("="*60)
    if not args.use_kite:
        logger.info("ℹ️  Running without Kite - using database prices (recommended)")
    if not args.use_nse_api:
        logger.info("ℹ️  Running without NSE API - setting historical features to 0.0")
    logger.info("="*60)
    
    backfill_nse_features(
        exchange=args.exchange,
        days=args.days,
        use_kite=args.use_kite,
        use_nse_api=args.use_nse_api
    )


if __name__ == '__main__':
    main()
