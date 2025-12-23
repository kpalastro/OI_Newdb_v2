"""
Historical backfill script for multi-expiry option chain data.

This script backfills historical data for the nse_multi_expiry_minute_data table.
It fetches data from Kite API for past dates and saves aggregated metrics.

Run this script from the project root directory:
    python nse_multi_expiry_collector/backfill_multi_expiry.py --start-date 2025-12-10 --end-date 2025-12-11
"""
import argparse
import logging
import sys
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Optional

# Add parent directory to path so we can import from root
# This allows the script to be run from either the root or the subdirectory
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from kite_trade import KiteApp, get_enctoken
from config import get_config
from time_utils import now_ist, to_ist
from database_new import save_multi_expiry_minute_data
from nse_multi_expiry_collector.kite_fetcher import (
    get_next_5_expiries,
    aggregate_multi_expiry_data
)
from nse_multi_expiry_collector.aggregator import calculate_base_strike

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_open_price_for_date(
    kite_obj: KiteApp,
    exchange: str,
    target_date: date,
    underlying_token: int
) -> Optional[float]:
    """
    Get market open price for a specific date from Kite historical data.
    
    Args:
        kite_obj: KiteApp instance
        exchange: Exchange name
        target_date: Target date
        underlying_token: Underlying instrument token
    
    Returns:
        Open price or None if not available
    """
    try:
        # Fetch daily data for the target date
        candles = kite_obj.historical_data(
            underlying_token,
            target_date.strftime('%Y-%m-%d'),
            target_date.strftime('%Y-%m-%d'),
            'day',
            continuous=False,
            oi=False
        )
        
        if candles and len(candles) > 0:
            # Get the first candle of the day (market open)
            return float(candles[0].get('open', 0) or 0)
        
        return None
    except Exception as e:
        logger.error(f"Error getting open price for {target_date}: {e}")
        return None


def get_underlying_token_for_exchange(
    kite_obj: KiteApp,
    exchange: str,
    exchange_config: dict
) -> Optional[int]:
    """
    Get underlying instrument token for an exchange.
    
    Args:
        kite_obj: KiteApp instance
        exchange: Exchange name (e.g., "NSE")
        exchange_config: Exchange configuration dict
    
    Returns:
        Instrument token or None
    """
    try:
        ltp_exchange = exchange_config.get('ltp_exchange', 'NSE')
        underlying_symbol = exchange_config.get('underlying_symbol', 'NIFTY 50')
        
        instruments = kite_obj.instruments(ltp_exchange)
        
        for inst in instruments:
            if inst.get('name') == underlying_symbol:
                return inst['instrument_token']
        
        return None
    except Exception as e:
        logger.error(f"Error getting underlying token: {e}")
        return None


def backfill_date(
    kite_obj: KiteApp,
    exchange: str,
    exchange_config: dict,
    target_date: date
) -> int:
    """
    Backfill data for a single date.
    
    Args:
        kite_obj: KiteApp instance
        exchange: Exchange name
        exchange_config: Exchange configuration
        target_date: Target date to backfill
    
    Returns:
        Number of records saved
    """
    logger.info(f"Backfilling data for {exchange} on {target_date}")
    
    # Get underlying token
    underlying_token = get_underlying_token_for_exchange(kite_obj, exchange, exchange_config)
    if not underlying_token:
        logger.error(f"Could not find underlying token for {exchange}")
        return 0
    
    # Get market open price
    open_price = get_open_price_for_date(kite_obj, exchange, target_date, underlying_token)
    if not open_price:
        logger.warning(f"Could not get open price for {target_date}. Skipping.")
        return 0
    
    # Calculate base strike
    strike_difference = exchange_config.get('strike_difference', 50)
    base_strike = calculate_base_strike(open_price, strike_difference)
    
    logger.info(f"Open Price: {open_price}, Base Strike: {base_strike}")
    
    # Get next 5 expiries (as of the target date)
    underlying_prefix = exchange_config.get('underlying_prefix', 'NIFTY')
    options_exchange = exchange_config.get('options_exchange', 'NFO')
    
    expiry_dates = get_next_5_expiries(
        kite_obj,
        exchange,
        underlying_prefix,
        options_exchange,
        current_date=target_date
    )
    
    if not expiry_dates or len(expiry_dates) < 1:
        logger.warning(f"Could not find expiry dates for {target_date}. Skipping.")
        return 0
    
    logger.info(f"Found {len(expiry_dates)} expiries: {expiry_dates}")
    
    # Verify we can find instruments for at least one expiry
    from nse_multi_expiry_collector.kite_fetcher import find_option_instruments
    test_ce, test_pe = find_option_instruments(
        kite_obj, underlying_prefix, base_strike, expiry_dates[0], options_exchange
    )
    if not test_ce or not test_pe:
        logger.error(
            f"Could not find option instruments for strike {base_strike}, "
            f"expiry {expiry_dates[0]}. This might indicate the strike/expiry combination doesn't exist."
        )
        return 0
    logger.info(f"Verified instruments exist: CE token {test_ce['instrument_token']}, PE token {test_pe['instrument_token']}")
    
    # Fetch data for each minute during market hours (9:15 AM - 3:30 PM)
    records_saved = 0
    market_open = datetime.combine(target_date, datetime.strptime('09:15', '%H:%M').time())
    market_close = datetime.combine(target_date, datetime.strptime('15:30', '%H:%M').time())
    
    # Convert to IST-aware datetimes (to_ist already imported at top)
    market_open = to_ist(market_open)
    market_close = to_ist(market_close)
    
    current_time = market_open
        
    # Get spot price history for IV calculation (fetch once per day)
    spot_price_history = {}
    try:
        spot_candles = kite_obj.historical_data(
            underlying_token,
            target_date.strftime('%Y-%m-%d'),
            target_date.strftime('%Y-%m-%d'),
            'minute',
            continuous=False,
            oi=False
        )
        if spot_candles:
            for candle in spot_candles:
                ts = candle.get('date')
                if ts:
                    # Convert to IST-aware datetime
                    candle_dt = to_ist(ts) if isinstance(ts, datetime) else ts
                    if isinstance(candle_dt, datetime):
                        # Round to minute and ensure IST-aware
                        minute_key = candle_dt.replace(second=0, microsecond=0)
                        if minute_key.tzinfo is None:
                            minute_key = to_ist(minute_key)
                        spot_price_history[minute_key] = float(candle.get('close', 0) or 0)
            logger.info(f"Fetched {len(spot_price_history)} spot price data points for {target_date}")
    except Exception as e:
        logger.warning(f"Could not fetch spot price history: {e}")
    
    # Store previous OI for change calculation
    previous_minute_oi = {}
    
    while current_time <= market_close:
        # Round to minute boundary and ensure IST-aware
        current_minute = current_time.replace(second=0, microsecond=0)
        # Ensure it's IST-aware if it somehow became naive
        if current_minute.tzinfo is None:
            current_minute = to_ist(current_minute)
        
        # Get spot price for this minute (use closest available or open_price)
        spot_price = open_price
        if spot_price_history:
            # Find closest spot price
            closest_ts = min(spot_price_history.keys(), 
                            key=lambda x: abs((x - current_minute).total_seconds()) 
                            if isinstance(x, datetime) else float('inf'))
            if isinstance(closest_ts, datetime) and abs((closest_ts - current_minute).total_seconds()) < 120:
                spot_price = spot_price_history[closest_ts]
        
        # Fetch and aggregate data with target_timestamp
        try:
            _, aggregated = aggregate_multi_expiry_data(
                kite_obj,
                underlying_prefix,
                base_strike,
                expiry_dates,
                options_exchange,
                use_websocket_data=None,  # No websocket for historical data
                previous_oi=previous_minute_oi if previous_minute_oi else None,  # Use previous minute OI
                spot_price=spot_price,
                target_timestamp=current_minute  # Pass target timestamp for historical fetch
            )
            
            # Update previous OI for next iteration (store aggregated OI from this minute)
            # Note: This is a simplified approach - ideally we'd store per-instrument OI
            # But for now, we'll rely on the internal historical comparison in fetch function
            
            # Check if we got valid data before saving
            has_valid_data = (
                aggregated.get('total_oi_call_all_expiries', 0) > 0 or
                aggregated.get('total_oi_put_all_expiries', 0) > 0 or
                aggregated.get('total_volume_call_all_expiries', 0) > 0 or
                aggregated.get('total_volume_put_all_expiries', 0) > 0
            )
            
            # Also check that change in OI is not zero for both
            oi_change_call = aggregated.get('total_oi_change_call_all_expiries', 0) or 0
            oi_change_put = aggregated.get('total_oi_change_put_all_expiries', 0) or 0
            has_oi_change = (oi_change_call != 0 or oi_change_put != 0)
            
            if not has_valid_data:
                logger.debug(f"All metrics zero for {current_minute}. Skipping save.")
                # Move to next minute
                current_time += timedelta(minutes=1)
                continue
            
            if not has_oi_change:
                logger.debug(
                    f"Change in OI is zero for both calls and puts at {current_minute}. "
                    f"Skipping save (no meaningful change detected)."
                )
                # Move to next minute
                current_time += timedelta(minutes=1)
                continue
            
            # Ensure timestamp is IST-aware before saving
            timestamp_ist = to_ist(current_minute) if current_minute.tzinfo is None else current_minute
            
            # Save to database
            success = save_multi_expiry_minute_data(
                exchange=exchange,
                timestamp=timestamp_ist,
                open_price=open_price,
                base_strike=base_strike,
                expiry_dates=expiry_dates,
                aggregated_metrics=aggregated
            )
            
            if success:
                records_saved += 1
                if records_saved % 30 == 0:  # Log progress every 30 minutes
                    logger.info(f"Saved {records_saved} records so far for {target_date}...")
        
        except Exception as e:
            logger.error(f"Error processing minute {current_minute}: {e}", exc_info=True)
        
        # Move to next minute
        current_time += timedelta(minutes=1)
    
    logger.info(f"Saved {records_saved} records for {target_date}")
    return records_saved


def main():
    parser = argparse.ArgumentParser(
        description='Backfill historical multi-expiry option chain data'
    )
    parser.add_argument(
        '--exchange',
        type=str,
        default='NSE',
        help='Exchange name (default: NSE)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        required=True,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        required=True,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--user-id',
        type=str,
        help='Zerodha user ID (default: from config/env)'
    )
    parser.add_argument(
        '--password',
        type=str,
        help='Zerodha password (default: from config/env)'
    )
    
    args = parser.parse_args()
    
    config = get_config()
    user_id = args.user_id or config.user_id
    password = args.password or config.password
    
    if not user_id or not password:
        logger.error("User ID and password required (via args or config/env)")
        return
    
    # Initialize Kite
    logger.info("Initializing Kite connection...")
    try:
        twofa_code = input("Enter 2FA code: ").strip()
        enctoken = get_enctoken(user_id, password, twofa_code)
        
        if not enctoken:
            logger.error("Failed to obtain enctoken")
            return
        
        kite_obj = KiteApp(enctoken=enctoken)
        profile = kite_obj.profile()
        logger.info(f"Connected as: {profile.get('user_id')}")
    except Exception as e:
        logger.error(f"Failed to initialize Kite: {e}")
        return
    
    # Get exchange config
    exchange_config = config.exchange_configs.get(args.exchange)
    if not exchange_config:
        logger.error(f"Exchange config not found for {args.exchange}")
        return
    
    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        return
    
    # Backfill each date
    current_date = start_date
    total_records = 0
    
    while current_date <= end_date:
        # Skip weekends (Saturday=5, Sunday=6)
        if current_date.weekday() < 5:  # Monday=0, Friday=4
            records = backfill_date(kite_obj, args.exchange, exchange_config, current_date)
            total_records += records
        else:
            logger.info(f"Skipping {current_date} (weekend)")
        
        current_date += timedelta(days=1)
    
    logger.info(f"Backfill complete. Total records saved: {total_records}")


if __name__ == '__main__':
    main()
