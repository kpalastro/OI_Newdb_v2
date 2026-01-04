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
import time
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

# Configure logging to both console and file
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / 'backfill_multi_expiry.log'

# Get root logger and clear any existing handlers
root_logger = logging.getLogger()
root_logger.handlers = []

# Create file handler with immediate flush
file_handler = logging.FileHandler(log_file, mode='a')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Add handlers to root logger
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)
root_logger.setLevel(logging.DEBUG)

# Get module logger
logger = logging.getLogger(__name__)

# Force flush and log immediately to verify logging works
def flush_logs():
    """Force flush all file handlers to ensure logs are written immediately."""
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()

flush_logs()
logger.info("=" * 80)
logger.info("BACKFILL MULTI-EXPIRY SCRIPT STARTED")
logger.info(f"Log file: {log_file.absolute()}")
logger.info("=" * 80)
flush_logs()


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
    
    # Test if we can fetch historical data for the target date
    try:
        test_date_str = target_date.strftime('%Y-%m-%d')
        test_candles = kite_obj.historical_data(
            test_ce['instrument_token'],
            test_date_str,
            test_date_str,
            'minute',
            continuous=False,
            oi=True
        )
        if test_candles:
            logger.info(f"Test fetch successful: Found {len(test_candles)} minute candles for test instrument on {target_date}")
            if len(test_candles) > 0:
                logger.info(f"First candle: {test_candles[0]}")
                logger.info(f"Last candle: {test_candles[-1]}")
        else:
            logger.warning(f"No historical candles found for test instrument on {target_date}. Data may not be available.")
    except Exception as e:
        logger.error(f"Error testing historical data fetch: {e}")
        logger.warning("Proceeding anyway, but data may not be available for this date.")
    
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
    
    skipped_zero_data = 0
    skipped_errors = 0
    total_iterations = 0
    save_attempts = 0
    
    # Calculate expected number of minutes
    expected_minutes = int((market_close - market_open).total_seconds() / 60) + 1
    logger.info(f"Starting minute-by-minute backfill from {market_open} to {market_close}")
    logger.info(f"Expected to process {expected_minutes} minutes (9:15 AM to 3:30 PM = 375 minutes)")
    logger.info(f"Market open: {market_open}, Market close: {market_close}")
    
    # Safety check: ensure we don't have an infinite loop
    max_iterations = expected_minutes + 100  # Allow some buffer
    
    while current_time <= market_close:
        # Safety check to prevent infinite loops
        if total_iterations > max_iterations:
            logger.error(f"LOOP SAFETY BREAK: Exceeded max iterations {max_iterations}. Breaking loop.")
            flush_logs()
            break
        total_iterations += 1
        
        # Log every iteration for first 10 to debug loop execution
        if total_iterations <= 10:
            logger.info(f"LOOP ITERATION #{total_iterations}: current_time={current_time}, market_close={market_close}, condition={current_time <= market_close}")
            flush_logs()
        
        # Round to minute boundary and ensure IST-aware
        current_minute = current_time.replace(second=0, microsecond=0)
        # Ensure it's IST-aware if it somehow became naive
        if current_minute.tzinfo is None:
            current_minute = to_ist(current_minute)
        
        # Log progress every 60 minutes
        if total_iterations % 60 == 0:
            logger.info(f"Processing minute {total_iterations}: {current_minute} (saved: {records_saved}, skipped: {skipped_zero_data}, errors: {skipped_errors})")
            flush_logs()
        
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
            # For historical data, the fetch function will calculate OI change internally
            # by comparing current minute to previous minute from historical candles
            # So we don't need to pass previous_oi - it will be calculated from historical data
            expiry_data_list, aggregated = aggregate_multi_expiry_data(
                kite_obj,
                underlying_prefix,
                base_strike,
                expiry_dates,
                options_exchange,
                use_websocket_data=None,  # No websocket for historical data
                previous_oi=None,  # Let fetch function calculate from historical candles
                spot_price=spot_price,
                target_timestamp=current_minute  # Pass target timestamp for historical fetch
            )
            
            # Validate aggregated data
            if aggregated is None:
                logger.error(f"Aggregated data is None for {current_minute} - this should not happen!")
                skipped_errors += 1
                current_time += timedelta(minutes=1)
                continue
            
            # Log aggregated data for first few iterations to debug
            if total_iterations <= 3:
                logger.info(
                    f"Minute {total_iterations} ({current_minute}) aggregated data: "
                    f"CE_OI={aggregated.get('total_oi_call_all_expiries', 0):,.0f}, "
                    f"PE_OI={aggregated.get('total_oi_put_all_expiries', 0):,.0f}, "
                    f"CE_Vol={aggregated.get('total_volume_call_all_expiries', 0):,.0f}, "
                    f"PE_Vol={aggregated.get('total_volume_put_all_expiries', 0):,.0f}, "
                    f"CE_OI_Chg={aggregated.get('total_oi_change_call_all_expiries', 0):,.0f}, "
                    f"PE_OI_Chg={aggregated.get('total_oi_change_put_all_expiries', 0):,.0f}"
                )
            
            # Check if we got valid data before saving
            total_oi_call = aggregated.get('total_oi_call_all_expiries', 0) or 0
            total_oi_put = aggregated.get('total_oi_put_all_expiries', 0) or 0
            total_volume_call = aggregated.get('total_volume_call_all_expiries', 0) or 0
            total_volume_put = aggregated.get('total_volume_put_all_expiries', 0) or 0
            
            has_valid_data = (
                total_oi_call > 0 or
                total_oi_put > 0 or
                total_volume_call > 0 or
                total_volume_put > 0
            )
            
            # For historical backfilling, save ALL records even if data is zero
            # This ensures we capture the complete historical timeline
            # Only skip if we got an error or exception (handled in except block)
            oi_change_call = aggregated.get('total_oi_change_call_all_expiries', 0) or 0
            oi_change_put = aggregated.get('total_oi_change_put_all_expiries', 0) or 0
            has_oi_change = (oi_change_call != 0 or oi_change_put != 0)
            
            # Log data status but don't skip - save all records for historical completeness
            if not has_valid_data:
                skipped_zero_data += 1
                if skipped_zero_data <= 5 or skipped_zero_data % 60 == 0:  # Log first 5 and every hour
                    logger.info(f"All metrics zero for {current_minute} but saving anyway for historical completeness. (Total zero-data: {skipped_zero_data})")
            
            # Log if OI change is zero
            if not has_oi_change and has_valid_data:
                logger.debug(
                    f"Change in OI is zero for both calls and puts at {current_minute}. "
                    f"Saving record for historical completeness."
                )
            
            # Ensure timestamp is IST-aware before saving
            timestamp_ist = to_ist(current_minute) if current_minute.tzinfo is None else current_minute
            
            # Save to database
            save_attempts += 1
            
            # Log first 10 attempts in detail to debug
            if save_attempts <= 10:
                logger.info(
                    f"Attempt {save_attempts}: Saving {current_minute} - "
                    f"CE_OI={total_oi_call:,.0f}, PE_OI={total_oi_put:,.0f}, "
                    f"CE_Vol={total_volume_call:,.0f}, PE_Vol={total_volume_put:,.0f}"
                )
            elif save_attempts % 60 == 0:
                logger.info(f"Save attempt {save_attempts} for {current_minute} (saved so far: {records_saved})")
            
            try:
                if total_iterations <= 5:
                    logger.info(f"About to call save_multi_expiry_minute_data for {current_minute} (iteration {total_iterations})")
                    flush_logs()
                
                # Call save function with timeout protection
                try:
                    success = save_multi_expiry_minute_data(
                        exchange=exchange,
                        timestamp=timestamp_ist,
                        open_price=open_price,
                        base_strike=base_strike,
                        expiry_dates=expiry_dates,
                        aggregated_metrics=aggregated
                    )
                    
                    if total_iterations <= 5:
                        logger.info(f"✓ save_multi_expiry_minute_data returned: {success} for {current_minute} (iteration {total_iterations})")
                        flush_logs()
                except Exception as save_func_error:
                    logger.error(f"Exception INSIDE save_multi_expiry_minute_data call: {save_func_error}", exc_info=True)
                    flush_logs()
                    success = False
                    raise  # Re-raise to be caught by outer except
                
                if success:
                    records_saved += 1
                    if records_saved == 1:
                        logger.info(f"✓ First record saved for {target_date} at {current_minute}")
                        flush_logs()  # Force flush after first save
                    elif records_saved <= 10:
                        logger.info(f"✓ Record {records_saved} saved for {current_minute} (attempt {save_attempts})")
                        flush_logs()  # Force flush for first 10 saves
                    elif records_saved % 30 == 0:  # Log progress every 30 minutes
                        logger.info(f"✓ Saved {records_saved} records so far for {target_date}... (attempted: {save_attempts}, skipped: {skipped_zero_data}, errors: {skipped_errors})")
                        flush_logs()
                else:
                    if save_attempts <= 10:
                        logger.error(f"✗ Save returned False for {current_minute} (attempt {save_attempts}) - check logs above for reason")
                        flush_logs()
                    else:
                        logger.warning(f"✗ Save returned False for {current_minute} (attempt {save_attempts})")
                
                if total_iterations <= 5:
                    logger.info(f"After save handling for iteration {total_iterations}, continuing...")
                    flush_logs()
            except Exception as save_error:
                logger.error(f"Exception during save for {current_minute}: {save_error}", exc_info=True)
                flush_logs()
                success = False
                skipped_errors += 1
        
        except Exception as e:
            skipped_errors += 1
            logger.error(f"ERROR processing minute {current_minute}: {e}", exc_info=True)
            flush_logs()
            if skipped_errors <= 5:  # Log first 5 errors in detail
                logger.error(f"Error details for {current_minute}: {e}")
                flush_logs()
            # Add delay after errors to avoid overwhelming the API
            if skipped_errors % 10 == 0:
                logger.info(f"Encountered {skipped_errors} errors. Pausing for 5 seconds to avoid rate limiting...")
                flush_logs()
                time.sleep(5)
        
        # CRITICAL: Log that we're continuing to next iteration - this MUST happen
        if total_iterations <= 10:
            logger.info(f"✓ Completed iteration {total_iterations}, moving to next minute...")
            flush_logs()
        
        # CRITICAL: This must happen to continue the loop
        # Add small delay between API calls to avoid rate limiting
        # Only delay every N minutes to speed up backfilling
        if total_iterations % 10 == 0:
            time.sleep(0.5)  # Small delay every 10 minutes
        
        # Move to next minute - CRITICAL: This must happen to continue the loop
        current_time += timedelta(minutes=1)
        
        # Log loop continuation for first 10 iterations
        if total_iterations <= 10:
            logger.info(f"After iteration {total_iterations}: Moved to next minute. New current_time={current_time}, still <= market_close? {current_time <= market_close}")
            flush_logs()
        
        # CRITICAL: Ensure we're still in the loop - log this for first few iterations
        if total_iterations <= 5:
            logger.info(f"Loop will continue? current_time ({current_time}) <= market_close ({market_close}) = {current_time <= market_close}")
            flush_logs()
    
    # Log final state when loop exits
    logger.info("=" * 80)
    logger.info(f"LOOP EXITED: current_time={current_time}, market_close={market_close}")
    logger.info(f"Loop condition: current_time <= market_close = {current_time <= market_close}")
    logger.info(f"Total iterations: {total_iterations}, Expected: {expected_minutes}")
    logger.info("=" * 80)
    flush_logs()
    
    logger.info(
        f"Backfill complete for {target_date}: "
        f"Total iterations: {total_iterations} (expected: {expected_minutes}), "
        f"Save attempts: {save_attempts}, "
        f"Saved {records_saved} records, "
        f"Skipped {skipped_zero_data} minutes with zero data, "
        f"Encountered {skipped_errors} errors"
    )
    flush_logs()
    
    if total_iterations != expected_minutes:
        logger.error(
            f"LOOP ISSUE: Expected {expected_minutes} iterations but only ran {total_iterations} times! "
            f"This indicates the loop exited early. Check for breaks or exceptions."
        )
    
    if records_saved == 0:
        logger.error(
            f"No records were saved for {target_date} out of {save_attempts} save attempts. "
            f"This could indicate: "
            f"1) No historical data available for this date, "
            f"2) All minutes returned zero data, "
            f"3) Save function is rejecting all records. "
            f"Check logs above for details."
        )
    elif records_saved == 1:
        logger.error(
            f"Only 1 record was saved for {target_date} out of {save_attempts} save attempts ({total_iterations} iterations). "
            f"This suggests the save function is rejecting most records. "
            f"Check save function logs above to see why records are being rejected."
        )
    elif records_saved < expected_minutes * 0.1:  # Less than 10% of expected
        logger.warning(
            f"Only {records_saved} records saved out of {expected_minutes} expected minutes ({save_attempts} attempts). "
            f"This suggests most minutes are returning zero data or being rejected. "
            f"Check if historical data is available for this date in Kite API."
        )
    
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
    
    # Check if dates are in the future
    today = now_ist().date()
    if start_date > today:
        logger.warning(f"Start date {start_date} is in the future. Historical data may not be available.")
    if end_date > today:
        logger.warning(f"End date {end_date} is in the future. Historical data may not be available.")
    
    # Calculate expected number of records (assuming 375 minutes per trading day: 9:15 AM to 3:30 PM)
    num_trading_days = sum(1 for d in [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)] if d.weekday() < 5)
    expected_records = num_trading_days * 375  # 375 minutes from 9:15 to 15:30
    logger.info(f"Starting backfill for {args.exchange} from {start_date} to {end_date}")
    logger.info(f"Expected to process {num_trading_days} trading days (~{expected_records} minute records)")
    
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
    try:
        logger.info("=" * 80)
        logger.info("SCRIPT STARTING")
        logger.info("=" * 80)
        flush_logs()
        main()
        logger.info("=" * 80)
        logger.info("SCRIPT COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        flush_logs()
    except KeyboardInterrupt:
        logger.warning("Script interrupted by user")
        flush_logs()
    except Exception as e:
        logger.error(f"FATAL ERROR in main: {e}", exc_info=True)
        flush_logs()
        raise
