"""
Kite API integration for fetching option chain data across multiple expiries.

Provides functions to:
- Get next 5 expiry dates for an exchange
- Fetch option chain data for a specific strike and expiry
- Aggregate data across multiple expiries
"""
import logging
import sys
from pathlib import Path
from datetime import date, datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Tuple
from threading import Lock

# Ensure we can import from parent directory
_module_dir = Path(__file__).parent
_project_root = _module_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from kite_trade import KiteApp
from time_utils import now_ist, to_ist

logger = logging.getLogger(__name__)

# Cache for instruments list with TTL (Time To Live)
_instruments_cache: Dict[str, Tuple[List[Dict], datetime]] = {}
_cache_lock = Lock()
_cache_ttl = timedelta(minutes=5)  # Refresh cache every 5 minutes


def get_next_5_expiries(
    kite_obj: KiteApp,
    exchange: str,
    underlying_prefix: str,
    options_exchange: str = "NFO",
    current_date: Optional[date] = None
) -> List[date]:
    """
    Get the next 5 expiry dates for the given exchange and underlying.
    
    Args:
        kite_obj: KiteApp instance
        exchange: Exchange name (e.g., "NSE")
        underlying_prefix: Underlying prefix (e.g., "NIFTY", "BANKNIFTY")
        options_exchange: Options exchange segment (e.g., "NFO", "BFO")
        current_date: Current date (defaults to today in IST)
    
    Returns:
        List of next 5 expiry dates (sorted, ascending)
    """
    if current_date is None:
        current_date = now_ist().date()
    
    try:
        # Fetch all instruments for the options exchange
        # Wrap in try-except to handle malformed CSV rows from Kite API
        try:
            instruments = kite_obj.instruments(options_exchange)
        except (IndexError, ValueError) as e:
            logger.error(f"Error fetching instruments from Kite API (malformed CSV row): {e}")
            # Try fetching all instruments without exchange filter as fallback
            try:
                all_instruments = kite_obj.instruments(None)
                instruments = [inst for inst in all_instruments if inst.get('exchange') == options_exchange]
                logger.info(f"Fetched {len(instruments)} instruments using fallback method")
            except Exception as e2:
                logger.error(f"Fallback instrument fetch also failed: {e2}")
                return []
        
        if not instruments:
            logger.warning(f"No instruments found for {options_exchange}")
            return []
        
        # Filter for options of the given underlying
        # Options have instrument_type as 'CE' or 'PE'
        option_instruments = []
        for inst in instruments:
            try:
                if (inst.get('instrument_type') in ('CE', 'PE') and
                    inst.get('name', '').startswith(underlying_prefix) and
                    inst.get('expiry') is not None and
                    inst.get('expiry') >= current_date):
                    option_instruments.append(inst)
            except (AttributeError, TypeError) as e:
                # Skip malformed instrument entries
                logger.debug(f"Skipping malformed instrument entry: {e}")
                continue
        
        if not option_instruments:
            logger.warning(f"No option instruments found for {underlying_prefix} in {options_exchange}")
            return []
        
        # Extract unique expiry dates and sort
        expiry_dates = sorted(set(inst['expiry'] for inst in option_instruments if inst.get('expiry')))
        
        # Get next 5 expiries
        next_5_expiries = expiry_dates[:5]
        
        logger.info(f"Found {len(next_5_expiries)} next expiries for {underlying_prefix}: {next_5_expiries}")
        return next_5_expiries
        
    except Exception as e:
        logger.error(f"Error getting next 5 expiries: {e}", exc_info=True)
        return []


def _get_cached_instruments(
    kite_obj: KiteApp,
    options_exchange: str
) -> List[Dict]:
    """
    Get instruments list with caching to reduce API calls.
    
    Args:
        kite_obj: KiteApp instance
        options_exchange: Options exchange segment (e.g., "NFO")
    
    Returns:
        List of instrument dictionaries
    """
    cache_key = options_exchange or "ALL"
    now = now_ist()
    
    with _cache_lock:
        # Check if we have a valid cached version
        if cache_key in _instruments_cache:
            cached_instruments, cache_time = _instruments_cache[cache_key]
            if now - cache_time < _cache_ttl:
                logger.debug(f"Using cached instruments for {options_exchange} (cached at {cache_time})")
                return cached_instruments
            else:
                logger.debug(f"Cache expired for {options_exchange}, refreshing...")
        
        # Fetch fresh instruments
        try:
            try:
                instruments = kite_obj.instruments(options_exchange)
            except (IndexError, ValueError) as e:
                logger.debug(f"Error fetching instruments with exchange filter (malformed CSV row): {e}")
                # Try fetching all instruments without exchange filter as fallback
                try:
                    all_instruments = kite_obj.instruments(None)
                    instruments = [inst for inst in all_instruments if inst.get('exchange') == options_exchange]
                    logger.info(f"Fetched {len(instruments)} instruments using fallback method")
                except Exception as e2:
                    logger.error(f"Fallback instrument fetch also failed: {e2}")
                    # Return cached version if available, even if expired
                    if cache_key in _instruments_cache:
                        logger.warning(f"Using expired cache due to fetch failure")
                        return _instruments_cache[cache_key][0]
                    return []
            
            # Update cache
            _instruments_cache[cache_key] = (instruments, now)
            logger.debug(f"Cached {len(instruments)} instruments for {options_exchange}")
            return instruments
            
        except Exception as e:
            logger.error(f"Error fetching instruments: {e}", exc_info=True)
            # Return cached version if available, even if expired
            if cache_key in _instruments_cache:
                logger.warning(f"Using expired cache due to exception")
                return _instruments_cache[cache_key][0]
            return []


def find_option_instruments(
    kite_obj: KiteApp,
    underlying_prefix: str,
    strike: float,
    expiry_date: date,
    options_exchange: str = "NFO"
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Find CE and PE option instruments for a given strike and expiry.
    
    Args:
        kite_obj: KiteApp instance
        underlying_prefix: Underlying prefix (e.g., "NIFTY")
        strike: Strike price (e.g., 25750.0)
        expiry_date: Expiry date
        options_exchange: Options exchange segment (e.g., "NFO")
    
    Returns:
        Tuple of (CE_instrument_dict, PE_instrument_dict)
        Each dict contains: instrument_token, tradingsymbol, etc.
        Returns (None, None) if not found
    """
    try:
        # Use cached instruments list
        instruments = _get_cached_instruments(kite_obj, options_exchange)
        
        if not instruments:
            logger.debug(f"No instruments found for {options_exchange}")
            return (None, None)
        
        # Normalize strike for comparison (round to avoid float precision issues)
        strike_normalized = round(strike, 2)
        
        # Normalize expiry_date - handle both date and datetime
        if isinstance(expiry_date, datetime):
            expiry_date_normalized = expiry_date.date()
        else:
            expiry_date_normalized = expiry_date
        
        # Find CE and PE for this strike and expiry
        ce_instrument = None
        pe_instrument = None
        matched_expiries = set()
        matched_strikes = set()
        matched_names = set()
        
        for inst in instruments:
            try:
                # Skip malformed entries
                if not isinstance(inst, dict):
                    continue
                
                inst_expiry = inst.get('expiry')
                # Normalize expiry - handle both date and datetime
                if inst_expiry:
                    if isinstance(inst_expiry, datetime):
                        inst_expiry = inst_expiry.date()
                    elif isinstance(inst_expiry, str):
                        # Try to parse if it's a string
                        try:
                            inst_expiry = datetime.strptime(inst_expiry, '%Y-%m-%d').date()
                        except:
                            continue
                
                inst_strike = inst.get('strike')
                inst_name = inst.get('name', '')
                inst_type = inst.get('instrument_type')
                
                # Debug: Track what we're seeing
                if inst_name.startswith(underlying_prefix) and inst_type in ('CE', 'PE'):
                    if inst_expiry:
                        matched_expiries.add(inst_expiry)
                    if inst_strike is not None:
                        matched_strikes.add(round(float(inst_strike), 2))
                    matched_names.add(inst_name.split()[0] if inst_name else '')
                
                # Check if this instrument matches our criteria
                if (inst_expiry == expiry_date_normalized and
                    inst_strike is not None and
                    abs(float(inst_strike) - strike_normalized) < 0.01 and  # Handle float comparison
                    inst_name.startswith(underlying_prefix)):
                    
                    if inst_type == 'CE':
                        ce_instrument = inst
                    elif inst_type == 'PE':
                        pe_instrument = inst
                    
                    # If we found both, we can break early
                    if ce_instrument and pe_instrument:
                        break
                        
            except (TypeError, ValueError, AttributeError) as e:
                # Skip malformed instrument entries
                logger.debug(f"Skipping malformed instrument entry: {e}")
                continue
        
        # Enhanced logging for debugging
        if not ce_instrument or not pe_instrument:
            logger.warning(
                f"Instruments not found for strike {strike}, expiry {expiry_date} "
                f"(underlying: {underlying_prefix}, exchange: {options_exchange}). "
                f"Available expiries for {underlying_prefix}: {sorted(matched_expiries)[:10]}, "
                f"Available strikes near {strike}: {sorted([s for s in matched_strikes if abs(s - strike) < 100])[:10]}"
            )
        
        return (ce_instrument, pe_instrument)
        
    except Exception as e:
        logger.error(f"Error finding option instruments: {e}", exc_info=True)
        return (None, None)


def _calculate_iv_from_price(
    option_type: str,
    option_price: float,
    spot_price: float,
    strike: float,
    expiry_date: date,
    risk_free_rate: float = 0.10
) -> Optional[float]:
    """
    Calculate IV from option price using Black-Scholes.
    
    Args:
        option_type: 'CE' or 'PE'
        option_price: Current option price
        spot_price: Current spot price
        strike: Strike price
        expiry_date: Expiry date
        risk_free_rate: Risk-free rate (default 10% for India)
    
    Returns:
        IV as percentage (e.g., 15.5 for 15.5%) or None if calculation fails
    """
    try:
        # Import IV calculation from oi_tracker_new (avoid circular import)
        import sys
        import importlib
        
        # Try to import the function dynamically
        try:
            oi_module = sys.modules.get('oi_tracker_new')
            if oi_module is None:
                import oi_tracker_new
                oi_module = oi_tracker_new
            
            calculate_implied_volatility = getattr(oi_module, 'calculate_implied_volatility', None)
            
            if calculate_implied_volatility is None:
                logger.warning("calculate_implied_volatility not found in oi_tracker_new")
                return None
        except ImportError:
            logger.warning("Could not import oi_tracker_new for IV calculation")
            return None
        
        # Calculate time to expiry in years
        current_time = now_ist()
        expiry_dt = datetime.combine(expiry_date, dt_time(15, 30))  # Expiry at 3:30 PM
        expiry_dt = to_ist(expiry_dt)
        time_to_expiry_seconds = (expiry_dt - current_time).total_seconds()
        
        if time_to_expiry_seconds <= 0:
            return None
        
        time_years = time_to_expiry_seconds / (365.25 * 24 * 3600)
        
        # Calculate IV - try to import from oi_tracker_new
        try:
            from oi_tracker_new import calculate_implied_volatility
            
            iv = calculate_implied_volatility(
                option_type=option_type,
                option_price=option_price,
                spot=spot_price,
                strike=strike,
                time_years=time_years,
                rate=risk_free_rate
            )
            
            return iv
        except (ImportError, AttributeError) as e:
            logger.debug(f"Could not import calculate_implied_volatility from oi_tracker_new: {e}. IV calculation skipped.")
            return None
        
    except Exception as e:
        logger.debug(f"Error calculating IV: {e}")
        return None


def fetch_kite_option_chain_for_expiry(
    kite_obj: KiteApp,
    underlying_prefix: str,
    strike: float,
    expiry_date: date,
    options_exchange: str = "NFO",
    use_websocket_data: Optional[Dict[int, Dict]] = None,
    previous_oi: Optional[Dict[int, float]] = None,
    spot_price: Optional[float] = None,
    target_timestamp: Optional[datetime] = None
) -> Dict[str, float]:
    """
    Fetch option chain data for a specific strike and expiry from Kite.
    
    Args:
        kite_obj: KiteApp instance
        underlying_prefix: Underlying prefix (e.g., "NIFTY")
        strike: Strike price
        expiry_date: Expiry date
        options_exchange: Options exchange segment
        use_websocket_data: Optional dict of {instrument_token: tick_data} from websocket
        previous_oi: Optional dict of {instrument_token: previous_oi} for calculating change in OI
        spot_price: Current spot price (needed for IV calculation if not in websocket data)
        target_timestamp: Optional target timestamp for historical data fetching (defaults to now_ist())
    
    Returns:
        Dictionary with metrics:
        - 'oi_call': Open Interest for CE
        - 'oi_put': Open Interest for PE
        - 'oi_change_call': Change in OI for CE (current - previous)
        - 'oi_change_put': Change in OI for PE (current - previous)
        - 'volume_call': Volume for CE
        - 'volume_put': Volume for PE
        - 'iv_call': Implied Volatility for CE (from API or calculated)
        - 'iv_put': Implied Volatility for PE (from API or calculated)
    """
    from datetime import time as dt_time
    from time_utils import to_ist
    
    # Use target_timestamp if provided, otherwise use current time
    if target_timestamp is None:
        target_timestamp = now_ist()
    else:
        target_timestamp = to_ist(target_timestamp)
    
    try:
        # Find instruments
        ce_instrument, pe_instrument = find_option_instruments(
            kite_obj, underlying_prefix, strike, expiry_date, options_exchange
        )
        
        if not ce_instrument or not pe_instrument:
            logger.warning(f"Instruments not found for strike {strike}, expiry {expiry_date}")
            return {
                'oi_call': 0.0, 'oi_put': 0.0,
                'oi_change_call': 0.0, 'oi_change_put': 0.0,
                'volume_call': 0.0, 'volume_put': 0.0,
                'iv_call': 0.0, 'iv_put': 0.0
            }
        
        ce_token = ce_instrument['instrument_token']
        pe_token = pe_instrument['instrument_token']
        
        # Try to use websocket data if available (most efficient)
        if use_websocket_data:
            ce_data = use_websocket_data.get(ce_token, {})
            pe_data = use_websocket_data.get(pe_token, {})
            
            ce_oi = float(ce_data.get('oi', 0) or 0)
            pe_oi = float(pe_data.get('oi', 0) or 0)
            
            # Calculate change in OI (compare to previous minute if available)
            ce_oi_change = 0.0
            pe_oi_change = 0.0
            if previous_oi:
                ce_prev_oi = previous_oi.get(ce_token, 0)
                pe_prev_oi = previous_oi.get(pe_token, 0)
                ce_oi_change = ce_oi - ce_prev_oi
                pe_oi_change = pe_oi - pe_prev_oi
            
            # Get IV from websocket data if available, otherwise calculate
            ce_iv = ce_data.get('iv')
            pe_iv = pe_data.get('iv')
            
            if ce_iv is None or ce_iv <= 0:
                # Try to calculate IV
                ce_price = ce_data.get('last_price')
                if ce_price and spot_price:
                    ce_iv = _calculate_iv_from_price('CE', ce_price, spot_price, strike, expiry_date)
            
            if pe_iv is None or pe_iv <= 0:
                pe_price = pe_data.get('last_price')
                if pe_price and spot_price:
                    pe_iv = _calculate_iv_from_price('PE', pe_price, spot_price, strike, expiry_date)
            
            return {
                'oi_call': ce_oi,
                'oi_put': pe_oi,
                'oi_change_call': ce_oi_change,
                'oi_change_put': pe_oi_change,
                'volume_call': float(ce_data.get('volume_traded', 0) or 0),
                'volume_put': float(pe_data.get('volume_traded', 0) or 0),
                'iv_call': float(ce_iv or 0),
                'iv_put': float(pe_iv or 0),
            }
        
        # Fallback: Fetch from historical_data
        # Use target_timestamp (which is already set above)
        current_time = target_timestamp
        market_open_time = datetime.combine(current_time.date(), dt_time(9, 15))
        market_open_time = to_ist(market_open_time)
        
        # Fetch from market open or last 30 minutes, whichever is more recent
        from_date = max(
            market_open_time,
            current_time - timedelta(minutes=30)
        )
        to_date = current_time
        
        ce_oi = 0.0
        ce_volume = 0.0
        ce_price = 0.0
        pe_oi = 0.0
        pe_volume = 0.0
        pe_price = 0.0
        ce_prev_oi = 0.0
        pe_prev_oi = 0.0
        
        # Round down to current minute for comparison
        current_minute = current_time.replace(second=0, microsecond=0)
        
        try:
            # Fetch CE data - get data from the target date
            date_str = current_time.strftime('%Y-%m-%d')
            ce_candles = kite_obj.historical_data(
                ce_token,
                date_str,
                date_str,
                'minute',
                continuous=False,
                oi=True
            )
            
            if ce_candles and len(ce_candles) > 0:
                # Filter candles to current minute or most recent
                filtered_candles_with_ts = []
                for candle in ce_candles:
                    candle_time = candle.get('date')
                    if candle_time:
                        # Strip timezone and round to minute
                        candle_dt = to_ist(candle_time) if hasattr(candle_time, 'replace') else candle_time
                        if isinstance(candle_dt, datetime):
                            candle_dt = candle_dt.replace(second=0, microsecond=0)
                        # Include candles up to and including current minute
                        if candle_dt <= current_minute:
                            filtered_candles_with_ts.append((candle_dt, candle))
                
                # Sort by timestamp
                filtered_candles_with_ts.sort(key=lambda x: x[0])
                filtered_candles = [c for _, c in filtered_candles_with_ts]
                
                if filtered_candles:
                    # Get latest candle (current or most recent)
                    latest_ce = filtered_candles[-1]
                    ce_oi = float(latest_ce.get('oi', 0) or 0)
                    ce_volume = float(latest_ce.get('volume', 0) or 0)
                    ce_price = float(latest_ce.get('close', 0) or 0)
                    
                    # Get previous minute OI if available (look for candle 1 minute earlier)
                    if len(filtered_candles) > 1:
                        prev_minute = current_minute - timedelta(minutes=1)
                        # Search in the tuple list for exact minute match
                        for candle_dt, candle in reversed(filtered_candles_with_ts[:-1]):
                            if candle_dt == prev_minute:
                                ce_prev_oi = float(candle.get('oi', 0) or 0)
                                break
                    
                    # If no previous minute found, use the second-to-last candle as fallback
                    if ce_prev_oi == 0.0 and len(filtered_candles) > 1:
                        prev_ce = filtered_candles[-2]
                        ce_prev_oi = float(prev_ce.get('oi', 0) or 0)
            else:
                logger.debug(f"No CE candles found for token {ce_token} on {current_time.date()}")
                
        except Exception as e:
            logger.warning(f"Error fetching CE data for token {ce_token}: {e}")
        
        try:
            # Fetch PE data - same logic as CE
            date_str = current_time.strftime('%Y-%m-%d')
            pe_candles = kite_obj.historical_data(
                pe_token,
                date_str,
                date_str,
                'minute',
                continuous=False,
                oi=True
            )
            
            if pe_candles and len(pe_candles) > 0:
                # Filter candles to current minute or most recent
                filtered_candles_with_ts = []
                for candle in pe_candles:
                    candle_time = candle.get('date')
                    if candle_time:
                        candle_dt = to_ist(candle_time) if hasattr(candle_time, 'replace') else candle_time
                        if isinstance(candle_dt, datetime):
                            candle_dt = candle_dt.replace(second=0, microsecond=0)
                        # Include candles up to and including current minute
                        if candle_dt <= current_minute:
                            filtered_candles_with_ts.append((candle_dt, candle))
                
                # Sort by timestamp
                filtered_candles_with_ts.sort(key=lambda x: x[0])
                filtered_candles = [c for _, c in filtered_candles_with_ts]
                
                if filtered_candles:
                    latest_pe = filtered_candles[-1]
                    pe_oi = float(latest_pe.get('oi', 0) or 0)
                    pe_volume = float(latest_pe.get('volume', 0) or 0)
                    pe_price = float(latest_pe.get('close', 0) or 0)
                    
                    # Get previous minute OI
                    if len(filtered_candles) > 1:
                        prev_minute = current_minute - timedelta(minutes=1)
                        # Search in the tuple list for exact minute match
                        for candle_dt, candle in reversed(filtered_candles_with_ts[:-1]):
                            if candle_dt == prev_minute:
                                pe_prev_oi = float(candle.get('oi', 0) or 0)
                                break
                    
                    if pe_prev_oi == 0.0 and len(filtered_candles) > 1:
                        prev_pe = filtered_candles[-2]
                        pe_prev_oi = float(prev_pe.get('oi', 0) or 0)
            else:
                logger.debug(f"No PE candles found for token {pe_token} on {current_time.date()}")
                
        except Exception as e:
            logger.warning(f"Error fetching PE data for token {pe_token}: {e}")
        
        # Calculate change in OI (always calculate difference, even if previous is 0)
        ce_oi_change = ce_oi - ce_prev_oi
        pe_oi_change = pe_oi - pe_prev_oi
        
        # Log if we got zero OI (might indicate data not available yet)
        if ce_oi == 0.0 or pe_oi == 0.0:
            logger.debug(
                f"Zero OI detected for strike {strike}, expiry {expiry_date}: "
                f"CE_OI={ce_oi}, PE_OI={pe_oi} "
                f"(CE_prev={ce_prev_oi}, PE_prev={pe_prev_oi})"
            )
        
        # Calculate IV if we have option prices and spot price
        ce_iv = 0.0
        pe_iv = 0.0
        if spot_price:
            if ce_price > 0:
                ce_iv = _calculate_iv_from_price('CE', ce_price, spot_price, strike, expiry_date) or 0.0
            if pe_price > 0:
                pe_iv = _calculate_iv_from_price('PE', pe_price, spot_price, strike, expiry_date) or 0.0
        
        result = {
            'oi_call': ce_oi,
            'oi_put': pe_oi,
            'oi_change_call': ce_oi_change,
            'oi_change_put': pe_oi_change,
            'volume_call': ce_volume,
            'volume_put': pe_volume,
            'iv_call': ce_iv,
            'iv_put': pe_iv,
        }
        
        # Log if result has all zeros (might indicate an issue)
        if all(v == 0.0 for k, v in result.items() if k not in ['oi_change_call', 'oi_change_put']):
            logger.warning(
                f"All metrics are zero for strike {strike}, expiry {expiry_date}. "
                f"This might indicate data is not available yet or instruments were not found."
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching option chain for strike {strike}, expiry {expiry_date}: {e}", exc_info=True)
        return {
            'oi_call': 0.0, 'oi_put': 0.0,
            'oi_change_call': 0.0, 'oi_change_put': 0.0,
            'volume_call': 0.0, 'volume_put': 0.0,
            'iv_call': 0.0, 'iv_put': 0.0
        }


def aggregate_multi_expiry_data(
    kite_obj: KiteApp,
    underlying_prefix: str,
    strike: float,
    expiry_dates: List[date],
    options_exchange: str = "NFO",
    use_websocket_data: Optional[Dict[int, Dict]] = None,
    previous_oi: Optional[Dict[int, float]] = None,
    spot_price: Optional[float] = None,
    target_timestamp: Optional[datetime] = None
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Fetch and aggregate option chain data across multiple expiries.
    
    Args:
        kite_obj: KiteApp instance
        underlying_prefix: Underlying prefix (e.g., "NIFTY")
        strike: Strike price
        expiry_dates: List of expiry dates to fetch
        options_exchange: Options exchange segment
        use_websocket_data: Optional websocket tick data
        previous_oi: Optional dict of {instrument_token: previous_oi} for calculating change in OI
        spot_price: Current spot price (needed for IV calculation)
        target_timestamp: Optional target timestamp for historical data fetching
    
    Returns:
        Tuple of (list of expiry data dicts, aggregated metrics dict)
    """
    expiry_data_list = []
    
    for expiry_date in expiry_dates:
        expiry_data = fetch_kite_option_chain_for_expiry(
            kite_obj, underlying_prefix, strike, expiry_date,
            options_exchange, use_websocket_data, previous_oi, spot_price,
            target_timestamp=target_timestamp
        )
        expiry_data_list.append(expiry_data)
    
    # Aggregate using aggregator module
    from .aggregator import aggregate_option_metrics
    aggregated = aggregate_option_metrics(expiry_data_list)
    
    return expiry_data_list, aggregated