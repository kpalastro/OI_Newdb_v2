# Implementation Plan: NSE Option Chain Features Based on ATM Strike

## Overview
Add 8 new features based on NSE option chain data fetched using the ATM (At The Money) strike price calculated from the market open price.

## Feature List
The following 8 features will be added:
1. **`nse_next_oi_call_total`** - Total Open Interest for CALL options
2. **`nse_next_oi_put_total`** - Total Open Interest for PUT options
3. **`nse_next_oi_change_call_total`** - Total Change in OI for CALL options
4. **`nse_next_oi_change_put_total`** - Total Change in OI for PUT options
5. **`nse_next_volume_call_total`** - Total Volume for CALL options
6. **`nse_next_volume_put_total`** - Total Volume for PUT options
7. **`nse_next_oi_change_diff_put_call`** - Difference in Change in OI (PUT - CALL)
8. **`oi_next_sentiment`** - Sentiment feature (same value as `nse_next_oi_change_diff_put_call`) - placed in Sentiment section

## Implementation Steps

### Step 1: Create Helper Function to Calculate Nearest ATM Strike
**Location**: `feature_engineering.py` (new function)

```python
def _calculate_nearest_atm_strike(open_price: float, strike_difference: int) -> float:
    """
    Calculate the nearest ATM strike price based on open price.
    
    Args:
        open_price: Market open price of the underlying
        strike_difference: Strike difference from config (e.g., 50 for NIFTY, 100 for BANKNIFTY)
    
    Returns:
        Nearest ATM strike price rounded to nearest strike_difference
    """
    # Round to nearest strike_difference
    nearest_strike = round(open_price / strike_difference) * strike_difference
    return nearest_strike
```

### Step 2: Create Function to Fetch NSE Option Chain Data
**Location**: `feature_engineering.py` (new function)

```python
def _fetch_nse_option_chain_data(strike: float) -> Optional[Dict]:
    """
    Fetch option chain data from NSE API for a given strike price.
    
    Args:
        strike: Strike price to fetch data for (e.g., 25950.00)
    
    Returns:
        Dictionary containing option chain data or None if fetch fails
    """
    import requests
    from time import sleep
    
    url = f"https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol=NIFTY&strike={strike:,.2f}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/option-chain",
        "X-Requested-With": "XMLHttpRequest"
    }
    
    try:
        # First, establish a session by visiting the main page
        session = requests.Session()
        session.get("https://www.nseindia.com/option-chain", headers=headers, timeout=10)
        sleep(0.5)  # Small delay to avoid rate limiting
        
        # Now fetch the option chain data
        response = session.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            logging.warning(f"NSE API returned status {response.status_code} for strike {strike}")
            return None
    except Exception as e:
        logging.error(f"Error fetching NSE option chain data for strike {strike}: {e}")
        return None
```

### Step 3: Create Function to Parse and Aggregate Option Chain Data
**Location**: `feature_engineering.py` (new function)

```python
def _parse_nse_option_chain_data(option_chain_data: Dict) -> Dict[str, float]:
    """
    Parse NSE option chain JSON response and aggregate OI, Change in OI, and Volume.
    
    Args:
        option_chain_data: Raw JSON response from NSE API
    
    Returns:
        Dictionary with aggregated metrics:
        - total_oi_call
        - total_oi_put
        - total_oi_change_call
        - total_oi_change_put
        - total_volume_call
        - total_volume_put
        - oi_change_diff_put_call
    """
    try:
        # Navigate through the JSON structure
        # Based on typical NSE API response structure:
        # data['records']['data'] contains array of strike data
        records = option_chain_data.get('records', {})
        data = records.get('data', [])
        
        total_oi_call = 0.0
        total_oi_put = 0.0
        total_oi_change_call = 0.0
        total_oi_change_put = 0.0
        total_volume_call = 0.0
        total_volume_put = 0.0
        
        for strike_data in data:
            # Extract CALL option data
            ce_data = strike_data.get('CE', {})
            if ce_data:
                total_oi_call += float(ce_data.get('openInterest', 0) or 0)
                total_oi_change_call += float(ce_data.get('changeinOpenInterest', 0) or 0)
                total_volume_call += float(ce_data.get('totalTradedVolume', 0) or 0)
            
            # Extract PUT option data
            pe_data = strike_data.get('PE', {})
            if pe_data:
                total_oi_put += float(pe_data.get('openInterest', 0) or 0)
                total_oi_change_put += float(pe_data.get('changeinOpenInterest', 0) or 0)
                total_volume_put += float(pe_data.get('totalTradedVolume', 0) or 0)
        
        # Calculate difference: PUT - CALL
        oi_change_diff_put_call = total_oi_change_put - total_oi_change_call
        
        return {
            'total_oi_call': total_oi_call,
            'total_oi_put': total_oi_put,
            'total_oi_change_call': total_oi_change_call,
            'total_oi_change_put': total_oi_change_put,
            'total_volume_call': total_volume_call,
            'total_volume_put': total_volume_put,
            'oi_change_diff_put_call': oi_change_diff_put_call
        }
    except Exception as e:
        logging.error(f"Error parsing NSE option chain data: {e}")
        # Return zeros if parsing fails
        return {
            'total_oi_call': 0.0,
            'total_oi_put': 0.0,
            'total_oi_change_call': 0.0,
            'total_oi_change_put': 0.0,
            'total_volume_call': 0.0,
            'total_volume_put': 0.0,
            'oi_change_diff_put_call': 0.0
        }
```

### Step 4: Create Main Function to Get Market Open Price
**Location**: `feature_engineering.py` (new function)

```python
def _get_market_open_price(handler) -> Optional[float]:
    """
    Get the market open price from handler's data reels.
    
    Args:
        handler: ExchangeDataHandler instance
    
    Returns:
        Open price for the current trading session or None if not available
    """
    try:
        if handler.underlying_token is None:
            return None
        
        # Get the first bar of the current session from data reels
        data_reel = handler.data_reels.get(handler.underlying_token)
        if not data_reel or len(data_reel) == 0:
            return None
        
        # The first entry in the reel should have the open price
        # Or we can get it from the first minute bar
        first_bar = data_reel[0] if len(data_reel) > 0 else None
        
        if first_bar:
            # Check if it has open_price field
            if 'open_price' in first_bar:
                return float(first_bar['open_price'])
            # Fallback: use the first LTP as open price
            elif 'ltp' in first_bar:
                return float(first_bar['ltp'])
        
        # Alternative: Get from latest_oi_data if available
        # (though this might be current price, not open)
        return None
    except Exception as e:
        logging.error(f"Error getting market open price: {e}")
        return None
```

### Step 5: Integrate into `engineer_live_feature_set`
**Location**: `feature_engineering.py` (modify existing function)

Add the following code in `engineer_live_feature_set()` function, after the sentiment features section (around line 300 where `features.update(sentiment_metrics)` is called) and after all other feature calculations:

```python
# NSE Option Chain Features (based on ATM strike from open price)
# Note: This should be added after the sentiment features section (line ~300)
try:
    # Get market open price
    open_price = _get_market_open_price(handler)
    
    if open_price is not None:
        # Calculate nearest ATM strike
        strike_difference = handler.config.get('strike_difference', 50)
        atm_strike = _calculate_nearest_atm_strike(open_price, strike_difference)
        
        # Fetch NSE option chain data (with caching)
        option_chain_data = _get_cached_nse_data(atm_strike)
        if option_chain_data is None:
            option_chain_data = _fetch_nse_option_chain_data(atm_strike)
            if option_chain_data:
                _cache_nse_data(atm_strike, option_chain_data)
        
        if option_chain_data:
            # Parse and aggregate the data
            nse_metrics = _parse_nse_option_chain_data(option_chain_data)
            
            # Add to features with prefix 'nse_next_'
            features['nse_next_oi_call_total'] = nse_metrics['total_oi_call']
            features['nse_next_oi_put_total'] = nse_metrics['total_oi_put']
            features['nse_next_oi_change_call_total'] = nse_metrics['total_oi_change_call']
            features['nse_next_oi_change_put_total'] = nse_metrics['total_oi_change_put']
            features['nse_next_volume_call_total'] = nse_metrics['total_volume_call']
            features['nse_next_volume_put_total'] = nse_metrics['total_volume_put']
            features['nse_next_oi_change_diff_put_call'] = nse_metrics['oi_change_diff_put_call']
            
            # Add sentiment feature (same value as the difference) - placed in Sentiment section
            features['oi_next_sentiment'] = nse_metrics['oi_change_diff_put_call']
        else:
            # Set to zero if fetch failed
            features.update({
                'nse_next_oi_call_total': 0.0,
                'nse_next_oi_put_total': 0.0,
                'nse_next_oi_change_call_total': 0.0,
                'nse_next_oi_change_put_total': 0.0,
                'nse_next_volume_call_total': 0.0,
                'nse_next_volume_put_total': 0.0,
                'nse_next_oi_change_diff_put_call': 0.0,
                'oi_next_sentiment': 0.0
            })
    else:
        # Set to zero if open price not available
        features.update({
            'nse_next_oi_call_total': 0.0,
            'nse_next_oi_put_total': 0.0,
            'nse_next_oi_change_call_total': 0.0,
            'nse_next_oi_change_put_total': 0.0,
            'nse_next_volume_call_total': 0.0,
            'nse_next_volume_put_total': 0.0,
            'nse_next_oi_change_diff_put_call': 0.0,
            'oi_next_sentiment': 0.0
        })
except Exception as e:
    logging.error(f"Error calculating NSE option chain features: {e}", exc_info=True)
    # Set to zero on error
    features.update({
        'nse_next_oi_call_total': 0.0,
        'nse_next_oi_put_total': 0.0,
        'nse_next_oi_change_call_total': 0.0,
        'nse_next_oi_change_put_total': 0.0,
        'nse_next_volume_call_total': 0.0,
        'nse_next_volume_put_total': 0.0,
        'nse_next_oi_change_diff_put_call': 0.0,
        'oi_next_sentiment': 0.0
    })
```

### Step 6: Update REQUIRED_FEATURE_COLUMNS
**Location**: `feature_engineering.py` (modify existing list)

Add the 8 new feature names to `REQUIRED_FEATURE_COLUMNS`:

```python
REQUIRED_FEATURE_COLUMNS = [
    # ... existing features ...
    # Feature 5: Sentiment Features (existing section)
    'sentiment_ad_ratio_50',
    'sentiment_ad_ratio_100',
    'sentiment_trin_50',
    'sentiment_trin_100',
    'sentiment_score',
    'sentiment_fii_net_crores',
    'sentiment_dii_net_crores',
    'sentiment_inst_net_crores',
    'oi_next_sentiment',  # NEW: Added to Sentiment section
    # ... other features ...
    # NSE Option Chain Features (new section at end)
    'nse_next_oi_call_total',
    'nse_next_oi_put_total',
    'nse_next_oi_change_call_total',
    'nse_next_oi_change_put_total',
    'nse_next_volume_call_total',
    'nse_next_volume_put_total',
    'nse_next_oi_change_diff_put_call',
]
```

**Note**: The `oi_next_sentiment` feature should be placed in the "Feature 5: Sentiment Features" section (after `sentiment_inst_net_crores`), while the 7 NSE features should be added in a new "NSE Option Chain Features" section.

### Step 7: Handle Rate Limiting and Caching (Optional but Recommended)
**Location**: `feature_engineering.py` (add caching mechanism)

To avoid making too many API calls, add a simple cache:

```python
# Module-level cache (will be cleared on restart)
_nse_option_chain_cache: Dict[float, Tuple[Dict, datetime]] = {}
_cache_timeout_seconds = 30  # Cache for 30 seconds

def _get_cached_nse_data(strike: float) -> Optional[Dict]:
    """Get cached NSE option chain data if available and fresh."""
    from datetime import datetime, timedelta
    
    global _nse_option_chain_cache
    
    if strike in _nse_option_chain_cache:
        data, cached_time = _nse_option_chain_cache[strike]
        if datetime.now() - cached_time < timedelta(seconds=_cache_timeout_seconds):
            return data
    
    return None

def _cache_nse_data(strike: float, data: Dict):
    """Cache NSE option chain data with timestamp."""
    global _nse_option_chain_cache
    _nse_option_chain_cache[strike] = (data, datetime.now())
```

## Data Flow Diagram

```
Market Open Price (from handler.data_reels)
    ↓
Calculate Nearest ATM Strike (round to strike_difference)
    ↓
Check Cache for NSE Option Chain Data
    ↓ (if not cached or expired)
Fetch from NSE API: https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol=NIFTY&strike={strike}
    ↓
Parse JSON Response
    ↓
Sum OI, Change in OI, Volume for Calls and Puts
    ↓
Calculate Difference: PUT Change in OI - CALL Change in OI
    ↓
Add 8 Features to Feature Dictionary (7 NSE features + 1 sentiment feature)
```

## Important Considerations

1. **Rate Limiting**: NSE API may have rate limits. We'll use caching to reduce API calls.

2. **Error Handling**: If API fetch fails or parsing fails, features will be set to 0.0 to ensure the pipeline continues.

3. **Open Price Source**: We'll get open price from the first bar in the data reel. If not available, features will default to 0.0.

4. **Strike Calculation**: Uses `strike_difference` from config (50 for NIFTY, 100 for BANKNIFTY/SENSEX).

5. **API Session**: NSE requires a session cookie. We'll establish a session before making the API call.

6. **Feature Names**: All NSE features prefixed with `nse_next_` to clearly identify their source. The sentiment feature `oi_next_sentiment` is placed in the Sentiment section.

## Testing Plan

1. **Unit Tests**:
   - Test `_calculate_nearest_atm_strike()` with various prices
   - Test `_parse_nse_option_chain_data()` with sample JSON
   - Test error handling when API fails

2. **Integration Tests**:
   - Verify features are added to `engineer_live_feature_set()` output
   - Verify features are included in training data preparation

3. **Manual Testing**:
   - Run live system and verify features are populated
   - Check logs for any API errors
   - Verify cache is working (check API call frequency)

## Notes

- The JSON structure from NSE API might need adjustment based on actual response format. We'll need to inspect the actual API response first.
- The URL format uses `strike=25,950.00` (with comma as thousand separator) - we'll format it correctly.
- This feature will only work for NSE/NIFTY. For BSE/SENSEX, we might need a different API endpoint.
