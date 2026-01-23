# Trade Hours Clarification - IST Market Hours

## Market Hours (IST)
- **Market Opens**: 9:15 AM IST
- **Market Closes**: 3:30 PM IST
- **Trading Window**: 9:15 AM - 3:30 PM IST (6 hours 15 minutes)

## Problem Hours Identified

### Hour 6 (6:00-6:59 AM IST) - PRE-MARKET ❌
- **Status**: BEFORE market open (3+ hours before)
- **Problem**: Trades entering based on stale data or pre-market signals
- **Performance**: 4.4% win rate (very poor)
- **Why it's bad**: 
  - Market isn't open yet
  - Data may be stale from previous day
  - No real-time price discovery
  - Features calculated on old data

**Recommendation**: **BLOCK all trades before 9:15 AM IST**

### Hour 7 (7:00-7:59 AM IST) - PRE-MARKET ❌
- **Status**: BEFORE market open (2+ hours before)
- **Problem**: Same as hour 6
- **Performance**: 0% win rate (1 trade sample, but still bad)
- **Why it's bad**: Same reasons as hour 6

**Recommendation**: **BLOCK all trades before 9:15 AM IST**

### Hour 15 (3:00-3:59 PM IST) - NEAR MARKET CLOSE ⚠️
- **Status**: Near market close (market closes at 3:30 PM)
- **Problem**: Trades entering too close to market close
- **Performance**: 35.1% win rate (poor)
- **Why it's bad**:
  - Only 30 minutes or less before market close
  - Not enough time for targets to be hit
  - Increased risk of overnight positions
  - End-of-day volatility and position squaring

**Recommendation**: **BLOCK trades after 3:00 PM IST** (or 2:45 PM for safety)

## Corrected Recommendations

### Updated Time Filter

```python
def is_valid_trading_time(timestamp) -> bool:
    """
    Check if current time is within valid trading hours.
    Market: 9:15 AM - 3:30 PM IST
    """
    hour = timestamp.hour
    minute = timestamp.minute
    
    # Before market open (before 9:15 AM)
    if hour < 9 or (hour == 9 and minute < 15):
        return False
    
    # After market close (after 3:30 PM)
    if hour > 15 or (hour == 15 and minute > 30):
        return False
    
    return True

# In your trading logic:
if not is_valid_trading_time(now_ist()):
    logging.info(f"Skipping trade - outside market hours (9:15 AM - 3:30 PM IST)")
    return
```

### Alternative: More Restrictive Filter

If you want to be more conservative and avoid the problematic hour 15:

```python
def is_valid_trading_time(timestamp) -> bool:
    """
    Conservative trading hours: 9:15 AM - 2:45 PM IST
    Avoids last 45 minutes before market close
    """
    hour = timestamp.hour
    minute = timestamp.minute
    
    # Before market open
    if hour < 9 or (hour == 9 and minute < 15):
        return False
    
    # After 2:45 PM (avoid last 45 minutes)
    if hour > 14 or (hour == 14 and minute > 45):
        return False
    
    return True
```

## Updated Analysis Understanding

### Why Hour 6 & 7 Are So Bad
1. **Pre-market trading**: Market isn't open, so no real price discovery
2. **Stale data**: Features calculated on previous day's closing data
3. **No liquidity**: Options may not have real bid/ask spreads
4. **Gap risk**: Market can gap significantly at open

### Why Hour 15 Is Bad
1. **Time constraint**: Only 30 minutes or less before close
2. **Position squaring**: End-of-day volatility from traders closing positions
3. **Overnight risk**: Positions may need to be held overnight
4. **Reduced liquidity**: Many traders have already closed positions

## Best Trading Hours (Within Market Hours)

Based on analysis, the best hours are:
- **Hour 11 (11:00-11:59 AM)**: 55.3% win rate
- **Hour 12 (12:00-12:59 PM)**: 76.6% win rate ⭐ BEST
- **Hour 10 (10:00-10:59 AM)**: 50% win rate

These are all within valid market hours (9:15 AM - 3:30 PM).

## Implementation Priority

### HIGH PRIORITY (Do Immediately)
1. ✅ Block all trades before 9:15 AM IST
2. ✅ Block all trades after 3:00 PM IST (or 2:45 PM for safety)

### MEDIUM PRIORITY
3. Focus trading during hours 11-12 (best performance)
4. Reduce activity during hours 13-14 (afternoon performance drops)

## Code Implementation

Add to `oi_tracker_new.py` or your trading execution logic:

```python
from time_utils import now_ist

def should_skip_trade_due_to_market_hours() -> bool:
    """
    Skip trades outside market hours or during problematic times.
    Market hours: 9:15 AM - 3:30 PM IST
    """
    current_time = now_ist()
    hour = current_time.hour
    minute = current_time.minute
    
    # Block pre-market (before 9:15 AM)
    if hour < 9 or (hour == 9 and minute < 15):
        return True
    
    # Block near market close (after 3:00 PM for safety, or 2:45 PM to be conservative)
    if hour > 14 or (hour == 14 and minute > 45):
        return True
    
    return False

# In your trade execution logic:
if should_skip_trade_due_to_market_hours():
    logging.info(f"Skipping trade - outside valid trading hours (9:15 AM - 2:45 PM IST)")
    return
```

## Expected Impact

- **Blocking hours 6-7 (pre-market)**: Should eliminate ~45-50 trades with 4.4% win rate
- **Blocking hour 15 (near close)**: Should eliminate ~37 trades with 35.1% win rate
- **Total impact**: Reduce losing trades by 10-15% as estimated
- **Additional benefit**: Prevents trading on stale data and reduces overnight risk

---

**Key Takeaway**: The "hour 6, 7, 15" issue is actually about trading OUTSIDE or TOO CLOSE TO market hours. The solution is to enforce proper market hours (9:15 AM - 3:30 PM IST) and optionally avoid the last 45 minutes before close.
