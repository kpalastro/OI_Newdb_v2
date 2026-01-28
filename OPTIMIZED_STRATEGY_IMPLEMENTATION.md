# Optimized Option Return Strategy - Independent Implementation

**Date:** January 27, 2026  
**Status:** ✅ READY TO RUN

---

## Overview

The **OptimizedOptionReturnStrategy** is an independent strategy that runs alongside existing strategies. It uses the improved backtest logic:

- ✅ BSE/NSE prediction scaling (0.21x / 0.70x)
- ✅ Early exit at 30%/40%/50% of predicted return
- ✅ Stop-loss at -5% of entry price
- ✅ Stricter entry filters (confidence 0.6+, price ₹50-200, time 13:00-15:30)
- ✅ Horizon preferences (15m, 10m - avoids 3m, 5m)
- ✅ Option type preferences (NSE: CE preferred)

---

## Architecture

### 1. Strategy Class
**File:** `strategies/optimized_option_return.py`

- Extends `BaseStrategy`
- Uses `OptionReturnPredictor` with automatic scaling
- Applies all entry filters
- Returns `TradeRecommendation` with strategy metadata

### 2. Strategy Router Integration
**File:** `execution/strategy_router.py`

- Automatically routes to `OptimizedOptionReturnStrategy` when:
  - Option return prediction is available in signal metadata
  - Predicted return > 0.2%
  - Confidence ≥ 0.6

### 3. Position Monitor
**File:** `execution/optimized_position_monitor.py`

- Separate monitor for positions opened by this strategy
- Applies improved exit logic:
  - Early exit at 30%/40%/50% targets
  - Stop-loss at -5%
  - Percentage-based exits

### 4. Auto Executor Integration
**File:** `execution/auto_executor.py`

- Stores strategy metadata in positions:
  - `strategy`: Strategy name ("OptimizedOptionReturn")
  - `predicted_return`: Predicted return percentage
  - `horizon`: Time horizon (15m, 10m)
  - `entry_timestamp`: Entry time for exit calculations

---

## How It Works

### Entry Flow

1. **Signal Generation** (`ml_core.py`)
   - Option return predictor generates predictions
   - Predictions added to signal metadata

2. **Strategy Routing** (`execution/strategy_router.py`)
   - Checks if option return prediction is strong enough
   - Routes to `OptimizedOptionReturnStrategy` if:
     - `option_return_prediction` in metadata
     - Max return > 0.2%
     - Max confidence ≥ 0.6

3. **Strategy Analysis** (`strategies/optimized_option_return.py`)
   - Applies entry filters:
     - Time: 13:00-15:30
     - Confidence: ≥ 0.6
     - Entry price: ₹50-200
     - Horizon: 15m or 10m only
     - Option type: NSE prefers CE
   - Returns `TradeRecommendation` with metadata

4. **Trade Execution** (`execution/auto_executor.py`)
   - Executes trade if all checks pass
   - Stores strategy metadata in position

### Exit Flow

1. **Position Monitoring** (`execution/optimized_position_monitor.py`)
   - Called every tick alongside `monitor_positions()`
   - Only monitors positions with `strategy == 'OptimizedOptionReturn'`

2. **Exit Logic**
   - **Priority 1:** End-of-Day exit (15:20 IST)
   - **Priority 2:** Early exit at 30%/40%/50% targets
   - **Priority 3:** Stop-loss at -5%
   - **Priority 4:** Fallback to 50% target or -5% stop-loss

3. **Early Exit Check**
   - Queries option price every minute
   - Checks if 30%/40%/50% of predicted return reached
   - Exits immediately when target hit

4. **Stop-Loss Check**
   - Monitors price every minute
   - Exits if loss exceeds -5% of entry price

---

## Configuration

### Entry Filters (Hardcoded in Strategy)

```python
# Time filter
Trading hours: 13:00 - 15:30 IST

# Confidence filter
Minimum confidence: 0.6 (60%)

# Entry price filter
Minimum: ₹50
Maximum: ₹200

# Horizon preference
Preferred: ['15m', '10m']
Avoid: ['3m', '5m']

# Option type preference
NSE: Prefers CE (if CE return > PE return)
BSE: Uses PE (only type available)
```

### Exit Settings (Hardcoded in Monitor)

```python
# Early exit targets
target_30_pct: True
target_40_pct: True
target_50_pct: True

# Stop-loss
stop_loss_pct: -5.0%

# Check interval
check_interval_minutes: 1
```

---

## Running the Strategy

### Automatic Activation

The strategy **automatically activates** when:
1. Option return models are loaded
2. Signal contains `option_return_prediction` in metadata
3. Prediction is strong enough (return > 0.2%, confidence ≥ 0.6)

### Manual Verification

Check logs for:
```
[EXCHANGE] Routing to OptimizedOptionReturnStrategy (return: X.XX%, conf: XX.X%)
[EXCHANGE] OptimizedOptionReturn: Closed position XXX - Early Exit 30% ...
```

### Position Identification

Positions opened by this strategy have:
- `strategy: 'OptimizedOptionReturn'`
- `predicted_return: <value>`
- `horizon: '15m' or '10m'`

---

## Expected Performance

Based on backtest improvements:

### Win Rate
- **Before:** 50-60% (with fixed point exits)
- **After:** 60-70% (with early exit at 30-50% targets)

### Average Holding Time
- **Before:** 15 minutes (full horizon)
- **After:** 2-4 minutes (early exit)

### Stop-Loss Protection
- **Before:** Fixed points (50 BSE, 25 NSE)
- **After:** -5% of entry price (percentage-based)

### Risk Management
- Entry price filter avoids illiquid options
- Time filter focuses on best-performing hours
- Horizon preference avoids poor-performing timeframes

---

## Monitoring

### Position Tags

All positions opened by this strategy are tagged with:
```python
{
    'strategy': 'OptimizedOptionReturn',
    'predicted_return': 2.5,  # Example: 2.5%
    'horizon': '15m',
    'entry_timestamp': datetime(...)
}
```

### Exit Reasons

Positions close with reasons like:
- `"Early Exit 30%"` - Exited at 30% of predicted return
- `"Early Exit 40%"` - Exited at 40% of predicted return
- `"Early Exit 50%"` - Exited at 50% of predicted return
- `"Stop Loss (-5.0%)"` - Stop-loss triggered
- `"End of Day Exit (15:20 IST)"` - EOD exit

---

## Files Modified/Created

### New Files
1. `strategies/optimized_option_return.py` - Strategy implementation
2. `execution/optimized_position_monitor.py` - Position monitor
3. `OPTIMIZED_STRATEGY_IMPLEMENTATION.md` - This document

### Modified Files
1. `execution/strategy_router.py` - Added routing logic
2. `execution/auto_executor.py` - Store strategy metadata
3. `oi_tracker_new.py` - Call optimized monitor

---

## Testing

### 1. Verify Strategy Loads

Check logs on startup:
```
[EXCHANGE] OptimizedOptionReturnStrategy initialized
```

### 2. Verify Routing

When option return prediction is strong:
```
[EXCHANGE] Routing to OptimizedOptionReturnStrategy (return: 2.50%, conf: 75.0%)
```

### 3. Verify Entry Filters

Check that trades only occur:
- Between 13:00-15:30
- With confidence ≥ 0.6
- With entry price ₹50-200
- Using 15m or 10m horizon

### 4. Verify Exit Logic

Check that positions close with:
- Early exit reasons (30%/40%/50%)
- Stop-loss at -5%
- Faster exits (2-4 minutes vs 15 minutes)

---

## Troubleshooting

### Strategy Not Activating

**Check:**
1. Option return models loaded? (`models/option_returns/{EXCHANGE}/`)
2. Signal contains `option_return_prediction`?
3. Prediction strong enough? (return > 0.2%, confidence ≥ 0.6)

### Positions Not Closing Early

**Check:**
1. Position has `strategy == 'OptimizedOptionReturn'`?
2. Position has `predicted_return` stored?
3. `monitor_optimized_positions()` being called?
4. Option prices available in database?

### Import Errors

**Fix:**
```bash
# Ensure all dependencies available
python3 -c "from strategies.optimized_option_return import OptimizedOptionReturnStrategy"
python3 -c "from execution.optimized_position_monitor import monitor_optimized_positions"
```

---

## Next Steps

1. **Run in Paper Trading Mode**
   - Monitor for 1-2 weeks
   - Compare results vs backtest expectations
   - Verify early exit frequency

2. **Fine-Tune Thresholds**
   - Adjust target percentages if needed
   - Adjust stop-loss percentage
   - Adjust entry filters

3. **Performance Analysis**
   - Track win rate
   - Track average holding time
   - Track early exit vs horizon exit ratio

---

## Summary

✅ **Strategy is ready to run independently alongside existing strategies**

- Automatically activates when conditions are met
- Uses improved exit logic (early exit + stop-loss)
- Applies strict entry filters
- Monitors positions separately from existing system

**No manual configuration needed** - it runs automatically when option return predictions are available and strong enough.
