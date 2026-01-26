# YouTube Strategy - Real-Time Signal & Auto-Trade Setup

## ✅ Current Status

**YES, the YouTube strategy WILL generate real-time signals and auto-trade!**

### Integration Status

1. **✅ Strategy Router**: YouTube strategy is registered and available
2. **✅ Auto Executor**: Integrated and can execute paper trades automatically
3. **✅ Real-Time Signal Generation**: Active in `feature_result_consumer()`
4. **✅ Auto-Execution**: Enabled by default (`auto_exec_enabled = True`)

## 📋 How It Works

### Signal Flow

```
Real-time Data
    ↓
Feature Engineering
    ↓
Strategy Router (AdvancedStrategyRouter)
    ↓
YouTube Strategy (if conditions match)
    ↓
TradeRecommendation (BUY/SELL/HOLD)
    ↓
Auto Executor (if BUY/SELL)
    ↓
Paper Trade Execution
```

### Current Implementation

1. **Real-time data** flows into `feature_result_consumer()`
2. **Strategy router** generates signals from features
3. **YouTube strategy** can be selected in two ways:
   - **Explicit**: Set `use_youtube_strategy=True` in features/metadata
   - **Automatic**: Used as fallback for intraday horizon when no other strategy matches
4. **Auto executor** executes trades when signal is BUY/SELL (not HOLD)

## ⚙️ Configuration

### Enable YouTube Strategy Explicitly

To force YouTube strategy to be used, you need to set:

```python
# In feature engineering or signal generation
features['use_youtube_strategy'] = True

# OR in signal metadata
signal.metadata['use_youtube_strategy'] = True
```

### Auto-Execution Settings

Auto-execution is **enabled by default**. Configuration in `config.py`:

```python
auto_exec_enabled: bool = True  # Default: enabled
auto_exec_min_kelly_fraction: float = 0.2
auto_exec_max_position_size_lots: int = 4
auto_exec_max_open_positions: int = 2
```

### Environment Variables

You can control auto-execution via `.env` file:

```bash
OI_TRACKER_AUTO_EXEC_ENABLED=True
OI_TRACKER_MIN_CONFIDENCE_FOR_TRADE=0.60
OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS=2
```

## 🔧 How to Enable YouTube Strategy for Real-Time Trading

### Option 1: Modify Strategy Router (Recommended)

Edit `execution/strategy_router.py` to prioritize YouTube strategy:

```python
# In route_strategy() method, add at the top:
if horizon == 'intraday':
    selected_strategy = self.strategies.get('youtube')
    if selected_strategy:
        LOGGER.debug(f"[{self.exchange}] Routing to YouTube strategy for intraday")
```

### Option 2: Set Feature Flag

Modify feature engineering to always set:

```python
features['use_youtube_strategy'] = True
```

### Option 3: Modify Signal Generation

In `ml_core.py` or wherever signals are generated, add:

```python
metadata['use_youtube_strategy'] = True
```

## 📊 Current Behavior

### Automatic Fallback

Currently, YouTube strategy is used as **fallback** for intraday horizon:

```python
# From strategy_router.py line 317-319
# Fallback: Use YouTube strategy for intraday if no other strategy selected
if not selected_strategy and horizon == 'intraday':
    selected_strategy = self.strategies.get('youtube')
```

This means:
- ✅ YouTube strategy **will** be used if no other strategy matches
- ⚠️ But other strategies (expiry_pin, gamma_scalping, oi_buildup, vol_expansion) take priority

### To Force YouTube Strategy

To ensure YouTube strategy is **always** used for intraday, modify the routing logic to check YouTube strategy first.

## 🎯 Verification Steps

1. **Check if auto-execution is enabled**:
   ```python
   from config import get_config
   config = get_config()
   print(f"Auto-exec enabled: {config.auto_exec_enabled}")
   ```

2. **Monitor logs** for YouTube strategy usage:
   ```
   [NSE] Routing to YouTube strategy (explicitly requested)
   [NSE] Auto trade executed: BUY NIFTY25JAN26000CE
   ```

3. **Check signal generation**:
   - Look for `use_youtube_strategy` in logs
   - Verify TradeRecommendation has `strategy_name='YouTubeStrategy'`

## ⚠️ Important Notes

1. **Paper Trading Only**: Auto-execution is in **paper mode** (not real money)
2. **Time Filter**: YouTube strategy only trades after 9:15 AM (flexible start)
3. **Strict Filters**: With current BSE/NSE filters, trades will be very selective
4. **Volume Filter**: Only active for BSE (disabled for NSE due to weak correlation)

## 🚀 Quick Start

To enable YouTube strategy for real-time auto-trading:

1. **Ensure auto-execution is enabled** (default: True)
2. **Set feature flag** to force YouTube strategy:
   ```python
   # In feature engineering or signal generation
   features['use_youtube_strategy'] = True
   ```
3. **Run the system**:
   ```bash
   python oi_tracker_new.py
   ```
4. **Monitor logs** for YouTube strategy signals and auto-execution

## 📈 Expected Behavior

With current filters:
- **BSE**: Very selective (2 trades in 26 days) with 100% win rate
- **NSE**: More frequent (9 trades in 31 days) with 100% win rate
- **Auto-execution**: Will execute when YouTube strategy returns BUY/SELL
- **Paper trades**: All trades are paper trades (not real money)

## 🔍 Troubleshooting

If YouTube strategy is not being used:

1. Check logs for routing decisions
2. Verify `use_youtube_strategy` flag is set
3. Check if other strategies are matching first
4. Verify horizon is 'intraday'
5. Check if time is after 9:15 AM

If auto-execution is not working:

1. Verify `auto_exec_enabled = True` in config
2. Check signal is BUY/SELL (not HOLD)
3. Verify confidence meets minimum threshold
4. Check position limits (max_open_positions)
5. Verify cooldown period has passed
