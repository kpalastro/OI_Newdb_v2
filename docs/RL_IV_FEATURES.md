# RL Model IV Change Features

This document explains the IV (Implied Volatility) change features that have been added to help the RL model detect sudden IV inclines on strike prices that drive market direction.

## Overview

Sudden IV inclines on specific strike prices are strong indicators of market direction changes. When IV spikes on certain strikes, it often signals:
- **Bullish moves**: IV increase on PUT options (especially ITM puts)
- **Bearish moves**: IV increase on CALL options (especially ITM calls)
- **Volatility expansion**: IV spikes across multiple strikes indicate increased uncertainty

## IV Change Features

The following features have been added to capture IV dynamics:

### 1. ATM IV Changes
- **`atm_iv_change_3m`**: Percentage change in ATM IV over 3 minutes
- **`atm_iv_change_5m`**: Percentage change in ATM IV over 5 minutes
- **`atm_iv_change_15m`**: Percentage change in ATM IV over 15 minutes
- **`atm_iv_spike`**: Detects sudden large increases (>5% in 3 minutes) in ATM IV

### 2. IV Momentum
- **`iv_momentum`**: Rate of change of IV (first derivative)
  - Positive = IV increasing
  - Negative = IV decreasing

### 3. IV Incline Intensity
- **`iv_incline_intensity`**: Weighted average of IV increases across all strikes
  - Weighted by OI (higher OI = more significant)
  - Weighted by proximity to ATM (closer strikes = more significant)
  - Captures overall IV expansion pressure

### 4. IV Spike Detection
- **`iv_spike_strikes_count`**: Number of strikes with IV spikes (>5% increase in 3 minutes)
  - Higher count = broader volatility expansion

### 5. ITM IV Changes
- **`itm_ce_iv_change_3m`**: Average IV change for ITM CALL options
- **`itm_pe_iv_change_3m`**: Average IV change for ITM PUT options

### 6. IV Incline Direction
- **`iv_incline_direction`**: Directional indicator
  - **Positive**: CE IV rising faster (bearish signal)
  - **Negative**: PE IV rising faster (bullish signal)
  - **Zero**: Balanced or no significant change

## How It Works

### IV History Tracking
The system maintains a rolling window of IV values for each option:
- Tracks IV for last 20 minutes per option
- Updates every minute when new IV is calculated
- Uses `handler._iv_history_window` dictionary

### Feature Calculation
1. **Find ATM Options**: Identifies call and put options closest to ATM strike
2. **Track IV History**: Maintains rolling window of IV values
3. **Calculate Changes**: Computes percentage changes over different time windows
4. **Weight by Significance**: 
   - Higher OI strikes get more weight
   - Strikes closer to ATM get more weight
5. **Detect Spikes**: Flags sudden large increases (>5% in 3 minutes)

## Integration with RL Model

### Automatic Inclusion
These features are **automatically included** in the RL model state because:
1. They're added to the feature dictionary in `engineer_live_feature_set()`
2. They're included in `REQUIRED_FEATURE_COLUMNS`
3. The RL `TradingEnvironment._get_state()` extracts all numeric features

### RL State Vector
The RL model receives these features as part of its state vector:
```python
state = [
    ... (all other features) ...,
    atm_iv_change_3m,
    atm_iv_change_5m,
    atm_iv_change_15m,
    atm_iv_spike,
    iv_incline_intensity,
    iv_momentum,
    itm_ce_iv_change_3m,
    itm_pe_iv_change_3m,
    iv_spike_strikes_count,
    iv_incline_direction,
    current_position,
    portfolio_value_ratio
]
```

## Usage in Trading Decisions

### Bullish Signals
- **`iv_incline_direction < 0`**: PE IV rising faster (bullish)
- **`itm_pe_iv_change_3m > 5%`**: Strong IV increase in ITM puts
- **`iv_spike_strikes_count > 3`**: Multiple strikes showing IV expansion

### Bearish Signals
- **`iv_incline_direction > 0`**: CE IV rising faster (bearish)
- **`itm_ce_iv_change_3m > 5%`**: Strong IV increase in ITM calls
- **`atm_iv_spike > 10%`**: Large ATM IV spike

### Volatility Expansion
- **`iv_incline_intensity > 3.0`**: Strong overall IV expansion
- **`iv_spike_strikes_count > 5`**: Broad-based IV increases
- **`iv_momentum > 2.0`**: Accelerating IV increases

## Training Considerations

### Retrain RL Models
After adding these features, you should retrain your RL models:

```bash
python train_rl.py --algorithm BOTH --exchange NSE --days 30
```

This ensures the models learn to use these new IV change features.

### Feature Importance
The RL model will learn which IV features are most predictive:
- PPO/DQN will automatically weight important features
- Ensemble mode combines both models' interpretations

## Monitoring IV Features

### Check Feature Values
```python
from feature_engineering import engineer_live_feature_set

features = engineer_live_feature_set(handler, calls, puts, spot, atm, now, vix)
print(f"ATM IV Change 3m: {features.get('atm_iv_change_3m', 0):.2f}%")
print(f"IV Incline Intensity: {features.get('iv_incline_intensity', 0):.2f}")
print(f"IV Spike Count: {features.get('iv_spike_strikes_count', 0)}")
```

### Log Analysis
Look for IV-related patterns in logs:
```bash
grep -i "iv.*change\|iv.*spike" oi_tracker.log
```

## Best Practices

1. **Monitor IV History**: Ensure IV cache is being populated
2. **Check Data Quality**: Verify IV calculations are accurate
3. **Feature Validation**: Confirm features are non-zero when IV changes occur
4. **Model Performance**: Track if RL model performance improves with IV features
5. **A/B Testing**: Compare RL performance with/without IV features

## Troubleshooting

### Features Always Zero
- Check if `handler.option_iv_cache` is populated
- Verify IV calculations are running
- Ensure options have valid IV values

### IV History Not Tracking
- Check if `handler._iv_history_window` is initialized
- Verify IV cache updates are happening
- Check for errors in IV calculation

### Model Not Using Features
- Retrain models after adding features
- Verify features are in `REQUIRED_FEATURE_COLUMNS`
- Check feature extraction in `_get_state()`

