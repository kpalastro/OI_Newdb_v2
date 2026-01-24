# Option Return Model: Train/Test Split Guide

## Overview

The `train_option_return_models.py` script now supports explicit date ranges, allowing you to:
1. **Train on known historical data** (e.g., Jan 1-20)
2. **Test on untrained data** (e.g., Jan 21-23) for proper out-of-sample validation

## Usage Examples

### Example 1: Train on Specific Date Range

```bash
# Train on Jan 1-20, 2026
python3 train_option_return_models.py \
    --exchange NSE \
    --start-date 2026-01-01 \
    --end-date 2026-01-20
```

### Example 2: Train on Last N Days (Default Behavior)

```bash
# Train on last 90 days (default)
python3 train_option_return_models.py --exchange NSE

# Train on last 60 days
python3 train_option_return_models.py --exchange NSE --days 60
```

### Example 3: Train/Test Split Workflow

```bash
# Step 1: Train on training period (e.g., Jan 1-20)
python3 train_option_return_models.py \
    --exchange NSE \
    --start-date 2026-01-01 \
    --end-date 2026-01-20

# Step 2: Backtest on test period (e.g., Jan 21-23)
python3 backtesting/option_return_backtest.py \
    --exchange NSE \
    --start 2026-01-21 \
    --end 2026-01-23 \
    --min-confidence 0.4 \
    --min-return 0.5
```

### Example 4: Walk-Forward Validation

```bash
# Week 1: Train on days 1-10, test on days 11-15
python3 train_option_return_models.py --exchange NSE --start-date 2026-01-01 --end-date 2026-01-10
python3 backtesting/option_return_backtest.py --exchange NSE --start 2026-01-11 --end 2026-01-15

# Week 2: Train on days 1-15, test on days 16-20
python3 train_option_return_models.py --exchange NSE --start-date 2026-01-01 --end-date 2026-01-15
python3 backtesting/option_return_backtest.py --exchange NSE --start 2026-01-16 --end 2026-01-20

# Week 3: Train on days 1-20, test on days 21-25
python3 train_option_return_models.py --exchange NSE --start-date 2026-01-01 --end-date 2026-01-20
python3 backtesting/option_return_backtest.py --exchange NSE --start 2026-01-21 --end 2026-01-25
```

## Parameters

### Date Parameters

- `--start-date` / `-s`: Start date in `YYYY-MM-DD` format
  - If provided, overrides `--days` parameter
  - Must be before `--end-date`
  
- `--end-date` / `-e`: End date in `YYYY-MM-DD` format
  - Defaults to today if not provided
  - Must be after `--start-date`

- `--days` / `-d`: Number of days of historical data
  - Only used if `--start-date` is not provided
  - Default: 90 days
  - Calculates: `start_date = end_date - days`

### Other Parameters

- `--exchange` / `-e`: Exchange name (default: NSE)
- `--horizons`: Time horizons to train (default: 3m 5m 10m 15m)
- `--min-price`: Minimum option price for quality filter (default: 1.0)

## Training Summary

The training summary now includes the date range:

```json
{
  "exchange": "NSE",
  "training_date": "2026-01-23T23:57:00.000000",
  "training_start_date": "2026-01-01T00:00:00",
  "training_end_date": "2026-01-20T00:00:00",
  "training_duration_days": 19,
  "n_samples": 109,
  "model_metrics": { ... }
}
```

## Best Practices

### 1. **Proper Train/Test Split**

**Recommended Split:**
- **Training**: 70-80% of available data
- **Testing**: 20-30% of available data
- **Gap**: 0 days (use consecutive dates)

**Example:**
```bash
# If you have data from Jan 1-30:
# Train: Jan 1-23 (23 days, ~77%)
# Test: Jan 24-30 (7 days, ~23%)
```

### 2. **Avoid Data Leakage**

- **Never test on data before training end date**
- Always ensure: `test_start_date > training_end_date`
- Use at least 1 day gap if possible (to avoid same-day issues)

### 3. **Walk-Forward Validation**

For robust validation, use walk-forward approach:
- Train on expanding window
- Test on next period
- Retrain after each test period
- Track performance over time

### 4. **Check Training Summary**

After training, check `models/option_returns/NSE/training_summary.json`:
- Verify `training_start_date` and `training_end_date`
- Check `n_samples` (should be sufficient, > 100)
- Review `model_metrics` for each horizon

## Example: Complete Train/Test Workflow

```bash
# 1. Train on Jan 1-20
python3 train_option_return_models.py \
    --exchange NSE \
    --start-date 2026-01-01 \
    --end-date 2026-01-20

# 2. Check training summary
cat models/option_returns/NSE/training_summary.json | python3 -m json.tool

# 3. Backtest on Jan 21-23 (untrained data)
python3 backtesting/option_return_backtest.py \
    --exchange NSE \
    --start 2026-01-21 \
    --end 2026-01-23 \
    --min-confidence 0.4 \
    --min-return 0.5 \
    --output test_results.json

# 4. Compare results
# Training metrics: models/option_returns/NSE/training_summary.json
# Test metrics: test_results.json
```

## Troubleshooting

### Issue: "start_date must be before end_date"
**Solution**: Ensure `--start-date` is before `--end-date`

### Issue: "No data loaded"
**Solution**: 
- Check if data exists for the date range in database
- Verify dates are in correct format (YYYY-MM-DD)
- Check if option_chain_snapshots table has data

### Issue: "Insufficient samples"
**Solution**:
- Use longer date range (more days)
- Lower `--min-price` threshold
- Check data quality in database

## Notes

- **Date Format**: Always use `YYYY-MM-DD` format
- **Timezone**: Dates are interpreted in IST (Indian Standard Time)
- **Inclusive**: Both start_date and end_date are inclusive
- **Overlap**: Training and testing can overlap if you want (not recommended for validation)
