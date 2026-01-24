# Option Return Backtest Guide

## Overview

This guide explains how to backtest the option return prediction models to evaluate their performance.

## Prerequisites

1. **Trained Models**: Ensure option return models are trained first:
   ```bash
   python3 train_option_return_models.py --exchange NSE --days 60
   ```

2. **Feature Columns**: The training script now saves `feature_columns.pkl` which is required for predictions.

## Running Backtest

### Basic Usage

```bash
python3 backtesting/option_return_backtest.py \
    --exchange NSE \
    --start 2026-01-19 \
    --end 2026-01-23 \
    --min-confidence 0.4 \
    --min-return 0.5
```

### Parameters

- `--exchange` / `-e`: Exchange name (default: NSE)
- `--start`: Start date in YYYY-MM-DD format
- `--end`: End date in YYYY-MM-DD format
- `--min-confidence`: Minimum confidence threshold (0.0-1.0, default: 0.5)
- `--min-return`: Minimum predicted return % to take trade (default: 0.5)
- `--output` / `-o`: Optional JSON file to save results

### Example: Backtest Last 5 Days

```bash
python3 backtesting/option_return_backtest.py \
    --exchange NSE \
    --start $(date -v-5d +%Y-%m-%d) \
    --end $(date +%Y-%m-%d) \
    --min-confidence 0.3 \
    --min-return 0.3 \
    --output backtest_results.json
```

## Understanding Results

The backtest outputs:

1. **Overall Metrics**:
   - Total Trades
   - Total PnL
   - Win Rate
   - Average Win/Loss
   - Profit Factor
   - Direction Accuracy (how often predicted direction matches actual)
   - Mean Absolute Error (prediction error)

2. **By Option Type**:
   - CE vs PE trade counts and PnL

3. **By Horizon**:
   - Performance breakdown for 3m, 5m, 10m, 15m predictions

4. **Sample Trades**: First 10 trades with details

## Troubleshooting

### No Trades Generated

If no trades are generated, check:

1. **Feature Columns**: Ensure `models/option_returns/NSE/feature_columns.pkl` exists
   - If missing, retrain models: `python3 train_option_return_models.py --exchange NSE --days 60`

2. **Lower Thresholds**: Try lower confidence/return thresholds:
   ```bash
   --min-confidence 0.1 --min-return 0.1
   ```

3. **Data Availability**: Check if option price data exists for the date range:
   ```sql
   SELECT COUNT(*) FROM option_chain_snapshots 
   WHERE timestamp >= '2026-01-19' AND timestamp <= '2026-01-23' AND exchange = 'NSE';
   ```

4. **Check Logs**: Look for diagnostic messages about:
   - Successful predictions
   - Filtered trades (turning point, low confidence, low return, missing price)

### Feature Mismatch Errors

If you see "number of features" errors:

1. Retrain models to regenerate feature columns
2. Ensure you're using the same feature engineering pipeline

## Integration with Main Backtest

The option return models are also integrated into the main backtesting engine (`backtesting/engine.py`). They automatically enhance signals in `MLSignalGenerator` when models are available.

## Next Steps

1. **Analyze Results**: Review which horizons and option types perform best
2. **Tune Thresholds**: Adjust `min_confidence` and `min_return` based on results
3. **Paper Trading**: Test in paper trading mode before live trading
4. **Retrain**: Periodically retrain with more recent data

## Example Output

```
================================================================================
OPTION RETURN BACKTEST RESULTS
================================================================================

Config:
  exchange: NSE
  start_date: 2026-01-19
  end_date: 2026-01-23
  min_confidence: 0.4
  min_predicted_return: 0.5

================================================================================
OVERALL METRICS
================================================================================
Total Trades: 45
Total PnL: ₹12,450.00
Win Rate: 62.2%
Average Win: ₹850.00
Average Loss: ₹-420.00
Profit Factor: 2.02
Direction Accuracy: 68.9%
Mean Absolute Error: 2.35%

================================================================================
BY OPTION TYPE
================================================================================
CE Trades: 28, PnL: ₹8,200.00
PE Trades: 17, PnL: ₹4,250.00

================================================================================
BY HORIZON
================================================================================
3m:
  Trades: 15
  PnL: ₹3,500.00
  Win Rate: 60.0%
5m:
  Trades: 18
  PnL: ₹5,200.00
  Win Rate: 66.7%
10m:
  Trades: 10
  PnL: ₹2,800.00
  Win Rate: 70.0%
15m:
  Trades: 2
  PnL: ₹950.00
  Win Rate: 100.0%
```
