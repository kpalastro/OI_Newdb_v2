# Dashboard: Backtest Results & Model Performance

## Overview

The monitoring dashboard now includes two new sections displaying:
1. **Backtest Results** - Performance metrics from option return backtests
2. **Model Performance** - Training metrics from option return models

## Features Added

### 1. Backtest Results Section

Displays comprehensive backtest performance metrics:

#### Summary Metrics:
- **Total Trades**: Number of trades executed
- **Total PnL**: Cumulative profit/loss (color-coded: green for profit, red for loss)
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Ratio of average win to average loss

#### Performance by Horizon:
- Breakdown of performance for each time horizon (3m, 5m, 10m, 15m)
- Shows trades, total PnL, average PnL, and win rate for each horizon

#### Performance by Option Type:
- Separate metrics for CE (Call) and PE (Put) options

### 2. Model Performance Section

Displays model training metrics:

#### Training Information:
- **Training Period**: Start and end dates of training data
- **Duration**: Number of days used for training
- **Samples**: Number of training samples

#### Model Metrics:
- **R²**: Coefficient of determination (model fit quality)
- **MAE**: Mean Absolute Error (prediction accuracy)
- **Direction Accuracy**: Percentage of correct direction predictions (color-coded: green ≥60%, yellow ≥50%, red <50%)
- **Return on Long**: Average return when model predicts long position

## API Endpoints

### GET `/api/backtest-results`

Fetches backtest results from the `option_return_backtest_trades` table.

**Parameters:**
- `exchange` (default: "NSE"): Exchange name
- `days` (default: 7): Number of days to look back

**Response:**
```json
{
  "success": true,
  "exchange": "NSE",
  "days": 7,
  "summary": {
    "total_trades": 80,
    "total_pnl": 134133.85,
    "avg_pnl": 1676.67,
    "win_rate": 70.0,
    "winning_trades": 56,
    "losing_trades": 24,
    "avg_win": 2542.50,
    "avg_loss": -293.39,
    "profit_factor": 8.67,
    "first_trade": "2026-01-19T09:15:00",
    "last_trade": "2026-01-19T15:03:00"
  },
  "by_horizon": [
    {
      "horizon": "3m",
      "trades": 12,
      "total_pnl": 981.21,
      "avg_pnl": 81.77,
      "win_rate": 91.7,
      "wins": 11,
      "losses": 1
    },
    ...
  ],
  "by_option_type": {
    "CE": {
      "trades": 52,
      "total_pnl": 131655.15,
      "avg_pnl": 2533.75
    },
    "PE": {
      "trades": 28,
      "total_pnl": 2478.70,
      "avg_pnl": 88.52
    }
  }
}
```

### GET `/api/model-performance`

Fetches model performance metrics from training summary files.

**Parameters:**
- `exchange` (default: "NSE"): Exchange name

**Response:**
```json
{
  "success": true,
  "exchange": "NSE",
  "training_date": "2026-01-24T01:19:44.946336",
  "training_start_date": "2026-01-01T00:00:00",
  "training_end_date": "2026-01-20T00:00:00",
  "training_duration_days": 19,
  "n_samples": 1164,
  "models": {
    "ce_3m": {
      "r2": 0.1537,
      "mae": 3.41,
      "direction_accuracy": 71.25,
      "return_on_long": 2.44
    },
    "pe_3m": {
      "r2": 0.1336,
      "mae": 2.72,
      "direction_accuracy": 65.52,
      "return_on_long": 1.48
    },
    ...
  }
}
```

## Dashboard Display

### Visual Features:

1. **Color Coding**:
   - **Green** (#4ade80): Positive PnL, high accuracy (≥60%)
   - **Yellow** (#fbbf24): Medium accuracy (≥50%)
   - **Red** (#f87171): Negative PnL, low accuracy (<50%)

2. **Auto-Refresh**:
   - Backtest results refresh every 30 seconds
   - Model performance refreshes every 30 seconds
   - Page auto-refreshes every 5 seconds

3. **Responsive Layout**:
   - Grid layout for summary metrics
   - Tables for detailed breakdowns
   - Mobile-friendly design

## Usage

### Accessing the Dashboard:

1. Start the dashboard server (usually runs automatically with the main application)
2. Navigate to `http://localhost:8000/` in your browser
3. The backtest results and model performance sections will load automatically

### Direct API Access:

```bash
# Get backtest results for last 7 days
curl http://localhost:8000/api/backtest-results?exchange=NSE&days=7

# Get model performance
curl http://localhost:8000/api/model-performance?exchange=NSE
```

## Data Sources

### Backtest Results:
- **Table**: `option_return_backtest_trades`
- **Fields Used**: 
  - `timestamp`, `exchange`, `option_type`, `horizon`
  - `net_pnl`, `entry_price`, `exit_price`
  - `predicted_return`, `actual_return`

### Model Performance:
- **File**: `models/option_returns/{exchange}/training_summary.json`
- **Fields Used**:
  - `training_start_date`, `training_end_date`, `training_duration_days`
  - `n_samples`
  - `model_metrics` (R², MAE, direction_accuracy, return_on_long)

## Notes

1. **Data Availability**: 
   - Backtest results require at least one backtest run
   - Model performance requires trained models

2. **Performance**:
   - Queries are optimized with indexes on `timestamp` and `exchange`
   - Results are cached in browser for 30 seconds

3. **Error Handling**:
   - Graceful error messages if data is unavailable
   - API returns `success: false` with error message on failure

4. **Future Enhancements**:
   - Add date range selector
   - Add charts/visualizations
   - Add comparison between different backtest runs
   - Add model performance trends over time
