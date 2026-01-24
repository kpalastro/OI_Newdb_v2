# Backtest Trades Table Guide

## Overview

The `option_return_backtest_trades` table stores detailed records of every trade executed during option return model backtesting. This table is similar to `paper_trading_metrics` but includes additional fields for model predictions, feature payloads, and model metadata.

## Table Schema

```sql
CREATE TABLE option_return_backtest_trades (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,          -- Entry timestamp
    exit_timestamp TIMESTAMP,              -- Exit timestamp (entry + horizon)
    exchange TEXT NOT NULL,
    option_type TEXT NOT NULL,              -- 'CE' or 'PE'
    option_symbol TEXT,                    -- Option symbol (e.g., NIFTY2612025550CE)
    horizon TEXT NOT NULL,                  -- '3m', '5m', '10m', '15m'
    predicted_return DOUBLE PRECISION,     -- Predicted return for chosen option
    actual_return DOUBLE PRECISION,        -- Actual return achieved
    predicted_ce_return DOUBLE PRECISION,  -- Predicted CE return
    predicted_pe_return DOUBLE PRECISION,  -- Predicted PE return
    ce_confidence DOUBLE PRECISION,        -- CE prediction confidence
    pe_confidence DOUBLE PRECISION,        -- PE prediction confidence
    turning_point_prob DOUBLE PRECISION,   -- Turning point probability
    recommendation TEXT,                   -- Model recommendation
    entry_price DOUBLE PRECISION,          -- Option entry price
    exit_price DOUBLE PRECISION,          -- Option exit price
    quantity_lots INTEGER,                 -- Number of lots traded
    gross_pnl DOUBLE PRECISION,            -- Gross PnL
    net_pnl DOUBLE PRECISION,              -- Net PnL (after costs)
    transaction_cost DOUBLE PRECISION,    -- Transaction cost
    confidence DOUBLE PRECISION,           -- Overall confidence
    feature_payload JSONB,                 -- Features used for prediction
    model_metadata JSONB,                  -- Additional model metadata
    created_at TIMESTAMP DEFAULT NOW()
);
```

## Key Features

### 1. **Entry and Exit Timestamps**
- `timestamp`: Entry time (when trade was initiated)
- `exit_timestamp`: Exit time (entry time + horizon duration)
  - For 3m horizon: exit = entry + 3 minutes
  - For 5m horizon: exit = entry + 5 minutes
  - For 10m horizon: exit = entry + 10 minutes
  - For 15m horizon: exit = entry + 15 minutes

### 2. **Option Symbol**
- `option_symbol`: Full option symbol (e.g., `NIFTY2612025550CE`)
  - Format: `{UNDERLYING}{DDMMMYY}{STRIKE}{CE/PE}`
  - Example: `NIFTY2612025550CE` = NIFTY, 26 Jan 2026, Strike 25550, Call
  - Useful for:
    - Tracking specific contracts
    - Cross-referencing with option chain data
    - Analyzing performance by strike price

### 3. **Feature Payload (JSONB)**
Stores the first 50 features used for prediction. This allows you to:
- Analyze which features led to good/bad trades
- Correlate feature values with outcomes
- Debug model predictions

Example:
```json
{
  "itm_oi_ce_pct_change_3m_wavg": 2.5,
  "itm_oi_pe_pct_change_3m_wavg": -1.2,
  "pcr": 1.15,
  "vix": 18.5,
  ...
}
```

### 4. **Model Metadata (JSONB)**
Stores model predictions and configuration:
```json
{
  "horizon": "10m",
  "predicted_ce_return": 5.2,
  "predicted_pe_return": 3.1,
  "ce_confidence": 0.85,
  "pe_confidence": 0.72,
  "turning_point_prob": 0.15,
  "recommendation": "BUY_CE",
  "backtest_config": {
    "min_confidence": 0.6,
    "min_predicted_return": 0.2,
    "use_turning_point_filter": true,
    "turning_point_threshold": 0.7
  }
}
```

## Usage

### Running Backtests

The backtest script automatically records trades to the database:

```bash
python3 backtesting/option_return_backtest.py \
    --exchange NSE \
    --start 2026-01-19 \
    --end 2026-01-19 \
    --min-confidence 0.6 \
    --min-return 0.2
```

All trades will be automatically saved to `option_return_backtest_trades` table.

### Querying Backtest Trades

#### 1. Daily Summary
```sql
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as total_trades,
    SUM(net_pnl) as total_pnl,
    AVG(net_pnl) as avg_pnl,
    COUNT(CASE WHEN net_pnl > 0 THEN 1 END) as winning_trades,
    ROUND(100.0 * COUNT(CASE WHEN net_pnl > 0 THEN 1 END) / COUNT(*), 2) as win_rate
FROM option_return_backtest_trades
WHERE exchange = 'NSE'
GROUP BY DATE(timestamp)
ORDER BY date DESC;
```

#### 2. Performance by Option Type
```sql
SELECT 
    option_type,
    COUNT(*) as trades,
    SUM(net_pnl) as total_pnl,
    AVG(net_pnl) as avg_pnl,
    COUNT(CASE WHEN net_pnl > 0 THEN 1 END) as wins
FROM option_return_backtest_trades
WHERE exchange = 'NSE'
  AND DATE(timestamp) >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY option_type;
```

#### 3. Performance by Horizon
```sql
SELECT 
    horizon,
    COUNT(*) as trades,
    SUM(net_pnl) as total_pnl,
    AVG(net_pnl) as avg_pnl
FROM option_return_backtest_trades
WHERE exchange = 'NSE'
GROUP BY horizon
ORDER BY horizon;
```

#### 4. Prediction Accuracy Analysis
```sql
SELECT 
    CASE 
        WHEN predicted_return > 0 AND actual_return > 0 THEN 'Correct Long'
        WHEN predicted_return < 0 AND actual_return < 0 THEN 'Correct Short'
        WHEN predicted_return > 0 AND actual_return < 0 THEN 'Wrong Long'
        WHEN predicted_return < 0 AND actual_return > 0 THEN 'Wrong Short'
    END as prediction_direction,
    COUNT(*) as trades,
    AVG(ABS(predicted_return - actual_return)) as avg_error,
    AVG(net_pnl) as avg_pnl
FROM option_return_backtest_trades
WHERE exchange = 'NSE'
GROUP BY prediction_direction;
```

#### 5. Feature Analysis
```sql
SELECT 
    timestamp,
    option_type,
    net_pnl,
    feature_payload->>'itm_oi_ce_pct_change_3m_wavg' as itm_ce_delta,
    feature_payload->>'itm_oi_pe_pct_change_3m_wavg' as itm_pe_delta,
    model_metadata->>'turning_point_prob' as tp_prob
FROM option_return_backtest_trades
WHERE exchange = 'NSE'
  AND feature_payload IS NOT NULL
ORDER BY timestamp DESC
LIMIT 20;
```

#### 6. Model Performance by Confidence
```sql
SELECT 
    CASE 
        WHEN confidence < 0.5 THEN 'Low'
        WHEN confidence < 0.7 THEN 'Medium'
        WHEN confidence < 0.9 THEN 'High'
        ELSE 'Very High'
    END as confidence_level,
    COUNT(*) as trades,
    AVG(net_pnl) as avg_pnl,
    COUNT(CASE WHEN net_pnl > 0 THEN 1 END) * 100.0 / COUNT(*) as win_rate
FROM option_return_backtest_trades
WHERE exchange = 'NSE'
GROUP BY confidence_level
ORDER BY confidence_level;
```

## Comparison with paper_trading_metrics

| Feature | paper_trading_metrics | option_return_backtest_trades |
|---------|----------------------|------------------------------|
| Purpose | Real-time paper trading | Historical backtesting |
| Timestamp | Real-time | Historical |
| Option Type | No | Yes (CE/PE) |
| Horizon | No | Yes (3m/5m/10m/15m) |
| Predicted Returns | No | Yes (CE & PE) |
| Actual Returns | No | Yes |
| Feature Payload | No | Yes (first 50 features) |
| Model Metadata | Limited | Full (predictions, config) |
| Entry/Exit Prices | No | Yes |
| Transaction Costs | No | Yes |

## Use Cases

### 1. **Model Debugging**
Analyze trades where predictions were wrong:
```sql
SELECT *
FROM option_return_backtest_trades
WHERE exchange = 'NSE'
  AND ABS(predicted_return - actual_return) > 20
  AND feature_payload IS NOT NULL
ORDER BY ABS(predicted_return - actual_return) DESC
LIMIT 10;
```

### 2. **Feature Importance Analysis**
Find features that correlate with good trades:
```sql
SELECT 
    feature_payload->>'itm_oi_ce_pct_change_3m_wavg' as itm_ce,
    AVG(net_pnl) as avg_pnl,
    COUNT(*) as trades
FROM option_return_backtest_trades
WHERE exchange = 'NSE'
  AND feature_payload IS NOT NULL
GROUP BY itm_ce
ORDER BY avg_pnl DESC;
```

### 3. **Horizon Optimization**
Determine which horizon performs best:
```sql
SELECT 
    horizon,
    COUNT(*) as trades,
    SUM(net_pnl) as total_pnl,
    AVG(net_pnl) as avg_pnl,
    COUNT(CASE WHEN net_pnl > 0 THEN 1 END) * 100.0 / COUNT(*) as win_rate
FROM option_return_backtest_trades
WHERE exchange = 'NSE'
GROUP BY horizon
ORDER BY total_pnl DESC;
```

### 4. **Confidence Threshold Tuning**
Find optimal confidence threshold:
```sql
SELECT 
    CASE 
        WHEN confidence >= 0.9 THEN '>= 0.9'
        WHEN confidence >= 0.8 THEN '0.8-0.9'
        WHEN confidence >= 0.7 THEN '0.7-0.8'
        WHEN confidence >= 0.6 THEN '0.6-0.7'
        ELSE '< 0.6'
    END as conf_range,
    COUNT(*) as trades,
    AVG(net_pnl) as avg_pnl,
    SUM(net_pnl) as total_pnl
FROM option_return_backtest_trades
WHERE exchange = 'NSE'
GROUP BY conf_range
ORDER BY conf_range DESC;
```

## Indexes

The table has the following indexes for fast queries:
- `idx_backtest_trades_timestamp` - On timestamp
- `idx_backtest_trades_exchange` - On exchange
- `idx_backtest_trades_option_type` - On option_type
- `idx_backtest_trades_horizon` - On horizon
- `idx_backtest_trades_feature_payload` - GIN index on feature_payload (JSONB)
- `idx_backtest_trades_model_metadata` - GIN index on model_metadata (JSONB)
- `idx_backtest_trades_exit_timestamp` - On exit_timestamp
- `idx_backtest_trades_option_symbol` - On option_symbol

## Maintenance

### Cleanup Old Data
```sql
-- Delete trades older than 90 days
DELETE FROM option_return_backtest_trades
WHERE created_at < NOW() - INTERVAL '90 days';
```

### Export to CSV
```sql
COPY (
    SELECT *
    FROM option_return_backtest_trades
    WHERE exchange = 'NSE'
      AND DATE(timestamp) >= '2026-01-01'
) TO '/tmp/backtest_trades.csv' WITH CSV HEADER;
```

## Python API

### Record a Trade (automatically done by backtest script)
```python
import database_new as db
from datetime import datetime

db.record_option_return_backtest_trade(
    exchange='NSE',
    timestamp=datetime.now(),
    option_type='CE',
    horizon='10m',
    predicted_return=5.2,
    actual_return=6.1,
    predicted_ce_return=5.2,
    predicted_pe_return=3.1,
    ce_confidence=0.85,
    pe_confidence=0.72,
    turning_point_prob=0.15,
    recommendation='BUY_CE',
    entry_price=100.0,
    exit_price=106.1,
    quantity_lots=1,
    gross_pnl=305.0,
    net_pnl=300.0,
    transaction_cost=5.0,
    confidence=0.85,
    feature_payload={'itm_oi_ce_pct_change_3m_wavg': 2.5, ...},
    model_metadata={'horizon': '10m', ...}
)
```

### Query Trades
```python
import database_new as db
import pandas as pd

conn = db.get_db_connection()
df = pd.read_sql_query("""
    SELECT *
    FROM option_return_backtest_trades
    WHERE exchange = 'NSE'
      AND DATE(timestamp) = '2026-01-19'
""", conn)
db.release_db_connection(conn)
```

## Notes

1. **Feature Payload**: Only the first 50 features are stored to keep the table size manageable. All features are available during backtesting.

2. **Automatic Recording**: Trades are automatically recorded when running the backtest script. No manual intervention needed.

3. **Performance**: The table is indexed for fast queries. Use date ranges in WHERE clauses for best performance.

4. **Storage**: JSONB columns are efficient for storing structured data and support JSON queries.

5. **Comparison**: Use this table alongside `paper_trading_metrics` to compare backtest performance with real-time paper trading results.
