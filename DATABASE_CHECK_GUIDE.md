# Database Check Guide - OptimizedOptionReturnStrategy

**Fixed:** Metadata is now being saved to `paper_trading_metrics` table.

---

## 📊 Where to Check in Database

### 1. **paper_trading_metrics** Table (Main Table for Live Trades)

This table stores all paper trading execution decisions, including strategy metadata.

#### Check All Recent Trades with Strategy Names

```sql
SELECT 
    exchange,
    timestamp,
    signal,
    confidence,
    executed,
    reason,
    quantity_lots,
    pnl,
    metadata->>'strategy_name' as strategy_name,
    metadata->>'predicted_return' as predicted_return,
    metadata->>'horizon' as horizon,
    metadata->>'option_type' as option_type,
    metadata
FROM paper_trading_metrics
WHERE timestamp >= NOW() - INTERVAL '2 hours'
ORDER BY timestamp DESC
LIMIT 50;
```

#### Check Only OptimizedOptionReturnStrategy Trades

```sql
SELECT 
    exchange,
    timestamp,
    signal,
    confidence,
    executed,
    reason,
    quantity_lots,
    pnl,
    metadata->>'predicted_return' as predicted_return,
    metadata->>'horizon' as horizon,
    metadata->>'option_type' as option_type,
    metadata->>'ce_return' as ce_return,
    metadata->>'pe_return' as pe_return,
    metadata
FROM paper_trading_metrics
WHERE timestamp >= NOW() - INTERVAL '1 day'
  AND metadata->>'strategy_name' = 'OptimizedOptionReturn'
ORDER BY timestamp DESC;
```

#### Check Strategy Distribution

```sql
SELECT 
    COALESCE(metadata->>'strategy_name', 'ML_Base') as strategy_name,
    COUNT(*) as trade_count,
    COUNT(CASE WHEN executed THEN 1 END) as executed_count,
    AVG(confidence) as avg_confidence,
    SUM(CASE WHEN executed THEN quantity_lots ELSE 0 END) as total_lots
FROM paper_trading_metrics
WHERE timestamp >= NOW() - INTERVAL '1 day'
GROUP BY metadata->>'strategy_name'
ORDER BY trade_count DESC;
```

#### Check Metadata Content (Debug)

```sql
SELECT 
    exchange,
    timestamp,
    signal,
    metadata
FROM paper_trading_metrics
WHERE timestamp >= NOW() - INTERVAL '1 hour'
  AND metadata IS NOT NULL
ORDER BY timestamp DESC
LIMIT 10;
```

---

### 2. **option_return_backtest_trades** Table (Backtest Results)

This table stores backtest trades (not live, but useful for comparison).

```sql
SELECT 
    exchange,
    timestamp,
    option_type,
    horizon,
    predicted_return,
    actual_return,
    entry_price,
    exit_price,
    net_pnl,
    confidence
FROM option_return_backtest_trades
WHERE timestamp >= NOW() - INTERVAL '7 days'
ORDER BY timestamp DESC
LIMIT 50;
```

---

### 3. **option_chain_snapshots** Table (Price Data)

Check actual option prices for exit verification:

```sql
-- Check price history for a specific option
SELECT 
    timestamp,
    ltp as price,
    oi,
    oi_change,
    volume
FROM option_chain_snapshots
WHERE exchange = 'BSE'
  AND symbol = 'SENSEX2612282600PE'
  AND timestamp >= NOW() - INTERVAL '1 hour'
ORDER BY timestamp DESC;
```

---

## 🔍 What to Look For

### ✅ Strategy is Working If You See:

1. **In `paper_trading_metrics`:**
   - `metadata->>'strategy_name' = 'OptimizedOptionReturn'`
   - `metadata->>'predicted_return'` has values (e.g., 2.5, 12.15)
   - `metadata->>'horizon'` is '15m' or '10m'
   - `metadata->>'option_type'` is 'CE' or 'PE'

2. **Trades Executed:**
   - `executed = true`
   - `quantity_lots > 0`
   - `reason` shows execution reason

### ⚠️ Strategy Not Working If:

1. **All metadata is NULL:**
   - Fix: Restart application (metadata fix is now in code)

2. **No trades with `strategy_name = 'OptimizedOptionReturn'`:**
   - Check: Are we in trading window (13:00-15:30 IST)?
   - Check: Are predictions strong enough (> 0.2% return, ≥ 0.6 confidence)?

3. **Metadata exists but strategy_name is 'ML_Base':**
   - Strategy router didn't select OptimizedOptionReturnStrategy
   - Check: Are option return predictions in signal metadata?

---

## 📋 Quick Check Queries

### Count Trades by Strategy (Last 24 Hours)

```sql
SELECT 
    COALESCE(metadata->>'strategy_name', 'ML_Base') as strategy,
    COUNT(*) as total,
    COUNT(CASE WHEN executed THEN 1 END) as executed
FROM paper_trading_metrics
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY metadata->>'strategy_name'
ORDER BY total DESC;
```

### Check Latest OptimizedOptionReturn Trades

```sql
SELECT 
    timestamp,
    exchange,
    signal,
    confidence,
    executed,
    metadata->>'predicted_return' as predicted_return,
    metadata->>'horizon' as horizon
FROM paper_trading_metrics
WHERE metadata->>'strategy_name' = 'OptimizedOptionReturn'
ORDER BY timestamp DESC
LIMIT 10;
```

### Check if Metadata Column Exists

```sql
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'paper_trading_metrics'
  AND column_name = 'metadata';
```

---

## 🔧 If Metadata is Still NULL

### Step 1: Verify Column Exists

```sql
SELECT column_name 
FROM information_schema.columns 
WHERE table_name = 'paper_trading_metrics' 
  AND column_name = 'metadata';
```

If it doesn't exist, add it:

```sql
ALTER TABLE paper_trading_metrics 
ADD COLUMN IF NOT EXISTS metadata JSONB;
```

### Step 2: Restart Application

The fix is now in the code, but you need to restart the application for it to take effect.

### Step 3: Verify After Restart

After restart, check new trades:

```sql
SELECT 
    timestamp,
    metadata
FROM paper_trading_metrics
WHERE timestamp >= NOW() - INTERVAL '10 minutes'
ORDER BY timestamp DESC
LIMIT 5;
```

---

## 📍 Database Connection Info

From `.env` file:
- **Host:** localhost
- **Port:** 5432
- **Database:** oi_db_live
- **User:** root

---

## Query Records with Horizon Metadata

### Basic Query - Find records with "horizon" key

```sql
SELECT 
    id,
    timestamp,
    exchange,
    signal,
    confidence,
    executed,
    metadata->>'horizon' as horizon,
    metadata->>'strategy_name' as strategy_name,
    metadata->>'predicted_return' as predicted_return
FROM paper_trading_metrics
WHERE metadata ? 'horizon'
ORDER BY timestamp DESC;
```

### Count Records with Horizon

```sql
SELECT 
    COUNT(*) as total_with_horizon,
    COUNT(DISTINCT exchange) as exchanges
FROM paper_trading_metrics
WHERE metadata ? 'horizon';
```

### Group by Horizon Value

```sql
SELECT 
    metadata->>'horizon' as horizon,
    COUNT(*) as count,
    COUNT(CASE WHEN executed THEN 1 END) as executed_count,
    AVG(confidence) as avg_confidence
FROM paper_trading_metrics
WHERE metadata ? 'horizon'
GROUP BY metadata->>'horizon'
ORDER BY count DESC;
```

### Filter by Specific Horizon (e.g., '15m')

```sql
SELECT 
    id,
    timestamp,
    exchange,
    signal,
    confidence,
    executed,
    metadata->>'horizon' as horizon,
    metadata->>'predicted_return' as predicted_return
FROM paper_trading_metrics
WHERE metadata->>'horizon' = '15m'
ORDER BY timestamp DESC;
```

**Note:** The `?` operator checks if a JSONB key exists. Use `metadata->>'horizon'` to extract the value as text.

---

## Summary

**Main Table:** `paper_trading_metrics`  
**Key Column:** `metadata` (JSONB)  
**Key Fields in Metadata:**
- `strategy_name` → Should be 'OptimizedOptionReturn'
- `predicted_return` → Predicted return percentage
- `horizon` → '15m' or '10m'
- `option_type` → 'CE' or 'PE'

**Fixed:** Metadata is now being saved. Restart application to see it in new trades.

**See also:** `scripts/query_paper_trading_with_horizon.sql` for more query examples.
