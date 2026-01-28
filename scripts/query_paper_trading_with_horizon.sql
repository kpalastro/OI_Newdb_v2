-- Query paper_trading_metrics where metadata contains "horizon" key

-- Method 1: Check if key exists (most efficient)
SELECT 
    id,
    timestamp,
    exchange,
    signal,
    confidence,
    executed,
    metadata->>'horizon' as horizon,
    metadata->>'strategy_name' as strategy_name,
    metadata->>'predicted_return' as predicted_return,
    metadata->>'option_type' as option_type
FROM paper_trading_metrics
WHERE metadata ? 'horizon'
ORDER BY timestamp DESC;

-- Method 2: Check if key exists and get count
SELECT 
    COUNT(*) as total_with_horizon,
    COUNT(DISTINCT exchange) as exchanges,
    COUNT(DISTINCT metadata->>'strategy_name') as strategies
FROM paper_trading_metrics
WHERE metadata ? 'horizon';

-- Method 3: Group by horizon value
SELECT 
    metadata->>'horizon' as horizon,
    COUNT(*) as count,
    COUNT(CASE WHEN executed THEN 1 END) as executed_count,
    AVG(confidence) as avg_confidence,
    AVG((metadata->>'predicted_return')::float) as avg_predicted_return
FROM paper_trading_metrics
WHERE metadata ? 'horizon'
GROUP BY metadata->>'horizon'
ORDER BY count DESC;

-- Method 4: Get detailed records with all metadata fields
SELECT 
    id,
    timestamp,
    exchange,
    signal,
    confidence,
    executed,
    reason,
    quantity_lots,
    pnl,
    metadata->>'strategy_name' as strategy_name,
    metadata->>'horizon' as horizon,
    metadata->>'predicted_return' as predicted_return,
    metadata->>'option_type' as option_type,
    metadata->>'ce_return' as ce_return,
    metadata->>'pe_return' as pe_return,
    metadata->>'ce_confidence' as ce_confidence,
    metadata->>'pe_confidence' as pe_confidence,
    metadata->>'turning_point_prob' as turning_point_prob,
    metadata  -- Full metadata JSONB
FROM paper_trading_metrics
WHERE metadata ? 'horizon'
ORDER BY timestamp DESC
LIMIT 50;

-- Method 5: Filter by specific horizon value (e.g., '15m')
SELECT 
    id,
    timestamp,
    exchange,
    signal,
    confidence,
    executed,
    metadata->>'horizon' as horizon,
    metadata->>'predicted_return' as predicted_return,
    metadata->>'option_type' as option_type
FROM paper_trading_metrics
WHERE metadata->>'horizon' = '15m'
ORDER BY timestamp DESC;

-- Method 6: Check for OptimizedOptionReturn strategy with horizon
SELECT 
    id,
    timestamp,
    exchange,
    signal,
    confidence,
    executed,
    metadata->>'horizon' as horizon,
    metadata->>'predicted_return' as predicted_return,
    metadata->>'option_type' as option_type,
    pnl
FROM paper_trading_metrics
WHERE metadata->>'strategy_name' = 'OptimizedOptionReturn'
  AND metadata ? 'horizon'
ORDER BY timestamp DESC;

-- Method 7: Summary statistics by horizon
SELECT 
    metadata->>'horizon' as horizon,
    metadata->>'strategy_name' as strategy,
    COUNT(*) as total_trades,
    COUNT(CASE WHEN executed THEN 1 END) as executed_trades,
    ROUND(AVG(confidence)::numeric, 3) as avg_confidence,
    ROUND(AVG((metadata->>'predicted_return')::float)::numeric, 2) as avg_predicted_return,
    ROUND(SUM(pnl)::numeric, 2) as total_pnl,
    ROUND(AVG(pnl)::numeric, 2) as avg_pnl
FROM paper_trading_metrics
WHERE metadata ? 'horizon'
GROUP BY metadata->>'horizon', metadata->>'strategy_name'
ORDER BY horizon, strategy;
