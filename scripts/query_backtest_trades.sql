-- Query backtest trades from option_return_backtest_trades table
-- Similar to paper_trading_metrics queries

-- 1. Daily PnL summary
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as total_trades,
    SUM(net_pnl) as total_pnl,
    AVG(net_pnl) as avg_pnl,
    COUNT(CASE WHEN net_pnl > 0 THEN 1 END) as winning_trades,
    COUNT(CASE WHEN net_pnl < 0 THEN 1 END) as losing_trades,
    ROUND(100.0 * COUNT(CASE WHEN net_pnl > 0 THEN 1 END) / COUNT(*), 2) as win_rate
FROM option_return_backtest_trades
WHERE exchange = 'NSE'
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- 2. PnL by option type
SELECT 
    option_type,
    COUNT(*) as trades,
    SUM(net_pnl) as total_pnl,
    AVG(net_pnl) as avg_pnl,
    COUNT(CASE WHEN net_pnl > 0 THEN 1 END) as wins,
    COUNT(CASE WHEN net_pnl < 0 THEN 1 END) as losses
FROM option_return_backtest_trades
WHERE exchange = 'NSE'
  AND DATE(timestamp) >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY option_type;

-- 3. PnL by horizon
SELECT 
    horizon,
    COUNT(*) as trades,
    SUM(net_pnl) as total_pnl,
    AVG(net_pnl) as avg_pnl,
    COUNT(CASE WHEN net_pnl > 0 THEN 1 END) as wins,
    COUNT(CASE WHEN net_pnl < 0 THEN 1 END) as losses
FROM option_return_backtest_trades
WHERE exchange = 'NSE'
  AND DATE(timestamp) >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY horizon
ORDER BY horizon;

-- 4. Prediction accuracy analysis
SELECT 
    CASE 
        WHEN predicted_return > 0 AND actual_return > 0 THEN 'Correct Long'
        WHEN predicted_return < 0 AND actual_return < 0 THEN 'Correct Short'
        WHEN predicted_return > 0 AND actual_return < 0 THEN 'Wrong Long'
        WHEN predicted_return < 0 AND actual_return > 0 THEN 'Wrong Short'
        ELSE 'Neutral'
    END as prediction_direction,
    COUNT(*) as trades,
    AVG(ABS(predicted_return - actual_return)) as avg_error,
    AVG(net_pnl) as avg_pnl
FROM option_return_backtest_trades
WHERE exchange = 'NSE'
  AND DATE(timestamp) >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY prediction_direction;

-- 5. Trades with high confidence but poor results
SELECT 
    timestamp,
    option_type,
    horizon,
    predicted_return,
    actual_return,
    confidence,
    net_pnl
FROM option_return_backtest_trades
WHERE exchange = 'NSE'
  AND confidence > 0.8
  AND net_pnl < -1000
ORDER BY timestamp DESC
LIMIT 20;

-- 6. Feature payload analysis (example: ITM features)
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
  AND DATE(timestamp) >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY timestamp DESC
LIMIT 20;

-- 7. Model metadata analysis
SELECT 
    horizon,
    AVG((model_metadata->>'predicted_ce_return')::float) as avg_pred_ce,
    AVG((model_metadata->>'predicted_pe_return')::float) as avg_pred_pe,
    AVG((model_metadata->>'ce_confidence')::float) as avg_ce_conf,
    AVG((model_metadata->>'pe_confidence')::float) as avg_pe_conf,
    AVG((model_metadata->>'turning_point_prob')::float) as avg_tp_prob
FROM option_return_backtest_trades
WHERE exchange = 'NSE'
  AND model_metadata IS NOT NULL
  AND DATE(timestamp) >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY horizon;
