-- Sample queries for the nse_multi_expiry_analytics view

-- 1. Get latest data with sentiment indicators for a specific exchange
SELECT 
    timestamp,
    exchange,
    base_strike,
    oi_change_diff_put_call,
    sentiment_score_oi_change,
    sentiment_label,
    trend_direction,
    is_turning_point,
    prediction_signal
FROM nse_multi_expiry_analytics
WHERE exchange = 'NSE'
ORDER BY timestamp DESC
LIMIT 100;

-- 2. Find all turning points in the last day
SELECT 
    timestamp,
    exchange,
    base_strike,
    oi_change_diff_put_call,
    prev_oi_change_diff,
    trend_direction,
    sentiment_label,
    prediction_signal
FROM nse_multi_expiry_analytics
WHERE is_turning_point = TRUE
  AND timestamp >= NOW() - INTERVAL '1 day'
ORDER BY timestamp DESC;

-- 3. Get sentiment trend over time (hourly aggregation)
SELECT 
    DATE_TRUNC('hour', timestamp) AS hour,
    exchange,
    AVG(sentiment_score_oi_change) AS avg_sentiment_score,
    AVG(oi_change_diff_put_call) AS avg_oi_change_diff,
    COUNT(*) FILTER (WHERE sentiment_label = 'STRONG_BULLISH') AS strong_bullish_count,
    COUNT(*) FILTER (WHERE sentiment_label = 'BULLISH') AS bullish_count,
    COUNT(*) FILTER (WHERE sentiment_label = 'NEUTRAL') AS neutral_count,
    COUNT(*) FILTER (WHERE sentiment_label = 'BEARISH') AS bearish_count,
    COUNT(*) FILTER (WHERE sentiment_label = 'STRONG_BEARISH') AS strong_bearish_count,
    COUNT(*) FILTER (WHERE is_turning_point = TRUE) AS turning_points
FROM nse_multi_expiry_analytics
WHERE timestamp >= NOW() - INTERVAL '1 day'
GROUP BY DATE_TRUNC('hour', timestamp), exchange
ORDER BY hour DESC;

-- 4. Get prediction signals for next day analysis
SELECT 
    timestamp,
    exchange,
    base_strike,
    sentiment_score_oi_change,
    sentiment_label,
    prediction_signal,
    pc_oi_ratio,
    iv_skew,
    is_volume_spike
FROM nse_multi_expiry_analytics
WHERE prediction_signal != 'NEUTRAL_PREDICT'
  AND timestamp >= NOW() - INTERVAL '1 day'
ORDER BY timestamp DESC;

-- 5. Compare current sentiment with moving averages (trend analysis)
SELECT 
    timestamp,
    exchange,
    oi_change_diff_put_call,
    oi_change_diff_ma5,
    oi_change_diff_ma15,
    oi_change_diff_ma30,
    CASE 
        WHEN oi_change_diff_put_call > oi_change_diff_ma30 THEN 'ABOVE_30MA'
        WHEN oi_change_diff_put_call < oi_change_diff_ma30 THEN 'BELOW_30MA'
        ELSE 'AT_30MA'
    END AS ma_position,
    sentiment_label
FROM nse_multi_expiry_analytics
WHERE timestamp >= NOW() - INTERVAL '1 day'
  AND oi_change_diff_ma30 IS NOT NULL
ORDER BY timestamp DESC
LIMIT 200;

-- 6. Volume spike analysis with sentiment
SELECT 
    timestamp,
    exchange,
    total_volume_all,
    volume_ma5,
    is_volume_spike,
    sentiment_label,
    trend_direction,
    prediction_signal
FROM nse_multi_expiry_analytics
WHERE is_volume_spike = TRUE
  AND timestamp >= NOW() - INTERVAL '1 day'
ORDER BY timestamp DESC;

-- 7. IV Skew analysis (bearish/bullish indicator)
SELECT 
    timestamp,
    exchange,
    iv_skew,
    iv_diff_put_call,
    sentiment_score_oi_change,
    sentiment_label,
    CASE 
        WHEN iv_skew > 1.2 THEN 'HIGH_BEARISH_SKEW'
        WHEN iv_skew > 1.1 THEN 'MODERATE_BEARISH_SKEW'
        WHEN iv_skew BETWEEN 0.9 AND 1.1 THEN 'NEUTRAL_SKEW'
        WHEN iv_skew < 0.9 THEN 'BULLISH_SKEW'
        ELSE 'UNKNOWN'
    END AS iv_skew_category
FROM nse_multi_expiry_analytics
WHERE iv_skew IS NOT NULL
  AND timestamp >= NOW() - INTERVAL '1 day'
ORDER BY timestamp DESC
LIMIT 200;

-- 8. End of day summary for prediction
SELECT 
    DATE(timestamp) AS trade_date,
    exchange,
    base_strike,
    MAX(sentiment_score_oi_change) AS max_sentiment_score,
    MIN(sentiment_score_oi_change) AS min_sentiment_score,
    AVG(sentiment_score_oi_change) AS avg_sentiment_score,
    MAX(oi_change_diff_put_call) AS max_oi_change_diff,
    MIN(oi_change_diff_put_call) AS min_oi_change_diff,
    AVG(oi_change_diff_put_call) AS avg_oi_change_diff,
    MODE() WITHIN GROUP (ORDER BY sentiment_label) AS most_common_sentiment,
    MODE() WITHIN GROUP (ORDER BY prediction_signal) AS most_common_prediction,
    COUNT(*) FILTER (WHERE is_turning_point = TRUE) AS turning_points_count
FROM nse_multi_expiry_analytics
WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY DATE(timestamp), exchange, base_strike
ORDER BY trade_date DESC, exchange;
