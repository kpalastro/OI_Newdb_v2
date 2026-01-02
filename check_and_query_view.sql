-- Step 1: Check if the view exists
SELECT 
    schemaname,
    viewname,
    definition
FROM pg_views 
WHERE viewname = 'nse_multi_expiry_analytics';

-- Step 2: Check the base table exists first
SELECT 
    table_name,
    table_type
FROM information_schema.tables
WHERE table_name IN ('nse_multi_expiry_minute_data', 'nse_multi_expiry_analytics')
ORDER BY table_name;

-- Step 3: Simple query to test the view (if it exists)
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
ORDER BY timestamp DESC
LIMIT 10;

-- Step 4: Count records in the view
SELECT 
    COUNT(*) AS total_records,
    COUNT(DISTINCT exchange) AS exchanges,
    COUNT(DISTINCT DATE(timestamp)) AS days,
    MIN(timestamp) AS earliest_timestamp,
    MAX(timestamp) AS latest_timestamp
FROM nse_multi_expiry_analytics;

-- Step 5: Get latest data for a specific exchange (most common use case)
SELECT 
    timestamp,
    exchange,
    base_strike,
    oi_change_diff_put_call,
    sentiment_score_oi_change,
    sentiment_label,
    trend_direction,
    is_turning_point,
    prediction_signal,
    pc_oi_ratio,
    iv_skew
FROM nse_multi_expiry_analytics
WHERE exchange = 'NSE'
ORDER BY timestamp DESC
LIMIT 50;

-- Step 6: Group by date to see record counts per day
SELECT 
    DATE(timestamp) AS trade_date,
    exchange,
    COUNT(*) AS record_count,
    MIN(timestamp) AS first_record,
    MAX(timestamp) AS last_record
FROM nse_multi_expiry_minute_data
WHERE exchange = 'NSE'
GROUP BY DATE(timestamp), exchange
ORDER BY trade_date DESC
LIMIT 100;
