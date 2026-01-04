-- Fixed query: Group by date instead of timestamp
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

-- Alternative: Just date and count (simpler)
SELECT 
    DATE(timestamp) AS trade_date,
    COUNT(*) AS record_count
FROM nse_multi_expiry_minute_data
WHERE exchange = 'NSE'
GROUP BY DATE(timestamp)
ORDER BY trade_date DESC
LIMIT 100;

-- With additional statistics per day
SELECT 
    DATE(timestamp) AS trade_date,
    exchange,
    COUNT(*) AS record_count,
    COUNT(DISTINCT base_strike) AS unique_strikes,
    MIN(base_strike) AS min_strike,
    MAX(base_strike) AS max_strike,
    MIN(timestamp) AS first_record_time,
    MAX(timestamp) AS last_record_time
FROM nse_multi_expiry_minute_data
WHERE exchange = 'NSE'
GROUP BY DATE(timestamp), exchange
ORDER BY trade_date DESC
LIMIT 100;
