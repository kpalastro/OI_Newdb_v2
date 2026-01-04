-- SQL Query to update ml_features table columns (nse_next_*) 
-- with corresponding columns from nse_multi_expiry_minute_data

-- Column Mappings:
-- nse_multi_expiry_minute_data -> ml_features
-- total_oi_call_all_expiries -> nse_next_oi_call_total
-- total_oi_put_all_expiries -> nse_next_oi_put_total
-- total_oi_change_call_all_expiries -> nse_next_oi_change_call_total
-- total_oi_change_put_all_expiries -> nse_next_oi_change_put_total
-- total_volume_call_all_expiries -> nse_next_volume_call_total
-- total_volume_put_all_expiries -> nse_next_volume_put_total
-- (total_oi_change_put_all_expiries - total_oi_change_call_all_expiries) -> nse_next_oi_change_diff_put_call
-- (total_oi_change_put_all_expiries - total_oi_change_call_all_expiries) -> oi_next_sentiment
-- All nse_next_* columns + oi_next_sentiment -> feature_payload JSON (merged with existing JSON)
-- (total_oi_change_put_all_expiries - total_oi_change_call_all_expiries) -> oi_next_sentiment
-- All nse_next_* columns + oi_next_sentiment -> feature_payload JSON (merged with existing JSON)

-- MAIN UPDATE QUERY: Match on timestamp (rounded to minute) and exchange
-- Updates nse_next_* columns, oi_next_sentiment, and feature_payload JSON
UPDATE ml_features AS mf
SET 
    nse_next_oi_call_total = multi.total_oi_call_all_expiries,
    nse_next_oi_put_total = multi.total_oi_put_all_expiries,
    nse_next_oi_change_call_total = multi.total_oi_change_call_all_expiries,
    nse_next_oi_change_put_total = multi.total_oi_change_put_all_expiries,
    nse_next_volume_call_total = multi.total_volume_call_all_expiries,
    nse_next_volume_put_total = multi.total_volume_put_all_expiries,
    nse_next_oi_change_diff_put_call = (multi.total_oi_change_put_all_expiries - multi.total_oi_change_call_all_expiries),
    oi_next_sentiment = (multi.total_oi_change_put_all_expiries - multi.total_oi_change_call_all_expiries),
    feature_payload = (
        COALESCE(mf.feature_payload::jsonb, '{}'::jsonb) ||
        jsonb_build_object(
            'nse_next_oi_call_total', multi.total_oi_call_all_expiries,
            'nse_next_oi_put_total', multi.total_oi_put_all_expiries,
            'nse_next_oi_change_call_total', multi.total_oi_change_call_all_expiries,
            'nse_next_oi_change_put_total', multi.total_oi_change_put_all_expiries,
            'nse_next_volume_call_total', multi.total_volume_call_all_expiries,
            'nse_next_volume_put_total', multi.total_volume_put_all_expiries,
            'nse_next_oi_change_diff_put_call', (multi.total_oi_change_put_all_expiries - multi.total_oi_change_call_all_expiries),
            'oi_next_sentiment', (multi.total_oi_change_put_all_expiries - multi.total_oi_change_call_all_expiries)
        )
    )::text
FROM nse_multi_expiry_minute_data AS multi
WHERE mf.exchange = multi.exchange
  AND DATE_TRUNC('minute', mf.timestamp) = DATE_TRUNC('minute', multi.timestamp)
  AND mf.exchange = 'NSE';

-- DRY RUN: Check how many records will be updated (run this first!)
SELECT 
    COUNT(*) AS records_to_update,
    COUNT(DISTINCT DATE_TRUNC('minute', mf.timestamp)) AS unique_minutes,
    MIN(mf.timestamp) AS earliest_timestamp,
    MAX(mf.timestamp) AS latest_timestamp
FROM ml_features AS mf
INNER JOIN nse_multi_expiry_minute_data AS multi
    ON mf.exchange = multi.exchange
    AND DATE_TRUNC('minute', mf.timestamp) = DATE_TRUNC('minute', multi.timestamp)
WHERE mf.exchange = 'NSE';

-- VERIFY: Check the update worked (run after UPDATE)
SELECT 
    mf.timestamp,
    mf.exchange,
    mf.nse_next_oi_call_total,
    mf.nse_next_oi_put_total,
    mf.nse_next_oi_change_diff_put_call,
    mf.oi_next_sentiment,
    multi.total_oi_call_all_expiries,
    multi.total_oi_put_all_expiries,
    (multi.total_oi_change_put_all_expiries - multi.total_oi_change_call_all_expiries) AS calculated_diff
FROM ml_features AS mf
INNER JOIN nse_multi_expiry_minute_data AS multi
    ON mf.exchange = multi.exchange
    AND DATE_TRUNC('minute', mf.timestamp) = DATE_TRUNC('minute', multi.timestamp)
WHERE mf.exchange = 'NSE'
  AND mf.nse_next_oi_call_total IS NOT NULL
ORDER BY mf.timestamp DESC
LIMIT 20;

-- VERIFY: Check feature_payload JSON was updated correctly
SELECT 
    mf.timestamp,
    mf.exchange,
    mf.feature_payload::jsonb->>'nse_next_oi_call_total' AS payload_oi_call,
    mf.feature_payload::jsonb->>'oi_next_sentiment' AS payload_sentiment,
    mf.nse_next_oi_call_total AS column_oi_call,
    mf.oi_next_sentiment AS column_sentiment
FROM ml_features AS mf
WHERE mf.exchange = 'NSE'
  AND mf.nse_next_oi_call_total IS NOT NULL
  AND mf.feature_payload IS NOT NULL
ORDER BY mf.timestamp DESC
LIMIT 10;

-- SUMMARY: Count updated records by date
SELECT 
    DATE(mf.timestamp) AS trade_date,
    COUNT(*) AS updated_records,
    SUM(CASE WHEN mf.nse_next_oi_call_total > 0 THEN 1 ELSE 0 END) AS records_with_data
FROM ml_features AS mf
WHERE mf.exchange = 'NSE'
  AND mf.nse_next_oi_call_total IS NOT NULL
GROUP BY DATE(mf.timestamp)
ORDER BY trade_date DESC
LIMIT 30;
