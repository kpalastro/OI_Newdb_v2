-- SQL Query to Update sentiment_score_50 and sentiment_score_100 in ml_features
-- from macro_signals table for each exchange
--
-- This query updates ml_features by matching records from macro_signals
-- based on exchange and timestamp (with nearest match tolerance)
--
-- Usage:
--   For PostgreSQL: Run as-is
--   For SQLite: May need adjustment for timestamp matching

-- ============================================================================
-- OPTION 1: Minute-Level Timestamp Match (Recommended)
-- ============================================================================
-- Updates ml_features by matching timestamps up to minutes (ignoring seconds)
-- This is useful when timestamps might differ by seconds but are in the same minute
-- Example: 2025-12-15 10:30:45 will match with 2025-12-15 10:30:23

-- PostgreSQL version (matches timestamps up to minutes, ignoring seconds):
UPDATE ml_features mf
SET 
    sentiment_score_50 = ms.sentiment_score_50,
    sentiment_score_100 = ms.sentiment_score_100
FROM macro_signals ms
WHERE mf.exchange = ms.exchange
  AND DATE_TRUNC('minute', mf.timestamp) = DATE_TRUNC('minute', ms.timestamp)
  AND ms.sentiment_score_50 IS NOT NULL
  AND ms.sentiment_score_100 IS NOT NULL;

-- SQLite version (matches timestamps up to minutes, ignoring seconds):
-- UPDATE ml_features
-- SET 
--     sentiment_score_50 = (
--         SELECT sentiment_score_50 
--         FROM macro_signals ms
--         WHERE ms.exchange = ml_features.exchange
--           AND strftime('%Y-%m-%d %H:%M', ms.timestamp) = strftime('%Y-%m-%d %H:%M', ml_features.timestamp)
--           AND ms.sentiment_score_50 IS NOT NULL
--           AND ms.sentiment_score_100 IS NOT NULL
--         LIMIT 1
--     ),
--     sentiment_score_100 = (
--         SELECT sentiment_score_100 
--         FROM macro_signals ms
--         WHERE ms.exchange = ml_features.exchange
--           AND strftime('%Y-%m-%d %H:%M', ms.timestamp) = strftime('%Y-%m-%d %H:%M', ml_features.timestamp)
--           AND ms.sentiment_score_50 IS NOT NULL
--           AND ms.sentiment_score_100 IS NOT NULL
--         LIMIT 1
--     )
-- WHERE EXISTS (
--     SELECT 1 FROM macro_signals ms
--     WHERE ms.exchange = ml_features.exchange
--       AND strftime('%Y-%m-%d %H:%M', ms.timestamp) = strftime('%Y-%m-%d %H:%M', ml_features.timestamp)
--       AND ms.sentiment_score_50 IS NOT NULL
--       AND ms.sentiment_score_100 IS NOT NULL
-- );


-- ============================================================================
-- OPTION 2: Nearest Timestamp Match (if exact match doesn't work)
-- ============================================================================
-- Updates ml_features using the nearest timestamp within a time window
-- Use this if timestamps don't match exactly but are close

-- PostgreSQL version with 5-minute window:
-- UPDATE ml_features mf
-- SET 
--     sentiment_score_50 = ms.sentiment_score_50,
--     sentiment_score_100 = ms.sentiment_score_100
-- FROM (
--     SELECT DISTINCT ON (mf.id) 
--         mf.id as ml_id,
--         ms.sentiment_score_50,
--         ms.sentiment_score_100
--     FROM ml_features mf
--     CROSS JOIN LATERAL (
--         SELECT sentiment_score_50, sentiment_score_100, timestamp
--         FROM macro_signals
--         WHERE macro_signals.exchange = mf.exchange
--           AND macro_signals.timestamp BETWEEN mf.timestamp - INTERVAL '5 minutes' 
--                                            AND mf.timestamp + INTERVAL '5 minutes'
--           AND macro_signals.sentiment_score_50 IS NOT NULL
--           AND macro_signals.sentiment_score_100 IS NOT NULL
--         ORDER BY ABS(EXTRACT(EPOCH FROM (macro_signals.timestamp - mf.timestamp)))
--         LIMIT 1
--     ) ms
-- ) nearest
-- WHERE mf.id = nearest.ml_id;


-- ============================================================================
-- OPTION 3: Latest Macro Signal per Exchange (Alternative approach)
-- ============================================================================
-- Updates ml_features using the latest available macro_signal for each exchange
-- This is useful if you want to use the most recent macro signal regardless of timestamp

-- PostgreSQL version:
-- UPDATE ml_features mf
-- SET 
--     sentiment_score_50 = latest_ms.sentiment_score_50,
--     sentiment_score_100 = latest_ms.sentiment_score_100
-- FROM (
--     SELECT DISTINCT ON (exchange)
--         exchange,
--         sentiment_score_50,
--         sentiment_score_100
--     FROM macro_signals
--     WHERE sentiment_score_50 IS NOT NULL
--       AND sentiment_score_100 IS NOT NULL
--     ORDER BY exchange, timestamp DESC
-- ) latest_ms
-- WHERE mf.exchange = latest_ms.exchange;


-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Check how many records will be updated (before running update):
-- Note: Uses minute-level matching (ignoring seconds)
-- 
-- PostgreSQL version:
-- SELECT 
--     mf.exchange,
--     COUNT(*) as total_ml_features,
--     COUNT(ms.sentiment_score_50) as records_with_macro_data
-- FROM ml_features mf
-- LEFT JOIN macro_signals ms 
--     ON mf.exchange = ms.exchange 
--     AND DATE_TRUNC('minute', mf.timestamp) = DATE_TRUNC('minute', ms.timestamp)
--     AND ms.sentiment_score_50 IS NOT NULL
--     AND ms.sentiment_score_100 IS NOT NULL
-- GROUP BY mf.exchange;
--
-- SQLite version:
-- SELECT 
--     mf.exchange,
--     COUNT(*) as total_ml_features,
--     SUM(CASE WHEN ms.sentiment_score_50 IS NOT NULL THEN 1 ELSE 0 END) as records_with_macro_data
-- FROM ml_features mf
-- LEFT JOIN macro_signals ms 
--     ON mf.exchange = ms.exchange 
--     AND strftime('%Y-%m-%d %H:%M', mf.timestamp) = strftime('%Y-%m-%d %H:%M', ms.timestamp)
--     AND ms.sentiment_score_50 IS NOT NULL
--     AND ms.sentiment_score_100 IS NOT NULL
-- GROUP BY mf.exchange;

-- Check records after update:
-- SELECT 
--     exchange,
--     COUNT(*) as total_records,
--     COUNT(sentiment_score_50) as records_with_score_50,
--     COUNT(sentiment_score_100) as records_with_score_100
-- FROM ml_features
-- GROUP BY exchange;

-- Sample records to verify (matching by minute):
-- 
-- PostgreSQL version:
-- SELECT 
--     mf.timestamp as ml_timestamp,
--     ms.timestamp as macro_timestamp,
--     mf.exchange,
--     mf.sentiment_score_50,
--     mf.sentiment_score_100,
--     ms.sentiment_score_50 as macro_score_50,
--     ms.sentiment_score_100 as macro_score_100
-- FROM ml_features mf
-- LEFT JOIN macro_signals ms 
--     ON mf.exchange = ms.exchange 
--     AND DATE_TRUNC('minute', mf.timestamp) = DATE_TRUNC('minute', ms.timestamp)
-- WHERE mf.exchange = 'NSE'
-- ORDER BY mf.timestamp DESC
-- LIMIT 10;
--
-- SQLite version:
-- SELECT 
--     mf.timestamp as ml_timestamp,
--     ms.timestamp as macro_timestamp,
--     mf.exchange,
--     mf.sentiment_score_50,
--     mf.sentiment_score_100,
--     ms.sentiment_score_50 as macro_score_50,
--     ms.sentiment_score_100 as macro_score_100
-- FROM ml_features mf
-- LEFT JOIN macro_signals ms 
--     ON mf.exchange = ms.exchange 
--     AND strftime('%Y-%m-%d %H:%M', mf.timestamp) = strftime('%Y-%m-%d %H:%M', ms.timestamp)
-- WHERE mf.exchange = 'NSE'
-- ORDER BY mf.timestamp DESC
-- LIMIT 10;
