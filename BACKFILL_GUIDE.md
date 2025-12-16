# Backfill Guide for NSE Option Chain Features

## Overview

This guide explains how to backfill the 8 new NSE option chain features for historical data in the `ml_features` table.

## Important Limitations

⚠️ **CRITICAL NOTE**: 
- **NSE API only provides CURRENT/live option chain data**, not historical data
- Historical option chain data is **not available** from public APIs
- For **historical dates**, the features will be set to **0.0**
- Only **TODAY's data** (if market is currently open) can fetch real NSE option chain features

## What the Script Does

The backfill script (`backfill_nse_features.py`):
1. Loads historical records from `ml_features` table for the last N days
2. Gets open price for each day (from Kite API or database)
3. Calculates nearest ATM strike price based on open price
4. For TODAY: Attempts to fetch NSE option chain data (if `--use-nse-api` flag is used)
5. For HISTORICAL dates: Sets all features to 0.0 (since historical data isn't available)
6. Updates database records with the calculated features

## Usage

### Basic Usage (Recommended - No Kite Required)

```bash
# Backfill last 10 days using database prices (no Kite needed)
# This will set all historical features to 0.0
# Kite is NOT required - script uses underlying_price from database
python backfill_nse_features.py --exchange NSE --days 10
```

**Note**: You do NOT need Kite to run the backfill. The script defaults to using `underlying_price` from your database records, which works perfectly for calculating ATM strikes.

### With Options

```bash
# Backfill with Kite API for open prices (requires 2FA)
python backfill_nse_features.py --exchange NSE --days 10 --use-kite

# Backfill and try NSE API for TODAY only (historical will still be 0.0)
python backfill_nse_features.py --exchange NSE --days 10 --use-nse-api

# Both options
python backfill_nse_features.py --exchange NSE --days 10 --use-kite --use-nse-api
```

### Command Line Arguments

- `--exchange`: Exchange name (default: `NSE`)
- `--days`: Number of days to backfill (default: `10`)
- `--use-kite`: Use Kite API to get open prices (requires credentials and 2FA)
- `--use-nse-api`: Try NSE API for TODAY's data only (historical dates will still be 0.0)

## Step-by-Step Instructions

### Step 1: Ensure Database Migration Ran

Make sure the new columns exist in the `ml_features` table:

```sql
SELECT column_name 
FROM information_schema.columns 
WHERE table_name = 'ml_features'
  AND (column_name LIKE '%nse_next%' OR column_name = 'oi_next_sentiment')
ORDER BY column_name;
```

If columns don't exist, restart your application to run the migration, or manually add them (see `DATABASE_UPDATES_SUMMARY.md`).

### Step 2: Run the Backfill Script

```bash
# Simple backfill (recommended for first run)
python backfill_nse_features.py --exchange NSE --days 10
```

### Step 3: Verify Results

Check if records were updated:

```sql
-- Check how many records have non-zero NSE features (should be 0 for historical)
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as total_records,
    SUM(CASE WHEN nse_next_oi_call_total > 0 THEN 1 ELSE 0 END) as records_with_nse_data
FROM ml_features
WHERE exchange = 'NSE'
  AND timestamp >= CURRENT_DATE - INTERVAL '10 days'
GROUP BY DATE(timestamp)
ORDER BY date DESC;
```

## What Gets Updated

The script updates 8 columns in the `ml_features` table:

1. `oi_next_sentiment` - Sentiment feature
2. `nse_next_oi_call_total` - Total CALL OI
3. `nse_next_oi_put_total` - Total PUT OI
4. `nse_next_oi_change_call_total` - CALL Change in OI
5. `nse_next_oi_change_put_total` - PUT Change in OI
6. `nse_next_volume_call_total` - CALL Volume
7. `nse_next_volume_put_total` - PUT Volume
8. `nse_next_oi_change_diff_put_call` - OI Change Difference

## Expected Behavior

### Historical Dates (Past 9 days)
- All 8 features will be set to **0.0**
- This is expected because NSE API doesn't provide historical option chain data
- The script will log: `Using zero values for NSE features for historical date YYYY-MM-DD`

### Today (If Market is Open)
- If `--use-nse-api` is used, the script will attempt to fetch real data
- If successful, features will have actual values
- If failed, features will be set to 0.0

## Troubleshooting

### Error: "Could not find underlying token for NSE"
- **This is OK if you're not using `--use-kite`**: The script works fine without Kite
- **If using `--use-kite`**: This means Kite couldn't find the token. The script will automatically fall back to using database prices
- **Solution**: Simply run without `--use-kite` flag (recommended):
  ```bash
  python backfill_nse_features.py --exchange NSE --days 10
  ```

### Error: "No historical records found"
- **Solution**: Check if you have data in `ml_features` table for the date range:
  ```sql
  SELECT COUNT(*) FROM ml_features 
  WHERE exchange = 'NSE' 
    AND timestamp >= CURRENT_DATE - INTERVAL '10 days';
  ```

### All features are 0.0
- **This is expected** for historical dates. NSE API only provides current data.
- For future data, features will be calculated automatically when the system runs.

## Alternative: Manual SQL Update (Set All to 0)

If you just want to initialize the columns with 0 values without running the full script:

```sql
-- For PostgreSQL
UPDATE ml_features
SET 
    oi_next_sentiment = 0.0,
    nse_next_oi_call_total = 0.0,
    nse_next_oi_put_total = 0.0,
    nse_next_oi_change_call_total = 0.0,
    nse_next_oi_change_put_total = 0.0,
    nse_next_volume_call_total = 0.0,
    nse_next_volume_put_total = 0.0,
    nse_next_oi_change_diff_put_call = 0.0
WHERE exchange = 'NSE'
  AND timestamp >= CURRENT_DATE - INTERVAL '10 days'
  AND (nse_next_oi_call_total IS NULL OR nse_next_oi_call_total = 0.0);
```

## Future Data

Going forward, the features will be automatically calculated and saved:
- During live trading when market is open
- Using the real-time NSE API data
- Stored automatically in the `ml_features` table

No backfilling needed for future dates - the system handles it automatically!
