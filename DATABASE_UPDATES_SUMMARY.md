# Database Table Updates Summary

## Table Being Updated: `ml_features`

This is the main table that stores ML features for training and analysis.

## 8 New Columns Being Added

### 1. Sentiment Feature (in Sentiment section)
- **Column Name**: `oi_next_sentiment`
- **Type**: `DOUBLE PRECISION`
- **Description**: Sentiment indicator (PUT - CALL difference in Change in OI)

### 2-7. NSE Option Chain Features
- **Column Name**: `nse_next_oi_call_total`
- **Type**: `DOUBLE PRECISION`
- **Description**: Total Open Interest for CALL options from NSE API

- **Column Name**: `nse_next_oi_put_total`
- **Type**: `DOUBLE PRECISION`
- **Description**: Total Open Interest for PUT options from NSE API

- **Column Name**: `nse_next_oi_change_call_total`
- **Type**: `DOUBLE PRECISION`
- **Description**: Total Change in Open Interest for CALL options

- **Column Name**: `nse_next_oi_change_put_total`
- **Type**: `DOUBLE PRECISION`
- **Description**: Total Change in Open Interest for PUT options

- **Column Name**: `nse_next_volume_call_total`
- **Type**: `DOUBLE PRECISION`
- **Description**: Total Volume for CALL options

- **Column Name**: `nse_next_volume_put_total`
- **Type**: `DOUBLE PRECISION`
- **Description**: Total Volume for PUT options

- **Column Name**: `nse_next_oi_change_diff_put_call`
- **Type**: `DOUBLE PRECISION`
- **Description**: Difference in Change in OI (PUT - CALL)

## Files Modified

### 1. `database_new.py`

#### Location 1: Migration Function (lines ~537-544)
**Function**: `migrate_database()`
**Purpose**: Adds new columns to existing table if they don't exist

```python
# Around line 537 in database_new.py
ml_feature_cols = [
    # ... existing columns ...
    # NSE Option Chain Features
    ('oi_next_sentiment', 'DOUBLE PRECISION'),
    ('nse_next_oi_call_total', 'DOUBLE PRECISION'),
    ('nse_next_oi_put_total', 'DOUBLE PRECISION'),
    ('nse_next_oi_change_call_total', 'DOUBLE PRECISION'),
    ('nse_next_oi_change_put_total', 'DOUBLE PRECISION'),
    ('nse_next_volume_call_total', 'DOUBLE PRECISION'),
    ('nse_next_volume_put_total', 'DOUBLE PRECISION'),
    ('nse_next_oi_change_diff_put_call', 'DOUBLE PRECISION')
]
```

#### Location 2: Save Function - Values Extraction (lines ~764-772)
**Function**: `save_option_chain_snapshot()`
**Purpose**: Extracts feature values from `ml_features_dict` to insert into database

```python
# Around line 764 in database_new.py
raw_vals = [
    # ... existing values ...
    # NSE Option Chain Features
    ml_features_dict.get('oi_next_sentiment'),
    ml_features_dict.get('nse_next_oi_call_total'),
    ml_features_dict.get('nse_next_oi_put_total'),
    ml_features_dict.get('nse_next_oi_change_call_total'),
    ml_features_dict.get('nse_next_oi_change_put_total'),
    ml_features_dict.get('nse_next_volume_call_total'),
    ml_features_dict.get('nse_next_volume_put_total'),
    ml_features_dict.get('nse_next_oi_change_diff_put_call'),
]
```

#### Location 3: Save Function - PostgreSQL INSERT (lines ~777-787)
**Function**: `save_option_chain_snapshot()`
**Purpose**: Column names for PostgreSQL INSERT statement

```python
# Around line 777 in database_new.py
cols = [
    # ... existing columns ...
    "sentiment_score_50", "sentiment_score_100", "trin_50", "trin_100",
    "oi_next_sentiment",
    "nse_next_oi_call_total", "nse_next_oi_put_total",
    "nse_next_oi_change_call_total", "nse_next_oi_change_put_total",
    "nse_next_volume_call_total", "nse_next_volume_put_total",
    "nse_next_oi_change_diff_put_call",
    "created_at", "feature_payload"
]
```

#### Location 4: Save Function - SQLite INSERT (lines ~801-812)
**Function**: `save_option_chain_snapshot()`
**Purpose**: Column names for SQLite INSERT statement (updated placeholder count from 32 to 40)

```python
# Around line 801 in database_new.py
INSERT OR REPLACE INTO ml_features 
(..., trin_100,
 oi_next_sentiment,
 nse_next_oi_call_total, nse_next_oi_put_total,
 nse_next_oi_change_call_total, nse_next_oi_change_put_total,
 nse_next_volume_call_total, nse_next_volume_put_total,
 nse_next_oi_change_diff_put_call,
 created_at, feature_payload)
VALUES ({', '.join([ph]*40)})  # Updated from 32 to 40
```

### 2. `feature_engineering.py`
- Already updated with the feature calculation logic
- Features are added to `REQUIRED_FEATURE_COLUMNS` list (lines 105-113)
- Features are calculated in `engineer_live_feature_set()` function (lines ~603-671)

## How Migration Works

1. **On Application Start**: 
   - `initialize_database()` runs (creates table if doesn't exist)
   - `migrate_database()` runs automatically (adds missing columns)

2. **Migration Process**:
   - Checks existing columns in `ml_features` table
   - Adds any missing columns from the `ml_feature_cols` list
   - Logs each column addition

## How to Verify Columns Exist

### Using PostgreSQL Query:
```sql
-- List all columns in ml_features table
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'ml_features'
  AND column_name LIKE '%nse_next%' OR column_name = 'oi_next_sentiment'
ORDER BY ordinal_position;
```

### Using psql Command Line:
```bash
psql -U your_user -d your_database -c "\d ml_features"
```

### Check if Migration Ran:
Look for log messages like:
```
Added column oi_next_sentiment to ml_features
Added column nse_next_oi_call_total to ml_features
...
```

## When Columns Will Appear

- **Automatic**: Next time you start the application, `migrate_database()` will run
- **Manual**: You can also manually run the migration by calling `migrate_database()` function
- **Data Population**: Columns will be populated with data once:
  1. Market is open
  2. Open price is available
  3. NSE API call succeeds
  4. Features are calculated in `engineer_live_feature_set()`
  5. Data is saved via `save_option_chain_snapshot()`

## Complete SQL to Add Columns (if needed manually)

```sql
-- For PostgreSQL
ALTER TABLE ml_features ADD COLUMN IF NOT EXISTS oi_next_sentiment DOUBLE PRECISION;
ALTER TABLE ml_features ADD COLUMN IF NOT EXISTS nse_next_oi_call_total DOUBLE PRECISION;
ALTER TABLE ml_features ADD COLUMN IF NOT EXISTS nse_next_oi_put_total DOUBLE PRECISION;
ALTER TABLE ml_features ADD COLUMN IF NOT EXISTS nse_next_oi_change_call_total DOUBLE PRECISION;
ALTER TABLE ml_features ADD COLUMN IF NOT EXISTS nse_next_oi_change_put_total DOUBLE PRECISION;
ALTER TABLE ml_features ADD COLUMN IF NOT EXISTS nse_next_volume_call_total DOUBLE PRECISION;
ALTER TABLE ml_features ADD COLUMN IF NOT EXISTS nse_next_volume_put_total DOUBLE PRECISION;
ALTER TABLE ml_features ADD COLUMN IF NOT EXISTS nse_next_oi_change_diff_put_call DOUBLE PRECISION;
```
