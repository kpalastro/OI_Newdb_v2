# Production Backfill Guide: ITM Volume Features & Enhanced ITM Dominance Features

This guide provides step-by-step instructions for backfilling new features in production.

## Overview

### Features to Backfill

**Base Features** (require backfilling):
- `itm_volume_ce_pct_change_3m_wavg` - ITM Call Volume % Change (volume-weighted)
- `itm_volume_pe_pct_change_3m_wavg` - ITM Put Volume % Change (volume-weighted)
- `pcr_total_volume` - Put Call Ratio of Volume (PCRV)

**Derived Features** (calculated automatically, no backfill needed):
- `itm_dominance_signal` - Direct PE vs CE comparison (PE - CE)
- `itm_divergence_strength` - Magnitude of divergence
- `itm_dominance_ratio` - Ratio-based signal
- `is_post_1145` - Time-of-day indicator for post-11:45 AM periods
- `itm_dominance_signal_weighted` - Time-weighted dominance signal
- `itm_volume_dominance_signal` - Volume-based dominance
- `itm_volume_divergence_strength` - Volume divergence magnitude
- `itm_combined_dominance_signal` - Combined OI + Volume signal
- `itm_signal_agreement` - Agreement between OI and Volume signals
- `itm_post_1145_enhanced_signal` - Time-weighted combined signal

## Prerequisites

1. **Database Access**: Ensure you have read/write access to the production database
2. **Python Environment**: Activate the virtual environment with all dependencies installed
3. **Data Availability**: Verify that historical option chain snapshots exist in the database
4. **Disk Space**: Ensure sufficient disk space for processing (minimal, but check logs)

## Step-by-Step Backfill Process

### Step 1: Verify Database Connection

```bash
# Test database connection
python -c "from database_new import get_db_connection, release_db_connection; conn = get_db_connection(); print('✓ Database connection successful'); release_db_connection(conn)"
```

### Step 2: Check Data Availability

```bash
# Check how many records exist for the date range you want to backfill
python -c "
from database_new import get_db_connection, release_db_connection
from datetime import datetime, timedelta
conn = get_db_connection()
cursor = conn.cursor()
end_date = datetime.now()
start_date = end_date - timedelta(days=90)
cursor.execute(\"\"\"
    SELECT COUNT(*) 
    FROM ml_features 
    WHERE exchange = 'NSE' 
    AND timestamp >= %s 
    AND timestamp <= %s
\"\"\", (start_date, end_date))
count = cursor.fetchone()[0]
print(f'Found {count} ml_features records for NSE in last 90 days')
release_db_connection(conn)
"
```

### Step 3: Run Backfill Script

#### For NSE Exchange

```bash
# Backfill last 90 days (recommended for training data)
python backfill_itm_volume_features.py --exchange NSE --days 90

# Or backfill specific number of days
python backfill_itm_volume_features.py --exchange NSE --days 60
```

#### For BSE Exchange

```bash
# Backfill last 90 days
python backfill_itm_volume_features.py --exchange BSE --days 90
```

### Step 4: Monitor Progress

The script will output progress logs every 50 updates:

```
2026-01-18 10:00:00 - INFO - Starting backfill for NSE (last 90 days)
2026-01-18 10:00:01 - INFO - Found 4500 timestamps to process
2026-01-18 10:05:30 - INFO - Progress: 50/4500 - Updated: 50, Skipped: 0, Errors: 0
2026-01-18 10:10:45 - INFO - Progress: 100/4500 - Updated: 100, Skipped: 0, Errors: 0
...
2026-01-18 12:30:00 - INFO - Backfill complete: Updated: 4450, Skipped: 50, Errors: 0
```

**Expected Processing Time:**
- ~1-2 minutes per day of data
- 90 days ≈ 90-180 minutes (1.5-3 hours)
- Processing is sequential to avoid database overload

### Step 5: Verify Backfill Results

```bash
# Verify that features were updated
python -c "
from database_new import get_db_connection, release_db_connection
import json
conn = get_db_connection()
cursor = conn.cursor()
cursor.execute(\"\"\"
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN feature_payload->>'itm_volume_ce_pct_change_3m_wavg' IS NOT NULL THEN 1 END) as has_ce_vol,
        COUNT(CASE WHEN feature_payload->>'itm_volume_pe_pct_change_3m_wavg' IS NOT NULL THEN 1 END) as has_pe_vol,
        COUNT(CASE WHEN pcr_total_volume IS NOT NULL THEN 1 END) as has_pcrv
    FROM ml_features
    WHERE exchange = 'NSE'
    AND timestamp >= NOW() - INTERVAL '90 days'
\"\"\")
row = cursor.fetchone()
print(f'Total records: {row[0]}')
print(f'Has ITM CE Volume: {row[1]} ({row[1]/row[0]*100:.1f}%)')
print(f'Has ITM PE Volume: {row[2]} ({row[2]/row[0]*100:.1f}%)')
print(f'Has PCRV: {row[3]} ({row[3]/row[0]*100:.1f}%)')
release_db_connection(conn)
"
```

**Expected Results:**
- All three features should be present in >95% of records
- Lower percentages may indicate missing option chain data for some timestamps

## Post-Backfill Steps

### Step 6: Retrain Models

After backfilling, retrain your models to incorporate the new features:

```bash
# Retrain NSE model with 45 days of data (includes new features)
python train_model.py --exchange NSE --days 45

# The training will automatically:
# 1. Load historical data with backfilled features
# 2. Calculate derived ITM dominance features on-the-fly
# 3. Train models with all 161 features (including 10 new ITM dominance features)
# 4. Register models in the model registry
```

### Step 7: Verify Model Training

```bash
# Check that new model was registered
python scripts/manage_models.py list --exchange NSE --production

# View latest model details
python scripts/manage_models.py show --exchange NSE --latest
```

### Step 8: Run Backtesting

```bash
# Test the new model on recent data
python -m backtesting.run --exchange NSE --start 2026-01-08 --end 2026-01-16 --min-confidence 0.40

# Results will be saved to reports/backtests/NSE.json
# View on Model Monitoring Dashboard: http://localhost:5000/monitoring
```

## Troubleshooting

### Issue: "No option data found" for many timestamps

**Cause**: Historical option chain snapshots may be missing for some timestamps.

**Solution**: 
- Check if `nse_option_chain_snapshots` or `bse_option_chain_snapshots` tables have data
- Skipped records are normal if option data wasn't collected at that time
- The script will continue processing other timestamps

### Issue: "Database connection timeout"

**Cause**: Long-running backfill may hit connection timeouts.

**Solution**:
- The script uses connection pooling, but if issues persist:
- Run backfill in smaller chunks: `--days 30` multiple times
- Ensure database `idle_in_transaction_session_timeout` is set appropriately

### Issue: "Low update percentage (<50%)"

**Cause**: Many records may not have corresponding option chain snapshots.

**Solution**:
- Verify option chain data exists: `SELECT COUNT(*) FROM nse_option_chain_snapshots WHERE timestamp >= ...`
- Check if timestamps in `ml_features` match timestamps in option chain tables
- Some timestamps may legitimately have no option data (market holidays, etc.)

### Issue: "Memory error during backfill"

**Cause**: Processing too many timestamps at once.

**Solution**:
- Reduce `--days` parameter and run multiple times
- The script processes sequentially, but very large batches may still cause issues
- Consider running during off-peak hours

## Production Deployment Checklist

- [ ] Database connection verified
- [ ] Data availability confirmed
- [ ] Backfill script tested on small date range (e.g., `--days 7`)
- [ ] Full backfill completed for NSE
- [ ] Full backfill completed for BSE (if applicable)
- [ ] Backfill results verified (>95% success rate)
- [ ] Models retrained with new features
- [ ] Model registry updated
- [ ] Backtesting completed and results reviewed
- [ ] Model Monitoring Dashboard shows updated metrics

## Performance Considerations

### Database Load

- The backfill script processes records sequentially to minimize database load
- Each update involves:
  1. Loading option chain snapshots
  2. Calculating features
  3. Updating `ml_features` table
- Average processing time: ~2-3 seconds per timestamp

### Recommended Schedule

- **Best Time**: During off-market hours (evenings/weekends)
- **Duration**: Plan for 2-3 hours for 90 days of data
- **Monitoring**: Check logs periodically for errors

### Resource Usage

- **CPU**: Low to moderate (feature calculations)
- **Memory**: Low (~100-200 MB)
- **Database I/O**: Moderate (sequential reads/writes)
- **Network**: Low (local database connection)

## Rollback Plan

If backfill causes issues:

1. **Stop the script**: `Ctrl+C` (safe, processes current record before stopping)
2. **Verify partial updates**: Check how many records were updated
3. **Database backup**: Restore from backup if needed (before starting backfill)
4. **Re-run selectively**: Use smaller date ranges to identify problematic periods

## Support

For issues or questions:
1. Check script logs for error messages
2. Verify database schema matches expected structure
3. Review `feature_engineering.py` for feature calculation logic
4. Check `docs/ITM_DOMINANCE_FEATURES.md` for feature documentation

## Next Steps

After successful backfill and model retraining:

1. **Monitor Model Performance**: Use Model Monitoring Dashboard to track metrics
2. **Compare Models**: Use `python scripts/manage_models.py compare` to evaluate improvements
3. **Production Deployment**: Promote new model if performance is better
4. **Continuous Backfill**: Set up scheduled job to backfill new data daily (optional)

---

**Last Updated**: January 2026
**Version**: 1.0
