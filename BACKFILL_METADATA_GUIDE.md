# Backfill Paper Trading Metadata Guide

**Date:** January 27, 2026  
**Purpose:** Backfill metadata for `paper_trading_metrics` records that were created before the metadata fix

---

## Overview

The `paper_trading_metrics` table has a `metadata` JSONB column that stores strategy-specific information. Trades recorded before the metadata fix (January 27, 2026) have `NULL` metadata.

This script attempts to reconstruct metadata from available sources:
1. **Recommendations log** (`logs/recommendations/YYYY-MM-DD.jsonl`) - Basic metadata like regime, kelly_fraction
2. **ml_features table** - Option return predictions, confidence scores, turning point probabilities

---

## Usage

### 1. Dry Run (Recommended First)

Check what would be updated without making changes:

```bash
# Check all exchanges for last 30 days
python scripts/backfill_paper_trading_metadata.py

# Check specific exchange
python scripts/backfill_paper_trading_metadata.py --exchange NSE

# Check specific date range
python scripts/backfill_paper_trading_metadata.py \
    --start-date 2026-01-20 \
    --end-date 2026-01-27
```

### 2. Execute Backfill

After reviewing the dry-run output, actually update the database:

```bash
# Update all exchanges
python scripts/backfill_paper_trading_metadata.py --execute

# Update specific exchange
python scripts/backfill_paper_trading_metadata.py --exchange NSE --execute

# Update specific date range
python scripts/backfill_paper_trading_metadata.py \
    --start-date 2026-01-20 \
    --end-date 2026-01-27 \
    --execute
```

---

## What Metadata Gets Backfilled

### From Recommendations Log:
- `regime` - Market regime (LOW_VOL_COMPRESSION, RANGE_BOUND, etc.)
- `kelly_fraction` - Kelly criterion fraction
- `recommended_lots` - Recommended position size
- `model_version` - Model version used

### From ml_features Table:
- `strategy_name` - 'OptimizedOptionReturn' (if option return prediction exists)
- `horizon` - Time horizon (15m, 10m, etc.)
- `predicted_return` - Predicted return percentage
- `option_type` - 'CE' or 'PE'
- `ce_return`, `pe_return` - Call/Put return predictions
- `ce_confidence`, `pe_confidence` - Confidence scores
- `turning_point_prob` - Turning point probability
- `option_return_prediction` - Full prediction object

### Inferred:
- `strategy_name` - 'ML_Base' (if no option return prediction) or 'OptimizedOptionReturn'

---

## Matching Logic

The script matches records using:
1. **Timestamp matching** - Within 30 seconds window
2. **Exchange matching** - Must match exactly
3. **Signal matching** - Optional fuzzy match on signal type

For `ml_features` matching:
- Finds closest `ml_features` record within ±5 minutes
- Extracts option return predictions if available

---

## Limitations

⚠️ **Not all metadata can be reconstructed:**
- Strategy-specific metadata that wasn't logged
- ITM evaluation results (if not in ml_features)
- Detailed feature payloads
- Some metadata may be incomplete

✅ **What can be reconstructed:**
- Basic strategy identification
- Option return predictions (if available in ml_features)
- Market regime and risk parameters
- Confidence scores

---

## Verification

After backfilling, verify the results:

```sql
-- Check how many records were updated
SELECT 
    COUNT(*) as total,
    COUNT(metadata) as with_metadata,
    COUNT(*) - COUNT(metadata) as still_null
FROM paper_trading_metrics
WHERE timestamp >= '2026-01-20'::date;

-- Check strategy distribution
SELECT 
    metadata->>'strategy_name' as strategy,
    COUNT(*) as count
FROM paper_trading_metrics
WHERE metadata IS NOT NULL
GROUP BY metadata->>'strategy_name'
ORDER BY count DESC;

-- Check OptimizedOptionReturn trades
SELECT 
    timestamp,
    exchange,
    signal,
    metadata->>'predicted_return' as predicted_return,
    metadata->>'horizon' as horizon
FROM paper_trading_metrics
WHERE metadata->>'strategy_name' = 'OptimizedOptionReturn'
ORDER BY timestamp DESC
LIMIT 10;
```

---

## Example Output

```
================================================================================
BACKFILLING PAPER TRADING METADATA
================================================================================
🔍 DRY RUN MODE - No changes will be made
Loading recommendations from 2025-12-28 to 2026-01-27...
Loaded 11840 recommendations
Found 245 records with NULL metadata
[DRY RUN] Would update record 1234 (NSE, 2026-01-27 09:17:45): {'regime': 'LOW_VOL_COMPRESSION', 'kelly_fraction': 0.309, 'strategy_name': 'ML_Base'}
...
================================================================================
BACKFILL SUMMARY
================================================================================
Total records: 245
Updated: 180
Skipped (no metadata found): 60
Failed: 5
```

---

## Troubleshooting

### No Recommendations Found
- Check that `logs/recommendations/` directory exists
- Verify date range matches log file dates
- Recommendations log may not have all metadata

### No ml_features Matches
- Option return predictions may not exist for all timestamps
- Check if `ml_features` table has data for the time period
- Matching window is ±5 minutes

### Low Match Rate
- Normal - not all trades have corresponding recommendations
- Some metadata simply wasn't logged before the fix
- Focus on trades with `OptimizedOptionReturn` strategy (these have better metadata)

---

## Notes

- **Dry run is safe** - No database changes are made
- **Execute mode updates database** - Review dry-run output first
- **Partial backfill is OK** - Some metadata is better than none
- **Future trades** - Will have metadata automatically (fix is in code)

---

## Related Files

- `scripts/backfill_paper_trading_metadata.py` - Backfill script
- `DATABASE_CHECK_GUIDE.md` - How to check metadata in database
- `metrics/phase2_metrics.py` - Current metadata recording logic
