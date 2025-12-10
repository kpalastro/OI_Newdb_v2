# Model Performance & Archiving Guide

## Overview

This guide covers how to evaluate ML model performance, determine when retraining is needed, and manage model versions through archiving.

---

## 1. Performance Metrics to Monitor

### Key Metrics

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| **Accuracy** | >60% | 55-60% | <55% |
| **Precision (BUY/SELL)** | >55% | 50-55% | <50% |
| **Sharpe Ratio** | >1.5 | 1.0-1.5 | <1.0 |
| **Win Rate** | >55% | 50-55% | <50% |
| **Profit Factor** | >1.3 | 1.1-1.3 | <1.1 |
| **Max Drawdown** | <10% | 10-15% | >15% |

### Where to Find Metrics

1. **Web UI**: Dashboard → Performance tab
2. **Trade Logs**: `trade_logs/YYYY-MM-DD_<exchange>_trades.csv`
3. **Training Reports**: Generated after each training run
4. **Database**: Query `ml_performance` table

---

## 2. Performance Evaluation Process

### Daily Performance Check (5 minutes)

1. Open Web UI Dashboard
2. Check today's paper trading PnL
3. Review signal accuracy for the day
4. Compare to expected regime performance

### Weekly Performance Review (30 minutes)

| Step | Action | What to Look For |
|------|--------|------------------|
| 1 | Export weekly trades | `trade_logs/` CSVs |
| 2 | Calculate win rate | Wins / Total trades |
| 3 | Calculate profit factor | Gross profit / Gross loss |
| 4 | Compare to previous week | >5% degradation = concern |
| 5 | Document findings | Keep log for trend analysis |

### Performance Query (Database)

```sql
-- Weekly model performance summary
SELECT 
    DATE_TRUNC('week', timestamp) as week,
    exchange,
    COUNT(*) as total_signals,
    SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
    ROUND(AVG(confidence), 2) as avg_confidence,
    SUM(pnl) as total_pnl
FROM trade_history
WHERE timestamp > NOW() - INTERVAL '30 days'
GROUP BY week, exchange
ORDER BY week DESC;
```

---

## 3. When to Retrain: Decision Framework

### Performance Degradation Thresholds

```
                    ┌─────────────────────────────────────┐
                    │     MODEL PERFORMANCE CHECK         │
                    └─────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            Accuracy > 55%                  Accuracy < 55%
                    │                               │
                    ▼                               ▼
            ┌───────────────┐              ┌───────────────┐
            │   MONITOR     │              │   RETRAIN     │
            │  Continue 1   │              │    URGENT!    │
            │    week       │              │               │
            └───────────────┘              └───────────────┘
```

### Retraining Triggers

| Condition | Action | Priority |
|-----------|--------|----------|
| Accuracy drops 5%+ from baseline | Retrain immediately | 🔴 High |
| Losing streak > 5 trades | Investigate, may retrain | 🟡 Medium |
| VIX regime change (>10 points) | Retrain within 3 days | 🟡 Medium |
| 30 days since last training | Scheduled retrain | 🟢 Normal |
| Major market event (election, RBI) | Retrain after event | 🟡 Medium |

---

## 4. Model Archiving Procedure

### Before Retraining - Archive Current Models

```bash
#!/bin/bash
# archive_models.sh - Run BEFORE retraining

# Create timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create archive directory
ARCHIVE_DIR="models/archive/${TIMESTAMP}"
mkdir -p "${ARCHIVE_DIR}"

# Archive NSE models
echo "Archiving NSE models..."
cp -r models/NSE/* "${ARCHIVE_DIR}/NSE/" 2>/dev/null || mkdir -p "${ARCHIVE_DIR}/NSE"

# Archive BSE models
echo "Archiving BSE models..."
cp -r models/BSE/* "${ARCHIVE_DIR}/BSE/" 2>/dev/null || mkdir -p "${ARCHIVE_DIR}/BSE"

# Archive NSE_MONTHLY
cp -r models/NSE_MONTHLY/* "${ARCHIVE_DIR}/NSE_MONTHLY/" 2>/dev/null || mkdir -p "${ARCHIVE_DIR}/NSE_MONTHLY"

# Archive BANKNIFTY_MONTHLY
cp -r models/BANKNIFTY_MONTHLY/* "${ARCHIVE_DIR}/BANKNIFTY_MONTHLY/" 2>/dev/null || mkdir -p "${ARCHIVE_DIR}/BANKNIFTY_MONTHLY"

# Create metadata file
cat > "${ARCHIVE_DIR}/archive_metadata.json" << EOF
{
    "archive_date": "$(date -Iseconds)",
    "reason": "Pre-retraining backup",
    "performance_before_archive": {
        "notes": "Fill in current performance metrics"
    }
}
EOF

echo "✓ Models archived to: ${ARCHIVE_DIR}"
```

### Archive Directory Structure

```
models/
├── NSE/                        # Current production models
│   ├── lightgbm_classifier.pkl
│   └── hmm_regime_model.pkl
├── BSE/                        # Current production models
├── archive/                    # Archived versions
│   ├── 20251201_120000/        # Archive timestamp
│   │   ├── NSE/
│   │   ├── BSE/
│   │   └── archive_metadata.json
│   ├── 20251115_180000/        # Older archive
│   └── ...
```

---

## 5. Rollback to Previous Model

### When to Rollback

- New model performs worse than archived version
- Critical bug discovered in new model
- Market conditions better suited to older model

### Rollback Procedure

```bash
#!/bin/bash
# rollback_models.sh <ARCHIVE_TIMESTAMP>

ARCHIVE_TIMESTAMP=$1

if [ -z "$ARCHIVE_TIMESTAMP" ]; then
    echo "Usage: ./rollback_models.sh <ARCHIVE_TIMESTAMP>"
    echo "Available archives:"
    ls -la models/archive/
    exit 1
fi

ARCHIVE_DIR="models/archive/${ARCHIVE_TIMESTAMP}"

if [ ! -d "$ARCHIVE_DIR" ]; then
    echo "Error: Archive not found: ${ARCHIVE_DIR}"
    exit 1
fi

# Backup current models before rollback
BACKUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "Creating safety backup of current models..."
mkdir -p "models/archive/${BACKUP_TIMESTAMP}_pre_rollback"
cp -r models/NSE "models/archive/${BACKUP_TIMESTAMP}_pre_rollback/"
cp -r models/BSE "models/archive/${BACKUP_TIMESTAMP}_pre_rollback/"

# Rollback
echo "Rolling back to: ${ARCHIVE_DIR}"
cp -r "${ARCHIVE_DIR}/NSE/"* models/NSE/ 2>/dev/null
cp -r "${ARCHIVE_DIR}/BSE/"* models/BSE/ 2>/dev/null
cp -r "${ARCHIVE_DIR}/NSE_MONTHLY/"* models/NSE_MONTHLY/ 2>/dev/null
cp -r "${ARCHIVE_DIR}/BANKNIFTY_MONTHLY/"* models/BANKNIFTY_MONTHLY/ 2>/dev/null

echo "✓ Rollback complete!"
echo "⚠ Restart the application to load rolled-back models"
```

### After Rollback

1. **Restart Application**: `python oi_tracker_new.py`
2. **Verify Model Loaded**: Check logs for "Loaded LightGBM model"
3. **Monitor Performance**: Track for 1-2 days
4. **Document**: Note why rollback was needed

---

## 6. Model Version Tracking

### Performance Log Template

Create `models/performance_log.csv`:

```csv
date,exchange,version,accuracy,precision_buy,precision_sell,sharpe,win_rate,notes
2025-12-01,NSE,20251201_120000,0.62,0.58,0.55,1.8,0.58,"Initial training"
2025-12-08,NSE,20251208_100000,0.59,0.52,0.51,1.2,0.54,"Performance drop after retraining"
2025-12-09,NSE,20251201_120000,0.61,0.57,0.54,1.7,0.57,"Rolled back to previous version"
```

### Keeping Track of Best Models

1. Tag high-performing archives:
   ```bash
   mv models/archive/20251201_120000 models/archive/20251201_120000_BEST
   ```

2. Keep at least 3 recent archives
3. Delete archives older than 60 days (unless tagged as BEST)

---

## 7. Quick Reference: Complete Workflow

```
┌──────────────────────────────────────────────────────────────┐
│                    MODEL LIFECYCLE                           │
└──────────────────────────────────────────────────────────────┘

  1. MONITOR          2. EVALUATE         3. DECIDE
  ─────────          ──────────          ────────
  Daily PnL  ───────► Check metrics ───► Retrain needed?
  Win rate           Compare to baseline      │
  Signals                                     │
                                    ┌────────┴────────┐
                                    │                 │
                                    ▼                 ▼
                               No: Continue      Yes: Archive
                                                      │
                                                      ▼
                                               4. ARCHIVE
                                               ──────────
                                               ./archive_models.sh
                                                      │
                                                      ▼
                                               5. RETRAIN
                                               ─────────
                                               train_orchestrator.py
                                                      │
                                                      ▼
                                               6. VALIDATE
                                               ──────────
                                               Compare new vs old
                                                      │
                                    ┌────────────────┴─────────────────┐
                                    │                                  │
                                    ▼                                  ▼
                            Better: Deploy                    Worse: Rollback
                                                             ./rollback_models.sh
```
