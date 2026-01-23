# Restart & Retrain Guide - ITM Enforcement

## Quick Answer

✅ **YES - Restart is REQUIRED** (code changes need to be loaded)  
✅ **YES - Retraining is RECOMMENDED** (models will learn ITM features better)

---

## Part 1: Restart Application (REQUIRED)

### Why Restart is Required

We modified these files:
- `ml_core.py` - ITM evaluation in signal generation
- `risk_manager.py` - ITM position multiplier
- `execution/auto_executor.py` - ITM safety net check
- `feature_engineering.py` - ITM optimal range features
- `utils/itm_feature_evaluator.py` - NEW module

**The application must be restarted to load the new code.**

### How to Restart

#### Option 1: If Application is Running via Web UI

1. **Logout from web UI** (this triggers a soft restart)
   - Go to: `http://localhost:5000` (or your configured port)
   - Click "Logout"
   - This will reset application state and reload code

2. **Or stop and restart manually**:
   ```bash
   # Find the process
   ps aux | grep oi_tracker_new.py
   
   # Kill it (replace PID with actual process ID)
   kill <PID>
   
   # Restart
   python3 oi_tracker_new.py
   ```

#### Option 2: If Running as Service/Background

```bash
# Stop the application
pkill -f oi_tracker_new.py

# Wait a few seconds
sleep 3

# Restart
python3 oi_tracker_new.py &
```

#### Option 3: Using Systemd (if configured)

```bash
sudo systemctl restart oi_tracker
```

### Verify Restart Worked

After restart, check logs for:

1. **ITM Evaluator Loaded**:
   ```
   [INFO] ITM feature evaluator loaded successfully
   ```

2. **No Import Errors**:
   ```
   [WARNING] ITM feature evaluator not available  # This should NOT appear
   ```

3. **Feature Engineering**:
   - Check that ITM optimal range features are being created
   - Look for features like `itm_pe_vol_in_optimal_range` in logs

---

## Part 2: Retrain Models (RECOMMENDED)

### Why Retraining is Recommended

**Current State:**
- ✅ ITM enforcement works **without retraining** (real-time filtering)
- ✅ Models can use existing ITM features
- ⚠️ Models haven't learned the new **ITM optimal range features**

**After Retraining:**
- ✅ Models will learn to use ITM optimal range features
- ✅ Better feature importance alignment
- ✅ Improved prediction accuracy
- ✅ Models will naturally prioritize ITM volume features

### What Gets Retrained

The new ITM optimal range features will be included:
- `itm_pe_vol_in_optimal_range` (binary indicator)
- `itm_pe_vol_warning` (binary indicator)
- `itm_ce_vol_in_optimal_range` (binary indicator)
- `itm_ce_vol_warning` (binary indicator)
- `itm_pe_delta_in_optimal_range` (binary indicator)
- `itm_ce_delta_in_optimal_range` (binary indicator)
- `itm_signals_agree` (binary indicator)
- `itm_combined_score` (0-1 score)

### How to Retrain

#### Step 1: Check Current Models

```bash
# List existing models
ls -lh models/NSE/*.pkl models/NSE/*.json
ls -lh models/BSE/*.pkl models/BSE/*.json
```

#### Step 2: Backup Existing Models (Optional but Recommended)

```bash
# Create backup directory
mkdir -p models_backup_$(date +%Y%m%d)

# Copy models
cp -r models/NSE models_backup_$(date +%Y%m%d)/
cp -r models/BSE models_backup_$(date +%Y%m%d)/
```

#### Step 3: Retrain All Models

```bash
# Retrain all models (NSE and BSE)
python3 train_all_models.py

# This will:
# - Train intraday models
# - Train swing models
# - Train expiry models
# - Include new ITM optimal range features
```

#### Step 4: Verify New Models

```bash
# Check model files were updated
ls -lh models/NSE/*.pkl
ls -lh models/BSE/*.pkl

# Check feature importance includes ITM features
python3 -c "
import json
with open('models/NSE/feature_importance.json') as f:
    features = json.load(f)
    itm_features = [f for f in features if 'itm' in f.lower()]
    print(f'ITM features in model: {len(itm_features)}')
    for f in itm_features[:10]:
        print(f'  - {f}')
"
```

### Retraining Time Estimate

- **Intraday models**: ~10-30 minutes per exchange
- **Swing models**: ~5-15 minutes per exchange
- **Expiry models**: ~5-15 minutes per exchange
- **Total**: ~30-60 minutes for both NSE and BSE

### Can I Skip Retraining?

**Short Answer**: Yes, but not recommended.

**ITM Enforcement Works Without Retraining:**
- ✅ Real-time ITM filtering works
- ✅ Confidence adjustments work
- ✅ Position sizing adjustments work

**But Models Will Perform Better After Retraining:**
- ✅ Models learn to use ITM optimal range features
- ✅ Better feature importance alignment
- ✅ Improved prediction accuracy
- ✅ Models naturally prioritize important ITM features

**Recommendation**: Retrain when convenient, but you can start paper trading immediately after restart.

---

## Recommended Sequence

### Option A: Quick Start (Restart Only)

1. ✅ Restart application
2. ✅ Start paper trading immediately
3. ⏳ Retrain models later (when convenient)

**Timeline**: 5 minutes  
**ITM Enforcement**: ✅ Active (real-time filtering)  
**Model Learning**: ⚠️ Limited (no new features yet)

### Option B: Full Setup (Restart + Retrain)

1. ✅ Restart application
2. ✅ Retrain all models
3. ✅ Start paper trading

**Timeline**: 30-60 minutes  
**ITM Enforcement**: ✅ Active (real-time filtering)  
**Model Learning**: ✅ Full (with new ITM features)

---

## Verification Checklist

After restart, verify:

- [ ] Application starts without errors
- [ ] No "ITM evaluator not available" warnings in logs
- [ ] ITM features are being evaluated (check logs for ITM metadata)
- [ ] Paper trading signals include ITM metadata

After retraining, verify:

- [ ] New model files created/updated
- [ ] ITM optimal range features in feature importance
- [ ] Models load without errors
- [ ] Feature count increased (should have 8 new ITM features)

---

## Troubleshooting

### Application Won't Start

1. **Check for syntax errors**:
   ```bash
   python3 -m py_compile ml_core.py risk_manager.py execution/auto_executor.py utils/itm_feature_evaluator.py
   ```

2. **Check imports**:
   ```bash
   python3 -c "from utils.itm_feature_evaluator import evaluate_itm_features; print('OK')"
   ```

3. **Check logs** for specific error messages

### Models Won't Load After Retraining

1. **Check feature count matches**:
   - Old models: X features
   - New models: X + 8 features (ITM optimal range)

2. **Verify feature names**:
   ```bash
   python3 -c "
   from feature_engineering import REQUIRED_FEATURE_COLUMNS
   itm_features = [f for f in REQUIRED_FEATURE_COLUMNS if 'itm' in f.lower()]
   print(f'ITM features in REQUIRED_FEATURE_COLUMNS: {len(itm_features)}')
   "
   ```

3. **Check model files exist**:
   ```bash
   ls -lh models/NSE/*.pkl models/BSE/*.pkl
   ```

---

## Summary

| Action | Required? | Time | Impact |
|--------|-----------|------|--------|
| **Restart Application** | ✅ **YES** | 1-2 min | Loads new code, ITM enforcement active |
| **Retrain Models** | ⭐ **RECOMMENDED** | 30-60 min | Models learn ITM features, better accuracy |

**My Recommendation**: 
1. **Restart now** (required) - Get ITM enforcement working immediately
2. **Retrain later** (when convenient) - Improve model performance with new features

---

**Status**: Ready to restart and retrain!
