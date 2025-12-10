# Model Training Guide

## Overview

This guide explains how to train and retrain ML models for different exchanges. Training is essential for the system to make accurate predictions.

---

## 1. Training Scripts Overview

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `train_model.py` | Single exchange training | Quick retraining, testing |
| `train_orchestrator.py` | Walk-forward AutoML | Full production training with hyperparameter tuning |

---

## 2. train_model.py - Basic Training

### Command Syntax

```bash
python train_model.py --exchange <EXCHANGE> --days <DAYS>
```

### Arguments Explained

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--exchange` | ✅ Yes | - | Exchange to train: `NSE`, `BSE` |
| `--days` | ❌ No | 90 | Days of historical data to use |

### Training Commands for All Exchanges

```bash
# NSE (NIFTY) - Primary Exchange
python train_model.py --exchange NSE --days 90

# BSE (SENSEX)
python train_model.py --exchange BSE --days 90
```

> ⚠️ **Note**: `train_model.py` currently supports only `NSE` and `BSE`.
> For `NSE_MONTHLY` and `BANKNIFTY_MONTHLY`, use `train_orchestrator.py`.

### What Happens During Training

1. **Data Loading**: Fetches features from database (last N days)
2. **HMM Regime Training**: Trains Hidden Markov Model for regime detection
3. **Target Labeling**: Creates Triple Barrier labels (profit/loss/neutral)
4. **Cross-Validation**: TimeSeriesSplit with 5 folds
5. **Model Training**: LightGBM classifier with balanced classes
6. **Model Saving**: Saves to `models/<EXCHANGE>/` directory

### Output Files

```
models/NSE/
├── lightgbm_classifier.pkl      # Main trading model
├── hmm_regime_model.pkl         # Regime detection (with mapping)
├── training_metadata.json       # Feature order, training date
└── cv_results.csv               # Cross-validation metrics
```

---

## 3. train_orchestrator.py - Production Training

### Command Syntax

```bash
python train_orchestrator.py --exchange <EXCHANGE> [options]
```

### Arguments Explained

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--exchange` | ✅ Yes | - | Exchange to train |
| `--days` | ❌ No | 120 | Total historical days to use |
| `--window-days` | ❌ No | 45 | Training window size (days) |
| `--step-days` | ❌ No | 15 | Step size between windows |
| `--families` | ❌ No | all | Model types: `lightgbm`, `xgboost`, `catboost` |
| `--optuna-trials` | ❌ No | 10 | Hyperparameter tuning trials |
| `--output` | ❌ No | auto | Output directory for results |

### Full Training Commands for All Exchanges

```bash
# NSE (NIFTY) - Full AutoML
python train_orchestrator.py --exchange NSE --days 120 --optuna-trials 20

# BSE (SENSEX) - Full AutoML
python train_orchestrator.py --exchange BSE --days 120 --optuna-trials 20

# NSE Monthly Contracts
python train_orchestrator.py --exchange NSE_MONTHLY --days 120 --optuna-trials 20

# BANKNIFTY Monthly Contracts
python train_orchestrator.py --exchange BANKNIFTY_MONTHLY --days 120 --optuna-trials 20
```

### Quick Training (Fewer Trials, Faster)

```bash
# Fast training for testing (5 trials, 60 days)
python train_orchestrator.py --exchange NSE --days 60 --optuna-trials 5
```

### What Happens During Orchestrator Training

1. **Walk-Forward Segmentation**: Splits data into rolling train/validation windows
2. **Multi-Model Evaluation**: Tests LightGBM, XGBoost, and CatBoost
3. **Optuna Hyperparameter Tuning**: Optimizes each model family
4. **Best Model Selection**: Picks winner based on validation performance
5. **Final Training**: Trains production model on all data
6. **Report Generation**: Creates detailed performance report

---

## 4. Training Schedule Recommendations

### Recommended Frequency

| Exchange | Frequency | Best Time | Command |
|----------|-----------|-----------|---------|
| NSE | Weekly (Sunday) | After market close | `train_orchestrator.py --exchange NSE --days 120` |
| BSE | Bi-weekly | After market close | `train_orchestrator.py --exchange BSE --days 90` |
| NSE_MONTHLY | Monthly | 1st weekend of month | `train_orchestrator.py --exchange NSE_MONTHLY --days 90` |
| BANKNIFTY_MONTHLY | Monthly | 1st weekend of month | `train_orchestrator.py --exchange BANKNIFTY_MONTHLY --days 90` |

### When to Retrain (Triggers)

| Trigger | Urgency | Action |
|---------|---------|--------|
| Model accuracy drops below 55% | 🔴 High | Immediate retraining |
| Regime change (VIX regime shift) | 🟡 Medium | Retrain within 1 week |
| Monthly expiry cycle complete | 🟢 Low | Scheduled retraining |
| New data >30 days accumulated | 🟢 Low | Scheduled retraining |

---

## 5. Training Best Practices

### Pre-Training Checklist

- [ ] Database has sufficient historical data (>60 days)
- [ ] No pending database maintenance
- [ ] System not in active trading (off-hours)
- [ ] Sufficient disk space for model files

### Training Duration Estimates

| Exchange | Days | Optuna Trials | Estimated Time |
|----------|------|---------------|----------------|
| NSE | 90 | 10 | ~15-30 minutes |
| NSE | 120 | 20 | ~45-60 minutes |
| All exchanges | 120 | 20 each | ~3-4 hours |

### Weekend Training Script (Recommended)

Create a script `weekend_train.sh`:

```bash
#!/bin/bash
# Weekend Training Script - Run every Sunday

echo "Starting weekend model training..."

# Activate virtual environment
source venv_newdb/bin/activate

# Train all exchanges
echo "Training NSE..."
python train_orchestrator.py --exchange NSE --days 120 --optuna-trials 15

echo "Training BSE..."
python train_orchestrator.py --exchange BSE --days 90 --optuna-trials 10

echo "Training NSE_MONTHLY..."
python train_orchestrator.py --exchange NSE_MONTHLY --days 90 --optuna-trials 10

echo "Training BANKNIFTY_MONTHLY..."
python train_orchestrator.py --exchange BANKNIFTY_MONTHLY --days 90 --optuna-trials 10

echo "Training complete!"
```

Make executable: `chmod +x weekend_train.sh`
Run: `./weekend_train.sh`

---

## 6. Troubleshooting Training Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Insufficient data" | <60 days in database | Wait for more data accumulation |
| "XGBoost not available" | Library not installed | `pip install xgboost` |
| "CatBoost not available" | Library not installed | `pip install catboost` |
| Training hangs | Memory issue | Reduce `--days` parameter |
| Low accuracy (<50%) | Insufficient features | Check database for missing data |
