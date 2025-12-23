# Complete Model Retraining Guide

This guide explains how to retrain all models from scratch using the master training script.

## Quick Start

### Train Everything (Recommended)

```bash
# Train all models for NSE and BSE
python train_all_models.py --all

# Train all models including RL
python train_all_models.py --all

# Skip RL training (faster, if you don't need RL)
python train_all_models.py --all --skip-rl
```

### Train Specific Model Types

```bash
# Only main trading models (LightGBM + HMM)
python train_all_models.py --main

# Only swing ensemble models (XGBoost + LightGBM)
python train_all_models.py --swing

# Only expiry models (Deep Learning)
python train_all_models.py --expiry

# Only RL models (PPO + DQN)
python train_all_models.py --rl

# Combine multiple types
python train_all_models.py --main --swing --expiry
```

### Train for Specific Exchanges

```bash
# Train for NSE only
python train_all_models.py --all --exchanges NSE

# Train for all supported exchanges
python train_all_models.py --all --exchanges NSE BSE NSE_MONTHLY BANKNIFTY_MONTHLY
```

### Use Walk-Forward Orchestrator (Production)

The orchestrator includes hyperparameter tuning and is recommended for production:

```bash
# Use orchestrator only (includes hyperparameter tuning)
python train_all_models.py --orchestrator-only

# Use orchestrator in addition to individual models
python train_all_models.py --all --orchestrator
```

## Model Types Explained

### 1. Main Trading Models (`--main`)
- **Script**: `train_model.py`
- **Models**: LightGBM classifier with HMM regime detection
- **Exchanges**: NSE, BSE
- **Purpose**: Primary trading signals (intraday)
- **Data**: 90 days default

### 2. Swing Ensemble Models (`--swing`)
- **Script**: `train_swing_ensemble.py`
- **Models**: XGBoost + LightGBM ensemble
- **Exchanges**: NSE, BSE
- **Purpose**: 1-3 day swing trading signals
- **Data**: 90 days default

### 3. Expiry Models (`--expiry`)
- **Script**: `train_expiry_model.py`
- **Models**: Deep Learning (ExpiryDayTransformer)
- **Exchanges**: NSE, BSE
- **Purpose**: Expiry day dynamics (0-1 DTE)
- **Data**: 365 days default (needs more data)

### 4. RL Models (`--rl`)
- **Script**: `train_rl.py`
- **Models**: PPO + DQN agents
- **Exchanges**: Exchange-agnostic (trained once, used for all)
- **Purpose**: Execution optimization (order placement)
- **Data**: 30 days default
- **Algorithms**: PPO, DQN, or BOTH

### 5. Walk-Forward Orchestrator (`--orchestrator`)
- **Script**: `train_orchestrator.py`
- **Models**: LightGBM, XGBoost, CatBoost, RL (with Optuna tuning)
- **Exchanges**: All (NSE, BSE, NSE_MONTHLY, BANKNIFTY_MONTHLY)
- **Purpose**: Production-grade training with hyperparameter optimization
- **Data**: 120 days default
- **Features**: Walk-forward validation, model selection, comprehensive reporting

## Advanced Options

### Customize Data Range

```bash
# Use more historical data
python train_all_models.py --all --days 180

# Different data ranges for different models
python train_all_models.py --main --days 90 --expiry-days 365 --rl-days 30
```

### RL Algorithm Selection

```bash
# Train only PPO
python train_all_models.py --rl --rl-algorithm PPO

# Train only DQN
python train_all_models.py --rl --rl-algorithm DQN

# Train both (default)
python train_all_models.py --rl --rl-algorithm BOTH
```

### Orchestrator Options

```bash
# More Optuna trials (better tuning, slower)
python train_all_models.py --orchestrator-only --optuna-trials 20

# More historical data
python train_all_models.py --orchestrator-only --orchestrator-days 180
```

## Training Time Estimates

| Model Type | Time (per exchange) | Notes |
|------------|---------------------|-------|
| Main Models | 5-15 minutes | Fast, single model |
| Swing Ensemble | 10-20 minutes | Two models (XGBoost + LightGBM) |
| Expiry Models | 15-30 minutes | Deep learning, more data |
| RL Models | 20-40 minutes | Two algorithms (PPO + DQN) |
| Orchestrator | 1-3 hours | Includes hyperparameter tuning |

**Total time for `--all`**: ~1-2 hours (depending on hardware and data size)

## Output Locations

All models are saved to:
```
models/
├── NSE/
│   ├── lightgbm_classifier.pkl
│   ├── hmm_regime_model.pkl
│   ├── regime_models.pkl
│   ├── swing_ensemble.pkl
│   ├── expiry_transformer.pth
│   └── training_metadata.json
├── BSE/
│   └── (same structure)
├── rl_ppo_model.zip
└── rl_dqn_model.zip
```

## Recommended Training Schedule

### Daily/Weekly
- **Main Models**: Weekly (or when market regime changes)
- **Swing Ensemble**: Weekly
- **RL Models**: Monthly (execution patterns change slowly)

### Monthly
- **Expiry Models**: Monthly (or when expiry dynamics change)
- **Full Retrain**: Monthly with `--all`

### Production Deployment
- **Orchestrator**: Monthly with `--orchestrator-only --optuna-trials 20`

## Troubleshooting

### Out of Memory
- Reduce `--days` parameter
- Train models one at a time instead of `--all`
- Skip RL training with `--skip-rl`

### Training Fails
- Check database has sufficient historical data
- Verify all dependencies are installed
- Check logs for specific error messages

### Models Not Loading
- Ensure models were saved successfully (check `models/` directory)
- Verify model files are not corrupted
- Check training metadata files exist

## Next Steps After Training

1. **Verify Models**: Check that model files exist in `models/` directory
2. **Restart Application**: Restart OI Tracker to load new models
3. **Monitor Performance**: Watch initial predictions and adjust if needed
4. **Backup Models**: Save trained models to version control or backup

## Example: Complete Retraining Workflow

```bash
# 1. Train all models from scratch
python train_all_models.py --all

# 2. Verify models were created
ls -lh models/NSE/
ls -lh models/BSE/
ls -lh models/*.zip

# 3. Restart the application
# (Your application startup command here)

# 4. Monitor logs for model loading
# Check that models load without errors
```

## Support

For issues or questions:
- Check individual training script documentation
- Review logs for specific error messages
- Verify database has sufficient data
- Ensure all dependencies are installed

allow_unsafe_werkzeug=True
python train_orchestrator.py --exchange BSE  --window-days 3 --step-days 1 --families lightgbm xgboost catboost rl-dqn rl-ppo