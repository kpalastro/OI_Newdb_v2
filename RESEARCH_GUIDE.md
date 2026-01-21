# Research Guide: Finding Best Models and Parameters

This guide explains how to use `train_orchestrator.py` and `run_research_experiments.py` to find the best performing models and hyperparameters for NSE and BSE.

## Quick Start

### Option 1: Use the Research Script (Recommended)

```bash
# Run research for both exchanges
python run_research_experiments.py --all

# Or run individually
python run_research_experiments.py --exchange NSE
python run_research_experiments.py --exchange BSE
```

### Option 2: Use train_orchestrator.py Directly

```bash
# NSE with default research settings
python train_orchestrator.py \
    --exchange NSE \
    --days 180 \
    --window-days 45 \
    --step-days 15 \
    --optuna-trials 20 \
    --families lightgbm xgboost catboost

# BSE with same settings
python train_orchestrator.py \
    --exchange BSE \
    --days 180 \
    --window-days 45 \
    --step-days 15 \
    --optuna-trials 20 \
    --families lightgbm xgboost catboost
```

## What Gets Tested

### Model Families
- **LightGBM**: Fast, memory-efficient gradient boosting
- **XGBoost**: Robust gradient boosting with regularization
- **CatBoost**: Handles categorical features well, good for noisy data
- **RL (PPO/DQN)**: Reinforcement learning (optional, slower)

### Walk-Forward Validation
The orchestrator uses **walk-forward analysis**:
- Splits historical data into rolling time windows
- Trains on each window, validates on the next period
- Prevents look-ahead bias
- Tests model robustness across different market conditions

### Hyperparameter Tuning
- Uses **Optuna** for automated hyperparameter optimization
- Tests different learning rates, tree depths, regularization, etc.
- Finds best parameters per model family per segment

## Understanding the Results

### Output Files
Results are saved to `reports/research/` as JSON files:
```
reports/research/
  ├── NSE_research_20241215_143022.json
  └── BSE_research_20241215_150145.json
```

### Key Metrics to Compare

Each JSON file contains:
1. **Per-Segment Results**: Performance for each time window
2. **Per-Family Results**: Aggregated metrics for each model family
3. **Best Parameters**: Optimal hyperparameters found by Optuna
4. **Sample Counts**: Training/validation sizes per segment

**Important Metrics:**
- `f1_macro`: Overall F1 score (higher is better)
- `accuracy`: Classification accuracy
- `precision`: Precision score
- `recall`: Recall score
- `optuna_score`: Best validation score from Optuna tuning

### Example: Analyzing Results

```python
import json
from pathlib import Path

# Load results
with open("reports/research/NSE_research_20241215_143022.json") as f:
    results = json.load(f)

# Find best performing family
family_scores = {}
for segment in results["segments"]:
    family = segment["family"]
    if family not in family_scores:
        family_scores[family] = []
    family_scores[family].append(segment["metrics"]["f1_macro"])

# Average F1 per family
for family, scores in family_scores.items():
    avg_f1 = sum(scores) / len(scores)
    print(f"{family}: {avg_f1:.3f} (avg F1)")

# Best parameters for top family
best_family = max(family_scores.keys(), 
                  key=lambda k: sum(family_scores[k])/len(family_scores[k]))
print(f"\nBest family: {best_family}")
```

## Recommended Research Settings

### Quick Research (Faster, Less Thorough)
```bash
python run_research_experiments.py \
    --exchange NSE \
    --days 120 \
    --window-days 30 \
    --step-days 10 \
    --optuna-trials 10 \
    --families lightgbm xgboost
```
**Time**: ~30-60 minutes per exchange

### Comprehensive Research (Slower, More Thorough)
```bash
python run_research_experiments.py \
    --exchange NSE \
    --days 240 \
    --window-days 60 \
    --step-days 15 \
    --optuna-trials 50 \
    --families lightgbm xgboost catboost
```
**Time**: ~2-4 hours per exchange

### Production Research (Balanced)
```bash
python run_research_experiments.py \
    --exchange NSE \
    --days 180 \
    --window-days 45 \
    --step-days 15 \
    --optuna-trials 20 \
    --families lightgbm xgboost catboost
```
**Time**: ~1-2 hours per exchange

## After Research: Training Production Models

Once you've identified the best model family and parameters:

1. **Review the research results** to identify:
   - Best performing model family (LightGBM/XGBoost/CatBoost)
   - Optimal hyperparameters from Optuna
   - Any exchange-specific differences

2. **Automatically update `train_model.py`** (Recommended):
   ```bash
   # Update with best LightGBM parameters from research
   python3 update_training_params.py reports/research/NSE_research_*.json \
       --exchange NSE \
       --family lightgbm
   
   # Preview changes first (dry-run)
   python3 update_training_params.py reports/research/NSE_research_*.json \
       --exchange NSE \
       --family lightgbm \
       --dry-run
   ```
   
   This script will:
   - Extract best parameters from research results
   - Update `DEFAULT_MODEL_PARAMS` in `train_model.py`
   - Create a backup of the original file
   - Preserve comments and formatting

3. **Or manually update `train_model.py`** if needed:
   - Modify `DEFAULT_MODEL_PARAMS` with best hyperparameters
   - Or create exchange-specific parameter sets

4. **Train production models**:
   ```bash
   # Train with best settings
   python3 train_model.py --exchange NSE --days 90
   python3 train_model.py --exchange BSE --days 90
   ```

## Tips for Research

1. **Start with Quick Research**: Run quick settings first to get initial insights
2. **Compare Across Exchanges**: NSE and BSE may have different optimal models
3. **Check Segment Stability**: Look for families that perform consistently across segments
4. **Consider Feature Importance**: Check `feature_importance.json` from production training to see which features matter most
5. **Validate on Recent Data**: Ensure recent segments perform well (market conditions change)

## Troubleshooting

### "No data loaded" Error
- **Solution**: Increase `--days` or check database has historical data

### "Zero windows" Error
- **Solution**: Reduce `--window-days` or increase `--days`

### Out of Memory
- **Solution**: Reduce `--optuna-trials` or test fewer families at once

### Very Slow Execution
- **Solution**: 
  - Reduce `--optuna-trials` (10-15 instead of 20+)
  - Test fewer families (just `lightgbm` first)
  - Reduce `--days` or `--window-days`

## Next Steps

After research:
1. ✅ Identify best model family per exchange
2. ✅ Extract optimal hyperparameters
3. ✅ Train production models with `train_model.py`
4. ✅ Validate on out-of-sample data
5. ✅ Deploy to live trading system
