# Fix LightGBM Training Warnings

## Issue

During model training, you're seeing these warnings:
1. `[LightGBM] [Warning] No further splits with positive gain, best gain: -inf`
2. `X does not have valid feature names, but LGBMClassifier was fitted with feature names`

## Root Causes

### 1. Feature Name Mismatch

**Problem**: Model was trained with feature names (DataFrame), but prediction receives numpy array without names.

**Solution**: Already implemented in `ml_core.py` - features are passed as DataFrame with column names.

### 2. NaN/Inf Values in New Features

**Problem**: New ITM features might have NaN or infinite values, causing LightGBM to fail.

**Solution**: Added NaN/inf handling in `feature_engineering.py`:
- Clean values before creating features
- Replace NaN/inf with 0.0
- Ensure all features are valid numbers

### 3. Missing Features During Prediction

**Problem**: New features added but `model_features.pkl` doesn't include them.

**Solution**: 
- Retrain models to update `model_features.pkl`
- Or manually add new features to the saved feature list

## Fixes Applied

### 1. Feature Engineering (`feature_engineering.py`)

**Live Features** (`engineer_live_feature_set`):
```python
# Handle NaN/inf values safely
if pd.isna(itm_ce_change) or pd.isna(itm_pe_change) or np.isinf(itm_ce_change) or np.isinf(itm_pe_change):
    # Set all features to 0.0
    features['itm_bearish_signal'] = 0.0
    # ... etc
```

**Training Features** (`prepare_training_features`):
```python
# Clean NaN and inf values
itm_ce_change = itm_ce_change.fillna(0.0).replace([np.inf, -np.inf], 0.0)
itm_pe_change = itm_pe_change.fillna(0.0).replace([np.inf, -np.inf], 0.0)
```

### 2. Feature Name Preservation (`ml_core.py`)

Already implemented:
```python
# Create DataFrame to preserve feature names (fixes LightGBM warnings)
feature_df = pd.DataFrame([ordered_features], columns=self.feature_columns)
```

## Next Steps

### 1. Retrain Models

After fixing NaN/inf issues, retrain models:

```bash
python3 train_model.py --exchange NSE
python3 train_model.py --exchange BSE
```

This will:
- Update `model_features.pkl` with new features
- Train models with clean feature values
- Fix feature name mismatches

### 2. Verify Feature List

Check that new features are in the saved feature list:

```python
import joblib
features = joblib.load('models/NSE/model_features.pkl')
print('itm_bearish_signal' in features)  # Should be True
print('itm_bullish_signal' in features)  # Should be True
```

### 3. Test Prediction

After retraining, test prediction:

```python
from ml_core import MLSignalGenerator
generator = MLSignalGenerator('NSE')
# Should not show feature name warnings
```

## Expected Behavior After Fix

1. ✅ **No "No further splits" warnings** - Features have valid values
2. ✅ **No "feature names" warnings** - Features passed as DataFrame
3. ✅ **Models train successfully** - All features are valid
4. ✅ **Predictions work** - Feature names match training

## Troubleshooting

### If warnings persist:

1. **Check for constant features**:
   ```python
   # In training data
   df[new_features].std()  # Should not be 0
   ```

2. **Check for missing features**:
   ```python
   # Ensure all new features are in REQUIRED_FEATURE_COLUMNS
   from feature_engineering import REQUIRED_FEATURE_COLUMNS
   assert 'itm_bearish_signal' in REQUIRED_FEATURE_COLUMNS
   ```

3. **Check feature selection**:
   ```python
   # Feature selector might be removing all features
   # Check selected feature count
   print(f"Selected {X_train_sel.shape[1]} features")
   ```

---

**Status**: ✅ **Fixes Applied** - Ready for retraining
