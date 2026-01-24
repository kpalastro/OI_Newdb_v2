# Option Return Training Implementation

## Overview

This implementation trains models to predict **option price returns directly** instead of index direction. This aligns model training with actual option trading outcomes, as we trade options, not the index.

## Key Features

1. **Direct Option Return Prediction**: Models predict CE and PE option price changes (3m, 5m, 10m, 15m horizons)
2. **Separate Models**: Independent models for CE and PE options
3. **Quality Filters**: Filters out illiquid options and data errors
4. **Turning Point Detection**: Separate model to detect reversal risk
5. **Walk-Forward Validation**: Time-series cross-validation to prevent overfitting
6. **Integration with Inference**: Option return predictions enhance/override existing signals

## Architecture

### Training Pipeline (`train_option_return_models.py`)

1. **Data Loading**: 
   - Loads `ml_features` data
   - Joins with `option_chain_snapshots` to get ATM option prices
   - Extracts price changes (3m, 5m, 10m, 15m)

2. **Quality Filtering**:
   - Removes rows with missing option prices
   - Filters out low-priced options (< ₹1.0, illiquid)
   - Removes extreme price changes (> ±200%, likely data errors)
   - Filters high spreads (> 10% of price, illiquid)

3. **Feature Preparation**:
   - Uses same features as existing models (`REQUIRED_FEATURE_COLUMNS`)
   - Adds ITM optimal range features for training

4. **Model Training**:
   - **CE Models**: Predict CE option price return for each horizon
   - **PE Models**: Predict PE option price return for each horizon
   - **Turning Point Model**: Binary classifier for reversal risk
   - All models use LightGBM with time-series cross-validation

5. **Model Saving**:
   - Models saved to `models/option_returns/{exchange}/`
   - Feature importance saved as CSV
   - Training summary saved as JSON

### Inference Pipeline (`models/option_return_predictor.py`)

1. **Model Loading**: Loads all trained option return models
2. **Prediction**: Predicts returns for all horizons and selects best
3. **Recommendation**: Generates trading recommendation (BUY_CE, BUY_PE, SELL_CE, SELL_PE, HOLD)
4. **Integration**: Integrated into `MLSignalGenerator` to enhance/override signals

## Usage

### Training Models

```bash
# Train with default settings (90 days, all horizons)
python3 train_option_return_models.py --exchange NSE

# Train with custom settings
python3 train_option_return_models.py \
    --exchange NSE \
    --days 120 \
    --horizons 3m 5m 10m \
    --min-price 2.0
```

### Model Output

Models are saved to:
```
models/option_returns/NSE/
├── ce_3m_model.pkl
├── ce_3m_importance.csv
├── pe_3m_model.pkl
├── pe_3m_importance.csv
├── ce_5m_model.pkl
├── pe_5m_model.pkl
├── turning_point_model.pkl
└── training_summary.json
```

### Inference

Option return predictions are automatically integrated into `MLSignalGenerator`:

```python
from ml_core import MLSignalGenerator

generator = MLSignalGenerator(exchange='NSE')
signal, confidence, rationale, metadata = generator.generate_signal(features_dict)

# Check option return prediction in metadata
if 'option_return_prediction' in metadata:
    option_pred = metadata['option_return_prediction']
    print(f"CE Return: {option_pred['ce_return']:.2f}%")
    print(f"PE Return: {option_pred['pe_return']:.2f}%")
    print(f"Recommendation: {option_pred['recommendation']}")
    print(f"Turning Point Risk: {option_pred['turning_point_prob']:.1%}")
```

## Model Performance Metrics

Training reports:
- **R² Score**: How well model explains variance in option returns
- **MAE**: Mean absolute error in return prediction
- **Direction Accuracy**: % of times model predicts correct direction
- **Avg Return on Long**: Average return when model predicts positive

Example output:
```
CE 3m:
  avg_r2: 0.6234
  avg_mae: 2.45
  avg_direction_accuracy: 0.7123
  avg_return_on_long: 3.21%

PE 3m:
  avg_r2: 0.5891
  avg_mae: 2.67
  avg_direction_accuracy: 0.6987
  avg_return_on_long: 2.98%
```

## Signal Enhancement Logic

Option return predictions enhance existing signals:

1. **Turning Point Risk**: If turning point probability > 70%, reduce confidence by 50%
2. **Strong Positive Returns**: If predicted return > 2%, override HOLD with BUY
3. **Strong Negative Returns**: If predicted return < -2%, override HOLD with SELL
4. **Confidence Boost**: Use option return confidence to set signal confidence

## Advantages Over Index-Based Training

1. **Direct Alignment**: Models predict what we actually trade (options)
2. **Better Correlations**: ITM features show 0.76-0.86 correlation with option returns
3. **Actionable Signals**: Direct recommendation (BUY_CE vs BUY_PE)
4. **Risk Management**: Turning point detection prevents entering at reversals
5. **Multiple Horizons**: Can choose best horizon dynamically

## Correlation Findings

From analysis:
- **CE Price Change vs ITM CE Vol Δ%**: 0.7662 (Very Strong!)
- **PE Price Change vs ITM PE Vol Δ%**: 0.8089 (Very Strong!)
- **CE Price Change vs ITM CE OI Δ%**: 0.7264
- **PE Price Change vs ITM PE OI Δ%**: 0.8595

These strong correlations validate using ITM features to predict option returns.

## Signal Performance (Historical)

From 5-day analysis:
- **BEARISH Signal**: CE options gain +7.87% on average, PE lose -0.55%
- **BULLISH Signal**: PE options gain +5.40% on average, CE lose -0.87%

## Next Steps

1. **Train Models**: Run training script to create option return models
2. **Validate**: Check training metrics and feature importance
3. **Monitor**: Track option return prediction accuracy in paper trading
4. **Tune**: Adjust quality filters and thresholds based on results
5. **Compare**: Compare option return model performance vs index-based models

## Files Created

- `train_option_return_models.py`: Main training script
- `models/option_return_predictor.py`: Inference class for option returns
- `ml_core.py`: Updated to integrate option return predictions

## Dependencies

- `lightgbm`: For regression models
- `sklearn`: For metrics and cross-validation
- `pandas`, `numpy`: Data manipulation
- `joblib`: Model serialization

All dependencies are already in the project.
