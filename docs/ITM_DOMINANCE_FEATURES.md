# Enhanced ITM Dominance Features for Model Training

## Problem Statement

Model performance degraded after 11:45 AM during downward trends. Analysis revealed:
- **Key Insight**: When ITM PE Δ% > ITM CE Δ%, the trend is usually **negative (bearish)**
- **Key Insight**: When ITM CE Δ% > ITM PE Δ%, the trend is usually **positive (bullish)**
- The model was not capturing this relationship effectively, especially in post-11:45 AM periods

## Solution: Enhanced ITM Dominance Features

### New Features Added

#### 1. **ITM Dominance Signal** (`itm_dominance_signal`)
- **Formula**: `ITM PE Δ% - ITM CE Δ%`
- **Interpretation**:
  - **Positive** = PE dominance → **Bearish signal**
  - **Negative** = CE dominance → **Bullish signal**
- **Purpose**: Direct comparison of PE vs CE deltas to capture the core relationship

#### 2. **ITM Divergence Strength** (`itm_divergence_strength`)
- **Formula**: `|ITM PE Δ% - ITM CE Δ%|`
- **Interpretation**: Magnitude of the difference between PE and CE deltas
- **Purpose**: Higher values indicate stronger directional signal (more confident prediction)

#### 3. **ITM Dominance Ratio** (`itm_dominance_ratio`)
- **Formula**: `ITM PE Δ% / ITM CE Δ%` (with safe division)
- **Interpretation**:
  - **> 1.0** = PE dominance (bearish)
  - **< 1.0** = CE dominance (bullish)
- **Purpose**: Ratio-based signal that handles relative magnitudes better

#### 4. **Post-11:45 AM Indicator** (`is_post_1145`)
- **Formula**: `1.0 if (hour == 11 && minute >= 45) || hour >= 12 else 0.0`
- **Purpose**: Binary flag to help model learn different patterns before/after 11:45 AM

#### 5. **Time-Weighted ITM Dominance Signal** (`itm_dominance_signal_weighted`)
- **Formula**: `itm_dominance_signal * (1.0 + 0.5 * is_post_1145)`
- **Purpose**: Amplifies the dominance signal after 11:45 AM when model performance typically degrades

#### 6. **Volume-Based ITM Dominance Signal** (`itm_volume_dominance_signal`)
- **Formula**: `ITM PE Volume Δ% - ITM CE Volume Δ%`
- **Purpose**: Similar to OI dominance but using volume deltas for confirmation

#### 7. **Volume Divergence Strength** (`itm_volume_divergence_strength`)
- **Formula**: `|ITM PE Volume Δ% - ITM CE Volume Δ%|`
- **Purpose**: Magnitude of volume-based divergence

#### 8. **Combined OI + Volume Dominance Signal** (`itm_combined_dominance_signal`)
- **Formula**: `0.6 * normalized_oi_signal + 0.4 * normalized_vol_signal`
- **Normalization**: Uses `tanh(signal / 10.0)` to scale to [-1, 1] range
- **Purpose**: Combines both OI and Volume signals for stronger directional confirmation

#### 9. **ITM Signal Agreement** (`itm_signal_agreement`)
- **Formula**: `1.0 if OI_sign * Volume_sign > 0 else 0.0`
- **Purpose**: Binary indicator showing whether OI and Volume signals agree (higher confidence when both agree)

#### 10. **Post-11:45 AM Enhanced Signal** (`itm_post_1145_enhanced_signal`)
- **Formula**: `itm_combined_dominance_signal * (1.0 + 0.7 * is_post_1145)`
- **Purpose**: Time-weighted combined signal that amplifies after 11:45 AM to address performance degradation

## Feature Engineering Implementation

### Live Features (`engineer_live_feature_set`)
- All features are calculated in real-time from current ITM OI and Volume deltas
- Features are added to the feature dictionary and included in `REQUIRED_FEATURE_COLUMNS`

### Historical Features (`prepare_training_features`)
- All features are recalculated from historical ITM OI and Volume deltas stored in the database
- Uses vectorized pandas operations for efficiency
- Ensures consistency between live and training features

## Expected Impact

1. **Better Trend Detection**: Direct PE vs CE comparison should improve trend direction prediction
2. **Post-11:45 AM Performance**: Time-weighted features should help model adapt to different market regimes
3. **Signal Confirmation**: Combined OI+Volume signals and agreement indicators should reduce false signals
4. **Stronger Features**: Divergence strength features should help model prioritize high-confidence signals

## Usage in Training

All new features are automatically included in:
- `REQUIRED_FEATURE_COLUMNS` list
- Feature selection process (LightGBM will learn which are most important)
- Model training pipeline (`train_model.py`)

## Monitoring

After retraining, monitor:
- Model performance metrics (F1, Precision, Recall) before and after 11:45 AM
- Feature importance scores to see which new features are most valuable
- Model predictions during downward trends to verify improved performance
