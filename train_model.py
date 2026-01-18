"""train_model.py

Institutional-grade training pipeline for the OI Gemini ML system.
FIXES:
1. Eliminates Look-Ahead Bias by fitting HMM inside CV loops.
2. Implements Volatility-Adjusted Target definitions.
3. Saves unified pipeline artifacts.
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, f1_score, precision_score
from sklearn.model_selection import TimeSeriesSplit


try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from models.expiry_transformer import ExpiryDayTransformer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not installed. Expiry model training disabled.")

import database_new as db
from time_utils import today_ist
from feature_engineering import (
    REQUIRED_FEATURE_COLUMNS,
    prepare_training_features,
)
try:
    from model_registry import ModelRegistry
    MODEL_REGISTRY_AVAILABLE = True
except ImportError:
    MODEL_REGISTRY_AVAILABLE = False
    logging.warning("model_registry not available. Model registration will be skipped.")

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging to output to stdout
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Use Z-score of PCR for stationarity (PCR can drift over time)
# If pcr_total_oi_zscore is not available, fall back to pcr_total_oi
REGIME_FEATURES = ['vix', 'realized_vol_5m', 'pcr_total_oi_zscore', 'price_roc_30m', 'breadth_divergence']
REGIME_FEATURES_FALLBACK = ['vix', 'realized_vol_5m', 'pcr_total_oi', 'price_roc_30m', 'breadth_divergence']

DEFAULT_MODEL_PARAMS: Dict[str, object] = {
    'objective': 'multiclass',
    'num_class': 3,
    'n_estimators': 500,  # Reduced from 1000
    'learning_rate': 0.02,
    'num_leaves': 32,
    'max_depth': 6,
    'class_weight': 'balanced',
    'n_jobs': -1,
    'random_state': 42,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'verbosity': -1,
    'min_child_samples': 20,  # Prevent overfitting on small splits
    'min_split_gain': 0.0,  # Allow splits even with minimal gain
}

class RegimeHMMTransformer(BaseEstimator, TransformerMixin):
    """
    Custom Transformer to ensure HMM is fit ONLY on training data
    to prevent look-ahead bias during Cross Validation.
    """
    def __init__(self, n_components: int = 4, n_iter: int = 100, random_state: int = 42):
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.valid_ = False
        self.regime_features_used_ = None  # Store which features were actually used

    def fit(self, X, y=None):
        # X is expected to be the full feature set; we extract REGIME_FEATURES
        # Use Z-score version if available, otherwise fall back to raw PCR
        if isinstance(X, pd.DataFrame):
            # Try to use Z-score version first, fall back if not available
            regime_features = REGIME_FEATURES.copy()
            if 'pcr_total_oi_zscore' not in X.columns and 'pcr_total_oi' in X.columns:
                # Replace Z-score with raw PCR in feature list
                regime_features = [f.replace('pcr_total_oi_zscore', 'pcr_total_oi') 
                                 if f == 'pcr_total_oi_zscore' else f 
                                 for f in regime_features]
            X_regime = X[regime_features].copy()
        else:
            # Fallback if numpy array (requires careful column mapping, assuming DF for this project)
            raise ValueError("RegimeHMMTransformer requires pandas DataFrame input")

        X_regime = X_regime.ffill().fillna(0.0)
        
        # Remove constant features (zero variance) that cause covariance issues
        feature_vars = X_regime.var()
        constant_features = feature_vars[feature_vars < 1e-8].index.tolist()
        if constant_features:
            logging.debug(f"Removing constant features for HMM: {constant_features}")
            X_regime = X_regime.drop(columns=constant_features)
        
        # Store which features were actually used (for transform consistency)
        self.regime_features_used_ = X_regime.columns.tolist()
        
        # Check if we have enough features left
        if X_regime.shape[1] < 2:
            logging.warning("HMM: Insufficient features after removing constants. Defaulting to single regime.")
            self.valid_ = False
            self.regime_features_used_ = None
            return self
        
        # Standardize features to prevent covariance issues
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_regime_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_regime),
            columns=X_regime.columns,
            index=X_regime.index
        )
        
        # Add small regularization jitter to prevent singular covariance matrices
        # This is critical when some features might be highly correlated
        np.random.seed(self.random_state)
        jitter = np.random.normal(0, 1e-5, X_regime_scaled.shape)
        X_regime_scaled = X_regime_scaled + jitter

        # Use 'diag' covariance type for more stability, or 'full' with better regularization
        try:
            self.model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type="full",  # Try full first
                n_iter=self.n_iter,
                random_state=self.random_state,
                init_params="stmc",
                min_covar=1e-3  # Increased from 1e-4 for better regularization
            )
            self.model.fit(X_regime_scaled.values)
            self.valid_ = True
        except (ValueError, np.linalg.LinAlgError) as e:
            # If full covariance fails, try diagonal (more stable but less expressive)
            logging.debug(f"HMM full covariance failed: {e}. Trying diagonal covariance...")
            try:
                self.model = hmm.GaussianHMM(
                    n_components=self.n_components,
                    covariance_type="diag",  # Diagonal is more stable
                    n_iter=self.n_iter,
                    random_state=self.random_state,
                    init_params="stmc",
                    min_covar=1e-3
                )
                self.model.fit(X_regime_scaled.values)
                self.valid_ = True
                logging.info("HMM fitted successfully with diagonal covariance")
            except Exception as e2:
                logging.warning(f"HMM Fit failed with both full and diagonal covariance: {e2}. Defaulting to single regime.")
                self.valid_ = False
        except Exception as e:
            logging.warning(f"HMM Fit failed: {e}. Defaulting to single regime.")
            self.valid_ = False
        return self

    def transform(self, X):
        if not self.valid_ or self.model is None:
            return np.zeros((len(X), 1))
        
        if isinstance(X, pd.DataFrame):
            # Use the exact same features that were used during fit
            if self.regime_features_used_ is None:
                # Fallback: try to use REGIME_FEATURES if we don't have stored features
                regime_features = REGIME_FEATURES.copy()
                if 'pcr_total_oi_zscore' not in X.columns and 'pcr_total_oi' in X.columns:
                    regime_features = [f.replace('pcr_total_oi_zscore', 'pcr_total_oi') 
                                     if f == 'pcr_total_oi_zscore' else f 
                                     for f in regime_features]
                X_regime = X[regime_features].copy()
            else:
                # Create a copy to avoid modifying original DataFrame
                X_regime = pd.DataFrame(index=X.index)
                
                # Add features in the exact order they were used during fit
                for feat in self.regime_features_used_:
                    if feat in X.columns:
                        X_regime[feat] = X[feat]
                    else:
                        # Feature missing - fill with zeros
                        logging.warning(f"HMM transform: Missing feature '{feat}', filling with zeros")
                        X_regime[feat] = 0.0
                
                # Ensure columns are in the exact same order as during fit
                X_regime = X_regime[self.regime_features_used_]
        else:
            raise ValueError("Input must be DataFrame")
            
        X_regime = X_regime.ffill().fillna(0.0)
        
        # Standardize using the same scaler from fit
        # The scaler expects features in the same order as during fit
        if hasattr(self, 'scaler') and self.scaler is not None:
            # Ensure feature names match exactly what the scaler expects
            # sklearn's StandardScaler validates feature names
            try:
                X_regime_scaled = pd.DataFrame(
                    self.scaler.transform(X_regime),
                    columns=X_regime.columns,
                    index=X_regime.index
                )
                X_regime = X_regime_scaled
            except ValueError as e:
                # If feature name validation fails, try to fix it
                if "feature names" in str(e).lower():
                    logging.warning(f"HMM transform: Feature name mismatch: {e}. Attempting to fix...")
                    # Reorder columns to match scaler's expected order
                    # The scaler stores feature names in feature_names_in_
                    if hasattr(self.scaler, 'feature_names_in_'):
                        expected_features = list(self.scaler.feature_names_in_)
                        # Reorder X_regime to match expected order
                        X_regime = X_regime.reindex(columns=expected_features, fill_value=0.0)
                        X_regime_scaled = pd.DataFrame(
                            self.scaler.transform(X_regime),
                            columns=X_regime.columns,
                            index=X_regime.index
                        )
                        X_regime = X_regime_scaled
                    else:
                        raise
                else:
                    raise
        
        try:
            hidden_states = self.model.predict(X_regime.values)
            return hidden_states.reshape(-1, 1)
        except Exception as e:
            logging.debug(f"HMM prediction failed: {e}. Returning zeros.")
            return np.zeros((len(X), 1))

def define_triple_barrier_target(
    df: pd.DataFrame,
    look_forward: int = 15,
    pt: float = 1.0,
    sl: float = 1.0
) -> pd.DataFrame:
    """
    Triple Barrier Method for labeling.
    Labels:
      1: Hit Upper Barrier (Profit Take) first
     -1: Hit Lower Barrier (Stop Loss) first
      0: Hit Vertical Barrier (Time Limit) first
    """
    logging.info("Defining Triple Barrier targets...")
    data = df.copy()
    
    # Calculate dynamic volatility (60m rolling std)
    returns = data['underlying_price'].pct_change()
    vol = returns.rolling(60).std()
    
    # Floor volatility to avoid near-zero barriers in quiet markets
    # 0.05% minimum daily move equivalent
    vol = np.maximum(vol, 0.0005)
    
    # Barriers are relative to the Close at time t
    # Using symmetrical barriers (1x Vol) for now, can be asymmetric
    upper_barrier = data['underlying_price'] * (1 + vol * pt)
    lower_barrier = data['underlying_price'] * (1 - vol * sl)
    
    # Initialize hit times with infinity
    hit_upper_time = pd.Series(np.inf, index=data.index)
    hit_lower_time = pd.Series(np.inf, index=data.index)
    
    # Vectorized check for each step in the look_forward window
    for k in range(1, look_forward + 1):
        future_price = data['underlying_price'].shift(-k)
        
        # Check Upper Breach
        # Only update if not already hit (hit_upper_time == inf)
        mask_u = (future_price > upper_barrier) & (hit_upper_time == np.inf)
        hit_upper_time[mask_u] = k
        
        # Check Lower Breach
        mask_l = (future_price < lower_barrier) & (hit_lower_time == np.inf)
        hit_lower_time[mask_l] = k
        
    # Assign labels based on which barrier was hit first
    # 1 if Upper < Lower (Profit first)
    # -1 if Lower < Upper (Stop first)
    # 0 if both are inf (Time limit reached)
    target = np.zeros(len(data))
    target = np.where(hit_upper_time < hit_lower_time, 1, target)
    target = np.where(hit_lower_time < hit_upper_time, -1, target)
    
    data['target'] = target
    
    # Remove last rows where we can't look forward
    data = data.iloc[:-look_forward]
    
    # Also drop initial rows where vol was NaN
    data = data.dropna(subset=['target', 'underlying_price'])
    
    logging.info(f"Target Distribution:\n{pd.Series(target).value_counts(normalize=True)}")
    return data

def train_regime_aware_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_splits: int = 5
) -> Dict[str, Any]:
    """
    Performs TimeSeriesSplit CV where HMM is refit in every fold.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fold_metrics = []
    
    X = df.reset_index(drop=True) # Ensure integer index for splitting
    y = df['target'].values
    
    logging.info(f"Starting Time-Series CV with {n_splits} splits...")
    print(f"DEBUG: Starting Time-Series CV with {n_splits} splits on {len(X)} samples...")
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"DEBUG: Starting fold {fold+1}/{n_splits}...")
        print(f"DEBUG: Fold {fold+1}: Train size={len(train_idx)}, Test size={len(test_idx)}")
        
        # 1. Split Data
        X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 2. Fit HMM on TRAIN data only (Prevents Leakage)
        print(f"DEBUG: Fold {fold+1}: Fitting HMM transformer (this may take 30-60 seconds)...")
        hmm_transformer = RegimeHMMTransformer(n_components=4, n_iter=50)  # Reduced iterations for speed
        hmm_transformer.fit(X_train_raw)
        print(f"DEBUG: Fold {fold+1}: HMM fitting complete.")
        
        # 3. Generate Regimes
        print(f"DEBUG: Fold {fold+1}: Generating regimes...")
        train_regimes = hmm_transformer.transform(X_train_raw).flatten()
        test_regimes = hmm_transformer.transform(X_test_raw).flatten()
        print(f"DEBUG: Fold {fold+1}: Regimes generated.")
        
        # 4. Train/Eval per Regime
        # For simplicity in reporting, we train one LGBM model that takes Regime as a Categorical Feature
        # This is often more robust than splitting data into 4 tiny buckets
        
        # Remove duplicate columns from DataFrames before selecting features
        X_train_raw = X_train_raw.loc[:, ~X_train_raw.columns.duplicated()]
        X_test_raw = X_test_raw.loc[:, ~X_test_raw.columns.duplicated()]
        
        # Ensure feature_cols doesn't have duplicates
        feature_cols_unique = list(dict.fromkeys(feature_cols))  # Preserves order
        
        # Filter to only columns that exist in the DataFrame
        feature_cols_available = [c for c in feature_cols_unique if c in X_train_raw.columns]
        
        X_train_feats = X_train_raw[feature_cols_available].copy()
        X_test_feats = X_test_raw[feature_cols_available].copy()
        
        X_train_feats['regime'] = train_regimes
        X_test_feats['regime'] = test_regimes
        
        # Feature Selection on Train
        print(f"DEBUG: Fold {fold+1}: Training feature selector on {len(X_train_feats)} samples, {len(feature_cols_available)} features (this may take 30-60 seconds)...")
        logging.info(f"Fold {fold+1}: Training feature selector on {len(X_train_feats)} samples, {len(feature_cols_available)} features...")
        lgb_selector = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbosity=-1)  # Reduced for speed
        lgb_selector.fit(X_train_feats, y_train)
        print(f"DEBUG: Fold {fold+1}: Feature selector training complete.")
        selector = SelectFromModel(lgb_selector, threshold='median', prefit=True)
        
        X_train_sel = selector.transform(X_train_feats)
        X_test_sel = selector.transform(X_test_feats)
        
        logging.info(f"Fold {fold+1}: Selected {X_train_sel.shape[1]} features from {X_train_feats.shape[1]} original features")
        
        # Check target distribution
        unique_targets, counts = np.unique(y_train, return_counts=True)
        logging.info(f"Fold {fold+1}: Target distribution: {dict(zip(unique_targets, counts))}")
        
        # Train Main Model with early stopping
        print(f"DEBUG: Fold {fold+1}: Training main model on {len(y_train)} samples...")
        logging.info(f"Fold {fold+1}: Training main model on {len(y_train)} samples...")
        clf = lgb.LGBMClassifier(**DEFAULT_MODEL_PARAMS)
        
        # Use early stopping if we have enough data
        if len(y_train) > 1000:
            print(f"DEBUG: Fold {fold+1}: Using early stopping (train size > 1000)...")
            # Split train into train/val for early stopping
            split_idx = int(len(y_train) * 0.8)
            X_train_split = X_train_sel[:split_idx]
            X_val_split = X_train_sel[split_idx:]
            y_train_split = y_train[:split_idx]
            y_val_split = y_train[split_idx:]
            
            clf.fit(
                X_train_split, y_train_split,
                eval_set=[(X_val_split, y_val_split)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
        else:
            # For small datasets, just fit without early stopping
            print(f"DEBUG: Fold {fold+1}: Fitting without early stopping (train size <= 1000)...")
            clf.fit(X_train_sel, y_train)
        
        print(f"DEBUG: Fold {fold+1}: Model training complete. Making predictions...")
        preds = clf.predict(X_test_sel)
        print(f"DEBUG: Fold {fold+1}: Predictions complete. Calculating metrics...")
        
        # Metrics
        precision = precision_score(y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        
        logging.info(f"Fold {fold+1}: Precision={precision:.3f}, F1={f1:.3f}")
        fold_metrics.append({'fold': fold, 'precision': precision, 'f1': f1})

    return fold_metrics


def _train_expiry_submodule(exchange: str, df_all: pd.DataFrame, lr: float = 0.001, epochs: int = 20):
    """
    Submodule to train the deep learning ExpiryDayTransformer using the loaded dataframe.
    """
    print("DEBUG: Expiry submodule - Checking PyTorch availability...")
    if not TORCH_AVAILABLE:
        print("DEBUG: Expiry submodule - PyTorch not available. Skipping.")
        return
    
    # Check if ExpiryDayTransformer can be imported and instantiated
    try:
        from models.expiry_transformer import ExpiryDayTransformer
        print("DEBUG: Expiry submodule - ExpiryDayTransformer imported successfully.")
    except Exception as e:
        print(f"DEBUG: Expiry submodule - Failed to import ExpiryDayTransformer: {e}")
        print("DEBUG: Expiry submodule - Skipping expiry training.")
        return

    logging.info("--> Training ExpiryDayTransformer submodule...")
    print("DEBUG: Expiry submodule - Starting training...")
    
    # Filter for Expiry Context (0-DTE and 1-DTE)
    if 'time_to_expiry_hours' not in df_all.columns:
        logging.warning("Missing time_to_expiry_hours. Skipping expiry training.")
        print("DEBUG: Expiry submodule - Missing time_to_expiry_hours. Skipping.")
        return
        
    print("DEBUG: Expiry submodule - Filtering expiry data...")
    df_expiry = df_all[df_all['time_to_expiry_hours'] <= 25].copy()
    print(f"DEBUG: Expiry submodule - Found {len(df_expiry)} rows with expiry <= 25 hours.")
    if len(df_expiry) < 50:
        logging.warning(f"Insufficient expiry data ({len(df_expiry)} rows). Skipping expiry training.")
        print(f"DEBUG: Expiry submodule - Insufficient data ({len(df_expiry)} rows). Skipping.")
        return

    # Use same target logic as main pipeline (already labeled in df_all), 
    # but we need specific targets for multi-head (Pin Risk, Gamma).
    # Re-calculate specific targets if needed or derive from 'target'.
    
    # 1. Pin Risk Target
    strike_step = 50.0 if exchange == 'NSE' else 100.0
    df_expiry['dist_to_strike'] = df_expiry['underlying_price'] % strike_step
    df_expiry['dist_to_strike'] = np.minimum(df_expiry['dist_to_strike'], strike_step - df_expiry['dist_to_strike'])
    df_expiry['pin_risk_target'] = (df_expiry['dist_to_strike'] < 10.0).astype(float)
    
    # 2. Gamma Flip (Placeholder)
    df_expiry['gamma_dist_target'] = 0.0
    
    # 3. Direction (-1, 0, 1) -> (0, 1, 2)
    # df_all has 'target' from define_triple_barrier_target
    df_expiry['direction_target'] = df_expiry['target'].astype(int) + 1
    
    # Prepare Tensors
    print("DEBUG: Expiry submodule - Preparing features...")
    feature_cols = [c for c in REQUIRED_FEATURE_COLUMNS if c in df_expiry.columns]
    print(f"DEBUG: Expiry submodule - Using {len(feature_cols)} features.")
    
    # Drop NaNs
    print("DEBUG: Expiry submodule - Dropping NaNs...")
    df_expiry.dropna(subset=feature_cols + ['direction_target', 'pin_risk_target'], inplace=True)
    print(f"DEBUG: Expiry submodule - After dropping NaNs: {len(df_expiry)} rows.")
    
    if df_expiry.empty:
        print("DEBUG: Expiry submodule - No data after dropping NaNs. Skipping.")
        return

    print("DEBUG: Expiry submodule - Converting to tensors...")
    X_np = df_expiry[feature_cols].values.astype(np.float32)
    y_dir_np = df_expiry['direction_target'].values.astype(np.int64)
    y_pin_np = df_expiry['pin_risk_target'].values.astype(np.float32)
    y_gamma_np = df_expiry['gamma_dist_target'].values.astype(np.float32)
    
    print("DEBUG: Expiry submodule - Creating dataset and dataloader...")
    dataset = TensorDataset(
        torch.from_numpy(X_np),
        torch.from_numpy(y_dir_np),
        torch.from_numpy(y_pin_np),
        torch.from_numpy(y_gamma_np)
    )
    
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Initialize Model
    print("DEBUG: Expiry submodule - Initializing model...")
    feature_dim = len(feature_cols)
    try:
        model = ExpiryDayTransformer(feature_dim=feature_dim)
        print(f"DEBUG: Expiry submodule - Model initialized with {feature_dim} features.")
    except Exception as e:
        print(f"DEBUG: Expiry submodule - Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        print("DEBUG: Expiry submodule - Skipping expiry training.")
        return
    
    criteria_pin = nn.BCELoss()
    criteria_gamma = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"DEBUG: Expiry submodule - Starting training for {epochs} epochs (this may take 1-2 minutes)...")
    print(f"DEBUG: Expiry submodule - DataLoader has {len(loader)} batches")
    model.train()
    batch_count = 0
    try:
        for epoch in range(epochs):
            if epoch % 5 == 0 or epoch == 0:
                print(f"DEBUG: Expiry submodule - Epoch {epoch+1}/{epochs}...")
            epoch_loss = 0.0
            print(f"DEBUG: Expiry submodule - Starting epoch {epoch+1} batch loop...")
            for batch_idx, (X_batch, y_dir_batch, y_pin_batch, y_gamma_batch) in enumerate(loader):
                if batch_idx == 0:
                    print(f"DEBUG: Expiry submodule - Processing first batch of epoch {epoch+1}...")
                optimizer.zero_grad()
                
                if batch_idx == 0:
                    print(f"DEBUG: Expiry submodule - Running forward pass...")
                    print(f"DEBUG: Expiry submodule - Input shape: {X_batch.shape}")
                
                try:
                    out = model(X_batch)
                    if batch_idx == 0:
                        print(f"DEBUG: Expiry submodule - Forward pass complete. Output keys: {out.keys() if isinstance(out, dict) else 'Not a dict'}")
                except Exception as e:
                    print(f"DEBUG: Expiry submodule - Forward pass failed: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
                
                if batch_idx == 0:
                    print(f"DEBUG: Expiry submodule - Forward pass complete. Calculating loss...")
                
                # NLLLoss for direction
                log_probs = torch.log(out['direction_probs'] + 1e-9)
                loss_dir = nn.NLLLoss()(log_probs, y_dir_batch)
                loss_pin = criteria_pin(out['pin_risk'].squeeze(-1), y_pin_batch)
                loss_gamma = criteria_gamma(out['gamma_dist'].squeeze(-1), y_gamma_batch)
                
                loss = loss_dir + 0.2 * loss_pin + 0.0 * loss_gamma
                
                if batch_idx == 0:
                    print(f"DEBUG: Expiry submodule - Loss calculated: {loss.item():.4f}. Running backward...")
                loss.backward()
                
                if batch_idx == 0:
                    print(f"DEBUG: Expiry submodule - Backward complete. Stepping optimizer...")
                optimizer.step()
                
                if batch_idx == 0:
                    print(f"DEBUG: Expiry submodule - First batch of epoch {epoch+1} complete!")
                
                epoch_loss += loss.item()
                batch_count += 1
                
                # Print progress every 10 batches
                if batch_count % 10 == 0:
                    print(f"DEBUG: Expiry submodule - Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
            
            if epoch % 5 == 0 or epoch == 0:
                print(f"DEBUG: Expiry submodule - Epoch {epoch+1}/{epochs} complete. Avg loss: {epoch_loss/(batch_idx+1):.4f}")
    except Exception as e:
        print(f"DEBUG: Expiry submodule - Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("DEBUG: Expiry submodule - Skipping expiry training due to error.")
        return
    print("DEBUG: Expiry submodule - Training complete.")
            
    # Save
    print("DEBUG: Expiry submodule - Saving model...")
    model_dir = Path('models') / exchange
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / "expiry_transformer.pt"
    torch.save(model.state_dict(), save_path)
    logging.info(f"✓ Expiry Model saved to {save_path}")
    print(f"DEBUG: Expiry submodule - Model saved to {save_path}")


def final_training_run(exchange: str, df: pd.DataFrame, feature_cols: List[str]):
    """
    Trains the final production model on ALL data.
    Saves artifacts for the live inference engine.
    """
    logging.info("Training Final Production Models...")
    print("DEBUG: Final training - Training expiry submodule...")
    
    # 0. Train Expiry Submodule (Deep Learning)
    # Skip expiry training for now - it's causing hangs and is not critical for main pipeline
    # TODO: Debug and fix ExpiryDayTransformer forward pass issue
    print("DEBUG: Final training - Skipping expiry submodule (known issue with transformer forward pass)")
    logging.info("Skipping expiry submodule training (optional component)")
    # try:
    #     _train_expiry_submodule(exchange, df)
    #     print("DEBUG: Final training - Expiry submodule complete.")
    # except Exception as e:
    #     print(f"DEBUG: Final training - Expiry submodule failed: {e}")
    #     print("DEBUG: Final training - Continuing without expiry model...")
    #     logging.warning(f"Expiry submodule training failed: {e}. Continuing without it.")
    
    # 1. Fit HMM on All Data
    print("DEBUG: Final training - Fitting HMM on all data (this may take 30-60 seconds)...")
    hmm_model = RegimeHMMTransformer(n_components=4, n_iter=50)  # Reduced for speed
    hmm_model.fit(df)
    print("DEBUG: Final training - HMM fitting complete.")
    print("DEBUG: Final training - Generating regimes...")
    regimes = hmm_model.transform(df).flatten()
    df['regime'] = regimes
    print("DEBUG: Final training - Regimes generated.")
    
    # 2. Train Regime-Specific Models
    # We train separate models per regime for the production inference engine
    # as this allows for specific tuning per market condition.
    
    regime_models = {}
    
    # Remove duplicate columns from df first
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Ensure feature_cols doesn't have duplicates and only includes existing columns
    feature_cols = list(dict.fromkeys([c for c in feature_cols if c in df.columns]))
    
    # Global Selector
    X_full = df[feature_cols]
    y_full = df['target']
    
    print(f"DEBUG: Final training - Training base model for feature selection on {len(y_full)} samples...")
    logging.info(f"Training base model for feature selection on {len(y_full)} samples...")
    base_model = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbosity=-1)  # Reduced for speed
    base_model.fit(X_full, y_full)
    print("DEBUG: Final training - Base model training complete.")
    selector = SelectFromModel(base_model, threshold='median', prefit=True)
    
    X_full_sel = selector.transform(X_full)
    logging.info(f"Selected {X_full_sel.shape[1]} features from {len(feature_cols)} original features")
    
    unique_regimes = np.unique(regimes)
    print(f"DEBUG: Final training - Training regime-specific models for {len(unique_regimes)} regimes...")
    logging.info(f"Training regime-specific models for {len(unique_regimes)} regimes...")
    
    for r in unique_regimes:
        print(f"DEBUG: Final training - Training regime {r} model...")
        mask = (df['regime'] == r)
        regime_count = mask.sum()
        if regime_count < 50:
            logging.warning(f"Regime {r} has insufficient data ({regime_count}). Skipping.")
            continue
        
        X_r = selector.transform(df.loc[mask, feature_cols])
        y_r = df.loc[mask, 'target']
        
        # Check target distribution for this regime
        unique_targets, counts = np.unique(y_r, return_counts=True)
        logging.info(f"Regime {r}: {regime_count} samples, target distribution: {dict(zip(unique_targets, counts))}")
        
        # Use early stopping for regime models if enough data
        model = lgb.LGBMClassifier(**DEFAULT_MODEL_PARAMS)
        
        if regime_count > 500:
            # Split for early stopping
            split_idx = int(regime_count * 0.8)
            X_r_train = X_r[:split_idx]
            X_r_val = X_r[split_idx:]
            y_r_train = y_r[:split_idx]
            y_r_val = y_r[split_idx:]
            
            model.fit(
                X_r_train, y_r_train,
                eval_set=[(X_r_val, y_r_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
        else:
            model.fit(X_r, y_r)
        
        regime_models[int(r)] = model
        logging.info(f"✓ Regime {r} model trained on {regime_count} samples.")
        print(f"DEBUG: Final training - Regime {r} model complete.")

    # Save Artifacts
    print("DEBUG: Final training - Saving model artifacts...")
    model_dir = Path('models') / exchange
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the internal HMM model from our transformer wrapper
    hmm_model_path = model_dir / 'hmm_regime_model.pkl'
    regime_models_path = model_dir / 'regime_models.pkl'
    feature_cols_path = model_dir / 'model_features.pkl'
    selector_path = model_dir / 'feature_selector.pkl'
    swing_ensemble_path = model_dir / 'swing_ensemble.pkl'
    
    joblib.dump(hmm_model.model, hmm_model_path)
    joblib.dump(regime_models, regime_models_path)
    joblib.dump(feature_cols, feature_cols_path)
    joblib.dump(selector, selector_path)
    
    # Save swing_ensemble.pkl for multi_horizon_ensemble.py compatibility
    # The inference engine expects this file to load trained tree-based models.
    # Without it, SwingTradingEnsemble falls back to uniform 1/3 probabilities.
    swing_ensemble_data = {
        'models': {'lightgbm': base_model},  # Use the fitted LightGBM model
        'weights': {'lightgbm': 1.0},
        '_is_fitted': True
    }
    joblib.dump(swing_ensemble_data, swing_ensemble_path)
    
    logging.info(f"✓ Models saved to {model_dir}")
    logging.info(f"✓ swing_ensemble.pkl saved for multi-horizon ensemble compatibility")
    print("DEBUG: Final training - Model artifacts saved.")
    
    # Return paths for registry registration
    print("DEBUG: Final training - Returning model paths...")
    return {
        'regime_models': regime_models_path,
        'hmm_model': hmm_model_path,
        'feature_selector': selector_path,
        'model_features': feature_cols_path,
        'swing_ensemble': swing_ensemble_path,
        'expiry_transformer': model_dir / 'expiry_transformer.pt' if (model_dir / 'expiry_transformer.pt').exists() else None
    }

def train(exchange: str, days: int = 90):
    # End date set to tomorrow to include all of today's data
    end_date = today_ist() + timedelta(days=1)
    start_date = today_ist() - timedelta(days=days)
    
    # 1. Load Data
    logging.info(f"Loading data from {start_date} to {end_date} for {exchange}...")
    print(f"DEBUG: Loading data from {start_date} to {end_date} for {exchange}...")
    raw_data = db.load_historical_data_for_ml(exchange, start_date, end_date)
    print(f"DEBUG: Loaded {len(raw_data)} rows.")
    if raw_data.empty:
        logging.error("No data found.")
        return

    # 2. Prepare Features
    print("DEBUG: Starting feature preparation...")
    df = prepare_training_features(raw_data, REQUIRED_FEATURE_COLUMNS)
    print(f"DEBUG: Feature preparation complete. DataFrame shape: {df.shape}")
    
    # 3. Define Target
    print("DEBUG: Defining target...")
    df = define_triple_barrier_target(df)
    print(f"DEBUG: Target defined. DataFrame shape: {df.shape}")
    
    # 4. Validate (CV)
    print("DEBUG: Starting cross-validation...")
    # Remove duplicate columns from df first
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Ensure feature_cols doesn't have duplicates
    feature_cols = list(dict.fromkeys([c for c in REQUIRED_FEATURE_COLUMNS if c in df.columns]))
    print(f"DEBUG: Using {len(feature_cols)} features for CV. DataFrame shape: {df.shape}")
    cv_results = train_regime_aware_model(df, feature_cols)
    print(f"DEBUG: Cross-validation complete. {len(cv_results)} folds processed.")
    
    print("DEBUG: Calculating CV metrics...")
    avg_f1 = np.mean([x['f1'] for x in cv_results])
    avg_precision = np.mean([x['precision'] for x in cv_results])
    avg_recall = np.mean([x.get('recall', avg_f1) for x in cv_results])
    logging.info(f"Cross-Validation Average F1: {avg_f1:.3f}, Precision: {avg_precision:.3f}, Recall: {avg_recall:.3f}")
    print(f"DEBUG: CV Metrics - F1: {avg_f1:.3f}, Precision: {avg_precision:.3f}, Recall: {avg_recall:.3f}")
    
    # 5. Final Train
    print("DEBUG: Starting final training run (this will train on ALL data)...")
    model_paths = final_training_run(exchange, df, feature_cols)
    print("DEBUG: Final training complete.")
    
    # 6. Register models in registry
    if MODEL_REGISTRY_AVAILABLE:
        print("DEBUG: Registering models in model registry...")
        try:
            registry = ModelRegistry()
            print("DEBUG: Model registry initialized.")
            
            # Register main LightGBM model (regime_models.pkl contains all regime models)
            main_model_path = model_paths.get('regime_models')
            if main_model_path and main_model_path.exists():
                registry.register_model(
                    exchange=exchange,
                    model_type='lightgbm_regime',
                    model_path=main_model_path,
                    validation_metrics={
                        'f1_score': avg_f1,
                        'precision': avg_precision,
                        'recall': avg_recall,
                        'accuracy': np.mean([x.get('accuracy', avg_f1) for x in cv_results])
                    },
                    training_data_start=start_date,
                    training_data_end=end_date,
                    training_samples=len(df),
                    cv_metrics=cv_results,
                    artifact_paths=[
                        str(model_paths.get('hmm_model', '')),
                        str(model_paths.get('feature_selector', '')),
                        str(model_paths.get('model_features', '')),
                        str(model_paths.get('swing_ensemble', ''))
                    ],
                    notes=f"Trained with {days} days of data, {len(feature_cols)} features"
                )
                logging.info("✓ Model registered in model registry")
                print("DEBUG: Main LightGBM model registered.")
            
            # Register HMM model separately
            print("DEBUG: Registering HMM model...")
            hmm_model_path = model_paths.get('hmm_model')
            if hmm_model_path and hmm_model_path.exists():
                registry.register_model(
                    exchange=exchange,
                    model_type='hmm_regime',
                    model_path=hmm_model_path,
                    validation_metrics={'cv_f1': avg_f1},
                    training_data_start=start_date,
                    training_data_end=end_date,
                    training_samples=len(df),
                    notes="HMM regime detection model"
                )
            
            # Register expiry transformer if it exists
            expiry_model_path = model_paths.get('expiry_transformer')
            if expiry_model_path and expiry_model_path.exists():
                registry.register_model(
                    exchange=exchange,
                    model_type='expiry_transformer',
                    model_path=expiry_model_path,
                    validation_metrics={'cv_f1': avg_f1},
                    training_data_start=start_date,
                    training_data_end=end_date,
                    training_samples=len(df),
                    notes="Expiry day transformer (PyTorch)"
                )
        except Exception as e:
            logging.warning(f"Failed to register models in registry: {e}")
            logging.warning("Training completed successfully, but model registration failed")
            print(f"DEBUG: Model registration failed: {e}")
    else:
        logging.info("Model registry not available. Skipping model registration.")
        print("DEBUG: Model registry not available. Skipping registration.")
    
    print("DEBUG: Training pipeline complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exchange', required=True, choices=['NSE', 'BSE'])
    parser.add_argument('--days', type=int, default=90)
    args = parser.parse_args()
    
    train(args.exchange, args.days)
