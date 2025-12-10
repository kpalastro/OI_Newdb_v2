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
    'n_estimators': 1000,
    'learning_rate': 0.02,
    'num_leaves': 32,
    'max_depth': 6,
    'class_weight': 'balanced',
    'n_jobs': -1,
    'random_state': 42,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'verbosity': -1
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
        self.valid_ = False

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
        
        # Add small jitter to prevent singular covariance matrices
        # This is critical when some features might be constant or highly correlated
        jitter = np.random.normal(0, 1e-6, X_regime.shape)
        X_regime = X_regime + jitter

        self.model = hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
            init_params="stmc",
            min_covar=1e-4  # Ensure non-zero variance
        )
        try:
            self.model.fit(X_regime)
            self.valid_ = True
        except Exception as e:
            logging.warning(f"HMM Fit failed: {e}. Defaulting to single regime.")
            self.valid_ = False
        return self

    def transform(self, X):
        if not self.valid_ or self.model is None:
            return np.zeros((len(X), 1))
        
        if isinstance(X, pd.DataFrame):
            # Use same feature selection as in fit (with fallback handling)
            regime_features = REGIME_FEATURES.copy()
            if 'pcr_total_oi_zscore' not in X.columns and 'pcr_total_oi' in X.columns:
                regime_features = [f.replace('pcr_total_oi_zscore', 'pcr_total_oi') 
                                 if f == 'pcr_total_oi_zscore' else f 
                                 for f in regime_features]
            X_regime = X[regime_features].copy()
        else:
            raise ValueError("Input must be DataFrame")
            
        X_regime = X_regime.ffill().fillna(0.0)
        try:
            hidden_states = self.model.predict(X_regime)
            return hidden_states.reshape(-1, 1)
        except Exception:
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
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        # 1. Split Data
        X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 2. Fit HMM on TRAIN data only (Prevents Leakage)
        hmm_transformer = RegimeHMMTransformer(n_components=4)
        hmm_transformer.fit(X_train_raw)
        
        # 3. Generate Regimes
        train_regimes = hmm_transformer.transform(X_train_raw).flatten()
        test_regimes = hmm_transformer.transform(X_test_raw).flatten()
        
        # 4. Train/Eval per Regime
        # For simplicity in reporting, we train one LGBM model that takes Regime as a Categorical Feature
        # This is often more robust than splitting data into 4 tiny buckets
        
        X_train_feats = X_train_raw[feature_cols].copy()
        X_test_feats = X_test_raw[feature_cols].copy()
        
        X_train_feats['regime'] = train_regimes
        X_test_feats['regime'] = test_regimes
        
        # Feature Selection on Train
        lgb_selector = lgb.LGBMClassifier(n_estimators=100, random_state=42)
        lgb_selector.fit(X_train_feats, y_train)
        selector = SelectFromModel(lgb_selector, threshold='median', prefit=True)
        
        X_train_sel = selector.transform(X_train_feats)
        X_test_sel = selector.transform(X_test_feats)
        
        # Train Main Model
        clf = lgb.LGBMClassifier(**DEFAULT_MODEL_PARAMS)
        clf.fit(X_train_sel, y_train)
        
        preds = clf.predict(X_test_sel)
        
        # Metrics
        precision = precision_score(y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        
        logging.info(f"Fold {fold+1}: Precision={precision:.3f}, F1={f1:.3f}")
        fold_metrics.append({'fold': fold, 'precision': precision, 'f1': f1})

    return fold_metrics


def _train_expiry_submodule(exchange: str, df_all: pd.DataFrame, lr: float = 0.001, epochs: int = 50):
    """
    Submodule to train the deep learning ExpiryDayTransformer using the loaded dataframe.
    """
    if not TORCH_AVAILABLE:
        return

    logging.info("--> Training ExpiryDayTransformer submodule...")
    
    # Filter for Expiry Context (0-DTE and 1-DTE)
    if 'time_to_expiry_hours' not in df_all.columns:
        logging.warning("Missing time_to_expiry_hours. Skipping expiry training.")
        return
        
    df_expiry = df_all[df_all['time_to_expiry_hours'] <= 25].copy()
    if len(df_expiry) < 50:
        logging.warning(f"Insufficient expiry data ({len(df_expiry)} rows). Skipping expiry training.")
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
    feature_cols = [c for c in REQUIRED_FEATURE_COLUMNS if c in df_expiry.columns]
    
    # Drop NaNs
    df_expiry.dropna(subset=feature_cols + ['direction_target', 'pin_risk_target'], inplace=True)
    
    if df_expiry.empty:
        return

    X_np = df_expiry[feature_cols].values.astype(np.float32)
    y_dir_np = df_expiry['direction_target'].values.astype(np.int64)
    y_pin_np = df_expiry['pin_risk_target'].values.astype(np.float32)
    y_gamma_np = df_expiry['gamma_dist_target'].values.astype(np.float32)
    
    dataset = TensorDataset(
        torch.from_numpy(X_np),
        torch.from_numpy(y_dir_np),
        torch.from_numpy(y_pin_np),
        torch.from_numpy(y_gamma_np)
    )
    
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Initialize Model
    feature_dim = len(feature_cols)
    model = ExpiryDayTransformer(feature_dim=feature_dim)
    
    criteria_pin = nn.BCELoss()
    criteria_gamma = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for X_batch, y_dir_batch, y_pin_batch, y_gamma_batch in loader:
            optimizer.zero_grad()
            out = model(X_batch)
            
            # NLLLoss for direction
            log_probs = torch.log(out['direction_probs'] + 1e-9)
            loss_dir = nn.NLLLoss()(log_probs, y_dir_batch)
            loss_pin = criteria_pin(out['pin_risk'].squeeze(-1), y_pin_batch)
            loss_gamma = criteria_gamma(out['gamma_dist'].squeeze(-1), y_gamma_batch)
            
            loss = loss_dir + 0.2 * loss_pin + 0.0 * loss_gamma
            loss.backward()
            optimizer.step()
            
    # Save
    model_dir = Path('models') / exchange
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / "expiry_transformer.pt"
    torch.save(model.state_dict(), save_path)
    logging.info(f"✓ Expiry Model saved to {save_path}")


def final_training_run(exchange: str, df: pd.DataFrame, feature_cols: List[str]):
    """
    Trains the final production model on ALL data.
    Saves artifacts for the live inference engine.
    """
    logging.info("Training Final Production Models...")
    
    # 0. Train Expiry Submodule (Deep Learning)
    _train_expiry_submodule(exchange, df)
    
    # 1. Fit HMM on All Data
    hmm_model = RegimeHMMTransformer(n_components=4)
    hmm_model.fit(df)
    regimes = hmm_model.transform(df).flatten()
    df['regime'] = regimes
    
    # 2. Train Regime-Specific Models
    # We train separate models per regime for the production inference engine
    # as this allows for specific tuning per market condition.
    
    regime_models = {}
    
    # Global Selector
    X_full = df[feature_cols]
    y_full = df['target']
    
    base_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
    base_model.fit(X_full, y_full)
    selector = SelectFromModel(base_model, threshold='median', prefit=True)
    
    unique_regimes = np.unique(regimes)
    for r in unique_regimes:
        mask = (df['regime'] == r)
        if mask.sum() < 50:
            logging.warning(f"Regime {r} has insufficient data ({mask.sum()}). Skipping.")
            continue
            
        X_r = selector.transform(df.loc[mask, feature_cols])
        y_r = df.loc[mask, 'target']
        
        model = lgb.LGBMClassifier(**DEFAULT_MODEL_PARAMS)
        model.fit(X_r, y_r)
        regime_models[int(r)] = model
        logging.info(f"Regime {r} model trained on {len(y_r)} samples.")

    # Save Artifacts
    model_dir = Path('models') / exchange
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the internal HMM model from our transformer wrapper
    joblib.dump(hmm_model.model, model_dir / 'hmm_regime_model.pkl')
    joblib.dump(regime_models, model_dir / 'regime_models.pkl')
    joblib.dump(feature_cols, model_dir / 'model_features.pkl')
    joblib.dump(selector, model_dir / 'feature_selector.pkl')
    
    # Save swing_ensemble.pkl for multi_horizon_ensemble.py compatibility
    # The inference engine expects this file to load trained tree-based models.
    # Without it, SwingTradingEnsemble falls back to uniform 1/3 probabilities.
    swing_ensemble_data = {
        'models': {'lightgbm': base_model},  # Use the fitted LightGBM model
        'weights': {'lightgbm': 1.0},
        '_is_fitted': True
    }
    joblib.dump(swing_ensemble_data, model_dir / 'swing_ensemble.pkl')
    
    logging.info(f"✓ Models saved to {model_dir}")
    logging.info(f"✓ swing_ensemble.pkl saved for multi-horizon ensemble compatibility")

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
    df = prepare_training_features(raw_data, REQUIRED_FEATURE_COLUMNS)
    
    # 3. Define Target
    df = define_triple_barrier_target(df)
    
    # 4. Validate (CV)
    feature_cols = [c for c in REQUIRED_FEATURE_COLUMNS if c in df.columns]
    cv_results = train_regime_aware_model(df, feature_cols)
    
    avg_f1 = np.mean([x['f1'] for x in cv_results])
    logging.info(f"Cross-Validation Average F1: {avg_f1:.3f}")
    
    # 5. Final Train
    final_training_run(exchange, df, feature_cols)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exchange', required=True, choices=['NSE', 'BSE'])
    parser.add_argument('--days', type=int, default=90)
    args = parser.parse_args()
    
    train(args.exchange, args.days)
