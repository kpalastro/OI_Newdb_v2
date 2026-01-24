"""
Option Return Predictor

Predicts CE and PE option price returns directly using ITM features.
This aligns predictions with actual option trading outcomes.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib
import pandas as pd
import numpy as np

LOGGER = logging.getLogger(__name__)


class OptionReturnPredictor:
    """
    Predicts option price returns for CE and PE options.
    
    Uses separate models for each option type and time horizon.
    """
    
    def __init__(self, exchange: str):
        self.exchange = exchange
        self.models: Dict[str, Any] = {}
        self.turning_point_model: Optional[Any] = None
        self.feature_columns: Optional[List[str]] = None
        self.models_loaded = False
        
        self._load_models()
    
    def _load_models(self) -> None:
        """Load all option return prediction models."""
        model_dir = Path(f"models/option_returns/{self.exchange}")
        
        if not model_dir.exists():
            LOGGER.warning(f"[{self.exchange}] Option return models not found at {model_dir}")
            return
        
        # Load models for each horizon and option type
        horizons = ['3m', '5m', '10m', '15m']
        option_types = ['CE', 'PE']
        
        for horizon in horizons:
            for opt_type in option_types:
                model_path = model_dir / f"{opt_type.lower()}_{horizon}_model.pkl"
                if model_path.exists():
                    try:
                        model = joblib.load(model_path)
                        self.models[f"{opt_type}_{horizon}"] = model
                        LOGGER.info(f"[{self.exchange}] Loaded {opt_type} {horizon} model")
                    except Exception as e:
                        LOGGER.error(f"[{self.exchange}] Failed to load {model_path}: {e}")
        
        # Load turning point model
        turning_point_path = model_dir / "turning_point_model.pkl"
        if turning_point_path.exists():
            try:
                self.turning_point_model = joblib.load(turning_point_path)
                LOGGER.info(f"[{self.exchange}] Loaded turning point model")
            except Exception as e:
                LOGGER.error(f"[{self.exchange}] Failed to load turning point model: {e}")
        
        # Load feature columns
        feature_cols_path = model_dir / "feature_columns.pkl"
        if feature_cols_path.exists():
            try:
                self.feature_columns = joblib.load(feature_cols_path)
                LOGGER.info(f"[{self.exchange}] Loaded feature columns: {len(self.feature_columns)} features")
            except Exception as e:
                LOGGER.warning(f"[{self.exchange}] Failed to load feature columns: {e}")
        
        if self.models:
            self.models_loaded = True
            LOGGER.info(f"[{self.exchange}] Loaded {len(self.models)} option return models")
        else:
            LOGGER.warning(f"[{self.exchange}] No option return models loaded")
    
    def predict_returns(
        self,
        features: pd.DataFrame,
        horizon: str = '3m'
    ) -> Dict[str, Any]:
        """
        Predict option returns for CE and PE.
        
        Args:
            features: DataFrame with feature columns (same as training)
            horizon: '3m', '5m', '10m', or '15m'
        
        Returns:
            {
                'ce_return': predicted CE return %,
                'pe_return': predicted PE return %,
                'ce_confidence': confidence score (0-1),
                'pe_confidence': confidence score (0-1),
                'turning_point_prob': probability of turning point,
                'recommendation': 'BUY_CE', 'BUY_PE', 'SELL_CE', 'SELL_PE', or 'HOLD'
            }
        """
        if not self.models_loaded:
            return {
                'ce_return': 0.0,
                'pe_return': 0.0,
                'ce_confidence': 0.0,
                'pe_confidence': 0.0,
                'turning_point_prob': 0.0,
                'recommendation': 'HOLD'
            }
        
        # Ensure features match training feature columns
        if self.feature_columns:
            # Add missing columns with 0.0
            for col in self.feature_columns:
                if col not in features.columns:
                    features[col] = 0.0
            
            # Select only the columns used during training, in the same order
            features = features[self.feature_columns].copy()
        else:
            LOGGER.warning(f"[{self.exchange}] Feature columns not loaded. Using provided features as-is.")
        
        # Fill NaN values
        features = features.fillna(0.0)
        
        result = {
            'ce_return': 0.0,
            'pe_return': 0.0,
            'ce_confidence': 0.0,
            'pe_confidence': 0.0,
            'turning_point_prob': 0.0,
            'recommendation': 'HOLD'
        }
        
        # Predict CE return
        ce_model_key = f"CE_{horizon}"
        if ce_model_key in self.models:
            try:
                ce_pred = self.models[ce_model_key].predict(features)[0]
                result['ce_return'] = float(ce_pred)
                # Confidence based on magnitude (normalized)
                result['ce_confidence'] = min(1.0, abs(ce_pred) / 10.0)  # 10% = max confidence
            except Exception as e:
                LOGGER.error(f"[{self.exchange}] CE prediction failed: {e}")
        
        # Predict PE return
        pe_model_key = f"PE_{horizon}"
        if pe_model_key in self.models:
            try:
                pe_pred = self.models[pe_model_key].predict(features)[0]
                result['pe_return'] = float(pe_pred)
                result['pe_confidence'] = min(1.0, abs(pe_pred) / 10.0)
            except Exception as e:
                LOGGER.error(f"[{self.exchange}] PE prediction failed: {e}")
        
        # Predict turning point
        if self.turning_point_model:
            try:
                turning_point_proba = self.turning_point_model.predict_proba(features)[0]
                result['turning_point_prob'] = float(turning_point_proba[1])  # Probability of turning point
            except Exception as e:
                LOGGER.error(f"[{self.exchange}] Turning point prediction failed: {e}")
        
        # Generate recommendation
        result['recommendation'] = self._generate_recommendation(result)
        
        return result
    
    def _generate_recommendation(self, result: Dict[str, Any]) -> str:
        """
        Generate trading recommendation based on predicted returns.
        
        Logic:
        - If turning point probability high, avoid trend-following
        - Choose option with higher expected return
        - Require minimum return threshold
        """
        ce_return = result['ce_return']
        pe_return = result['pe_return']
        turning_point_prob = result['turning_point_prob']
        
        # Minimum return threshold (0.5%)
        min_return_threshold = 0.5
        
        # If turning point likely, be cautious
        if turning_point_prob > 0.7:
            return 'HOLD'  # Avoid trend-following during reversals
        
        # Compare returns
        if ce_return > pe_return and ce_return > min_return_threshold:
            return 'BUY_CE'
        elif pe_return > ce_return and pe_return > min_return_threshold:
            return 'BUY_PE'
        elif ce_return < -min_return_threshold:
            return 'SELL_CE'
        elif pe_return < -min_return_threshold:
            return 'SELL_PE'
        else:
            return 'HOLD'
    
    def get_best_horizon(
        self,
        features: pd.DataFrame
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Predict returns for all horizons and return the best one.
        
        Returns:
            (best_horizon, best_result)
        """
        horizons = ['3m', '5m', '10m', '15m']
        results = {}
        
        for horizon in horizons:
            results[horizon] = self.predict_returns(features, horizon)
        
        # Choose horizon with highest expected return
        best_horizon = '3m'
        best_expected_return = 0.0
        
        for horizon, result in results.items():
            max_return = max(result['ce_return'], result['pe_return'])
            if max_return > best_expected_return:
                best_expected_return = max_return
                best_horizon = horizon
        
        return best_horizon, results[best_horizon]
