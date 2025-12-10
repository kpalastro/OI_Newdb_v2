import logging
import numpy as np
from typing import Dict, Any, List
import pandas as pd

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

class SwingTradingEnsemble:
    """
    Ensemble of Tree-based models (XGBoost + LightGBM) for Swing Trading (1-3 days).
    Uses weighted average of probabilities from constituent models.
    """
    def __init__(self):
        self.models = {}
        self.weights = {}
        self._is_fitted = False
        
        if XGBClassifier:
            self.models['xgboost'] = XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                objective='multi:softprob',
                n_jobs=-1
            )
            self.weights['xgboost'] = 0.5
            
        if LGBMClassifier:
            self.models['lightgbm'] = LGBMClassifier(
                n_estimators=500,
                num_leaves=31,
                learning_rate=0.05,
                objective='multiclass',
                n_jobs=-1
            )
            self.weights['lightgbm'] = 0.5
            
        if not self.models:
            logging.warning("No tree-based models available (XGBoost/LightGBM missing). Swing ensemble disabled.")

    def fit(self, X, y):
        """Train all sub-models."""
        for name, model in self.models.items():
            logging.info(f"Training {name} for swing ensemble...")
            model.fit(X, y)
        self._is_fitted = True

    def predict_proba(self, X) -> np.ndarray:
        """
        Weighted average of prediction probabilities.
        Returns: (n_samples, n_classes) array
        """
        if not self._is_fitted:
            # Return uniform probabilities if not fitted
            return np.ones((X.shape[0], 3)) / 3.0
            
        final_probs = np.zeros((X.shape[0], 3))
        total_weight = 0.0
        
        for name, model in self.models.items():
            try:
                probs = model.predict_proba(X)
                weight = self.weights[name]
                final_probs += probs * weight
                total_weight += weight
            except Exception as e:
                logging.error(f"Error predicting with {name}: {e}")
                
        if total_weight > 0:
            final_probs /= total_weight
            
        return final_probs

    def predict(self, features: Any) -> Dict[str, Any]:
        """
        Single sample inference.
        Args:
            features: Array-like (1, n_features) or DataFrame
        """
        # Ensure 2D array or DataFrame
        X = features
        if not isinstance(X, (pd.DataFrame, pd.Series)):
             if hasattr(features, 'values'):
                 X = features.values
             else:
                 X = np.array(features)

        if hasattr(X, 'ndim') and X.ndim == 1:
            if isinstance(X, pd.Series):
                 # Convert Series to DataFrame (transposed) to keep names if possible? 
                 # Usually passed as DF. If Series, to_frame().T
                 X = X.to_frame().T
            else:
                 X = X.reshape(1, -1)
             
        probs = self.predict_proba(X)[0]
        return {
            'class': int(probs.argmax()),
            'probabilities': probs.tolist()
        }

