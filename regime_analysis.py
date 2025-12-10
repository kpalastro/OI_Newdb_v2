"""
Regime Analysis Module.

This module analyzes market regimes using a combination of HMM and rule-based
logic. It classifies the market into 5 distinct regimes:
1. TRENDING_UP: Strong upward momentum (High ROC, Low Vol).
2. TRENDING_DOWN: Strong downward momentum (Negative ROC, Rising Vol).
3. RANGE_BOUND: Mean reverting (Low ROC, Low/Med Vol).
4. HIGH_VOL_CRASH: Panic selling (Extreme VIX, Negative ROC).
5. LOW_VOL_COMPRESSION: Squeeze setup (Very Low VIX, Flat ROC).
"""

import argparse
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from hmmlearn import hmm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Regime features used for HMM
REGIME_FEATURES = ['vix', 'realized_vol_5m', 'pcr_total_oi_zscore', 'price_roc_30m', 'breadth_divergence']

class MarketRegimeDetector:
    """
    Detects market regime using a hybrid approach (HMM + Heuristics).
    
    The HMM model and regime mapping are loaded from disk. If no mapping is available,
    the detector will infer regime labels from HMM cluster characteristics or fall back
    to rule-based classification.
    """
    
    # Default regime names for reference
    REGIME_NAMES = [
        'TRENDING_UP',
        'TRENDING_DOWN', 
        'RANGE_BOUND',
        'HIGH_VOL_CRASH',
        'LOW_VOL_COMPRESSION'
    ]
    
    def __init__(self, exchange: str = 'NSE'):
        self.exchange = exchange
        self.hmm_model: Optional[hmm.GaussianHMM] = None
        self.regime_map: Dict[int, str] = {}  # HMM state ID -> semantic regime name
        self.feature_order: list = REGIME_FEATURES  # Feature order used during training
        self._load_hmm_model()

    def _load_hmm_model(self):
        """
        Load the pre-trained HMM model and regime mapping from disk.
        
        Expected file format (joblib):
            {
                'model': GaussianHMM object,
                'regime_map': {0: 'TRENDING_UP', 1: 'RANGE_BOUND', ...},
                'feature_order': ['vix', 'realized_vol_5m', ...],
                'cluster_stats': {...}  # Optional: mean/std per cluster for analysis
            }
        
        Falls back to loading just the model if old format is detected.
        """
        model_path = Path(f'models/{self.exchange}/hmm_regime_model.pkl')
        
        if not model_path.exists():
            logging.warning(f"HMM model not found at {model_path}. Using fallback logic.")
            return
            
        try:
            saved_data = joblib.load(model_path)
            
            # Handle new format (dict with model + metadata)
            if isinstance(saved_data, dict):
                self.hmm_model = saved_data.get('model')
                self.regime_map = saved_data.get('regime_map', {})
                self.feature_order = saved_data.get('feature_order', REGIME_FEATURES)
                
                if self.regime_map:
                    logging.info(f"✓ Loaded HMM model with regime mapping: {self.regime_map}")
                else:
                    logging.info(f"✓ Loaded HMM model (no regime mapping - will infer)")
                    self._infer_regime_mapping()
                    
            # Handle old format (just the model object)
            elif hasattr(saved_data, 'predict'):
                self.hmm_model = saved_data
                logging.info(f"✓ Loaded HMM model (legacy format)")
                self._infer_regime_mapping()
            else:
                logging.error(f"Unknown HMM model format in {model_path}")
                
        except Exception as e:
            logging.error(f"Error loading HMM model: {e}")

    def _infer_regime_mapping(self):
        """
        Infer regime labels from HMM cluster characteristics (means).
        
        This analyzes the cluster centers to determine which state corresponds
        to which regime based on the feature values.
        
        Feature order: ['vix', 'realized_vol_5m', 'pcr_total_oi_zscore', 'price_roc_30m', 'breadth_divergence']
        """
        if self.hmm_model is None or not hasattr(self.hmm_model, 'means_'):
            return
            
        try:
            means = self.hmm_model.means_  # Shape: (n_states, n_features)
            n_states = means.shape[0]
            
            # Feature indices (based on REGIME_FEATURES order)
            VIX_IDX = 0
            REALIZED_VOL_IDX = 1
            PCR_ZSCORE_IDX = 2
            PRICE_ROC_IDX = 3
            BREADTH_DIV_IDX = 4
            
            for state_id in range(n_states):
                vix_mean = means[state_id, VIX_IDX]
                roc_mean = means[state_id, PRICE_ROC_IDX]
                vol_mean = means[state_id, REALIZED_VOL_IDX]
                
                # Classification logic based on cluster characteristics
                if vix_mean > 25 and roc_mean < -0.5:
                    regime = 'HIGH_VOL_CRASH'
                elif vix_mean < 15 and abs(roc_mean) < 0.3:
                    regime = 'LOW_VOL_COMPRESSION'
                elif roc_mean > 0.5:
                    regime = 'TRENDING_UP'
                elif roc_mean < -0.5:
                    regime = 'TRENDING_DOWN'
                else:
                    regime = 'RANGE_BOUND'
                    
                self.regime_map[state_id] = regime
                
            logging.info(f"Inferred regime mapping: {self.regime_map}")
            
        except Exception as e:
            logging.warning(f"Could not infer regime mapping: {e}")

    def detect_regime(self, market_data: Dict[str, float]) -> str:
        """
        Detect current market regime.
        
        Args:
            market_data: Dictionary containing key features:
                - vix
                - realized_vol_5m
                - pcr_total_oi_zscore
                - price_roc_30m
                - breadth_divergence
                - adx (optional)
                
        Returns:
            str: One of [TRENDING_UP, TRENDING_DOWN, RANGE_BOUND, HIGH_VOL_CRASH, LOW_VOL_COMPRESSION]
        """
        # 1. Rule-Based Overrides (Strong signals that bypass HMM)
        vix = market_data.get('vix', 0.0)
        roc = market_data.get('price_roc_30m', 0.0)
        
        # Extreme conditions override HMM
        if vix > 30.0 and roc < -1.0:
            return 'HIGH_VOL_CRASH'
            
        if vix < 12.0 and abs(roc) < 0.2:
            return 'LOW_VOL_COMPRESSION'
            
        # 2. HMM Prediction with persisted mapping
        if self.hmm_model is not None and self.regime_map:
            try:
                # Prepare feature vector in correct order
                features = [market_data.get(f, 0.0) for f in self.feature_order]
                X = np.array(features).reshape(1, -1)
                hmm_state = int(self.hmm_model.predict(X)[0])
                
                # Use persisted mapping
                if hmm_state in self.regime_map:
                    return self.regime_map[hmm_state]
                else:
                    logging.warning(f"HMM state {hmm_state} not in regime_map, using heuristic")
                    
            except Exception as e:
                logging.warning(f"HMM prediction failed: {e}")
        
        # 3. Fallback Heuristics (when HMM not available or mapping failed)
        if roc > 0.5:
            return 'TRENDING_UP'
        elif roc < -0.5:
            return 'TRENDING_DOWN'
        else:
            return 'RANGE_BOUND'

    def get_strategy_weights(self, regime: str) -> Dict[str, float]:
        """
        Get strategy weights and risk scaling factor for the given regime.
        
        Returns:
            Dict: {
                'risk_scale': float (0.0 to 1.0),
                'trend_weight': float,
                'mean_reversion_weight': float
            }
        """
        config = {
            'TRENDING_UP': {
                'risk_scale': 1.0,
                'trend_weight': 0.8,
                'mean_reversion_weight': 0.2
            },
            'TRENDING_DOWN': {
                'risk_scale': 0.8, # Slightly reduced due to panic risk
                'trend_weight': 0.9,
                'mean_reversion_weight': 0.1
            },
            'RANGE_BOUND': {
                'risk_scale': 0.7, # Reduced size in chop
                'trend_weight': 0.2,
                'mean_reversion_weight': 0.8
            },
            'HIGH_VOL_CRASH': {
                'risk_scale': 0.3, # Capital preservation mode
                'trend_weight': 1.0, # Pure momentum (down)
                'mean_reversion_weight': 0.0 # Don't catch knives
            },
            'LOW_VOL_COMPRESSION': {
                'risk_scale': 0.5, # Wait for breakout
                'trend_weight': 0.5,
                'mean_reversion_weight': 0.5
            }
        }
        return config.get(regime, config['RANGE_BOUND'])

def calculate_adx(high, low, close, period=14):
    """Placeholder for ADX calculation if TA-Lib not available."""
    # Full ADX implementation requires history. 
    # For snapshot-based inference, we might need to pass computed ADX from handler.
    return 20.0 

def analyze_regimes(exchange: str):
    """Analyze HMM model to extract regime characteristics (Legacy wrapper)."""
    # This keeps backward compatibility with CLI
    detector = MarketRegimeDetector(exchange)
    # ... legacy analysis code ...
    pass 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze HMM regimes')
    parser.add_argument('--exchange', required=True, choices=['NSE', 'BSE'], 
                       help='Exchange to analyze')
    args = parser.parse_args()
    
    # Just instantiate to verify loading
    detector = MarketRegimeDetector(args.exchange)
    print("Regime Detector initialized successfully.")
