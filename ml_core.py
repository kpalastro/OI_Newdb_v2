"""
ml_core.py

Runtime ML signal engine for OI Gemini.
Integrates Multi-Horizon Models (Phase 2) and Enhanced Regime Detection (Phase 5).
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

import database_new as db
from risk_manager import get_optimal_position_size
from time_utils import now_ist
from models.multi_horizon_ensemble import MultiHorizonEnsemble
from regime_analysis import MarketRegimeDetector
from feature_engineering import REQUIRED_FEATURE_COLUMNS

# ITM Feature Evaluator (optional - only import if available)
try:
    from utils.itm_feature_evaluator import evaluate_itm_features, create_itm_optimal_range_features
    ITM_EVALUATOR_AVAILABLE = True
except ImportError:
    ITM_EVALUATOR_AVAILABLE = False
    logging.warning("ITM feature evaluator not available. ITM enforcement disabled.")

SIGNAL_MAP = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}


class MLSignalGenerator:
    """
    Core ML engine that loads models and generates trading signals.
    """
    def __init__(self, exchange: str, use_swing_ensemble: Optional[bool] = None):
        self.exchange = exchange
        
        # Phase 2: Multi-Horizon Ensemble
        # Load config if not explicitly provided
        if use_swing_ensemble is None:
            try:
                from config import get_config
                config = get_config()
                use_swing_ensemble = config.use_swing_ensemble
            except Exception:
                use_swing_ensemble = True  # Default to True for backward compatibility
        
        self.model_ensemble = MultiHorizonEnsemble(exchange, use_swing_ensemble=use_swing_ensemble)
        self.models_loaded = True 

        # Phase 5: Enhanced Regime Detector
        self.regime_detector = MarketRegimeDetector(exchange)
        
        self.strategy_metrics = {
            'win_rate': 0.58,
            'avg_w_l_ratio': 1.55,
        }
        
        # State tracking
        self.signal_history = deque(maxlen=50)
        self.feedback_window = deque(maxlen=100)
        self.accuracy_history = deque(maxlen=50)
        self.pending_predictions: Dict[str, Dict[str, Any]] = {}
        self.degrade_threshold = 0.55
        self.needs_retrain = False
        self.last_feedback_timestamp: Optional[datetime] = None
        self.signal_sequence = 0
        
        # Rolling buffer for Sequence Models (if needed by ensemble)
        self.regime_feature_buffer: deque = deque(maxlen=60)

        # Load feature definition to ensure correct ordering
        self.feature_columns = self._load_feature_definition()

    def _load_feature_definition(self) -> List[str]:
        """Load the exact feature list used during training."""
        try:
            feature_path = Path("models") / self.exchange / "model_features.pkl"
            if feature_path.exists():
                features = joblib.load(feature_path)
                if isinstance(features, list) and len(features) > 0:
                    logging.info(f"[{self.exchange}] Loaded {len(features)} feature definitions.")
                    return features
        except Exception as e:
            logging.warning(f"[{self.exchange}] Failed to load feature definition: {e}")
        
        logging.warning(f"[{self.exchange}] Using default feature definition (may mismatch model).")
        return REQUIRED_FEATURE_COLUMNS

    def generate_signal(self, features_dict: Dict[str, Any]) -> Tuple[str, float, str, Dict]:
        """
        Generate a trading signal based on input features.
        """
        try:
            # 1. Detect Regime (Phase 5)
            # We pass the raw feature dict; the detector handles extraction
            current_regime = self.regime_detector.detect_regime(features_dict)
            regime_config = self.regime_detector.get_strategy_weights(current_regime)
            
            # 2. Prepare Features for Ensemble
            # Ensure features are ordered exactly as model expects
            ordered_features = [features_dict.get(col, 0.0) for col in self.feature_columns]
            
            # Create DataFrame to preserve feature names (fixes LightGBM warnings)
            # Reshape is handled by DataFrame constructor (single row)
            feature_df = pd.DataFrame([ordered_features], columns=self.feature_columns)
            
            # For sequence models, update buffer
            self.regime_feature_buffer.append(ordered_features)
            
            ensemble_input = {
                'vector': feature_df, # Pass DataFrame instead of numpy array
                'sequence': np.array(list(self.regime_feature_buffer)) if len(self.regime_feature_buffer) > 0 else None,
                'time_to_expiry_hours': features_dict.get('time_to_expiry_hours', 24.0)
            }
            
            # 3. Get Prediction from Ensemble (Phase 2)
            ensemble_result = self.model_ensemble.predict(ensemble_input)
            
            signal = ensemble_result.get('signal', 'HOLD')
            confidence = ensemble_result.get('confidence', 0.0)
            horizon = ensemble_result.get('horizon', 'unknown')
            probabilities = ensemble_result.get('probabilities', [0.0, 0.0, 0.0]) # Sell, Hold, Buy
            
            # --- ITM Feature Enforcement (NEW) ---
            # Evaluate ITM features and apply filters/multipliers based on reverse engineering
            itm_evaluation = None
            if ITM_EVALUATOR_AVAILABLE:
                try:
                    # Add ITM optimal range features to features_dict for evaluation
                    itm_optimal_features = create_itm_optimal_range_features(features_dict)
                    features_dict_with_optimal = {**features_dict, **itm_optimal_features}
                    
                    itm_evaluation = evaluate_itm_features(features_dict_with_optimal)
                    
                    # Apply ITM skip filter (CRITICAL)
                    if itm_evaluation.get('should_skip_trade', False):
                        signal = 'HOLD'
                        confidence = 0.0
                        rationale = f"ITM Filter: {', '.join(itm_evaluation.get('warnings', []))}"
                        metadata = {
                            'regime': current_regime,
                            'horizon': horizon,
                            'itm_filter_applied': True,
                            'itm_skip_reason': itm_evaluation.get('warnings', []),
                            'itm_score': itm_evaluation.get('itm_score', 0.0)
                        }
                        return signal, confidence, rationale, metadata
                    
                    # Apply ITM confidence multiplier
                    confidence = min(0.95, confidence * itm_evaluation.get('confidence_multiplier', 1.0))
                    
                    # Apply validated BEARISH/BULLISH signals (Chart Correlation Analysis)
                    # BEARISH signal (CE>PE, CE+, PE-) = contrarian bearish (predicts decline)
                    # BULLISH signal (PE>CE, PE+, CE-) = contrarian bullish (predicts rise)
                    if itm_evaluation.get('peak_detection', False):
                        # Peak detection - very strong bearish signal
                        if signal == 'BUY':
                            signal = 'HOLD'  # Skip long trades at peaks
                            confidence = 0.0
                            rationale = f"Peak BEARISH Signal: {', '.join(itm_evaluation.get('warnings', []))}"
                    elif itm_evaluation.get('bearish_signal', False):
                        # BEARISH signal - reduce confidence for long trades
                        if signal == 'BUY':
                            confidence = confidence * 0.7  # Reduce confidence for long
                            rationale += f" | BEARISH Signal (CE>PE, CE+, PE-)"
                        elif signal == 'SELL':
                            confidence = min(0.95, confidence * 1.1)  # Boost confidence for short
                            rationale += f" | BEARISH Signal (CE>PE, CE+, PE-)"
                    elif itm_evaluation.get('bullish_signal', False):
                        # BULLISH signal - boost confidence for long trades
                        if signal == 'BUY':
                            confidence = min(0.95, confidence * 1.15)  # Boost confidence for long
                            rationale += f" | BULLISH Signal (PE>CE, PE+, CE-)"
                        elif signal == 'SELL':
                            confidence = confidence * 0.8  # Reduce confidence for short
                            rationale += f" | BULLISH Signal (PE>CE, PE+, CE-)"
                    
                except Exception as e:
                    logging.warning(f"[{self.exchange}] ITM evaluation failed: {e}")
                    itm_evaluation = None
            
            # --- ITM OI Divergence Boost (Rule-Based) ---
            # Explicitly boost signal confidence if ITM OI shows strong directional divergence
            # This handles the specific "Red Market" scenario where models might be conservative.
            
            rationale = f"Horizon {horizon}"  # Initialize base rationale
            
            bearish_div = features_dict.get('itm_oi_bearish_divergence', 0.0)
            bullish_div = features_dict.get('itm_oi_bullish_divergence', 0.0)
            eod_winding = features_dict.get('eod_position_winding', 0.0)
            
            # Threshold for divergence action importance (experimentally set to 0.5)
            DIVERGENCE_THRESHOLD = 0.5
            
            # 1. Bearish Boost: ITM Calls Increasing + ITM Puts Decreasing
            if bearish_div > DIVERGENCE_THRESHOLD:
                if signal in ['HOLD', 'SELL']:
                    if signal == 'HOLD':
                        signal = 'SELL'
                        confidence = min(0.6, confidence + 0.3)
                        rationale = f"Boosted to SELL by Bearish ITM Divergence ({bearish_div:.2f})."
                    else:
                        confidence = min(0.95, confidence + 0.15)
                        rationale = f"{rationale} | Bearish ITM Boost (+0.15)."

            # 2. Bullish Boost: ITM Puts Increasing + ITM Calls Decreasing
            elif bullish_div > DIVERGENCE_THRESHOLD:
                if signal in ['HOLD', 'BUY']:
                    if signal == 'HOLD':
                        signal = 'BUY'
                        confidence = min(0.6, confidence + 0.3)
                        rationale = f"Boosted to BUY by Bullish ITM Divergence ({bullish_div:.2f})."
                    else:
                        confidence = min(0.95, confidence + 0.15)
                        rationale = f"{rationale} | Bullish ITM Boost (+0.15)."

            # 3. EOD Winding Suppression: Both decreasing after 14:30
            if eod_winding > 0.3:
                # Reduce confidence as this is likely just position squaring, not directional conviction
                confidence = confidence * 0.7
                rationale = f"{rationale} | EOD Winding (-30% conf)."

            # 4. Regime Adjustment (Phase 5)
            # Downgrade signal if regime is hostile
            if current_regime == 'HIGH_VOL_CRASH' and signal == 'BUY':
                signal = 'HOLD'
                confidence = 0.0
                rationale = "Signal suppressed by High Vol Crash regime."
            elif current_regime == 'LOW_VOL_COMPRESSION' and confidence < 0.7:
                 # Filter weak signals in chop
                 signal = 'HOLD' 
                 rationale = "Weak signal filtered in Low Vol regime."
            else:
                 if 'Boost' not in rationale and 'Winding' not in rationale:
                    rationale = f"Horizon {horizon} | Regime {current_regime} | Conf {confidence:.1%} | Signal {signal}"

            # 5. Risk Sizing
            risk_payload = {'fraction': 0.0, 'recommended_lots': 0, 'kelly_fraction': 0.0}
            if signal != 'HOLD':
                current_vol = float(features_dict.get('vix', 20.0)) / 100.0
                
                # Get ITM position multiplier if available
                itm_position_mult = 1.0
                if itm_evaluation:
                    itm_position_mult = itm_evaluation.get('position_size_multiplier', 1.0)
                
                risk_payload = get_optimal_position_size(
                    ml_confidence=confidence,
                    win_rate=self.strategy_metrics['win_rate'],
                    avg_win_loss_ratio=self.strategy_metrics['avg_w_l_ratio'],
                    current_volatility=current_vol,
                    regime_risk_scale=regime_config['risk_scale'],  # Phase 5 Scaling
                    itm_position_multiplier=itm_position_mult  # NEW: ITM-based position sizing
                )

            metadata = {
                'regime': current_regime,
                'horizon': horizon,
                'buy_prob': probabilities[2] if len(probabilities) > 2 else 0.0,
                'sell_prob': probabilities[0] if len(probabilities) > 0 else 0.0,
                'position_size_frac': risk_payload.get('fraction', 0.0),
                'kelly_fraction': risk_payload.get('kelly_fraction', 0.0),
                'recommended_lots': risk_payload.get('recommended_lots', 0),
                'confidence': confidence,
                'regime_risk_scale': regime_config['risk_scale'],
                'rolling_accuracy': self._rolling_accuracy(),
                'last_feedback_at': self.last_feedback_timestamp.isoformat() if self.last_feedback_timestamp else None,
            }
            
            # Add ITM evaluation metadata if available
            if itm_evaluation:
                metadata.update({
                    'itm_score': itm_evaluation.get('itm_score', 0.0),
                    'itm_confidence_multiplier': itm_evaluation.get('confidence_multiplier', 1.0),
                    'itm_position_multiplier': itm_evaluation.get('position_size_multiplier', 1.0),
                    'itm_reasons': itm_evaluation.get('reasons', []),
                    'itm_warnings': itm_evaluation.get('warnings', []),
                    'itm_raw_values': itm_evaluation.get('raw_values', {}),
                    # Validated CE/PE signals
                    'itm_bearish_signal': itm_evaluation.get('bearish_signal', False),
                    'itm_bullish_signal': itm_evaluation.get('bullish_signal', False),
                    'itm_peak_detection': itm_evaluation.get('peak_detection', False)
                })
                
                # Add ITM reasons to rationale
                if itm_evaluation.get('reasons'):
                    rationale += f" | ITM: {', '.join(itm_evaluation['reasons'][:2])}"

            self.signal_history.append({'signal': signal, 'confidence': confidence, 'regime': current_regime})
            metadata['signal_history'] = list(self.signal_history)[-5:]

            signal_id = self._register_prediction(signal, metadata.get('buy_prob'), metadata.get('sell_prob'))
            metadata['signal_id'] = signal_id

            return signal, confidence, rationale, metadata

        except Exception as err:
            logging.error(f"[{self.exchange}] Error during signal generation: {err}", exc_info=True)
            return 'HOLD', 0.0, 'Error during inference.', {}

    def predict_and_learn(self, features_dict: Dict[str, Any], actual_outcome: Optional[int] = None):
        """Single entry point for prediction with optional immediate feedback."""
        signal, confidence, rationale, metadata = self.generate_signal(features_dict)
        if actual_outcome is not None and metadata.get('signal_id'):
            self.record_feedback(metadata['signal_id'], actual_outcome)
        return signal, confidence, rationale, metadata

    def record_feedback(self, signal_id: str, actual_outcome: int) -> Optional[Dict[str, Any]]:
        """Record realised outcome for a prior prediction to update rolling accuracy."""
        if signal_id not in self.pending_predictions:
            logging.warning("[%s] Feedback received for unknown signal id %s", self.exchange, signal_id)
            return None

        predicted = self.pending_predictions.pop(signal_id)
        predicted_direction = predicted.get('direction', 0)
        
        success = 1.0 if actual_outcome == predicted_direction else 0.0
        self.feedback_window.append(success)
        rolling_accuracy = self._rolling_accuracy()
        self.accuracy_history.append(rolling_accuracy)
        self.last_feedback_timestamp = now_ist()
        
        degrade_triggered = (
            len(self.feedback_window) == self.feedback_window.maxlen
            and rolling_accuracy < self.degrade_threshold
        )
        self.needs_retrain = degrade_triggered

        summary = {
            'exchange': self.exchange,
            'signal_id': signal_id,
            'rolling_accuracy': rolling_accuracy,
            'degrade_triggered': degrade_triggered,
        }
        return summary

    def _register_prediction(self, signal: str, buy_prob: float | None, sell_prob: float | None) -> str:
        """Store latest prediction metadata for future feedback correlation."""
        self.signal_sequence += 1
        signal_id = f"{self.exchange}-{int(time.time() * 1000)}-{self.signal_sequence}"
        direction = 1 if signal == 'BUY' else -1 if signal == 'SELL' else 0
        
        self.pending_predictions[signal_id] = {
            'direction': direction,
            'timestamp': now_ist().isoformat(),
            'buy_prob': buy_prob,
            'sell_prob': sell_prob,
        }
        if len(self.pending_predictions) > 500:
            stale_keys = list(self.pending_predictions.keys())[:-500]
            for key in stale_keys:
                self.pending_predictions.pop(key, None)
        return signal_id

    def _rolling_accuracy(self) -> float:
        if not self.feedback_window:
            return 0.0
        return float(sum(self.feedback_window) / len(self.feedback_window))
