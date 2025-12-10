import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import torch
except ImportError:
    torch = None

try:
    import joblib
except ImportError:
    joblib = None

from models.intraday_lstm import IntradayLSTMModel
from models.swing_ensemble import SwingTradingEnsemble
from models.expiry_transformer import ExpiryDayTransformer
from models.horizon_router import HorizonRouter

LOGGER = logging.getLogger(__name__)


class MultiHorizonEnsemble:
    """
    Unified interface for multi-horizon predictions.
    Delegates to specialized models based on the HorizonRouter.
    """
    def __init__(self, exchange: str):
        self.exchange = exchange
        self.router = HorizonRouter()
        self.weights_loaded = False
        
        # Initialize Sub-Models
        try:
            self.intraday_model = IntradayLSTMModel()
        except ImportError:
            LOGGER.warning("IntradayLSTMModel unavailable (missing torch).")
            self.intraday_model = None
            
        self.swing_model = SwingTradingEnsemble()
        
        try:
            self.expiry_model = ExpiryDayTransformer()
        except ImportError:
            LOGGER.warning("ExpiryDayTransformer unavailable (missing torch).")
            self.expiry_model = None
        
        self._load_weights()

    def _load_weights(self):
        """
        Load pre-trained weights for all models from disk.
        
        Expected file structure:
            models/{exchange}/intraday_lstm.pt      - PyTorch LSTM weights
            models/{exchange}/expiry_transformer.pt - PyTorch Transformer weights
            models/{exchange}/swing_ensemble.pkl    - Joblib serialized ensemble
        """
        model_dir = Path(f"models/{self.exchange}")
        
        if not model_dir.exists():
            LOGGER.info(f"Model directory {model_dir} not found. Using untrained models.")
            return
        
        loaded_count = 0
        
        # 1. Load LSTM weights (PyTorch)
        lstm_path = model_dir / "intraday_lstm.pt"
        if lstm_path.exists() and self.intraday_model is not None and torch is not None:
            try:
                state_dict = torch.load(lstm_path, map_location='cpu')
                self.intraday_model.load_state_dict(state_dict)
                self.intraday_model.eval()  # Set to evaluation mode
                LOGGER.info(f"✓ Loaded LSTM weights from {lstm_path}")
                loaded_count += 1
            except Exception as e:
                LOGGER.warning(f"Failed to load LSTM weights: {e}")
        
        # 2. Load Transformer weights (PyTorch)
        transformer_path = model_dir / "expiry_transformer.pt"
        if transformer_path.exists() and self.expiry_model is not None and torch is not None:
            try:
                state_dict = torch.load(transformer_path, map_location='cpu')
                self.expiry_model.load_state_dict(state_dict)
                self.expiry_model.eval()  # Set to evaluation mode
                LOGGER.info(f"✓ Loaded Transformer weights from {transformer_path}")
                loaded_count += 1
            except Exception as e:
                LOGGER.warning(f"Failed to load Transformer weights: {e}")
        
        # 3. Load Swing Ensemble (Joblib - XGBoost/LightGBM)
        swing_path = model_dir / "swing_ensemble.pkl"
        if swing_path.exists() and self.swing_model is not None and joblib is not None:
            try:
                saved_data = joblib.load(swing_path)
                
                # Handle different save formats
                if isinstance(saved_data, dict):
                    # Format: {'models': {...}, 'weights': {...}, '_is_fitted': True}
                    if 'models' in saved_data:
                        self.swing_model.models = saved_data['models']
                    if 'weights' in saved_data:
                        self.swing_model.weights = saved_data['weights']
                    self.swing_model._is_fitted = saved_data.get('_is_fitted', True)
                elif isinstance(saved_data, SwingTradingEnsemble):
                    # Direct object serialization
                    self.swing_model = saved_data
                    
                LOGGER.info(f"✓ Loaded Swing Ensemble from {swing_path}")
                loaded_count += 1
            except Exception as e:
                LOGGER.warning(f"Failed to load Swing Ensemble: {e}")
        
        self.weights_loaded = loaded_count > 0
        
        if loaded_count > 0:
            LOGGER.info(f"Model loading complete: {loaded_count} model(s) loaded for {self.exchange}")
        else:
            LOGGER.info(f"No pre-trained weights found for {self.exchange}. Models will use default initialization.")

    def predict(self, features: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main prediction entry point.
        
        Args:
            features: Feature vector (DataFrame/Array) for Swing, or Sequence for LSTM.
                      Ideally, this is a tuple (feature_vector, feature_sequence) or similar.
                      For compatibility, we assume ml_core passes a dict with keys 'vector' and 'sequence'.
            context: Additional context (strategy mode, time, time_to_expiry_hours).
            
        Returns:
            Dict containing signal, probabilities, and used horizon.
        """
        # Unwrap input if it's our special struct
        feature_vector = features
        feature_sequence = None
        
        if isinstance(features, dict) and 'vector' in features:
            feature_vector = features['vector']
            feature_sequence = features.get('sequence')
            
        # Determine horizon
        # We need a dict for determination, if feature_vector is DF, convert first row to dict
        feature_dict = {}
        if hasattr(feature_vector, 'iloc'):
            feature_dict = feature_vector.iloc[0].to_dict()
        elif context:
            feature_dict = context # Fallback
            
        horizon = self.router.determine_horizon(feature_dict, context)
        
        result = {
            'horizon': horizon,
            'signal': 'HOLD',
            'confidence': 0.0,
            'probabilities': [0.0, 1.0, 0.0] # [SELL, HOLD, BUY] default
        }
        
        try:
            # 1. Intraday (LSTM)
            if horizon == 'intraday':
                if self.intraday_model and feature_sequence is not None:
                    # Expect sequence to be (seq_len, features)
                    # Convert to tensor if needed inside predict_single
                    pred = self.intraday_model.predict_single(feature_sequence)
                    result.update({
                        'signal': ['SELL', 'HOLD', 'BUY'][pred['class']], # Assuming 0=Sell, 1=Hold, 2=Buy? Check mapping
                        'probabilities': pred['probabilities'],
                        'confidence': max(pred['probabilities'])
                    })
                    return result
                else:
                    # Fallback to swing if model missing or sequence missing
                    logging.debug("Intraday fallback to swing (model or sequence missing)")
                    horizon = 'swing'
                    result['horizon'] = 'swing'
            
            # 2. Expiry (Transformer)
            if horizon == 'expiry':
                if self.expiry_model:
                    # Convert to array if DF
                    if hasattr(feature_vector, 'values'):
                        fv = feature_vector.values
                    else:
                        fv = np.array(feature_vector)
                        
                    if fv.ndim == 2:
                        fv = fv[0] # Take first row if batch
                        
                    try:
                        pred = self.expiry_model.predict(fv)
                        result.update({
                            'signal': ['SELL', 'HOLD', 'BUY'][pred['direction']],
                            'probabilities': pred['probabilities'],
                            'confidence': max(pred['probabilities'])
                        })
                    except RuntimeError as re:
                        # Catch shape mismatches (e.g. 117 features vs 64 expected)
                        logging.warning(f"Expiry model shape mismatch (safely falling back to swing): {re}")
                        horizon = 'swing'
                        result['horizon'] = 'swing'
                    except Exception as e:
                        logging.error(f"Expiry model prediction failed: {e}")
                        horizon = 'swing'
                        result['horizon'] = 'swing'
                else:
                    horizon = 'swing'
                    result['horizon'] = 'swing'
                    
            # 3. Swing (Ensemble)
            if horizon == 'swing':
                if self.swing_model:
                    pred = self.swing_model.predict(feature_vector)
                    # Mapping: 0=SELL, 1=HOLD, 2=BUY is standard for our pipeline?
                    # Check MLSignalGenerator logic: {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
                    # Usually classifiers output 0, 1, 2 indices. 
                    # Let's map 0->SELL, 1->HOLD, 2->BUY for now, need verification with train script.
                    classes = ['SELL', 'HOLD', 'BUY'] 
                    result.update({
                        'signal': classes[pred['class']],
                        'probabilities': pred['probabilities'],
                        'confidence': max(pred['probabilities'])
                    })
                
        except Exception as e:
            logging.error(f"Error in MultiHorizonEnsemble predict ({horizon}): {e}", exc_info=True)
            
        return result

