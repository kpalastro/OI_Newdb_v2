"""
Strategy router for dynamic model selection and strategy execution (Phase 2).

Routes trading signals through different model families (LightGBM, DL, RL)
and allocates them to specific trading strategies (Gamma Scalping, OI Buildup, etc.)
based on regime metadata and performance metrics.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

from ml_core import MLSignalGenerator

# Strategy Imports
try:
    from strategies.base_strategy import TradeRecommendation
    from strategies.gamma_scalping import GammaScalpingStrategy
    from strategies.oi_buildup import OIBuildupStrategy
    from strategies.vol_expansion import VolatilityExpansionStrategy
    from strategies.expiry_pin import ExpiryPinStrategy
    STRATEGIES_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).warning(f"Strategy imports failed: {e}")
    STRATEGIES_AVAILABLE = False
    TradeRecommendation = None

try:
    from models.deep_learning import DeepLearningPredictor
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    DeepLearningPredictor = None

try:
    from models.reinforcement_learning import RLStrategy, RLState
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    RLStrategy = None
    RLState = None

LOGGER = logging.getLogger(__name__)


@dataclass
class StrategySignal:
    """Unified signal output from strategy router."""
    signal: str  # BUY, SELL, HOLD
    confidence: float
    source: str  # 'lightgbm', 'dl', 'rl', 'ensemble'
    rationale: str
    metadata: Dict[str, Any]


class AdvancedStrategyRouter:
    """
    Dynamic strategy router that selects the best model and executes specific 
    trading strategies based on regime and performance.
    """
    
    def __init__(self, exchange: str):
        self.exchange = exchange
        self.lightgbm_predictor: Optional[MLSignalGenerator] = None
        self.dl_predictor: Optional[DeepLearningPredictor] = None
        self.rl_strategy: Optional[RLStrategy] = None
        
        # Strategies
        if STRATEGIES_AVAILABLE:
            self.strategies = {
                'gamma_scalping': GammaScalpingStrategy(),
                'oi_buildup': OIBuildupStrategy(),
                'vol_expansion': VolatilityExpansionStrategy(),
                'expiry_pin': ExpiryPinStrategy()
            }
        else:
            self.strategies = {}

        # Performance tracking
        self.model_performance: Dict[str, Dict[str, float]] = {
            'lightgbm': {'accuracy': 0.58, 'sharpe': 0.0},
            'dl': {'accuracy': 0.0, 'sharpe': 0.0},
            'rl': {'accuracy': 0.0, 'sharpe': 0.0},
        }
        
        # Routing rules
        self.routing_mode = 'adaptive'  # 'lightgbm', 'dl', 'rl', 'ensemble', 'adaptive'
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize all available model types."""
        # LightGBM (always available)
        try:
            self.lightgbm_predictor = MLSignalGenerator(self.exchange)
            if not self.lightgbm_predictor.models_loaded:
                LOGGER.warning(f"[{self.exchange}] LightGBM models not loaded")
        except Exception as e:
            LOGGER.error(f"[{self.exchange}] LightGBM initialization failed: {e}")
        
        # Deep Learning (optional)
        if DL_AVAILABLE:
            try:
                self.dl_predictor = DeepLearningPredictor(self.exchange, model_type="lstm")
                if not self.dl_predictor.model_loaded:
                    LOGGER.debug(f"[{self.exchange}] Deep learning models not available")
            except Exception as e:
                LOGGER.debug(f"[{self.exchange}] Deep learning initialization skipped: {e}")
        
        # Reinforcement Learning (optional)
        if RL_AVAILABLE:
            try:
                self.rl_strategy = RLStrategy(self.exchange, algorithm="PPO")
                if not self.rl_strategy.model_loaded:
                    LOGGER.debug(f"[{self.exchange}] RL models not available")
            except Exception as e:
                LOGGER.debug(f"[{self.exchange}] RL initialization skipped: {e}")
    
    def generate_signal(
        self,
        features_dict: Dict[str, Any],
        feature_sequence: Optional[Any] = None,  # np.ndarray for DL
        state: Optional[RLState] = None,  # For RL
    ) -> StrategySignal:
        """
        Generate raw ML signal using the selected routing strategy (LightGBM/DL/RL).
        
        Args:
            features_dict: Feature dictionary for LightGBM
            feature_sequence: Optional sequence for deep learning (shape: seq_len, features)
            state: Optional RL state
        
        Returns:
            StrategySignal with unified output
        """
        signals = []
        
        # LightGBM signal
        if self.lightgbm_predictor and self.lightgbm_predictor.models_loaded:
            try:
                signal, confidence, rationale, metadata = self.lightgbm_predictor.generate_signal(features_dict)
                signals.append({
                    'signal': signal,
                    'confidence': confidence,
                    'source': 'lightgbm',
                    'rationale': rationale,
                    'metadata': metadata,
                })
            except Exception as e:
                LOGGER.debug(f"[{self.exchange}] LightGBM signal generation failed: {e}")
        
        # Deep Learning signal
        if self.dl_predictor and self.dl_predictor.model_loaded and feature_sequence is not None:
            try:
                signal_class, confidence = self.dl_predictor.predict(feature_sequence)
                signal_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
                signal = signal_map.get(signal_class, 'HOLD')
                signals.append({
                    'signal': signal,
                    'confidence': confidence,
                    'source': 'dl',
                    'rationale': f'DL {self.dl_predictor.model_type.upper()} prediction',
                    'metadata': {'model_type': self.dl_predictor.model_type},
                })
            except Exception as e:
                LOGGER.debug(f"[{self.exchange}] DL signal generation failed: {e}")
        
        # RL signal
        if self.rl_strategy and self.rl_strategy.model_loaded and state is not None:
            try:
                action = self.rl_strategy.predict(state.features)
                signal_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
                signal = signal_map.get(action.signal, 'HOLD')
                signals.append({
                    'signal': signal,
                    'confidence': abs(action.position_size),  # Use position size as confidence proxy
                    'source': 'rl',
                    'rationale': f'RL {self.rl_strategy.algorithm} action',
                    'metadata': {'position_size': action.position_size},
                })
            except Exception as e:
                LOGGER.debug(f"[{self.exchange}] RL signal generation failed: {e}")
        
        # Route based on mode
        if not signals:
            return StrategySignal(
                signal='HOLD',
                confidence=0.0,
                source='none',
                rationale='No models available',
                metadata={}
            )
        
        if self.routing_mode == 'lightgbm':
            selected = next((s for s in signals if s['source'] == 'lightgbm'), signals[0])
        elif self.routing_mode == 'dl':
            selected = next((s for s in signals if s['source'] == 'dl'), signals[0])
        elif self.routing_mode == 'rl':
            selected = next((s for s in signals if s['source'] == 'rl'), signals[0])
        elif self.routing_mode == 'ensemble':
            selected = self._ensemble_vote(signals)
        else:  # adaptive
            selected = self._adaptive_select(signals, features_dict)
        
        return StrategySignal(
            signal=selected['signal'],
            confidence=selected['confidence'],
            source=selected['source'],
            rationale=selected['rationale'],
            metadata=selected['metadata']
        )

    def route_strategy(self, signal: StrategySignal, features: Dict[str, Any]) -> TradeRecommendation:
        """
        Refine the raw ML signal by routing it to a specific trading strategy logic.
        
        Args:
            signal: The raw strategy signal from generate_signal
            features: The feature set used
            
        Returns:
            TradeRecommendation with specific strategy details
        """
        if not STRATEGIES_AVAILABLE or signal.signal == 'HOLD':
             return TradeRecommendation(
                signal=signal.signal,
                confidence=signal.confidence,
                strategy_name="ML_Base",
                rationale=signal.rationale,
                suggested_contract="ATM",
                metadata=signal.metadata
            )

        # Extract context
        horizon = signal.metadata.get('horizon', 'intraday')
        regime = features.get('regime', 'neutral')
        # Some fallback logic for regime if not in features (maybe stored in signal metadata)
        if 'regime' in signal.metadata:
            regime = signal.metadata['regime']

        # Routing Logic
        selected_strategy = None
        
        if horizon == 'expiry':
            selected_strategy = self.strategies.get('expiry_pin')
            
        elif regime == 'low_vol' and horizon == 'intraday':
            selected_strategy = self.strategies.get('gamma_scalping')
            
        elif regime in ['trending_up', 'trending_down'] or horizon == 'swing':
            selected_strategy = self.strategies.get('oi_buildup')
            
        elif features.get('is_squeeze', False) or features.get('bb_width', 1.0) < 0.15:
            # Detect squeeze condition implicitly if not passed
            selected_strategy = self.strategies.get('vol_expansion')
            
        # Fallback / Default
        if selected_strategy:
            # signal is StrategySignal object, convert to dict-like for strategy.analyze
            signal_dict = {
                'signal': signal.signal, 
                'confidence': signal.confidence, 
                'source': signal.source
            }
            return selected_strategy.analyze(signal_dict, features, {})
        else:
            # Return original as base recommendation
             return TradeRecommendation(
                signal=signal.signal,
                confidence=signal.confidence,
                strategy_name="ML_Base_Fallback",
                rationale=signal.rationale,
                suggested_contract="ATM",
                metadata=signal.metadata
            )
    
    def _ensemble_vote(self, signals: list) -> Dict[str, Any]:
        """Vote-based ensemble of all available signals."""
        buy_votes = sum(1 for s in signals if s['signal'] == 'BUY')
        sell_votes = sum(1 for s in signals if s['signal'] == 'SELL')
        hold_votes = sum(1 for s in signals if s['signal'] == 'HOLD')
        
        # Weight by confidence
        buy_weight = sum(s['confidence'] for s in signals if s['signal'] == 'BUY')
        sell_weight = sum(s['confidence'] for s in signals if s['signal'] == 'SELL')
        
        if buy_weight > sell_weight and buy_weight > 0.5:
            signal = 'BUY'
            confidence = buy_weight / len(signals) if signals else 0.0
        elif sell_weight > buy_weight and sell_weight > 0.5:
            signal = 'SELL'
            confidence = sell_weight / len(signals) if signals else 0.0
        else:
            signal = 'HOLD'
            confidence = 0.0
        
        return {
            'signal': signal,
            'confidence': confidence,
            'source': 'ensemble',
            'rationale': f'Ensemble: {buy_votes}B/{sell_votes}S/{hold_votes}H',
            'metadata': {'votes': {'buy': buy_votes, 'sell': sell_votes, 'hold': hold_votes}},
        }
    
    def _adaptive_select(self, signals: list, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adaptively select model based on regime and performance.
        Currently defaults to LightGBM if available, falls back to others.
        """
        # Simple adaptive logic: prefer LightGBM, fall back to others
        regime = features_dict.get('vix', 0.0)
        
        # High volatility: prefer DL or RL
        if regime > 25.0:
            for s in signals:
                if s['source'] in ['dl', 'rl']:
                    return s
        
        # Default to LightGBM
        for s in signals:
            if s['source'] == 'lightgbm':
                return s
        
        # Fallback to first available
        return signals[0] if signals else {
            'signal': 'HOLD',
            'confidence': 0.0,
            'source': 'none',
            'rationale': 'No signals available',
            'metadata': {}
        }
    
    def update_performance(self, source: str, accuracy: float, sharpe: float) -> None:
        """Update model performance metrics for adaptive routing."""
        if source in self.model_performance:
            self.model_performance[source]['accuracy'] = accuracy
            self.model_performance[source]['sharpe'] = sharpe


# Backward compatibility alias
StrategyRouter = AdvancedStrategyRouter
