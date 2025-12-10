"""
Gamma Scalping Strategy (Intraday / Low Volatility).
"""
from typing import Dict, Any
from .base_strategy import BaseStrategy, TradeRecommendation

class GammaScalpingStrategy(BaseStrategy):
    """
    Intraday strategy exploiting Gamma in low volatility environments.
    Focuses on ATM options with high Gamma when IV is low, anticipating
    movement or scalping small delta changes.
    """
    
    def analyze(self, signal: Dict[str, Any], features: Dict[str, Any], market_state: Dict[str, Any]) -> TradeRecommendation:
        # Extract key metrics
        iv = features.get('iv', 0.0)
        iv_rank = features.get('iv_rank', 50.0) # Default to 50 if missing
        gamma = features.get('gamma_exposure', 0.0)
        
        # Sentiment Check
        ad_ratio = features.get('sentiment_ad_ratio_50', 1.0)
        sentiment_score = features.get('sentiment_score', 50.0)
        
        # Logic: Entry when IV is low (IV Rank < 30)
        is_low_vol = iv_rank < 30.0
        
        strategy_signal = "HOLD"
        confidence = 0.0
        rationale = []
        
        # Base signal from ML
        ml_signal = signal.get('signal', 'HOLD')
        ml_conf = signal.get('confidence', 0.0)
        
        if is_low_vol:
            rationale.append(f"Low IV Rank ({iv_rank:.1f}) favors Gamma Scalping")
            
            if ml_signal != "HOLD":
                # Confirm with Gamma checks if available
                strategy_signal = ml_signal
                confidence = ml_conf * 1.1 # Boost confidence in favorable regime
                rationale.append("ML Signal confirmed by Volatility Regime")
                
                # Sentiment Confirmation for Gamma Scalping
                # Extreme breadth (very high or very low) supports trend continuation/gamma expansion
                if ad_ratio > 2.0 or ad_ratio < 0.5:
                     confidence *= 1.1
                     rationale.append(f"Strong Breadth (A/D {ad_ratio:.2f}) supports momentum")
                elif 0.8 <= ad_ratio <= 1.2:
                     # Neutral breadth - good for range scalping, but maybe less directional conviction
                     rationale.append(f"Neutral Breadth (A/D {ad_ratio:.2f}) - Pure Scalp Mode")
                
                # In gamma scalping, we often want ATM or slightly OTM
                suggested_contract = "ATM"
            else:
                rationale.append("Waiting for directional catalyst")
        else:
            rationale.append(f"High IV Rank ({iv_rank:.1f}) unfavorable for long gamma")
            confidence = ml_conf * 0.8 # Penalty
            strategy_signal = ml_signal # Pass through
            
        return TradeRecommendation(
            signal=strategy_signal,
            confidence=min(confidence, 1.0),
            strategy_name="GammaScalping",
            rationale="; ".join(rationale),
            suggested_contract="ATM",
            metadata={
                'iv_rank': iv_rank,
                'regime': 'low_vol' if is_low_vol else 'high_vol',
                'sentiment_score': sentiment_score
            }
        )

