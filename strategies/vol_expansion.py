"""
Volatility Expansion Strategy (Pre-event / Squeeze).
"""
from typing import Dict, Any
from .base_strategy import BaseStrategy, TradeRecommendation

class VolatilityExpansionStrategy(BaseStrategy):
    """
    Strategy targeting volatility expansion events (Squeezes).
    Uses Bollinger Band compression and IV Rank to identify
    potential explosive moves.
    """
    
    def analyze(self, signal: Dict[str, Any], features: Dict[str, Any], market_state: Dict[str, Any]) -> TradeRecommendation:
        # Features needed: BB Width, IV Rank, Historical Vol
        bb_width = features.get('bb_width', 1.0)
        iv_rank = features.get('iv_rank', 50.0)
        
        # Squeeze logic: Low BB Width often precedes expansion
        is_squeeze = bb_width < 0.10 # Threshold for 'tight' bands, example value
        
        # Expecting expansion if IV is relatively low compared to Hist Vol?
        # Or if IV is rising from a bottom.
        
        ml_signal = signal.get('signal', 'HOLD')
        confidence = signal.get('confidence', 0.0)
        rationale = []
        strategy_signal = "HOLD"
        
        if is_squeeze:
            rationale.append("Bollinger Squeeze detected")
            
            if iv_rank < 20:
                # Long Volatility Setup (Long Straddle/Strangle)
                # Cheap options before the move
                strategy_signal = "BUY" # In this context, BUY means "Enter Long Vol Position"
                confidence = max(confidence, 0.6) # Base confidence from structural setup
                rationale.append(f"Low IV Rank ({iv_rank:.1f}) - Potential Long Vol play")
                suggested_contract = "STRADDLE" # Special type
                
            elif ml_signal != "HOLD":
                # Directional breakout likely
                strategy_signal = ml_signal
                confidence *= 1.2
                rationale.append("Directional breakout from squeeze")
                suggested_contract = "ATM"
            else:
                rationale.append("Squeeze detected but no trigger")
                
        else:
            # No squeeze, standard operation or mean reversion if BB width is huge
            if bb_width > 0.5: # Overextended
                rationale.append("Volatility Extended (Wide BB)")
                # potentially look for mean reversion if signal supports it
            
            strategy_signal = ml_signal
            suggested_contract = "ATM"
            
        return TradeRecommendation(
            signal=strategy_signal,
            confidence=min(confidence, 1.0),
            strategy_name="VolExpansion",
            rationale="; ".join(rationale),
            suggested_contract=suggested_contract if 'suggested_contract' in locals() else "ATM",
            metadata={
                'bb_width': bb_width,
                'is_squeeze': is_squeeze
            }
        )

