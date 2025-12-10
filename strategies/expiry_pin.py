"""
Expiry Pin Strategy (0-DTE).
"""
from typing import Dict, Any
from .base_strategy import BaseStrategy, TradeRecommendation

class ExpiryPinStrategy(BaseStrategy):
    """
    Expiry day strategy focusing on Max Pain and Gamma Pinning.
    Attempts to fade moves away from the 'Pin' level or capture
    theta decay around it.
    """
    
    def analyze(self, signal: Dict[str, Any], features: Dict[str, Any], market_state: Dict[str, Any]) -> TradeRecommendation:
        # Features: Max Pain, Gamma Flip, Pin Prob
        max_pain = features.get('max_pain', 0.0)
        current_price = features.get('close', 0.0)
        pin_prob = features.get('pin_risk_probability', 0.0) # From ExpiryTransformer
        
        if current_price == 0:
            return TradeRecommendation("HOLD", 0.0, "ExpiryPin", "Invalid price", "ATM", {})

        dist_to_pain = (current_price - max_pain) / current_price
        
        ml_signal = signal.get('signal', 'HOLD')
        ml_conf = signal.get('confidence', 0.0)
        
        strategy_signal = "HOLD"
        confidence = ml_conf
        rationale = []
        suggested_contract = "ATM"
        
        # Logic:
        # If High Pin Probability -> Fade moves towards Max Pain
        # If Low Pin Probability -> Trust Directional Momentum
        
        if pin_prob > 0.6:
            rationale.append(f"High Pin Probability ({pin_prob:.1%}) to {max_pain}")
            
            # Mean Reversion Logic
            if dist_to_pain > 0.005: # Price is 0.5% above Max Pain
                # Expect drop to Max Pain
                strategy_signal = "SELL" # Short the underlying / Buy Put
                confidence = max(confidence, pin_prob)
                rationale.append("Reverting to Max Pain (Over)")
                suggested_contract = "ITM" # Sell Call / Buy Put
            elif dist_to_pain < -0.005: # Price is 0.5% below Max Pain
                # Expect rise to Max Pain
                strategy_signal = "BUY"
                confidence = max(confidence, pin_prob)
                rationale.append("Reverting to Max Pain (Under)")
                suggested_contract = "ITM" # Buy Call
            else:
                # At the pin
                strategy_signal = "HOLD" # Or Sell Straddle (Short Vol)
                rationale.append("At Pin Level - Theta Harvest")
                suggested_contract = "IRON_FLY"
                
        else:
            rationale.append("Low Pin Risk - Following Momentum")
            strategy_signal = ml_signal
            
        return TradeRecommendation(
            signal=strategy_signal,
            confidence=min(confidence, 1.0),
            strategy_name="ExpiryPin",
            rationale="; ".join(rationale),
            suggested_contract=suggested_contract,
            metadata={
                'max_pain': max_pain,
                'pin_prob': pin_prob,
                'dist_to_pain': dist_to_pain
            }
        )

