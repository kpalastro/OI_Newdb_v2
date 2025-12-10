"""
OI Buildup Strategy (Swing / Directional).
"""
from typing import Dict, Any
from .base_strategy import BaseStrategy, TradeRecommendation

class OIBuildupStrategy(BaseStrategy):
    """
    Swing strategy based on Open Interest accumulation patterns.
    Looks for divergence between Price and OI, and PCR confirmation.
    """
    
    def analyze(self, signal: Dict[str, Any], features: Dict[str, Any], market_state: Dict[str, Any]) -> TradeRecommendation:
        # Extract OI metrics
        oi_change_pct = features.get('oi_change_pct', 0.0)
        pcr = features.get('pcr', 1.0)
        
        # Sentiment Metrics
        ad_ratio = features.get('sentiment_ad_ratio_50', 1.0)
        inst_flow = features.get('sentiment_inst_net_crores', 0.0)
        
        ml_signal = signal.get('signal', 'HOLD')
        ml_conf = signal.get('confidence', 0.0)
        
        strategy_signal = "HOLD"
        confidence = ml_conf
        rationale = []
        
        # Logic: Detect Long Buildup or Short Buildup
        # Long Buildup: Price UP + OI UP
        # Short Buildup: Price DOWN + OI UP
        
        if ml_signal == "BUY":
            # Check for Long Buildup support
            if oi_change_pct > 0.05: # > 5% OI increase
                confidence *= 1.2
                rationale.append(f"Long Buildup detected (OI +{oi_change_pct:.1%})")
                strategy_signal = "BUY"
            elif oi_change_pct < -0.05:
                # Short Covering? Price UP + OI DOWN
                confidence *= 1.1
                rationale.append(f"Short Covering detected (OI {oi_change_pct:.1%})")
                strategy_signal = "BUY"
            else:
                rationale.append("Weak OI confirmation")
                
            # Sentiment Confirmation
            if ad_ratio > 1.2:
                confidence *= 1.1
                rationale.append(f"Breadth Bullish (A/D {ad_ratio:.2f})")
            elif ad_ratio < 0.8:
                confidence *= 0.8
                rationale.append(f"Breadth Divergence (A/D {ad_ratio:.2f})")
                
            # Institutional Flow Confirmation
            if inst_flow > 500: # > 500 Cr Net Buy
                 confidence *= 1.1
                 rationale.append("Inst. Buying Detected")

            # PCR Confirmation
            if pcr < 0.7: 
                rationale.append(f"PCR Bullish ({pcr:.2f})")
            elif pcr > 1.5:
                confidence *= 0.8
                rationale.append(f"PCR Bearish Divergence ({pcr:.2f})")
                
        elif ml_signal == "SELL":
            # Check for Short Buildup support
            if oi_change_pct > 0.05:
                confidence *= 1.2
                rationale.append(f"Short Buildup detected (OI +{oi_change_pct:.1%})")
                strategy_signal = "SELL"
            elif oi_change_pct < -0.05:
                # Long Unwinding? Price DOWN + OI DOWN
                confidence *= 1.1
                rationale.append(f"Long Unwinding detected (OI {oi_change_pct:.1%})")
                strategy_signal = "SELL"
            
            # Sentiment Confirmation
            if ad_ratio < 0.8:
                confidence *= 1.1
                rationale.append(f"Breadth Bearish (A/D {ad_ratio:.2f})")
            elif ad_ratio > 1.2:
                confidence *= 0.8
                rationale.append(f"Breadth Divergence (A/D {ad_ratio:.2f})")

            # Inst Flow
            if inst_flow < -500: # > 500 Cr Net Sell
                 confidence *= 1.1
                 rationale.append("Inst. Selling Detected")
            
            # PCR Confirmation
            if pcr > 1.2:
                rationale.append(f"PCR Bearish ({pcr:.2f})")
                
        else:
            rationale.append("No directional signal")

        return TradeRecommendation(
            signal=strategy_signal,
            confidence=min(confidence, 1.0),
            strategy_name="OIBuildup",
            rationale="; ".join(rationale),
            suggested_contract="OTM-1", # Swing trades often target slightly OTM
            metadata={
                'oi_change_pct': oi_change_pct,
                'pcr': pcr,
                'ad_ratio': ad_ratio,
                'inst_flow': inst_flow
            }
        )

