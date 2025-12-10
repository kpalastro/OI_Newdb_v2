import logging
from typing import Dict, Any, Optional

class HorizonRouter:
    """
    Routes prediction requests to the appropriate model based on market context.
    
    Logic:
    - Expiry Day (0-DTE): Use Expiry Transformer (Gamma/Pin risk focus).
    - Intraday (Scalping): Use LSTM (Microstructure/Order Flow focus).
    - Swing (Positional): Use Tree Ensemble (OI Build-up/Macro focus).
    """
    
    def __init__(self):
        pass
        
    def determine_horizon(self, features: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        """
        Determine the trading horizon based on features and context.
        
        Args:
            features: Dictionary of current market features.
            context: External context (e.g., user strategy preference).
            
        Returns:
            str: 'expiry', 'intraday', or 'swing'
        """
        # 1. Check for Expiry Day
        # time_to_expiry_hours is typically in features
        tte = features.get('time_to_expiry_hours', 24.0)
        
        if tte < 6.5: # Less than 1 trading day (approx 6.15 hours)
            return 'expiry'
            
        # 2. Check for explicit strategy override
        if context and context.get('strategy_mode'):
            mode = context.get('strategy_mode').upper()
            if mode == 'SCALP':
                return 'intraday'
            elif mode == 'SWING':
                return 'swing'
                
        # 3. Default Logic based on volatility or time of day?
        # For now, default to Swing as it's the most robust/general purpose base
        return 'swing'

