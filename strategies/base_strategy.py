"""
Base Strategy Interface for OI Gemini (Phase 2).
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

@dataclass
class TradeRecommendation:
    """Standardized output from any strategy."""
    signal: str  # BUY, SELL, HOLD
    confidence: float
    strategy_name: str
    rationale: str
    suggested_contract: str = "ATM" # e.g., 'ATM', 'OTM-1', 'ITM+1'
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    @abstractmethod
    def analyze(self, signal: Dict[str, Any], features: Dict[str, Any], market_state: Dict[str, Any]) -> TradeRecommendation:
        """
        Analyze signal and market state to generate specific trade recommendation.
        
        Args:
            signal: The raw ML signal (e.g., {'signal': 'BUY', 'confidence': 0.8})
            features: Feature dictionary used for prediction
            market_state: Additional context (portfolio, order book, etc.)
            
        Returns:
            TradeRecommendation object
        """
        pass

