"""
Advanced Risk Management for OI Gemini.

Multi-layer risk framework with Greeks limits, correlation checks, VaR, and regime-based sizing.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from scipy.stats import norm

from time_utils import now_ist


@dataclass
class RiskLimitsConfig:
    """Configuration for risk limits."""
    
    # Horizon-specific position limits (as fraction of capital)
    max_position_size: Dict[str, float] = field(default_factory=lambda: {
        'intraday': 0.10,    # 10% of capital per intraday trade
        'swing': 0.20,       # 20% of capital per swing trade
        'expiry': 0.05,      # 5% of capital per expiry trade
        'default': 0.15      # 15% default
    })
    
    # Horizon-specific loss limits (as fraction of capital)
    max_loss_limits: Dict[str, float] = field(default_factory=lambda: {
        'intraday': 0.02,    # 2% of capital max loss per day (intraday)
        'swing': 0.05,       # 5% of capital max loss (swing)
        'expiry': 0.01,      # 1% of capital max loss (expiry day)
        'daily': 0.02        # 2% daily loss limit (system-wide)
    })
    
    # Greeks exposure limits
    greeks_limits: Dict[str, float] = field(default_factory=lambda: {
        'max_delta': 0.50,           # 50% of capital max delta exposure
        'max_gamma_exposure': 1000000000.0,  # Max gamma exposure (absolute)
        'max_vega_exposure': 0.30,   # 30% of capital max vega exposure
        'max_theta_exposure': -0.01, # Minimum theta (must be positive, i.e., theta > -0.01)
        'max_vanna_exposure': 0.20,  # 20% of capital max vanna exposure
    })
    
    # Correlation limits
    max_correlation_threshold: float = 0.80  # Max correlation between positions
    
    # VaR limits
    var_confidence_level: float = 0.95       # 95% VaR
    max_var_fraction: float = 0.03           # 3% of capital max VaR
    
    # Liquidity limits
    max_bid_ask_spread_pct: float = 0.01     # 1% max spread
    min_order_book_depth: float = 1000000.0  # Minimum order book depth (INR)
    
    # Market regime adjustments
    regime_adjustments: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'high_vol': {
            'position_size_multiplier': 0.7,  # Reduce size by 30%
            'max_loss_multiplier': 0.8        # Reduce loss limit by 20%
        },
        'low_vol': {
            'position_size_multiplier': 1.2,  # Increase size by 20%
            'max_loss_multiplier': 1.1        # Increase loss limit by 10%
        },
        'trending': {
            'position_size_multiplier': 1.1,
            'max_loss_multiplier': 1.0
        },
        'range_bound': {
            'position_size_multiplier': 0.9,
            'max_loss_multiplier': 1.0
        }
    })


class AdvancedRiskManager:
    """
    Multi-layer risk management framework.
    
    Provides comprehensive pre-trade checks, portfolio monitoring, and circuit breakers.
    """
    
    def __init__(self, config: Optional[RiskLimitsConfig] = None, account_size: float = 1_000_000.0):
        """
        Args:
            config: Risk limits configuration (defaults to conservative limits)
            account_size: Account size in INR
        """
        self.config = config or RiskLimitsConfig()
        self.account_size = account_size
        self.portfolio_positions: List[Dict] = []
        self.historical_returns: List[float] = []
        self.daily_pnl_history: List[float] = []
        self.contract_multiplier: int = 50  # NIFTY lot size
        
    def check_trade(
        self,
        proposed_trade: Dict,
        portfolio_state: Dict,
        market_regime: Optional[str] = None,
        current_prices: Optional[Dict[str, float]] = None,
        option_greeks: Optional[Dict[str, Dict[str, float]]] = None,
        spot_price: Optional[float] = None
    ) -> Tuple[bool, List[str]]:
        """
        Comprehensive pre-trade risk checks.
        
        Args:
            proposed_trade: Dict with keys:
                - 'signal': 'BUY' or 'SELL'
                - 'quantity_lots': int
                - 'strike': float
                - 'option_type': 'CE' or 'PE'
                - 'horizon': 'intraday', 'swing', or 'expiry'
                - 'price': float (current option price)
                - 'confidence': float
                - 'bid_ask_spread_pct': float (optional)
                - 'order_book_depth': float (optional)
            portfolio_state: Dict with keys:
                - 'net_delta': float
                - 'total_gamma_exposure': float
                - 'total_vega_exposure': float
                - 'total_theta_exposure': float
                - 'total_vanna_exposure': float
                - 'total_mtm': float
                - 'daily_pnl': float
            market_regime: Optional market regime ('high_vol', 'low_vol', 'trending', 'range_bound')
            current_prices: Optional dict mapping symbols to current prices
            option_greeks: Optional dict mapping option symbols to Greeks dicts with keys:
                - 'delta', 'gamma', 'vega', 'theta', 'vanna'
            spot_price: Optional spot price for gamma exposure calculation
        
        Returns:
            Tuple of (allowed: bool, violations: List[str])
        """
        violations = []
        
        horizon = proposed_trade.get('horizon', 'default')
        signal = proposed_trade.get('signal', 'HOLD')
        quantity_lots = proposed_trade.get('quantity_lots', 0)
        option_type = proposed_trade.get('option_type', 'CE')
        strike = proposed_trade.get('strike', 0.0)
        price = proposed_trade.get('price', 0.0)
        
        # Apply regime adjustments
        regime_multipliers = self._get_regime_adjustments(market_regime)
        
        # 1. Position Size Check (Horizon-based)
        max_size_fraction = self.config.max_position_size.get(horizon, self.config.max_position_size['default'])
        max_size_fraction *= regime_multipliers['position_size_multiplier']
        
        position_value = quantity_lots * price * self.contract_multiplier
        max_position_value = self.account_size * max_size_fraction
        
        if position_value > max_position_value:
            violations.append(
                f"Position size {position_value:.0f} exceeds limit {max_position_value:.0f} "
                f"for horizon {horizon} ({max_size_fraction:.1%})"
            )
        
        # 2. Greeks Exposure Check
        if option_greeks:
            # Get Greeks for this option (key could be symbol or option_type)
            greeks = None
            symbol = proposed_trade.get('symbol')
            if symbol and symbol in option_greeks:
                greeks = option_greeks[symbol]
            elif option_type in option_greeks:
                greeks = option_greeks[option_type]
            
            if greeks:
                # Calculate proposed Greeks contributions
                proposed_delta = self._calculate_proposed_delta(
                    signal, quantity_lots, option_type, greeks.get('delta', 0.0)
                )
                proposed_gamma = self._calculate_proposed_gamma(
                    signal, quantity_lots, option_type, greeks.get('gamma', 0.0), 
                    strike, spot_price
                )
                proposed_vega = self._calculate_proposed_vega(
                    signal, quantity_lots, option_type, greeks.get('vega', 0.0)
                )
                proposed_theta = self._calculate_proposed_theta(
                    signal, quantity_lots, option_type, greeks.get('theta', 0.0)
                )
                proposed_vanna = self._calculate_proposed_vanna(
                    signal, quantity_lots, option_type, greeks.get('vanna', 0.0)
                )
                
                # Check delta limit
                current_delta = portfolio_state.get('net_delta', 0.0)
                new_delta = abs(current_delta + proposed_delta)
                max_delta = self.account_size * self.config.greeks_limits['max_delta']
                if new_delta > max_delta:
                    violations.append(
                        f"Net delta {new_delta:.1f} would exceed limit {max_delta:.1f} "
                        f"({self.config.greeks_limits['max_delta']:.1%} of capital)"
                    )
                
                # Check gamma exposure limit
                current_gamma_exp = portfolio_state.get('total_gamma_exposure', 0.0)
                new_gamma_exp = current_gamma_exp + proposed_gamma
                max_gamma_exp = self.config.greeks_limits['max_gamma_exposure']
                if abs(new_gamma_exp) > max_gamma_exp:
                    violations.append(
                        f"Gamma exposure {new_gamma_exp:.0f} would exceed limit {max_gamma_exp:.0f}"
                    )
                
                # Check vega exposure limit
                current_vega_exp = portfolio_state.get('total_vega_exposure', 0.0)
                new_vega_exp = current_vega_exp + proposed_vega
                max_vega_exp = self.account_size * self.config.greeks_limits['max_vega_exposure']
                if abs(new_vega_exp) > max_vega_exp:
                    violations.append(
                        f"Vega exposure {new_vega_exp:.1f} would exceed limit {max_vega_exp:.1f}"
                    )
                
                # Check theta limit (must be positive, i.e., theta > threshold)
                current_theta_exp = portfolio_state.get('total_theta_exposure', 0.0)
                new_theta_exp = current_theta_exp + proposed_theta
                min_theta = self.config.greeks_limits['max_theta_exposure']
                if new_theta_exp < min_theta:
                    violations.append(
                        f"Theta exposure {new_theta_exp:.4f} would be below minimum {min_theta:.4f} "
                        "(insufficient positive theta)"
                    )
                
                # Check vanna exposure limit
                current_vanna_exp = portfolio_state.get('total_vanna_exposure', 0.0)
                new_vanna_exp = current_vanna_exp + proposed_vanna
                max_vanna_exp = self.account_size * self.config.greeks_limits['max_vanna_exposure']
                if abs(new_vanna_exp) > max_vanna_exp:
                    violations.append(
                        f"Vanna exposure {new_vanna_exp:.1f} would exceed limit {max_vanna_exp:.1f}"
                    )
        
        # 3. Portfolio Correlation Check
        correlation_violation = self._check_correlation(proposed_trade, self.portfolio_positions)
        if correlation_violation:
            violations.append(correlation_violation)
        
        # 4. Liquidity Check
        liquidity_violation = self._check_liquidity(proposed_trade, current_prices)
        if liquidity_violation:
            violations.append(liquidity_violation)
        
        # 5. VaR Check
        var_violation = self._check_var(proposed_trade, portfolio_state)
        if var_violation:
            violations.append(var_violation)
        
        # 6. Daily Loss Limit Check (Horizon-specific)
        daily_pnl = portfolio_state.get('daily_pnl', 0.0)
        max_loss_fraction = self.config.max_loss_limits.get(horizon, self.config.max_loss_limits['daily'])
        max_loss_fraction *= regime_multipliers['max_loss_multiplier']
        max_loss = self.account_size * max_loss_fraction
        
        if daily_pnl < -max_loss:
            violations.append(
                f"Daily loss {daily_pnl:.0f} exceeds limit {max_loss:.0f} "
                f"({max_loss_fraction:.1%} of capital for {horizon})"
            )
        
        return len(violations) == 0, violations
    
    def _get_regime_adjustments(self, regime: Optional[str]) -> Dict[str, float]:
        """Get position size and loss limit multipliers for market regime."""
        if not regime or regime not in self.config.regime_adjustments:
            return {'position_size_multiplier': 1.0, 'max_loss_multiplier': 1.0}
        return self.config.regime_adjustments[regime]
    
    def _calculate_proposed_delta(
        self,
        signal: str,
        quantity_lots: int,
        option_type: str,
        delta: float
    ) -> float:
        """Calculate proposed delta contribution."""
        direction = 1.0 if signal == 'BUY' else -1.0
        delta_sign = 1.0 if option_type == 'CE' else -1.0
        return direction * delta_sign * delta * quantity_lots * self.contract_multiplier
    
    def _calculate_proposed_gamma(
        self,
        signal: str,
        quantity_lots: int,
        option_type: str,
        gamma: float,
        strike: float,
        spot_price: Optional[float] = None
    ) -> float:
        """Calculate proposed gamma exposure contribution."""
        # GEX = OI × Gamma × Spot² × Multiplier
        # Simplified: use strike as proxy for spot if not provided
        spot = spot_price or strike
        direction = 1.0 if signal == 'BUY' else -1.0
        gamma_sign = 1.0 if option_type == 'CE' else -1.0
        return direction * gamma_sign * gamma * (spot ** 2) * quantity_lots * self.contract_multiplier
    
    def _calculate_proposed_vega(
        self,
        signal: str,
        quantity_lots: int,
        option_type: str,
        vega: float
    ) -> float:
        """Calculate proposed vega exposure contribution."""
        direction = 1.0 if signal == 'BUY' else -1.0
        return direction * vega * quantity_lots * self.contract_multiplier
    
    def _calculate_proposed_theta(
        self,
        signal: str,
        quantity_lots: int,
        option_type: str,
        theta: float
    ) -> float:
        """Calculate proposed theta exposure contribution."""
        direction = 1.0 if signal == 'BUY' else -1.0
        return direction * theta * quantity_lots * self.contract_multiplier
    
    def _calculate_proposed_vanna(
        self,
        signal: str,
        quantity_lots: int,
        option_type: str,
        vanna: float
    ) -> float:
        """Calculate proposed vanna exposure contribution."""
        direction = 1.0 if signal == 'BUY' else -1.0
        return direction * vanna * quantity_lots * self.contract_multiplier
    
    def _check_correlation(
        self,
        proposed_trade: Dict,
        existing_positions: List[Dict]
    ) -> Optional[str]:
        """
        Check if proposed trade is too correlated with existing positions.
        
        Simplified correlation check: same strike/expiry = high correlation.
        Full implementation would calculate historical correlation.
        """
        proposed_strike = proposed_trade.get('strike')
        proposed_expiry = proposed_trade.get('expiry_date')
        
        if not proposed_strike or not existing_positions:
            return None
        
        # Count positions with same strike (high correlation)
        same_strike_count = sum(
            1 for pos in existing_positions
            if pos.get('strike') == proposed_strike
        )
        
        # Count positions with same expiry (moderate correlation)
        same_expiry_count = sum(
            1 for pos in existing_positions
            if pos.get('expiry_date') == proposed_expiry
        )
        
        # Warn if too many correlated positions
        if same_strike_count >= 2:
            return f"High correlation: {same_strike_count} existing positions at strike {proposed_strike}"
        
        if same_expiry_count >= 3:
            return f"Moderate correlation: {same_expiry_count} existing positions at same expiry"
        
        return None
    
    def _check_liquidity(
        self,
        proposed_trade: Dict,
        current_prices: Optional[Dict[str, float]]
    ) -> Optional[str]:
        """
        Check liquidity constraints.
        
        Args:
            proposed_trade: Trade dict with 'spread', 'bid_ask_spread_pct', 'order_book_depth'
            current_prices: Current prices dict
        """
        spread_pct = proposed_trade.get('bid_ask_spread_pct', 0.0)
        if spread_pct > self.config.max_bid_ask_spread_pct:
            return (
                f"Bid-ask spread {spread_pct:.2%} exceeds limit "
                f"{self.config.max_bid_ask_spread_pct:.2%}"
            )
        
        order_book_depth = proposed_trade.get('order_book_depth', float('inf'))
        if order_book_depth < self.config.min_order_book_depth:
            return (
                f"Order book depth {order_book_depth:.0f} below minimum "
                f"{self.config.min_order_book_depth:.0f}"
            )
        
        return None
    
    def _check_var(
        self,
        proposed_trade: Dict,
        portfolio_state: Dict
    ) -> Optional[str]:
        """Check if proposed trade would exceed VaR limit."""
        var_result = self.calculate_var(portfolio_state, self.config.var_confidence_level)
        var_absolute = var_result.get('var_absolute', 0.0)
        max_var = self.account_size * self.config.max_var_fraction
        
        if var_absolute > max_var:
            return (
                f"VaR {var_absolute:.0f} exceeds limit {max_var:.0f} "
                f"({self.config.max_var_fraction:.1%} of capital)"
            )
        
        return None
    
    def calculate_var(
        self,
        portfolio_state: Dict,
        confidence_level: float = 0.95,
        lookback_days: int = 30
    ) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) for the portfolio.
        
        Uses historical simulation method.
        
        Args:
            portfolio_state: Portfolio state with 'total_mtm'
            confidence_level: VaR confidence level (default 0.95 = 95%)
            lookback_days: Number of days for historical returns
        
        Returns:
            Dict with 'var_absolute', 'var_percentage', 'expected_shortfall'
        """
        if not self.historical_returns or len(self.historical_returns) < lookback_days:
            # Fallback: use simple parametric VaR
            # Assume normal distribution with 20% annualized volatility
            annual_vol = 0.20
            daily_vol = annual_vol / np.sqrt(252)
            z_score = norm.ppf(1 - confidence_level)
            current_value = portfolio_state.get('total_mtm', 0.0) + self.account_size
            
            var_absolute = abs(z_score * daily_vol * current_value)
            var_percentage = abs(z_score * daily_vol)
            
            return {
                'var_absolute': var_absolute,
                'var_percentage': var_percentage,
                'expected_shortfall': var_absolute * 1.2,  # ES ~ 1.2x VaR for normal dist
                'confidence_level': confidence_level
            }
        
        # Historical simulation method
        returns = np.array(self.historical_returns[-lookback_days:])
        if len(returns) == 0:
            return {
                'var_absolute': 0.0,
                'var_percentage': 0.0,
                'expected_shortfall': 0.0,
                'confidence_level': confidence_level
            }
        
        percentile = (1 - confidence_level) * 100
        var_percentage = np.percentile(returns, percentile)
        current_value = portfolio_state.get('total_mtm', 0.0) + self.account_size
        var_absolute = abs(var_percentage * current_value)
        
        # Expected Shortfall (Conditional VaR) - average of losses beyond VaR
        tail_losses = returns[returns <= np.percentile(returns, percentile)]
        expected_shortfall = abs(np.mean(tail_losses) * current_value) if len(tail_losses) > 0 else var_absolute * 1.2
        
        return {
            'var_absolute': var_absolute,
            'var_percentage': abs(var_percentage),
            'expected_shortfall': expected_shortfall,
            'confidence_level': confidence_level
        }
    
    def update_portfolio_state(
        self,
        positions: List[Dict],
        current_prices: Dict[str, float],
        option_greeks: Optional[Dict[str, Dict[str, float]]] = None,
        spot_price: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Update and return current portfolio state.
        
        Args:
            positions: List of position dicts with keys: 'symbol', 'strike', 'option_type', 
                      'side' (B/S), 'quantity_lots', 'entry_price'
            current_prices: Dict mapping symbol to current price
            option_greeks: Optional dict mapping symbol to Greeks
            spot_price: Optional spot price for gamma exposure
        
        Returns:
            Dict with portfolio metrics
        """
        total_mtm = 0.0
        net_delta = 0.0
        total_gamma_exposure = 0.0
        total_vega_exposure = 0.0
        total_theta_exposure = 0.0
        total_vanna_exposure = 0.0
        total_exposure = 0.0
        
        for pos in positions:
            symbol = pos.get('symbol', '')
            strike = pos.get('strike', 0.0)
            option_type = pos.get('option_type', 'CE')
            side = pos.get('side', 'B')
            quantity_lots = pos.get('quantity_lots', 0)
            entry_price = pos.get('entry_price', 0.0)
            current_price = current_prices.get(symbol, entry_price)
            
            # Calculate MTM
            if side == 'B':
                mtm = (current_price - entry_price) * quantity_lots * self.contract_multiplier
            else:
                mtm = (entry_price - current_price) * quantity_lots * self.contract_multiplier
            total_mtm += mtm
            
            # Calculate Greeks if available
            if option_greeks and symbol in option_greeks:
                greeks = option_greeks[symbol]
                direction = 1.0 if side == 'B' else -1.0
                
                delta = greeks.get('delta', 0.0)
                delta_sign = 1.0 if option_type == 'CE' else -1.0
                net_delta += direction * delta_sign * delta * quantity_lots * self.contract_multiplier
                
                gamma = greeks.get('gamma', 0.0)
                gamma_sign = 1.0 if option_type == 'CE' else -1.0
                spot = spot_price or strike
                total_gamma_exposure += direction * gamma_sign * gamma * (spot ** 2) * quantity_lots * self.contract_multiplier
                
                vega = greeks.get('vega', 0.0)
                total_vega_exposure += direction * vega * quantity_lots * self.contract_multiplier
                
                theta = greeks.get('theta', 0.0)
                total_theta_exposure += direction * theta * quantity_lots * self.contract_multiplier
                
                vanna = greeks.get('vanna', 0.0)
                total_vanna_exposure += direction * vanna * quantity_lots * self.contract_multiplier
            
            # Calculate exposure
            total_exposure += quantity_lots * self.contract_multiplier * current_price
        
        # Update portfolio positions
        self.portfolio_positions = positions
        
        return {
            'total_mtm': total_mtm,
            'net_delta': net_delta,
            'total_gamma_exposure': total_gamma_exposure,
            'total_vega_exposure': total_vega_exposure,
            'total_theta_exposure': total_theta_exposure,
            'total_vanna_exposure': total_vanna_exposure,
            'total_exposure': total_exposure,
            'exposure_pct': (total_exposure / self.account_size) * 100 if self.account_size > 0 else 0.0,
            'return_pct': (total_mtm / self.account_size) * 100 if self.account_size > 0 else 0.0
        }
    
    def add_daily_pnl(self, daily_pnl: float):
        """Add daily PnL to history for VaR calculation."""
        self.daily_pnl_history.append(daily_pnl)
        if len(self.daily_pnl_history) > 252:  # Keep 1 year of history
            self.daily_pnl_history.pop(0)
        
        # Calculate return percentage
        if self.account_size > 0:
            return_pct = (daily_pnl / self.account_size) * 100
            self.historical_returns.append(return_pct)
            if len(self.historical_returns) > 252:
                self.historical_returns.pop(0)

