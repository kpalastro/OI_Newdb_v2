"""
Risk management utilities for the OI Gemini ML stack.

Implements Kelly-based position sizing with confidence adjustment,
plus helper routines to evaluate trading performance metrics.
Enhanced with portfolio-level constraints for Phase 2.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


DEFAULT_ACCOUNT_SIZE = 1_000_000  # INR
DEFAULT_MARGIN_PER_LOT = 175_000   # Approx margin required per NIFTY lot


def get_optimal_position_size(
    ml_confidence: float,
    win_rate: float,
    avg_win_loss_ratio: float,
    max_risk: float = 0.02,
    account_size: float = DEFAULT_ACCOUNT_SIZE,
    margin_per_lot: float = DEFAULT_MARGIN_PER_LOT,
    kelly_multiplier: float = 0.3,
    current_volatility: Optional[float] = None,
    target_volatility: float = 0.20,
    regime_risk_scale: float = 1.0,  # New: Regime-based scaling factor
) -> Dict[str, float]:
    """
    Kelly Criterion with confidence adjustment, risk caps, and Volatility Targeting.

    Returns a dict containing the recommended capital fraction, kelly fraction,
    and lot suggestion (rounded down).
    """
    ml_confidence = float(max(0.0, min(1.0, ml_confidence)))
    win_rate = float(max(0.0, min(1.0, win_rate)))
    avg_win_loss_ratio = float(max(avg_win_loss_ratio, 0.01))

    if ml_confidence < 0.5 or win_rate <= 0.0:
        return {
            'fraction': 0.0,
            'kelly_fraction': 0.0,
            'recommended_lots': 0,
            'capital_allocated': 0.0,
        }

    b = avg_win_loss_ratio
    q = 1.0 - win_rate
    kelly_fraction = max((win_rate * b - q) / b, 0.0)

    # Apply Fractional Kelly Multiplier (Recommendation: 0.3 to 0.5)
    adjusted_fraction = kelly_fraction * ml_confidence * kelly_multiplier
    
    # Apply Regime Scaling (New Feature)
    adjusted_fraction *= regime_risk_scale
    
    # Volatility Targeting
    # If market is 2x more volatile than target, cut size by half.
    if current_volatility and current_volatility > 0:
        vol_scalar = target_volatility / current_volatility
        # Cap scalar to avoid massive sizing in low vol environments (e.g. max 2x leverage)
        vol_scalar = min(vol_scalar, 2.0) 
        adjusted_fraction *= vol_scalar

    safe_fraction = min(adjusted_fraction, max_risk)
    capital_allocated = account_size * safe_fraction
    recommended_lots = int(capital_allocated // margin_per_lot) if margin_per_lot else 0

    # For paper trading and conservative sizing, ensure at least 1 lot
    # whenever there is a non-zero risk fraction and positive Kelly edge.
    if recommended_lots <= 0 and safe_fraction > 0.0 and kelly_fraction > 0.0:
        recommended_lots = 1

    return {
        'fraction': round(safe_fraction, 4),
        'kelly_fraction': round(kelly_fraction, 4),
        'recommended_lots': max(recommended_lots, 0),
        'capital_allocated': round(capital_allocated, 2),
    }

def check_circuit_breaker(
    current_daily_loss_pct: float,
    max_daily_loss_pct: float = 0.02,
    consecutive_losses: int = 0,
    max_consecutive_losses: int = 5
) -> Tuple[bool, str]:
    """
    System-wide circuit breakers.
    Returns: (is_triggered, reason)
    """
    if current_daily_loss_pct <= -max_daily_loss_pct:
        return True, f"Daily Loss Limit Hit: {current_daily_loss_pct:.2%}"
    
    if consecutive_losses >= max_consecutive_losses:
        return True, f"Consecutive Loss Limit Hit: {consecutive_losses}"
        
    return False, ""



def calculate_trading_metrics(
    predictions: Sequence[float],
    actual_returns: Sequence[float],
    position_sizes: Sequence[float],
) -> Dict[str, float]:
    """
    Compute trading KPIs such as PnL, Sharpe, and hit rate.
    """
    preds = np.array(predictions, dtype=float)
    rets = np.array(actual_returns, dtype=float)
    sizes = np.array(position_sizes, dtype=float)

    pnl_series = preds * rets * sizes
    total_pnl = float(np.nansum(pnl_series))
    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series < 0]

    sharpe = np.nan
    if pnl_series.std(ddof=1) > 0:
        sharpe = float(np.nanmean(pnl_series) / pnl_series.std(ddof=1) * np.sqrt(252))

    max_drawdown = _calculate_max_drawdown(np.nancumsum(pnl_series))

    metrics = {
        'total_pnl': total_pnl,
        'win_rate': float((pnl_series > 0).mean()) if len(pnl_series) else 0.0,
        'profit_factor': float(np.sum(wins) / abs(np.sum(losses))) if len(losses) else np.nan,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'calmar_ratio': float(total_pnl / max_drawdown) if max_drawdown else np.nan,
        'hit_rate': float((np.sign(preds) == np.sign(rets)).mean()) if len(preds) else 0.0,
        'avg_win': float(np.mean(wins)) if len(wins) else 0.0,
        'avg_loss': float(np.mean(losses)) if len(losses) else 0.0,
        'signal_frequency': float(np.mean(preds != 0)) if len(preds) else 0.0,
        'false_signal_rate': float(np.mean((preds != 0) & (preds * rets < 0))) if len(preds) else 0.0,
    }
    return metrics


def _calculate_max_drawdown(equity_curve: Iterable[float]) -> float:
    equity = np.array(list(equity_curve), dtype=float)
    if equity.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdowns = running_max - equity
    return float(np.max(drawdowns)) if drawdowns.size else 0.0


def check_portfolio_constraints(
    current_positions: Sequence[Dict],
    proposed_signal: str,
    proposed_size_lots: int,
    max_net_delta: Optional[float] = None,
    max_position_size_lots: Optional[int] = None,
    max_total_exposure: Optional[float] = None,
) -> tuple[bool, str]:
    """
    Check if proposed trade violates portfolio-level constraints.
    
    Args:
        current_positions: List of current position dicts with 'side' (B/S), 'qty', 'type' (CE/PE)
        proposed_signal: 'BUY' or 'SELL'
        proposed_size_lots: Proposed position size in lots
        max_net_delta: Maximum allowed net delta (None = no limit)
        max_position_size_lots: Maximum position size per trade (None = no limit)
        max_total_exposure: Maximum total capital exposure (None = no limit)
    
    Returns:
        Tuple of (allowed, reason_string)
    """
    # Check position size limit
    if max_position_size_lots is not None and proposed_size_lots > max_position_size_lots:
        return False, f"Position size {proposed_size_lots} exceeds limit {max_position_size_lots}"
    
    # Calculate current net delta
    if max_net_delta is not None:
        net_delta = 0.0
        for pos in current_positions:
            delta_contribution = 1.0 if pos.get('type') == 'CE' else -1.0
            if pos.get('side') == 'S':
                delta_contribution *= -1
            net_delta += delta_contribution * (pos.get('qty', 0) / 50)  # Convert qty to lots
        
        # Calculate proposed delta change
        proposed_delta = 1.0 if proposed_signal == 'BUY' else -1.0
        new_net_delta = abs(net_delta + proposed_delta * proposed_size_lots)
        
        if new_net_delta > max_net_delta:
            return False, f"Net delta {new_net_delta:.1f} would exceed limit {max_net_delta:.1f}"
    
    # Check total exposure
    if max_total_exposure is not None:
        current_exposure = sum(
            (pos.get('qty', 0) / 50) * DEFAULT_MARGIN_PER_LOT
            for pos in current_positions
        )
        proposed_exposure = proposed_size_lots * DEFAULT_MARGIN_PER_LOT
        new_exposure = current_exposure + proposed_exposure
        
        if new_exposure > max_total_exposure:
            return False, f"Total exposure {new_exposure:.0f} would exceed limit {max_total_exposure:.0f}"
    
    return True, "All constraints satisfied"


def calculate_portfolio_metrics(
    positions: Sequence[Dict],
    current_prices: Dict[str, float],
    initial_capital: float = DEFAULT_ACCOUNT_SIZE,
) -> Dict[str, float]:
    """
    Calculate portfolio-level risk metrics.
    
    Args:
        positions: List of position dicts
        current_prices: Dict mapping symbol to current price
        initial_capital: Initial capital
    
    Returns:
        Dictionary with portfolio metrics
    """
    total_mtm = 0.0
    net_delta = 0.0
    total_exposure = 0.0
    
    for pos in positions:
        symbol = pos.get('symbol')
        current_price = current_prices.get(symbol, pos.get('entry_price', 0.0))
        entry_price = pos.get('entry_price', 0.0)
        qty = pos.get('qty', 0)
        side = pos.get('side', 'B')
        opt_type = pos.get('type', 'CE')
        
        # Calculate MTM
        if side == 'B':
            mtm = (current_price - entry_price) * qty
        else:
            mtm = (entry_price - current_price) * qty
        total_mtm += mtm
        
        # Calculate delta contribution
        delta_sign = 1.0 if opt_type == 'CE' else -1.0
        if side == 'S':
            delta_sign *= -1
        net_delta += delta_sign * (qty / 50)  # Convert to lots
        
        # Calculate exposure
        total_exposure += (qty / 50) * DEFAULT_MARGIN_PER_LOT
    
    return {
        'total_mtm': total_mtm,
        'net_delta': net_delta,
        'total_exposure': total_exposure,
        'exposure_pct': (total_exposure / initial_capital) * 100 if initial_capital > 0 else 0.0,
        'return_pct': (total_mtm / initial_capital) * 100 if initial_capital > 0 else 0.0,
    }

