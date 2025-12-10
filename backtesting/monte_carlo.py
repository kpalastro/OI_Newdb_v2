"""
Monte Carlo Simulation for Robustness Testing.

This module implements block bootstrap resampling to generate thousands of
synthetic equity curves from backtest results. It validates strategy robustness
against sequence risk and market regime variations.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging

class MonteCarloTester:
    """
    Monte Carlo simulation engine using Block Bootstrap method.
    """
    def __init__(self):
        pass

    def run_simulation(
        self, 
        backtest_results: 'BacktestResult', 
        n_simulations: int = 1000, 
        block_size: int = 5
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation on trade returns.
        
        Args:
            backtest_results: Result object from BacktestEngine.
            n_simulations: Number of synthetic equity curves to generate.
            block_size: Size of blocks for bootstrap (preserves volatility clustering).
            
        Returns:
            Dict containing risk metrics (VaR, CVaR, Prob of Loss, etc.)
        """
        trades = backtest_results.trades
        if not trades:
            return self._empty_report()
            
        # Extract percentage returns per trade
        # We need trade-by-trade returns, not time-series returns for this specific test
        # Calculating trade PnL % = (Net PnL) / (Capital Allocated)
        returns = []
        for t in trades:
            if t.capital_allocated > 0:
                ret = t.net_pnl / t.capital_allocated
                returns.append(ret)
            else:
                returns.append(0.0)
                
        if not returns:
            return self._empty_report()
            
        returns_array = np.array(returns)
        n_trades = len(returns_array)
        
        # Block Bootstrap
        # 1. Create indices for blocks
        # 2. Resample blocks
        
        simulated_metrics = {
            'total_return': [],
            'max_drawdown': [],
            'sharpe_ratio': [],
            'win_rate': []
        }
        
        # Pre-calculate block starts
        if n_trades < block_size:
            block_size = 1
            
        block_starts = np.arange(0, n_trades - block_size + 1)
        
        if len(block_starts) == 0:
             # Fallback for very few trades
             block_starts = np.arange(n_trades)
             block_size = 1
        
        num_blocks_needed = (n_trades // block_size) + 1
        
        for _ in range(n_simulations):
            # Randomly select blocks
            chosen_starts = np.random.choice(block_starts, size=num_blocks_needed)
            
            # Construct synthetic trade sequence
            synthetic_returns = []
            for start in chosen_starts:
                synthetic_returns.extend(returns_array[start : start + block_size])
                
            # Trim to original length
            synthetic_returns = np.array(synthetic_returns[:n_trades])
            
            # Calculate Equity Curve
            # Assume constant capital reinvestment or simple sum?
            # Using simple sum of % returns for speed, or compounded
            # Let's use compounded curve starting at 1.0
            equity_curve = np.cumprod(1 + synthetic_returns)
            
            # Metrics
            total_ret = equity_curve[-1] - 1.0
            
            # Drawdown
            running_max = np.maximum.accumulate(equity_curve)
            drawdowns = (running_max - equity_curve) / running_max
            max_dd = np.max(drawdowns)
            
            # Sharpe (Annualized assuming 252 days? difficult with trade series)
            # Just use mean/std of per-trade returns * sqrt(trades per year approx?)
            # Simplified Sharpe: Mean / Std
            mean_ret = np.mean(synthetic_returns)
            std_ret = np.std(synthetic_returns)
            sharpe = mean_ret / std_ret if std_ret > 0 else 0.0
            
            # Win Rate
            win_rate = np.sum(synthetic_returns > 0) / n_trades
            
            simulated_metrics['total_return'].append(total_ret)
            simulated_metrics['max_drawdown'].append(max_dd)
            simulated_metrics['sharpe_ratio'].append(sharpe)
            simulated_metrics['win_rate'].append(win_rate)
            
        return self._generate_report(simulated_metrics)

    def _generate_report(self, metrics: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate percentiles and risk statistics."""
        
        def get_percentiles(data):
            return {
                'p05': float(np.percentile(data, 5)),
                'p50': float(np.percentile(data, 50)),
                'p95': float(np.percentile(data, 95))
            }
            
        total_returns = np.array(metrics['total_return'])
        max_drawdowns = np.array(metrics['max_drawdown'])
        
        # Probability of Loss (Total Return < 0)
        prob_loss = np.sum(total_returns < 0) / len(total_returns)
        
        # VaR (Value at Risk) 95% - The 5th percentile of returns
        var_95 = np.percentile(total_returns, 5)
        
        # CVaR (Conditional VaR) - Mean of returns below VaR
        cvar_95 = np.mean(total_returns[total_returns <= var_95]) if np.any(total_returns <= var_95) else var_95
        
        return {
            'simulations': len(total_returns),
            'risk_metrics': {
                'probability_of_loss': float(prob_loss),
                'var_95': float(var_95),
                'cvar_95': float(cvar_95),
                'worst_case_drawdown': float(np.max(max_drawdowns))
            },
            'distribution': {
                'total_return': get_percentiles(metrics['total_return']),
                'max_drawdown': get_percentiles(metrics['max_drawdown']),
                'sharpe_ratio': get_percentiles(metrics['sharpe_ratio']),
                'win_rate': get_percentiles(metrics['win_rate'])
            }
        }

    def _empty_report(self):
        return {
            'simulations': 0,
            'risk_metrics': {},
            'distribution': {}
        }

