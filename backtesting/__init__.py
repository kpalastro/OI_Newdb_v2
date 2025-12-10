"""
Backtesting utilities for the OI Gemini roadmap.

This package bundles the minute-by-minute replay engine plus CLI helpers
so researchers can evaluate ML signals with configurable transaction costs,
slippage, and holding windows.
"""

from .engine import BacktestConfig, BacktestEngine, BacktestResult, TradeRecord  # noqa: F401

