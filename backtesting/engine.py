"""
Vectorised backtesting engine for OI Gemini.

Implements the Phase 1 roadmap requirement to replay saved minute-level
snapshots, route features through the ML signal generator, and evaluate
PnL with configurable transaction costs and slippage.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import database_new as db
from feature_engineering import REQUIRED_FEATURE_COLUMNS, FeatureEngineeringError, prepare_training_features
from ml_core import MLSignalGenerator
from risk_manager import calculate_trading_metrics, get_optimal_position_size
from backtesting.monte_carlo import MonteCarloTester

LOGGER = logging.getLogger(__name__)


def _ensure_date(value: date | datetime | str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return datetime.fromisoformat(value).date()
    raise ValueError(f"Unsupported date input: {value!r}")


def _compute_drawdown(equity_curve: Sequence[float]) -> float:
    if not equity_curve:
        return 0.0
    arr = np.array(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(arr)
    drawdowns = running_max - arr
    return float(np.max(drawdowns))


@dataclass
class BacktestConfig:
    exchange: str
    start: date | datetime | str
    end: date | datetime | str
    strategy: str = "ml_signal"
    holding_period_minutes: int = 15
    transaction_cost_bps: float = 2.0
    slippage_bps: float = 1.0
    min_confidence: float = 0.55
    max_trades: Optional[int] = None
    account_size: float = 1_000_000.0
    margin_per_lot: float = 75_000.0
    max_risk_per_trade: float = 0.02
    limit_rows: Optional[int] = None

    def __post_init__(self) -> None:
        self.start = _ensure_date(self.start)
        self.end = _ensure_date(self.end)
        if self.end < self.start:
            raise ValueError("End date must be >= start date.")
        if self.holding_period_minutes <= 0:
            raise ValueError("holding_period_minutes must be positive.")
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("min_confidence must be between 0 and 1.")


@dataclass
class TradeRecord:
    timestamp: str
    signal: str
    direction: int
    confidence: float
    rationale: str
    future_return: float
    gross_pnl: float
    net_pnl: float
    transaction_cost: float
    capital_allocated: float
    position_fraction: float
    recommended_lots: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload['metadata'] = dict(self.metadata or {})
        return payload


@dataclass
class BacktestResult:
    config: BacktestConfig
    trades: List[TradeRecord]
    metrics: Dict[str, float]
    equity_curve: List[Dict[str, float]]
    raw_rows: int = 0
    monte_carlo_report: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                **{k: v for k, v in asdict(self.config).items() if k not in {"start", "end"}},
                "start": self.config.start.isoformat(),
                "end": self.config.end.isoformat(),
            },
            "metrics": self.metrics,
            "equity_curve": self.equity_curve,
            "trades": [trade.to_dict() for trade in self.trades],
            "raw_rows": self.raw_rows,
            "monte_carlo_report": self.monte_carlo_report
        }


class BacktestEngine:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.signal_engine = MLSignalGenerator(config.exchange)

    def run(self) -> BacktestResult:
        frame = self._prepare_frame()
        if frame.empty:
            LOGGER.warning("No data available for %s between %s and %s", self.config.exchange, self.config.start, self.config.end)
            return BacktestResult(self.config, [], {}, [], raw_rows=0)

        if not self.signal_engine.models_loaded:
            LOGGER.error("Models not loaded for exchange %s. Aborting backtest.", self.config.exchange)
            return BacktestResult(self.config, [], {}, [], raw_rows=len(frame))

        trades: List[TradeRecord] = []
        predictions: List[int] = []
        actual_returns: List[float] = []
        position_sizes: List[float] = []
        equity_curve: List[Dict[str, float]] = []
        gross_equity = 0.0
        net_equity = 0.0
        net_pnls: List[float] = []

        trade_limit = self.config.max_trades or float("inf")
        total_cost_rate = (self.config.transaction_cost_bps + self.config.slippage_bps) / 10000.0

        for _, row in frame.iterrows():
            if len(trades) >= trade_limit:
                break

            features = {col: float(row.get(col, 0.0)) for col in REQUIRED_FEATURE_COLUMNS}
            signal, confidence, rationale, metadata = self.signal_engine.generate_signal(features)

            if signal == 'HOLD' or confidence < self.config.min_confidence:
                continue

            direction = 1 if signal == 'BUY' else -1 if signal == 'SELL' else 0
            if direction == 0:
                continue

            risk = get_optimal_position_size(
                ml_confidence=confidence,
                win_rate=self.signal_engine.strategy_metrics['win_rate'],
                avg_win_loss_ratio=self.signal_engine.strategy_metrics['avg_w_l_ratio'],
                max_risk=self.config.max_risk_per_trade,
                account_size=self.config.account_size,
                margin_per_lot=self.config.margin_per_lot,
            )
            capital_allocated = risk.get('capital_allocated', 0.0)
            if capital_allocated <= 0.0:
                continue

            future_return = float(row['future_return'])
            gross_pnl = direction * future_return * capital_allocated
            transaction_cost = abs(capital_allocated) * total_cost_rate
            net_pnl = gross_pnl - transaction_cost

            predictions.append(direction)
            actual_returns.append(future_return)
            position_sizes.append(capital_allocated)
            net_pnls.append(net_pnl)

            gross_equity += gross_pnl
            net_equity += net_pnl
            equity_curve.append({
                "timestamp": row['timestamp'],
                "gross_equity": round(gross_equity, 2),
                "net_equity": round(net_equity, 2),
            })

            trade = TradeRecord(
                timestamp=row['timestamp'],
                signal=signal,
                direction=direction,
                confidence=confidence,
                rationale=rationale,
                future_return=future_return,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                transaction_cost=transaction_cost,
                capital_allocated=capital_allocated,
                position_fraction=risk.get('fraction', 0.0),
                recommended_lots=risk.get('recommended_lots', 0),
                metadata=metadata or {},
            )
            trades.append(trade)

        metrics: Dict[str, float] = {}
        if predictions:
            metrics = calculate_trading_metrics(predictions, actual_returns, position_sizes)
            metrics['net_total_pnl'] = float(np.sum(net_pnls))
            metrics['gross_total_pnl'] = float(metrics.get('total_pnl', 0.0))
            metrics['avg_trade_pnl'] = float(np.mean(net_pnls)) if net_pnls else 0.0
            metrics['num_trades'] = len(trades)
            metrics['cost_per_trade'] = float(total_cost_rate * self.config.account_size * self.config.max_risk_per_trade)
            metrics['net_max_drawdown'] = _compute_drawdown([point['net_equity'] for point in equity_curve])
            metrics['gross_max_drawdown'] = _compute_drawdown([point['gross_equity'] for point in equity_curve])
        else:
            LOGGER.warning("No qualifying trades generated for %s.", self.config.exchange)

        # Run Monte Carlo Simulation
        result = BacktestResult(self.config, trades, metrics, equity_curve, raw_rows=len(frame))
        
        if trades:
            try:
                mc_tester = MonteCarloTester()
                mc_report = mc_tester.run_simulation(result, n_simulations=1000)
                result.monte_carlo_report = mc_report
                LOGGER.info(f"Monte Carlo Simulation completed. VaR 95%: {mc_report['risk_metrics']['var_95']:.2%}")
            except Exception as e:
                LOGGER.error(f"Monte Carlo simulation failed: {e}", exc_info=True)

        return result

    def _prepare_frame(self) -> pd.DataFrame:
        try:
            raw = db.load_historical_data_for_ml(self.config.exchange, self.config.start, self.config.end)
        except Exception as exc:
            LOGGER.error("Failed to load historical data: %s", exc, exc_info=True)
            return pd.DataFrame()

        if raw is None or raw.empty:
            return pd.DataFrame()

        try:
            feature_frame = prepare_training_features(raw, required_columns=REQUIRED_FEATURE_COLUMNS)
        except FeatureEngineeringError as exc:
            LOGGER.error("Feature preparation failed: %s", exc)
            return pd.DataFrame()

        feature_frame = feature_frame.copy()
        feature_frame.reset_index(inplace=True)
        feature_frame.rename(columns={'index': 'timestamp'}, inplace=True)
        feature_frame['timestamp'] = feature_frame['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        feature_frame = feature_frame[feature_frame['underlying_price'] > 0]
        feature_frame['future_price'] = feature_frame['underlying_price'].shift(-self.config.holding_period_minutes)
        feature_frame['future_return'] = (
            feature_frame['future_price'] / feature_frame['underlying_price'] - 1.0
        )
        feature_frame.dropna(subset=['future_return'], inplace=True)
        feature_frame.drop(columns=['future_price'], inplace=True)

        if self.config.limit_rows:
            feature_frame = feature_frame.head(self.config.limit_rows)

        return feature_frame

