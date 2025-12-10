"""
CLI wrapper for the OI Gemini backtesting engine.

Usage:
    python backtesting/run.py --exchange NSE --start 2025-10-01 --end 2025-11-18 \
        --strategy ml_signal --holding-period 15 --cost-bps 2.0 --slippage-bps 1.5
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from .engine import BacktestConfig, BacktestEngine


def _parse_date(value: str):
    try:
        return datetime.fromisoformat(value).date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date '{value}'. Use YYYY-MM-DD.") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay ML signals over historical data.")
    parser.add_argument("--exchange", required=True, choices=["NSE", "BSE"])
    parser.add_argument("--start", required=True, type=_parse_date, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, type=_parse_date, help="End date (YYYY-MM-DD).")
    parser.add_argument("--strategy", default="ml_signal", help="Strategy id (default: ml_signal).")
    parser.add_argument("--holding-period", type=int, default=15, help="Holding window in minutes.")
    parser.add_argument("--cost-bps", type=float, default=2.0, help="Transaction cost in basis points.")
    parser.add_argument("--slippage-bps", type=float, default=1.0, help="Slippage in basis points.")
    parser.add_argument("--min-confidence", type=float, default=0.55, help="Minimum ML confidence to trade.")
    parser.add_argument("--max-trades", type=int, default=None, help="Upper bound on number of trades.")
    parser.add_argument("--account-size", type=float, default=1_000_000.0, help="Account notional in INR.")
    parser.add_argument("--margin-per-lot", type=float, default=75_000.0, help="Margin per index lot.")
    parser.add_argument("--max-risk", type=float, default=0.02, help="Max risk per trade (fraction).")
    parser.add_argument("--limit-rows", type=int, default=None, help="Optional cap on rows for dry-runs.")
    parser.add_argument("--output", type=Path, default=None, help="Path to dump JSON results.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")

    config = BacktestConfig(
        exchange=args.exchange,
        start=args.start,
        end=args.end,
        strategy=args.strategy,
        holding_period_minutes=args.holding_period,
        transaction_cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        min_confidence=args.min_confidence,
        max_trades=args.max_trades,
        account_size=args.account_size,
        margin_per_lot=args.margin_per_lot,
        max_risk_per_trade=args.max_risk,
        limit_rows=args.limit_rows,
    )

    engine = BacktestEngine(config)
    result = engine.run()

    if not result.metrics:
        logging.warning("No metrics returned. Ensure models exist and the date range has data.")
    else:
        logging.info("Backtest complete: %s trades | Net PnL %.2f | Sharpe %.2f",
                     result.metrics.get("num_trades", 0),
                     result.metrics.get("net_total_pnl", 0.0),
                     result.metrics.get("sharpe_ratio", float("nan")))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=2)
        logging.info("Saved backtest report to %s", args.output)


if __name__ == "__main__":
    main()

