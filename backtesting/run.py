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
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from .engine import BacktestConfig, BacktestEngine


def _sanitize_for_json(obj: Any) -> Any:
    """
    Recursively replace NaN and Infinity values with None (which becomes null in JSON).
    JSON doesn't support NaN/Infinity, so we convert them to null.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    else:
        return obj


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
    parser.add_argument("--min-confidence", type=float, default=0.6, help="Minimum ML confidence to trade.")
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

    # Print daily summary: target hits vs stop losses, total profit per day
    daily = getattr(result, "daily_summary", None) or []
    if daily:
        logging.info("")
        logging.info("=== Daily summary (Target hits vs Stop losses, Total PnL) ===")
        logging.info("%-12s | %6s | %8s | %10s | %8s | %14s | %14s",
                     "Date", "Trades", "Target", "StopLoss", "Breakeven", "Total Net PnL", "Total Gross PnL")
        logging.info("-" * 90)
        for row in daily:
            logging.info("%-12s | %6d | %8d | %10d | %8d | %14.2f | %14.2f",
                         row.get("date", ""),
                         row.get("num_trades", 0),
                         row.get("target_hits", 0),
                         row.get("stop_losses", 0),
                         row.get("breakeven", 0),
                         row.get("total_net_pnl", 0.0),
                         row.get("total_gross_pnl", 0.0))
        logging.info("-" * 90)
        total_net = sum(r.get("total_net_pnl", 0) for r in daily)
        total_trades = sum(r.get("num_trades", 0) for r in daily)
        total_wins = sum(r.get("target_hits", 0) for r in daily)
        total_losses = sum(r.get("stop_losses", 0) for r in daily)
        total_be = sum(r.get("breakeven", 0) for r in daily)
        logging.info("TOTAL        | %6d | %8d | %10d | %8d | %14.2f |",
                     total_trades, total_wins, total_losses, total_be, total_net)
        logging.info("")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result_dict = _sanitize_for_json(result.to_dict())
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result_dict, handle, indent=2)
        logging.info("Saved backtest report to %s", args.output)
    else:
        # Auto-save to dashboard location if no output specified
        backtest_dir = Path('reports') / 'backtests'
        backtest_dir.mkdir(parents=True, exist_ok=True)
        dashboard_file = backtest_dir / f'{args.exchange.upper()}.json'
        try:
            result_dict = _sanitize_for_json(result.to_dict())
            with open(dashboard_file, "w", encoding="utf-8") as handle:
                json.dump(result_dict, handle, indent=2)
            # Verify the file was written correctly
            file_size = dashboard_file.stat().st_size
            num_trades_saved = result_dict.get('metrics', {}).get('num_trades', 0)
            logging.warning("Saved backtest report to dashboard location: %s", dashboard_file)
            logging.warning("File size: %d bytes, Trades saved: %d, Date range: %s to %s", 
                          file_size, num_trades_saved, 
                          result_dict.get('config', {}).get('start'),
                          result_dict.get('config', {}).get('end'))
        except Exception as e:
            logging.error("Failed to save backtest report: %s", e, exc_info=True)


if __name__ == "__main__":
    main()

