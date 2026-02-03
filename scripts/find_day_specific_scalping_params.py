#!/usr/bin/env python3
"""
Find day-specific parameters that work best for ITM scalping (15 pts NSE / 30 pts BSE).

Sweeps divergence_min, min_hour, hold_minutes (and optionally volume thresholds),
runs the scalping backtest with each param set, groups trades by day_of_week,
and picks the param set that maximizes total points per weekday (with minimum trades).

Usage:
  python scripts/find_day_specific_scalping_params.py --exchange NSE --days 60
  python scripts/find_day_specific_scalping_params.py --exchange BSE --start 2026-01-01 --end 2026-01-30 --output reports/day_params_bse.json
"""

import argparse
import importlib.util
import itertools
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# Load backtest module from same directory (run from project root: python scripts/find_day_specific_scalping_params.py)
_script_dir = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("backtest_itm", _script_dir / "backtest_itm_scalping_15_30.py")
_bt = importlib.util.module_from_spec(_spec)
sys.path.insert(0, str(_script_dir.parent))
_spec.loader.exec_module(_bt)
DEFAULT_PARAMS = _bt.DEFAULT_PARAMS
load_ml_features = _bt.load_ml_features
run_backtest = _bt.run_backtest

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Param grid for sweep (keep small for speed; expand if needed)
PARAM_GRID = {
    "divergence_min": [3.0, 5.0, 7.0],
    "min_hour": [10, 11, 12],
    "hold_minutes": [15, 30],
}
# Optional: volume thresholds (uncomment to sweep)
# "nse_pe_vol_bullish_max": [-5.0, -3.0, -1.0],
# "bse_pe_vol_bullish_min": [2.0, 3.8, 5.0],

MIN_TRADES_PER_DAY = 3  # Require at least this many trades for a day/param combo to count


def _param_grid_to_list() -> list:
    """Turn PARAM_GRID into list of (label, params_dict)."""
    keys = list(PARAM_GRID.keys())
    values = [PARAM_GRID[k] for k in keys]
    out = []
    for combo in itertools.product(*values):
        params = dict(DEFAULT_PARAMS)
        params.update(dict(zip(keys, combo)))
        label = "_".join(f"{k}={v}" for k, v in zip(keys, combo))
        out.append((label, params))
    return out


def run_sweep(
    df: pd.DataFrame,
    exchange: str,
) -> list:
    """Run backtest for each param set; return list of (param_label, trades)."""
    results = []
    grid = _param_grid_to_list()
    for i, (label, params) in enumerate(grid):
        trades, _ = run_backtest(df, exchange, params=params, include_day_fields=True)
        results.append((label, params, trades))
    return results


def best_params_by_day(
    sweep_results: list,
    exchange: str,
    min_trades: int = MIN_TRADES_PER_DAY,
) -> dict:
    """
    For each day_of_week, find param set that maximizes total points (with >= min_trades).
    sweep_results: list of (param_label, params_dict, trades_list).
    """
    # Build: (day_of_week, param_label) -> [trades]
    by_day_param = {}
    for label, _params, trades in sweep_results:
        for t in trades:
            dow = t.get("day_of_week")
            if dow is None:
                continue
            key = (dow, label)
            if key not in by_day_param:
                by_day_param[key] = []
            by_day_param[key].append(t)

    # Aggregate: (day_of_week, param_label) -> {trades, wins, total_points, win_rate}
    agg = {}
    for (dow, label), trade_list in by_day_param.items():
        n = len(trade_list)
        wins = sum(1 for t in trade_list if t.get("hit_target"))
        total_pts = sum(t.get("points", 0) for t in trade_list)
        agg[(dow, label)] = {
            "trades": n,
            "wins": wins,
            "total_points": round(total_pts, 2),
            "win_rate_pct": round(100.0 * wins / n, 1) if n else 0,
        }

    # Get unique param labels from sweep_results
    param_labels = list({label for label, _p, _t in sweep_results})
    # For each day_of_week, pick best param_label by total_points (only if trades >= min_trades)
    best_by_day = {}
    for dow in range(7):
        candidates = [
            (label, agg[(dow, label)])
            for label in param_labels
            if (dow, label) in agg and agg[(dow, label)]["trades"] >= min_trades
        ]
        if not candidates:
            best_by_day[dow] = {"day_name": DAY_NAMES[dow], "best_label": None, "stats": None, "params": None}
            continue
        best_label = max(candidates, key=lambda x: x[1]["total_points"])[0]
        stats = agg[(dow, best_label)]
        params = next(p for l, p, _ in sweep_results if l == best_label)
        best_by_day[dow] = {
            "day_name": DAY_NAMES[dow],
            "best_label": best_label,
            "stats": stats,
            "params": params,
        }
    return best_by_day


def main() -> None:
    parser = argparse.ArgumentParser(description="Find day-specific best params for ITM scalping")
    parser.add_argument("--exchange", type=str, default="NSE", choices=["NSE", "BSE"])
    parser.add_argument("--days", type=int, default=60, help="Last N days of data")
    parser.add_argument("--start", type=str, default="", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="", help="End date YYYY-MM-DD")
    parser.add_argument("--min-trades", type=int, default=MIN_TRADES_PER_DAY, help="Min trades to pick best params per day")
    parser.add_argument("--output", type=str, default="", help="Write JSON report here")
    args = parser.parse_args()
    exchange = args.exchange.upper()

    end_date = datetime.now(timezone.utc).date()
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        start_date = end_date - timedelta(days=args.days)

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

    print("=" * 72)
    print("DAY-SPECIFIC SCALPING PARAMS (15 pts NSE / 30 pts BSE)")
    print("=" * 72)
    print(f"Exchange: {exchange}, Range: {start_date} to {end_date}")
    print(f"Param grid: {PARAM_GRID}")
    print(f"Min trades per day to qualify: {args.min_trades}")
    print()

    df = load_ml_features(exchange, start_dt, end_dt)
    if df.empty:
        print("No ml_features data for this range.")
        sys.exit(1)
    print(f"Loaded {len(df)} minute rows.")

    sweep_results = [(label, params, trades) for label, params, trades in run_sweep(df, exchange)]
    best_by_day = best_params_by_day(sweep_results, exchange, min_trades=args.min_trades)

    print("\n--- Best params by day of week ---")
    for dow in range(7):
        info = best_by_day[dow]
        name = info["day_name"]
        if info["best_label"] is None:
            print(f"  {name}: (no param set with >={args.min_trades} trades)")
            continue
        st = info["stats"]
        p = info["params"]
        print(f"  {name}: {info['best_label']}")
        print(f"    trades={st['trades']}, win_rate={st['win_rate_pct']}%, total_points={st['total_points']}")
        print(f"    params: divergence_min={p.get('divergence_min')}, min_hour={p.get('min_hour')}, hold_minutes={p.get('hold_minutes')}")

    if args.output:
        report = {
            "exchange": exchange,
            "start": str(start_date),
            "end": str(end_date),
            "param_grid": PARAM_GRID,
            "min_trades": args.min_trades,
            "best_by_day": {
                DAY_NAMES[dow]: {
                    "best_label": best_by_day[dow]["best_label"],
                    "stats": best_by_day[dow]["stats"],
                    "params": best_by_day[dow]["params"],
                }
                for dow in range(7)
            },
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
