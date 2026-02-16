"""
Analyze historic trade logs to suggest times to AVOID (more losses) and
PROFITABLE times (day-wise and hour-wise).

Only includes trades with entry between 9:15 AM and 3:30 PM IST (market hours).
Uses trade_logs/trades_*.csv. Outputs:
  - By calendar date: worst vs best days
  - By day-of-week: avoid vs prefer days
  - By hour (entry): avoid vs prefer hours

Usage:
  python scripts/analyze_trade_timing.py [--trade-logs-dir trade_logs] [--out reports/]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import time
from pathlib import Path

import pandas as pd

# Market hours (IST): include only trades with entry between 9:15 AM and 3:30 PM
MARKET_START = time(9, 15)
MARKET_END = time(15, 30)
# For "max win rate" band: use only trades from last N trading days (not min-trade count)
LAST_N_TRADING_DAYS = 10

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TRADE_LOG_COLUMNS = [
    "entry_timestamp", "exit_timestamp", "exchange", "position_id",
    "symbol", "type", "side", "quantity", "entry_price", "exit_price",
    "pnl", "entry_reason", "exit_reason", "status", "confidence",
    "kelly_fraction", "constraint_violation", "signal_id",
]

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load_all_trades(trade_logs_dir: Path) -> pd.DataFrame:
    """Load and concatenate all trades_*.csv; add entry_ts, entry_date, entry_hour, day_of_week."""
    if not trade_logs_dir.exists():
        LOGGER.warning("Trade logs dir does not exist: %s", trade_logs_dir)
        return pd.DataFrame()

    frames = []
    for f in sorted(trade_logs_dir.glob("trades_*.csv")):
        try:
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
            frames.append(df)
        except Exception as e:
            LOGGER.warning("Skip %s: %s", f.name, e)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)

    if "pnl" in out.columns:
        out["pnl"] = pd.to_numeric(out["pnl"], errors="coerce")
    if "confidence" in out.columns:
        out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")
    if "entry_timestamp" in out.columns:
        out["entry_ts"] = pd.to_datetime(out["entry_timestamp"], errors="coerce")
        out["entry_date"] = out["entry_ts"].dt.date
        out["entry_hour"] = out["entry_ts"].dt.hour
        out["day_of_week"] = out["entry_ts"].dt.dayofweek  # 0=Mon, 6=Sun

    return out


def _agg_pnl(g: pd.DataFrame) -> pd.Series:
    pnl = g["pnl"].dropna()
    losses = pnl[pnl < 0]
    wins = pnl[pnl > 0]
    return pd.Series({
        "trades": len(pnl),
        "total_pnl": pnl.sum(),
        "total_loss": losses.sum() if len(losses) else 0,
        "total_profit": wins.sum() if len(wins) else 0,
        "loss_count": (pnl < 0).sum(),
        "win_count": (pnl > 0).sum(),
        "win_rate_pct": (pnl > 0).mean() * 100 if len(pnl) else 0,
    })


def analyze_timing(df: pd.DataFrame):
    """Compute day-wise and hour-wise PnL; return dict of DataFrames and recommendations."""
    if df.empty or "pnl" not in df.columns:
        return None

    df = df.dropna(subset=["pnl"]).copy()
    if df.empty:
        return None

    # By calendar date
    by_date = df.groupby("entry_date", dropna=False, group_keys=False).apply(_agg_pnl, include_groups=False).reset_index()
    by_date = by_date.sort_values("total_pnl")

    # By day-of-week
    by_dow = df.groupby("day_of_week", dropna=False, group_keys=False).apply(_agg_pnl, include_groups=False).reset_index()
    by_dow["day_name"] = by_dow["day_of_week"].map(lambda x: DAY_NAMES[int(x)] if pd.notna(x) else "?")
    by_dow = by_dow.sort_values("total_pnl")

    # By hour
    by_hour = df.groupby("entry_hour", dropna=False, group_keys=False).apply(_agg_pnl, include_groups=False).reset_index()
    by_hour = by_hour.sort_values("total_pnl")

    # Recommendations: worst vs best (by total_pnl)
    n_avoid_days = min(2, max(1, len(by_dow) // 3))
    n_avoid_hours = min(4, max(2, len(by_hour) // 3))
    avoid_days = by_dow.head(n_avoid_days)
    prefer_days = by_dow.tail(n_avoid_days)
    avoid_hours = by_hour.head(n_avoid_hours)
    prefer_hours = by_hour.tail(n_avoid_hours)

    return {
        "by_date": by_date,
        "by_dow": by_dow,
        "by_hour": by_hour,
        "avoid_days": avoid_days,
        "prefer_days": prefer_days,
        "avoid_hours": avoid_hours,
        "prefer_hours": prefer_hours,
        "total_trades": len(df),
        "overall_pnl": df["pnl"].sum(),
        "overall_win_rate": (df["pnl"] > 0).mean() * 100,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze trade timing from historic logs")
    parser.add_argument("--trade-logs-dir", type=Path, default=Path("trade_logs"), help="Path to trade_logs folder")
    parser.add_argument("--out", type=Path, default=None, help="Write by_date, by_dow, by_hour to CSV in this dir (e.g. reports/)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    trade_logs_dir = args.trade_logs_dir if args.trade_logs_dir.is_absolute() else project_root / args.trade_logs_dir

    LOGGER.info("Loading trades from %s", trade_logs_dir)
    df = load_all_trades(trade_logs_dir)
    if df.empty:
        LOGGER.error("No trades loaded.")
        return 1

    df = df.dropna(subset=["pnl"])
    if df.empty:
        LOGGER.error("No closed trades with PnL.")
        return 1

    # Filter: entry between 9:15 AM and 3:30 PM IST
    if "entry_ts" in df.columns:
        entry_time = df["entry_ts"].dt.time
        mask = (entry_time >= MARKET_START) & (entry_time <= MARKET_END)
        df = df.loc[mask].copy()
        LOGGER.info("Filtered to market hours (9:15–15:30 IST): %d trades", len(df))

    if df.empty:
        LOGGER.error("No trades in market hours 9:15–15:30.")
        return 1

    LOGGER.info("Total closed trades with PnL (market hours): %d", len(df))

    # PnL for trades with 0.6 < confidence < 0.9
    if "confidence" in df.columns:
        mask_conf = (df["confidence"] > 0.6) & (df["confidence"] < 0.9)
        df_conf = df.loc[mask_conf]
        if not df_conf.empty:
            pnl_conf = df_conf["pnl"].sum()
            n_conf = len(df_conf)
            win_rate_conf = (df_conf["pnl"] > 0).mean() * 100
            LOGGER.info("")
            LOGGER.info("=== PnL for confidence in (0.6, 0.9) ===")
            LOGGER.info("Trades: %d | Total PnL: %s | Win rate: %.1f%%", n_conf, pnl_conf, win_rate_conf)
            LOGGER.info("")

        # Win rate by confidence band; find band with maximum win rate
        df_with_conf = df.dropna(subset=["confidence"]).copy()
        if not df_with_conf.empty:
            bins = [0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 2.0]
            labels = ["0-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.75", "0.75-0.8", "0.8-0.85", "0.85-0.9", "0.9-0.95", "0.95-1.0", "1.0+"]
            df_with_conf["conf_band"] = pd.cut(df_with_conf["confidence"], bins=bins, labels=labels, include_lowest=True)
            by_conf = df_with_conf.groupby("conf_band", observed=True).agg(
                trades=("pnl", "count"),
                win_rate_pct=("pnl", lambda s: (s > 0).mean() * 100 if len(s) else 0),
                total_pnl=("pnl", "sum"),
            ).reset_index()
            by_conf = by_conf[by_conf["trades"] > 0]
            if not by_conf.empty:
                LOGGER.info("=== Win rate by confidence band ===")
                for _, r in by_conf.iterrows():
                    LOGGER.info("  %s  trades: %d  win_rate: %.1f%%  total_pnl: %s",
                                r["conf_band"], int(r["trades"]), r["win_rate_pct"], r["total_pnl"])
                # Max win rate: use only last N trading days
                if "entry_date" in df_with_conf.columns:
                    unique_dates = sorted(df_with_conf["entry_date"].dropna().unique())
                    last_n_dates = unique_dates[-LAST_N_TRADING_DAYS:] if len(unique_dates) >= LAST_N_TRADING_DAYS else unique_dates
                    df_recent = df_with_conf[df_with_conf["entry_date"].isin(last_n_dates)]
                else:
                    last_n_dates = []
                    df_recent = df_with_conf
                by_conf_recent = df_recent.groupby("conf_band", observed=True).agg(
                    trades=("pnl", "count"),
                    win_rate_pct=("pnl", lambda s: (s > 0).mean() * 100 if len(s) else 0),
                    total_pnl=("pnl", "sum"),
                ).reset_index()
                by_conf_recent = by_conf_recent[by_conf_recent["trades"] > 0]
                if not by_conf_recent.empty:
                    best = by_conf_recent.loc[by_conf_recent["win_rate_pct"].idxmax()]
                    n_days_used = len(last_n_dates)
                    LOGGER.info("")
                    LOGGER.info("=== Confidence band with MAXIMUM win rate (last %d trading days) ===", n_days_used)
                    LOGGER.info("Band: %s  |  Win rate: %.1f%%  |  Trades: %d  |  Total PnL: %s",
                                best["conf_band"], best["win_rate_pct"], int(best["trades"]), best["total_pnl"])
                LOGGER.info("")

        # Same analysis by exchange (NSE / BSE); max win rate from last N trading days
        if "exchange" in df.columns and not df_with_conf.empty:
            unique_dates = sorted(df_with_conf["entry_date"].dropna().unique()) if "entry_date" in df_with_conf.columns else []
            last_n_dates = unique_dates[-LAST_N_TRADING_DAYS:] if len(unique_dates) >= LAST_N_TRADING_DAYS else unique_dates
            for ex in ["NSE", "BSE"]:
                df_ex = df_with_conf[df_with_conf["exchange"] == ex]
                if df_ex.empty:
                    continue
                by_conf_ex = df_ex.groupby("conf_band", observed=True).agg(
                    trades=("pnl", "count"),
                    win_rate_pct=("pnl", lambda s: (s > 0).mean() * 100 if len(s) else 0),
                    total_pnl=("pnl", "sum"),
                ).reset_index()
                by_conf_ex = by_conf_ex[by_conf_ex["trades"] > 0]
                if by_conf_ex.empty:
                    continue
                LOGGER.info("=== %s: Win rate by confidence band ===", ex)
                for _, r in by_conf_ex.iterrows():
                    LOGGER.info("  %s  trades: %d  win_rate: %.1f%%  total_pnl: %s",
                                r["conf_band"], int(r["trades"]), r["win_rate_pct"], r["total_pnl"])
                df_ex_recent = df_ex[df_ex["entry_date"].isin(last_n_dates)] if (last_n_dates and "entry_date" in df_ex.columns) else df_ex
                by_ex_recent = df_ex_recent.groupby("conf_band", observed=True).agg(
                    trades=("pnl", "count"),
                    win_rate_pct=("pnl", lambda s: (s > 0).mean() * 100 if len(s) else 0),
                    total_pnl=("pnl", "sum"),
                ).reset_index()
                by_ex_recent = by_ex_recent[by_ex_recent["trades"] > 0]
                if not by_ex_recent.empty:
                    best = by_ex_recent.loc[by_ex_recent["win_rate_pct"].idxmax()]
                    LOGGER.info("  -> Max win rate band (last %d trading days): %s  |  Win rate: %.1f%%  |  Trades: %d  |  PnL: %s",
                                len(last_n_dates), best["conf_band"], best["win_rate_pct"], int(best["trades"]), best["total_pnl"])
                LOGGER.info("")

    res = analyze_timing(df)
    if not res:
        LOGGER.error("Analysis failed.")
        return 1

    # --- Summary ---
    LOGGER.info("")
    LOGGER.info("=== OVERALL ===")
    LOGGER.info("Total PnL: %s | Win rate: %.1f%%", res["overall_pnl"], res["overall_win_rate"])
    LOGGER.info("")

    # --- Worst / best calendar days (sample) ---
    by_date = res["by_date"]
    worst_dates = by_date.head(5)
    best_dates = by_date.tail(5)
    LOGGER.info("=== BY CALENDAR DATE (sample: worst 5 / best 5) ===")
    LOGGER.info("Worst days (consider avoiding similar conditions):")
    for _, r in worst_dates.iterrows():
        LOGGER.info("  %s  PnL: %s  trades: %d  win_rate: %.1f%%", r["entry_date"], r["total_pnl"], int(r["trades"]), r["win_rate_pct"])
    LOGGER.info("Best days:")
    for _, r in best_dates.iterrows():
        LOGGER.info("  %s  PnL: %s  trades: %d  win_rate: %.1f%%", r["entry_date"], r["total_pnl"], int(r["trades"]), r["win_rate_pct"])
    LOGGER.info("")

    # --- Day-of-week: AVOID vs PREFER ---
    LOGGER.info("=== BY DAY OF WEEK ===")
    print(res["by_dow"][["day_name", "total_pnl", "trades", "win_rate_pct", "total_loss", "total_profit"]].to_string(index=False))
    LOGGER.info("")
    LOGGER.info("--- RECOMMENDATION: DAY OF WEEK ---")
    avoid_d = res["avoid_days"]
    prefer_d = res["prefer_days"]
    if not avoid_d.empty:
        avoid_names = ", ".join(avoid_d["day_name"].astype(str).tolist())
        LOGGER.info("  AVOID or reduce trading (most loss / lowest PnL): %s", avoid_names)
    if not prefer_d.empty:
        prefer_names = ", ".join(prefer_d["day_name"].astype(str).tolist())
        LOGGER.info("  PREFER / focus (most profit): %s", prefer_names)
    LOGGER.info("")

    # --- Hour: AVOID vs PREFER ---
    LOGGER.info("=== BY ENTRY HOUR (IST) ===")
    print(res["by_hour"][["entry_hour", "total_pnl", "trades", "win_rate_pct", "total_loss", "total_profit"]].to_string(index=False))
    LOGGER.info("")
    LOGGER.info("--- RECOMMENDATION: TIME OF DAY ---")
    avoid_h = res["avoid_hours"]
    prefer_h = res["prefer_hours"]
    if not avoid_h.empty:
        avoid_hrs = ", ".join(avoid_h["entry_hour"].astype(int).astype(str).tolist())
        LOGGER.info("  AVOID or reduce trading at entry hours (IST): %s", avoid_hrs)
    if not prefer_h.empty:
        prefer_hrs = ", ".join(prefer_h["entry_hour"].astype(int).astype(str).tolist())
        LOGGER.info("  PREFER entry hours (IST): %s", prefer_hrs)
    LOGGER.info("")

    if args.out:
        out_dir = args.out if args.out.is_absolute() else project_root / args.out
        if out_dir.suffix:
            out_dir = out_dir.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        res["by_date"].to_csv(out_dir / "timing_by_date.csv", index=False)
        res["by_dow"].to_csv(out_dir / "timing_by_day_of_week.csv", index=False)
        res["by_hour"].to_csv(out_dir / "timing_by_hour.csv", index=False)
        LOGGER.info("Wrote timing_by_date.csv, timing_by_day_of_week.csv, timing_by_hour.csv to %s", out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
