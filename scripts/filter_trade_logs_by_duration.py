"""
Filter original trade logs: exclude trades outside 9:16–15:20 IST, remove rows where
entry time and exit time differ by <= 3 seconds, then report total PnL by exchange (NSE, BSE).

Usage:
  python scripts/filter_trade_logs_by_duration.py [--trade-logs-dir trade_logs] [--max-diff-sec 3]
"""

from __future__ import annotations

import argparse
import sys
from datetime import time
from pathlib import Path

import pandas as pd

# Only consider trades with entry between 9:16 and 15:20 IST
ENTRY_TIME_START = time(9, 16, 0)   # 9:16 AM IST
ENTRY_TIME_END = time(15, 20, 0)    # 3:20 PM IST

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_ts(ts):
    if pd.isna(ts):
        return None
    try:
        t = pd.to_datetime(ts)
        if t.tzinfo is None:
            t = t.tz_localize("Asia/Kolkata", ambiguous=True)
        else:
            t = t.tz_convert("Asia/Kolkata")
        return t
    except Exception:
        return None


def load_all_trades(trade_logs_dir: Path) -> pd.DataFrame:
    if not trade_logs_dir.exists():
        return pd.DataFrame()
    frames = []
    for f in sorted(trade_logs_dir.glob("trades_*.csv")):
        try:
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["pnl"] = pd.to_numeric(out["pnl"], errors="coerce")
    return out


def main():
    parser = argparse.ArgumentParser(description="Filter trade logs by entry–exit duration and report PnL by exchange")
    parser.add_argument("--trade-logs-dir", type=Path, default=Path("trade_logs"))
    parser.add_argument("--max-diff-sec", type=float, default=3.0, help="Remove rows where exit - entry <= this many seconds")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    trade_logs_dir = root / args.trade_logs_dir if not args.trade_logs_dir.is_absolute() else args.trade_logs_dir

    df = load_all_trades(trade_logs_dir)
    if df.empty:
        print("No trades loaded.")
        return 1

    df["_entry_ts"] = df["entry_timestamp"].map(parse_ts)
    df["_exit_ts"] = df["exit_timestamp"].map(parse_ts)
    df["_duration_sec"] = df.apply(
        lambda r: (r["_exit_ts"] - r["_entry_ts"]).total_seconds() if r["_entry_ts"] is not None and r["_exit_ts"] is not None else None,
        axis=1,
    )

    # Exclude trades outside 9:16–15:20 IST (by entry time)
    in_window = df["_entry_ts"].apply(
        lambda ts: ts.time() >= ENTRY_TIME_START and ts.time() <= ENTRY_TIME_END if ts is not None and hasattr(ts, "time") else False
    )
    df = df[in_window].copy()

    before = len(df)
    filtered = df[df["_duration_sec"].notna() & (df["_duration_sec"] > args.max_diff_sec)].copy()
    after = len(filtered)
    removed = before - after

    if "exchange" not in filtered.columns:
        print("No 'exchange' column in trade logs.")
        return 1

    pnl_by_exchange = filtered.groupby("exchange")["pnl"].agg(["sum", "count"])
    total_pnl = filtered["pnl"].sum()

    print("Filters: entry 9:16–15:20 IST only; remove (exit - entry) <= {} sec".format(args.max_diff_sec))
    print("Rows (after time window) before duration filter: {}, after: {} (removed by duration: {})".format(before, after, removed))
    print("")
    print("PnL by exchange (after filter):")
    print(pnl_by_exchange.to_string())
    print("")
    print("Total PnL (all exchanges): {:.2f}".format(total_pnl))
    if "NSE" in pnl_by_exchange.index:
        print("NSE PnL: {:.2f}".format(pnl_by_exchange.loc["NSE", "sum"]))
    if "BSE" in pnl_by_exchange.index:
        print("BSE PnL: {:.2f}".format(pnl_by_exchange.loc["BSE", "sum"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
