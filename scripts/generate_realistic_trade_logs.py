"""
Generate fresh (realistic) trade logs in a separate folder using DB snapshot LTPs.

Reads existing trade_logs/*.csv. Entry: LTP at entry_timestamp (nearest snapshot).
Exit: scans snapshots after entry and finds the first minute where target or stop
is hit (NSE: 25 pts, BSE: 50 pts); records that time and price as exit_timestamp,
exit_price, exit_reason (Target Hit / Stop Loss). So exit reflects model decision
outcome, not old records. If neither hit, falls back to original exit time and LTP.
Only trades with entry between 9:16 and 15:20 IST are included. Writes new CSVs to
trade_logs_realistic/ (or --out-dir). pnl_source: target_stop | db_snapshot | original.

Usage:
  python scripts/generate_realistic_trade_logs.py [--trade-logs-dir trade_logs] [--out-dir trade_logs_realistic] [--db-only]
  --db-only: only include trades when DB had LTP at entry (and exit when not target/stop); skip others.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import time
from pathlib import Path
from typing import Optional, Tuple

# Only consider trades with entry between 9:16 and 15:20 IST
ENTRY_TIME_START = time(9, 16, 0)   # 9:16 AM IST
ENTRY_TIME_END = time(15, 20, 0)    # 3:20 PM IST

import pandas as pd

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)

TRADE_LOG_COLUMNS = [
    "entry_timestamp", "exit_timestamp", "exchange", "position_id",
    "symbol", "type", "side", "quantity", "entry_price", "exit_price",
    "pnl", "entry_reason", "exit_reason", "status", "confidence",
    "kelly_fraction", "constraint_violation", "signal_id",
]


def build_ltp_cache(conn, df: pd.DataFrame) -> dict:
    """
    Pre-load option_chain_snapshots for all (exchange, symbol) and time range in df.
    Returns dict: (exchange, symbol) -> sorted list of (timestamp_utc, ltp).
    """
    from database_new import _get_placeholder
    ph = _get_placeholder()
    cursor = conn.cursor()
    try:
        # Unique (exchange, symbol) and global min/max UTC
        subset = df[["exchange", "symbol", "_entry_ts", "_exit_ts"]].dropna(subset=["_entry_ts", "_exit_ts"])
        if subset.empty:
            return {}
        min_ts = subset["_entry_ts"].min().tz_convert("UTC").to_pydatetime()
        max_ts = subset["_exit_ts"].max().tz_convert("UTC").to_pydatetime()
        pairs = subset[["exchange", "symbol"]].drop_duplicates()

        cache = {}
        for _, r in pairs.iterrows():
            ex, sym = r["exchange"], r["symbol"]
            cursor.execute(
                f"""
                SELECT timestamp, ltp FROM option_chain_snapshots
                WHERE exchange = {ph} AND symbol = {ph} AND timestamp >= {ph} AND timestamp <= {ph}
                ORDER BY timestamp
                """,
                (ex, sym, min_ts, max_ts),
            )
            rows = cursor.fetchall()
            cache[(ex, sym)] = [(row[0], float(row[1])) for row in rows if row[1] is not None]
        LOGGER.info("Loaded LTP cache for %d (exchange, symbol) pairs", len(cache))
        return cache
    finally:
        cursor.close()


def get_ltp_from_cache(cache: dict, exchange: str, symbol: str, ts_utc) -> Optional[float]:
    """
    From pre-loaded cache, return LTP at or just before ts_utc (nearest snapshot).
    ts_utc: timezone-aware datetime (UTC) or naive.
    """
    key = (exchange, symbol)
    if key not in cache or not cache[key]:
        return None
    series = cache[key]
    # Normalize to naive for comparison (DB may return tz-aware UTC)
    if hasattr(ts_utc, "tzinfo") and ts_utc.tzinfo is not None:
        ts_utc = ts_utc.replace(tzinfo=None)
    best = None
    for t, ltp in series:
        t_naive = t.replace(tzinfo=None) if hasattr(t, "tzinfo") and t.tzinfo else t
        if t_naive <= ts_utc:
            best = ltp
        else:
            break
    return best


def find_exit_from_target_stop(
    cache: dict,
    exchange: str,
    symbol: str,
    entry_dt_utc,
    entry_ltp: float,
    side: str,
    target_points: int,
    stop_points: int,
) -> Tuple[Optional[object], Optional[float], Optional[str]]:
    """
    Scan snapshots after entry_dt and return (exit_timestamp_utc, exit_price, exit_reason)
    when target or stop is first hit. All auto-trades are long (BUY options); side 'B'.
    NSE: 25 points, BSE: 50 points for both target and stop.
    Returns (None, None, None) if neither is hit in the snapshot range.
    """
    key = (exchange, symbol)
    if key not in cache or not cache[key]:
        return None, None, None
    series = cache[key]
    if hasattr(entry_dt_utc, "tzinfo") and entry_dt_utc.tzinfo is not None:
        entry_naive = entry_dt_utc.replace(tzinfo=None)
    else:
        entry_naive = entry_dt_utc
    target_level = entry_ltp + target_points
    stop_level = entry_ltp - stop_points
    for t, ltp in series:
        t_naive = t.replace(tzinfo=None) if hasattr(t, "tzinfo") and t.tzinfo else t
        if t_naive <= entry_naive:
            continue
        # First snapshot strictly after entry
        if side in ("BUY", "B"):
            if ltp >= target_level:
                return t, ltp, f"Target Hit (+{target_points})"
            if ltp <= stop_level:
                return t, ltp, f"Stop Loss (-{stop_points})"
        else:
            if ltp <= entry_ltp - target_points:
                return t, ltp, f"Target Hit (-{target_points})"
            if ltp >= entry_ltp + stop_points:
                return t, ltp, f"Stop Loss (+{stop_points})"
    return None, None, None


def load_all_trades(trade_logs_dir: Path) -> pd.DataFrame:
    """Load and concatenate all trades_*.csv in trade_logs_dir."""
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
    for col in ["entry_price", "exit_price", "pnl", "quantity"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def parse_ts(ts) -> Optional[pd.Timestamp]:
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


def generate_realistic_logs(
    trade_logs_dir: Path,
    out_dir: Path,
    db_only: bool,
) -> None:
    from database_new import get_db_connection, release_db_connection

    df = load_all_trades(trade_logs_dir)
    if df.empty:
        LOGGER.error("No trades loaded from %s", trade_logs_dir)
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = get_db_connection()
    try:
        df["_entry_ts"] = df["entry_timestamp"].map(parse_ts)
        df["_exit_ts"] = df["exit_timestamp"].map(parse_ts)

        def date_from_entry(ts):
            if pd.isna(ts):
                return None
            t = parse_ts(ts)
            return t.date() if t is not None else None

        df["_date"] = df["entry_timestamp"].map(date_from_entry)

        ltp_cache = build_ltp_cache(conn, df)

        updated = []
        for idx, row in df.iterrows():
            entry_ts = row.get("_entry_ts")
            exit_ts = row.get("_exit_ts")
            exchange = row.get("exchange")
            symbol = row.get("symbol")
            quantity = row.get("quantity")
            side = (row.get("side") or "BUY").upper()
            status = (row.get("status") or "").strip()

            new_row = row.drop(labels=[c for c in df.columns if c.startswith("_")], errors="ignore").to_dict()
            new_row["_date"] = row.get("_date")  # keep for per-day output

            # Only consider trades with entry between 9:16 and 15:20 IST
            if entry_ts is not None:
                entry_time_ist = entry_ts.time() if hasattr(entry_ts, "time") else None
                if entry_time_ist is not None and (entry_time_ist < ENTRY_TIME_START or entry_time_ist > ENTRY_TIME_END):
                    continue  # skip row, do not include in output

            # Only recompute for closed trades with timestamps and symbol
            if status != "CLOSED" or pd.isna(exit_ts) or pd.isna(entry_ts) or not symbol or not exchange:
                new_row["pnl_source"] = "original"
                updated.append(new_row)
                continue

            entry_dt = entry_ts.tz_convert("UTC").to_pydatetime() if entry_ts.tzinfo else entry_ts.tz_localize("Asia/Kolkata").tz_convert("UTC").to_pydatetime()
            exit_dt = exit_ts.tz_convert("UTC").to_pydatetime() if exit_ts.tzinfo else exit_ts.tz_localize("Asia/Kolkata").tz_convert("UTC").to_pydatetime()
            entry_ltp = get_ltp_from_cache(ltp_cache, exchange, symbol, entry_dt)

            if entry_ltp is None:
                if db_only:
                    continue
                new_row["pnl_source"] = "original"
                updated.append(new_row)
                continue

            target_points = 50 if exchange == "BSE" else 25
            stop_points = target_points
            exit_t_utc, exit_price, exit_reason = find_exit_from_target_stop(
                ltp_cache, exchange, symbol, entry_dt, entry_ltp, side, target_points, stop_points
            )

            if exit_t_utc is not None and exit_price is not None and exit_reason is not None:
                # Exit from target/stop hit: use new exit time and price (model decision outcome)
                ex_ts = pd.Timestamp(exit_t_utc)
                if ex_ts.tzinfo is None:
                    ex_ts = ex_ts.tz_localize("UTC", ambiguous=True)
                else:
                    ex_ts = ex_ts.tz_convert("UTC")
                exit_ts_ist = ex_ts.tz_convert("Asia/Kolkata")
                new_row["exit_timestamp"] = exit_ts_ist.strftime("%Y-%m-%d %H:%M:%S")
                new_row["exit_price"] = round(exit_price, 2)
                new_row["exit_reason"] = exit_reason
                qty = int(quantity) if quantity == quantity else 0
                pnl = (exit_price - entry_ltp) * qty if side in ("BUY", "B") else (entry_ltp - exit_price) * qty
                new_row["pnl"] = round(pnl, 2)
                new_row["pnl_source"] = "target_stop"
            else:
                # Fallback: use original exit time and LTP from DB at that time
                exit_ltp = get_ltp_from_cache(ltp_cache, exchange, symbol, exit_dt)
                if exit_ltp is None:
                    if db_only:
                        continue
                    new_row["pnl_source"] = "original"
                    updated.append(new_row)
                    continue
                qty = int(quantity) if quantity == quantity else 0
                pnl = (exit_ltp - entry_ltp) * qty if side in ("BUY", "B") else (entry_ltp - exit_ltp) * qty
                new_row["exit_price"] = round(exit_ltp, 2)
                new_row["pnl"] = round(pnl, 2)
                new_row["pnl_source"] = "db_snapshot"

            new_row["entry_price"] = round(entry_ltp, 2)
            updated.append(new_row)
            LOGGER.debug(
                "%s %s entry_ltp=%.2f exit_price=%.2f pnl=%.2f source=%s",
                symbol, exchange, entry_ltp, new_row["exit_price"], new_row["pnl"], new_row["pnl_source"],
            )
    finally:
        release_db_connection(conn)

    out_df = pd.DataFrame(updated)
    if out_df.empty:
        LOGGER.warning("No rows to write (db_only=True may have skipped all).")
        return

    # Write per-day files to match original structure
    out_cols = [c for c in TRADE_LOG_COLUMNS if c in out_df.columns]
    if "pnl_source" in out_df.columns:
        out_cols = out_cols + ["pnl_source"]

    for date_key, group in out_df.groupby("_date", dropna=False):
        if pd.isna(date_key):
            date_str = "unknown"
        else:
            date_str = str(date_key)
        filepath = out_dir / f"trades_{date_str}.csv"
        g = group.drop(columns=[c for c in group.columns if c.startswith("_")], errors="ignore")
        cols = [c for c in out_cols if c in g.columns]
        g[cols].to_csv(filepath, index=False)
        LOGGER.info("Wrote %s (%d rows)", filepath, len(g))

    # Summary
    if "pnl_source" in out_df.columns:
        n_target_stop = (out_df["pnl_source"] == "target_stop").sum()
        n_db = (out_df["pnl_source"] == "db_snapshot").sum()
        n_orig = (out_df["pnl_source"] == "original").sum()
        LOGGER.info(
            "Realistic PnL: %d target/stop exit, %d DB snapshot exit, %d original",
            n_target_stop, n_db, n_orig,
        )
    total_pnl = out_df["pnl"].sum()
    LOGGER.info("Total PnL (realistic): %.2f", total_pnl)


def main():
    parser = argparse.ArgumentParser(
        description="Generate fresh trade logs with realistic PnL from DB snapshot LTPs",
    )
    parser.add_argument(
        "--trade-logs-dir",
        type=Path,
        default=Path("trade_logs"),
        help="Path to existing trade_logs folder",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("trade_logs_realistic"),
        help="Output folder for realistic trade logs",
    )
    parser.add_argument(
        "--db-only",
        action="store_true",
        help="Only include trades where DB had LTP at entry/exit; skip others",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    trade_logs_dir = args.trade_logs_dir if args.trade_logs_dir.is_absolute() else project_root / args.trade_logs_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else project_root / args.out_dir

    LOGGER.info("Reading from %s", trade_logs_dir)
    LOGGER.info("Writing to %s", out_dir)
    generate_realistic_logs(trade_logs_dir, out_dir, args.db_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
