#!/usr/bin/env python3
"""
Analyze paper_trading_metrics metadata with "horizon" key and correlate to real PnL
from trade_logs.

Why this exists:
- paper_trading_metrics.metadata contains horizon-like tags (e.g. 'swing', 'expiry')
- trade_logs/*.csv contains the real executed trades with entry/exit + pnl
- paper_trading_metrics.pnl is often NULL or not consistently filled for executed rows

This script:
1) Loads trade logs for a date range
2) Loads paper_trading_metrics rows where metadata ? 'horizon'
3) Matches trades -> metrics using (exchange, timestamp proximity, signal/confidence)
4) Produces a report:
   - win rate / avg pnl by metadata.horizon (+ exchange/type bins)
   - worst segments (loss-heavy) and suggested filters to reduce stop-loss trades

Run:
  python scripts/analyze_horizon_losses.py --start 2026-01-20 --end 2026-01-28
  python scripts/analyze_horizon_losses.py --start 2026-01-27 --end 2026-01-27 --out reports/horizon_loss_report_2026-01-27.md
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import database_new as db


def _parse_ymd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def load_trade_logs(start: date, end: date) -> pd.DataFrame:
    paths: List[str] = []
    cur = start
    while cur <= end:
        paths.append(f"trade_logs/trades_{cur.strftime('%Y-%m-%d')}.csv")
        cur = cur + timedelta(days=1)

    frames: List[pd.DataFrame] = []
    for p in paths:
        if Path(p).exists():
            df = pd.read_csv(p)
            df["trade_log_file"] = p
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    # Normalize columns
    out["entry_timestamp"] = pd.to_datetime(out["entry_timestamp"], errors="coerce")
    out["exit_timestamp"] = pd.to_datetime(out["exit_timestamp"], errors="coerce")
    out["pnl"] = pd.to_numeric(out["pnl"], errors="coerce")
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")
    out["is_win"] = out["pnl"].fillna(0) > 0
    out["hold_minutes"] = (
        (out["exit_timestamp"] - out["entry_timestamp"]).dt.total_seconds() / 60.0
    )
    # Common naming
    out["option_type"] = out.get("type")
    out["signal"] = out.get("side")

    return out


def load_metrics_with_horizon(start: date, end: date) -> pd.DataFrame:
    # DB stores timestamptz; use [start, end+1day)
    start_dt = datetime.combine(start, datetime.min.time())
    end_dt = datetime.combine(end + timedelta(days=1), datetime.min.time())

    conn = db.get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, timestamp, exchange, executed, reason, signal, confidence, quantity_lots, pnl, constraint_violation, metadata
        FROM paper_trading_metrics
        WHERE metadata ? 'horizon'
          AND timestamp >= %s
          AND timestamp < %s
        ORDER BY timestamp ASC
        """,
        (start_dt, end_dt),
    )
    rows = cur.fetchall()
    db.release_db_connection(conn)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=[
            "id",
            "timestamp",
            "exchange",
            "executed",
            "reason",
            "signal",
            "confidence",
            "quantity_lots",
            "db_pnl",
            "constraint_violation",
            "metadata",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    meta = pd.json_normalize(df["metadata"])
    # keep only key fields + horizon
    df = pd.concat([df.drop(columns=["metadata"]), meta], axis=1)

    # normalize
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["horizon"] = df.get("horizon")
    df["strategy_name"] = df.get("strategy_name")

    return df


@dataclass
class MatchParams:
    max_seconds: int = 90
    confidence_tol: float = 0.08


def match_trades_to_metrics(
    trades: pd.DataFrame, metrics: pd.DataFrame, params: MatchParams
) -> pd.DataFrame:
    """
    Heuristic match:
    - exchange exact
    - timestamp nearest within params.max_seconds
    - signal matches if possible (BUY/SELL)
    - confidence close within tolerance

    Note:
    - trade logs have naive timestamps (local IST strings). DB has timestamptz.
      We try both:
        A) treat trade timestamps as UTC
        B) treat trade timestamps as IST and convert to UTC (+05:30 -> UTC)
      We pick the match producing the smallest time diff.
    """
    if trades.empty or metrics.empty:
        return pd.DataFrame()

    m = metrics.copy()
    m = m[m["executed"] == True].copy()  # only executed decisions

    # prepare two versions of trade entry timestamps in UTC
    t = trades.copy()
    t["entry_ts_naive"] = pd.to_datetime(t["entry_timestamp"], errors="coerce")
    # Version A: assume already UTC
    t["entry_ts_utc_a"] = t["entry_ts_naive"].dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")
    # Version B: assume IST then convert to UTC
    try:
        t["entry_ts_utc_b"] = t["entry_ts_naive"].dt.tz_localize("Asia/Kolkata", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    except Exception:
        t["entry_ts_utc_b"] = pd.NaT

    out_rows: List[Dict[str, Any]] = []

    # Pre-index metrics by exchange for speed
    metrics_by_ex = {ex: df for ex, df in m.groupby("exchange")}

    for idx, row in t.iterrows():
        ex = row.get("exchange")
        if ex not in metrics_by_ex:
            continue
        cand = metrics_by_ex[ex]

        best = None
        best_diff = None

        for ts_col in ["entry_ts_utc_a", "entry_ts_utc_b"]:
            ts = row.get(ts_col)
            if pd.isna(ts):
                continue

            # narrow to window
            lo = ts - pd.Timedelta(seconds=params.max_seconds)
            hi = ts + pd.Timedelta(seconds=params.max_seconds)
            c = cand[(cand["timestamp"] >= lo) & (cand["timestamp"] <= hi)].copy()
            if c.empty:
                continue

            # signal match (trade side is BUY/SELL)
            side = row.get("signal")
            if side in ("BUY", "SELL") and "signal" in c.columns:
                c2 = c[c["signal"] == side]
                if not c2.empty:
                    c = c2

            # confidence tolerance
            conf = _safe_float(row.get("confidence"))
            if conf is not None and "confidence" in c.columns:
                c["conf_diff"] = (pd.to_numeric(c["confidence"], errors="coerce") - conf).abs()
                c = c[c["conf_diff"] <= params.confidence_tol]
                if c.empty:
                    continue

            # closest by time
            c["time_diff_s"] = (c["timestamp"] - ts).abs().dt.total_seconds()
            pick = c.sort_values(["time_diff_s", "conf_diff" if "conf_diff" in c.columns else "time_diff_s"]).iloc[0]
            diff = float(pick["time_diff_s"])

            if best is None or diff < (best_diff or 1e18):
                best = pick
                best_diff = diff

        if best is None:
            continue

        merged = dict(row)
        merged.update({f"ptm_{k}": best.get(k) for k in best.index})
        merged["match_time_diff_s"] = best_diff
        out_rows.append(merged)

    if not out_rows:
        return pd.DataFrame()

    matched = pd.DataFrame(out_rows)
    # convenience columns
    matched["ptm_horizon"] = matched.get("ptm_horizon")
    matched["ptm_strategy_name"] = matched.get("ptm_strategy_name")
    return matched


def build_recommendations(matched: pd.DataFrame) -> List[str]:
    """
    Convert loss clusters into actionable suggestions.
    We focus on:
    - stop-loss exits
    - exchange/type concentration
    - time-of-day windows
    """
    if matched.empty:
        return []

    recs: List[str] = []
    losses = matched[matched["is_win"] == False].copy()
    if losses.empty:
        return ["No losing trades in matched sample."]

    # Biggest loss drivers
    top_exit = (
        losses.groupby("exit_reason")
        .agg(trades=("pnl", "size"), total_pnl=("pnl", "sum"))
        .sort_values("total_pnl")
        .head(5)
        .reset_index()
    )
    if not top_exit.empty:
        recs.append(
            "Loss drivers by exit_reason: "
            + "; ".join(
                f"{r.exit_reason} (trades={int(r.trades)}, total_pnl={r.total_pnl:.2f})"
                for r in top_exit.itertuples(index=False)
            )
        )

    # Worst segments by (horizon, exchange, option_type)
    seg = (
        matched.groupby(["ptm_horizon", "exchange", "option_type"])
        .agg(trades=("pnl", "size"), win_rate=("is_win", "mean"), total_pnl=("pnl", "sum"))
        .reset_index()
    )
    seg = seg[seg["trades"] >= 5].sort_values("total_pnl").head(10)
    if not seg.empty:
        recs.append(
            "Worst segments (consider blocking or tightening filters): "
            + "; ".join(
                f"horizon={r.ptm_horizon}, {r.exchange} {r.option_type} (trades={int(r.trades)}, win_rate={r.win_rate:.2f}, total_pnl={r.total_pnl:.2f})"
                for r in seg.itertuples(index=False)
            )
        )

    # Time-of-day: entry hour bins
    matched["entry_hour"] = pd.to_datetime(matched["entry_timestamp"], errors="coerce").dt.hour
    bins = [0, 10, 11, 12, 13, 14, 15, 24]
    labels = ["<10", "10-11", "11-12", "12-13", "13-14", "14-15", "15+"]
    matched["hour_bin"] = pd.cut(matched["entry_hour"], bins=bins, labels=labels, right=False)
    tod = (
        matched.groupby(["ptm_horizon", "hour_bin"])
        .agg(trades=("pnl", "size"), win_rate=("is_win", "mean"), total_pnl=("pnl", "sum"))
        .reset_index()
    )
    tod = tod[tod["trades"] >= 5].sort_values("total_pnl").head(8)
    if not tod.empty:
        recs.append(
            "Worst time windows (block entries or require higher confidence): "
            + "; ".join(
                f"horizon={r.ptm_horizon}, hour={r.hour_bin} (trades={int(r.trades)}, win_rate={r.win_rate:.2f}, total_pnl={r.total_pnl:.2f})"
                for r in tod.itertuples(index=False)
            )
        )

    # General fixes tied to your observed pattern: stop-loss clustering
    recs.append(
        "Tactical fixes that usually convert stop-loss losers into avoided trades: "
        "add a per-exchange cooldown after any stop-loss (e.g. 10–15 minutes), "
        "block re-entry on the same symbol for N minutes, "
        "and tighten entry filters for the loss-heavy segment (commonly NSE PE)."
    )
    recs.append(
        "If you want to convert losers into winners via exits (not just avoiding entries): "
        "use earlier time-based exit (e.g. if not green by +X minutes, exit), "
        "and force-exit on first signal flip instead of waiting."
    )
    return recs


def render_report(trades: pd.DataFrame, metrics: pd.DataFrame, matched: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("# Horizon Loss Analysis Report")
    lines.append("")

    lines.append("## Data coverage")
    lines.append(f"- Trade logs loaded: **{len(trades)}** trades")
    lines.append(f"- paper_trading_metrics rows with `metadata ? 'horizon'`: **{len(metrics)}** rows")
    lines.append(f"- Matched trades ↔ metrics: **{len(matched)}** trades")
    lines.append("")

    if trades.empty:
        lines.append("No trade logs found for this window.")
        return "\n".join(lines)
    if metrics.empty:
        lines.append("No DB metrics with horizon found for this window.")
        return "\n".join(lines)
    if matched.empty:
        lines.append("Could not match trades to horizon-metrics (likely timestamp mismatch / missing metadata).")
        lines.append("Next fix: store `signal_id` inside `paper_trading_metrics.metadata` so joins are exact.")
        return "\n".join(lines)

    # Summary by horizon
    g = (
        matched.groupby(["ptm_horizon"])
        .agg(trades=("pnl", "size"), win_rate=("is_win", "mean"), total_pnl=("pnl", "sum"), avg_pnl=("pnl", "mean"))
        .reset_index()
        .sort_values("total_pnl")
    )
    lines.append("## Performance by metadata.horizon")
    lines.append("")
    lines.append(g.to_markdown(index=False))
    lines.append("")

    # Breakdown by horizon/exchange/type
    g2 = (
        matched.groupby(["ptm_horizon", "exchange", "option_type"])
        .agg(trades=("pnl", "size"), win_rate=("is_win", "mean"), total_pnl=("pnl", "sum"), avg_pnl=("pnl", "mean"))
        .reset_index()
        .sort_values("total_pnl")
    )
    lines.append("## Performance by horizon × exchange × option_type")
    lines.append("")
    lines.append(g2.head(30).to_markdown(index=False))
    lines.append("")

    # Exit reasons per horizon
    er = (
        matched.groupby(["ptm_horizon", "exit_reason"])
        .agg(trades=("pnl", "size"), win_rate=("is_win", "mean"), total_pnl=("pnl", "sum"))
        .reset_index()
        .sort_values("total_pnl")
    )
    lines.append("## Loss concentration by exit_reason")
    lines.append("")
    lines.append(er.head(30).to_markdown(index=False))
    lines.append("")

    lines.append("## How to convert losing trades to winners (actionable)")
    lines.append("")
    for rec in build_recommendations(matched):
        lines.append(f"- {rec}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--max-seconds", type=int, default=90, help="Max seconds diff for matching")
    ap.add_argument("--confidence-tol", type=float, default=0.08, help="Max abs confidence diff for matching")
    ap.add_argument("--out", default="", help="Optional markdown output path")
    args = ap.parse_args()

    start = _parse_ymd(args.start)
    end = _parse_ymd(args.end)

    trades = load_trade_logs(start, end)
    metrics = load_metrics_with_horizon(start, end)

    params = MatchParams(max_seconds=args.max_seconds, confidence_tol=args.confidence_tol)
    matched = match_trades_to_metrics(trades, metrics, params)

    report = render_report(trades, metrics, matched)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"Wrote report to {out_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()

