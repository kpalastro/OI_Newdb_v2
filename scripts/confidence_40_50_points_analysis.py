#!/usr/bin/env python3
"""
How many signals were "right" by exchange when we adjust confidence to maximise profit?

- NSE: 25 points (pts/share = pnl/quantity). Right = >= 25 pts.
- BSE: 40–50 points. Right = >= 40 pts; band = [40, 50] pts.

Loads closed trades from trade_logs, sweeps confidence thresholds, finds threshold that
maximizes total PnL per exchange, reports stats with exchange-specific point targets.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Exchange-specific point targets: NSE 25 pts, BSE 40–50 pts
NSE_MIN_PTS = 25.0
NSE_BAND_MAX = 30.0   # band for NSE: 25–30 (or just >= 25)
BSE_MIN_PTS = 40.0
BSE_MAX_PTS = 50.0


def load_closed_trades(days: int = 30) -> pd.DataFrame:
    """Load closed trades from trade_logs for the last N days."""
    log_dir = Path(__file__).resolve().parent.parent / "trade_logs"
    if not log_dir.exists():
        return pd.DataFrame()

    dfs = []
    for i in range(days):
        dt = (datetime.now() - timedelta(days=i)).date()
        p = log_dir / f"trades_{dt}.csv"
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)
    out = out[out["status"].astype(str).str.upper() == "CLOSED"].copy()
    if "pnl" in out.columns:
        out["pnl"] = pd.to_numeric(out["pnl"], errors="coerce")
    if "confidence" in out.columns:
        out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")
    if "exchange" not in out.columns:
        out["exchange"] = ""
    return out


def run_analysis_by_exchange(
    days: int = 30,
    threshold_min: float = 0.40,
    threshold_max: float = 0.95,
    step: float = 0.02,
):
    """
    NSE: 25 points (right = >= 25 pts/share).
    BSE: 40–50 points (right = >= 40; band = [40, 50] pts/share).
    For each exchange: sweep confidence, find threshold that maximizes total PnL; report stats.
    """
    df = load_closed_trades(days=days)
    if df.empty or "pnl" not in df.columns or "confidence" not in df.columns:
        print("No closed trades with pnl/confidence in trade_logs.")
        return

    df = df.dropna(subset=["pnl", "confidence"]).copy()
    if "quantity" not in df.columns:
        df["quantity"] = 1
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1)
    df["points"] = df["pnl"] / df["quantity"]

    nse = df[df["exchange"].astype(str).str.upper() == "NSE"]
    bse = df[df["exchange"].astype(str).str.upper() == "BSE"]

    def run_one(name: str, sub: pd.DataFrame, min_pts: float, max_pts: float, band_label: str):
        if len(sub) < 3:
            print(f"\n--- {name} ---\nToo few trades ({len(sub)}). Skipped.\n")
            return
        thresholds = np.arange(threshold_min, threshold_max + step / 2, step)
        rows = []
        for th in thresholds:
            s = sub[sub["confidence"] >= th]
            if len(s) < 2:
                continue
            total_pnl = float(s["pnl"].sum())
            n_right = int((s["points"] >= min_pts).sum())
            n_band = int(((s["points"] >= min_pts) & (s["points"] <= max_pts)).sum())
            wins = int((s["pnl"] > 0).sum())
            wr = 100.0 * wins / len(s) if len(s) else 0
            rows.append({
                "conf_min": round(th, 2),
                "trades": len(s),
                "total_pnl": round(total_pnl, 2),
                "right": n_right,
                "in_band": n_band,
                "win_rate_pct": round(wr, 1),
            })
        if not rows:
            return
        tbl = pd.DataFrame(rows)
        best_idx = tbl["total_pnl"].idxmax()
        best = tbl.loc[best_idx]
        print(f"\n--- {name} (target: {band_label}) ---")
        print(f"  Total closed trades: {len(sub)}")
        print(f"  Optimal confidence (max PnL): >= {best['conf_min']:.2f}")
        print(f"  Trades at optimal: {int(best['trades'])}, Total PnL: ₹{best['total_pnl']:,.2f}")
        print(f"  Signals 'right' (>= {min_pts} pts): {int(best['right'])}")
        print(f"  Signals in band [{min_pts}, {max_pts}]: {int(best['in_band'])}")
        print(f"  Win rate: {best['win_rate_pct']:.1f}%")
        print("\n  Threshold sweep:")
        print(tbl.to_string(index=False))
        return best

    print("=" * 72)
    print("CONFIDENCE vs PROFIT BY EXCHANGE (NSE 25 pts | BSE 40–50 pts)")
    print("Points = pts per share (pnl/quantity)")
    print("=" * 72)
    print(f"Days loaded: {days}")

    res_nse = run_one("NSE", nse, NSE_MIN_PTS, NSE_BAND_MAX, ">= 25 pts")
    res_bse = run_one("BSE", bse, BSE_MIN_PTS, BSE_MAX_PTS, "40–50 pts")

    print("\n" + "=" * 72)
    print("SUMMARY (at profit-maximising confidence per exchange)")
    print("=" * 72)
    if res_nse is not None:
        print(f"NSE (25 pts): conf >= {res_nse['conf_min']:.2f} → PnL ₹{res_nse['total_pnl']:,.2f}, "
              f"signals right (>= 25 pts): {int(res_nse['right'])}, win rate: {res_nse['win_rate_pct']:.1f}%")
    if res_bse is not None:
        print(f"BSE (40–50 pts): conf >= {res_bse['conf_min']:.2f} → PnL ₹{res_bse['total_pnl']:,.2f}, "
              f"right (>= 40 pts): {int(res_bse['right'])}, in [40,50] band: {int(res_bse['in_band'])}, win rate: {res_bse['win_rate_pct']:.1f}%")


def run_analysis(
    days: int = 30,
    min_profit_points: float = 40.0,
    max_profit_points: float = 50.0,
    threshold_min: float = 0.40,
    threshold_max: float = 0.95,
    step: float = 0.02,
    points_as_rupees: bool = False,
):
    """
    Sweep confidence thresholds; find threshold that maximizes total PnL.
    Report how many signals were "right" (profit >= min_profit_points, and in [min, max]) at that threshold.
    points_as_rupees: if True, "points" = raw PnL (rupees); if False, "points" = pnl/quantity (points per share).
    """
    df = load_closed_trades(days=days)
    if df.empty or "pnl" not in df.columns or "confidence" not in df.columns:
        print("No closed trades with pnl/confidence in trade_logs.")
        return

    df = df.dropna(subset=["pnl", "confidence"]).copy()
    if "quantity" not in df.columns:
        df["quantity"] = 1
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1)
    # Points: per-share (option premium move) or raw rupees
    df["points"] = (df["pnl"] / df["quantity"]) if not points_as_rupees else df["pnl"]

    if len(df) < 5:
        print("Too few closed trades for analysis.")
        return

    thresholds = np.arange(threshold_min, threshold_max + step / 2, step)
    rows = []

    for th in thresholds:
        sub = df[df["confidence"] >= th]
        if len(sub) < 3:
            continue
        total_pnl = float(sub["pnl"].sum())
        n_right_ge = int((sub["points"] >= min_profit_points).sum())
        n_right_band = int(((sub["points"] >= min_profit_points) & (sub["points"] <= max_profit_points)).sum())
        wins = int((sub["pnl"] > 0).sum())
        win_rate = 100.0 * wins / len(sub) if len(sub) else 0
        rows.append({
            "confidence_min": round(th, 2),
            "trades": len(sub),
            "total_pnl": round(total_pnl, 2),
            "right_ge_40": n_right_ge,
            "right_40_50": n_right_band,
            "win_rate_pct": round(win_rate, 1),
        })

    if not rows:
        print("No threshold band had enough trades.")
        return

    tbl = pd.DataFrame(rows)

    # Optimal threshold = maximize total PnL
    best_idx = tbl["total_pnl"].idxmax()
    best = tbl.loc[best_idx]

    pts_label = "rupees" if points_as_rupees else "pts/share (pnl/qty)"
    print("=" * 70)
    print("CONFIDENCE vs PROFIT (closed trades from trade_logs)")
    print(f"Points: {pts_label}. 'Right' = >= {min_profit_points} pts; '40–50' = in [{min_profit_points}, {max_profit_points}]")
    print("=" * 70)
    print(f"Days loaded: {days}")
    print(f"Total closed trades: {len(df)}")
    print()
    print("--- Threshold sweep (confidence >= X) ---")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(tbl.to_string(index=False))
    print()
    print("--- Optimal confidence (maximises total PnL) ---")
    print(f"  Confidence >= {best['confidence_min']:.2f}")
    print(f"  Trades: {int(best['trades'])}")
    print(f"  Total PnL: ₹{best['total_pnl']:,.2f}")
    print(f"  Signals 'right' (profit >= 40 pts): {int(best['right_ge_40'])}")
    print(f"  Signals in 40–50 pts band: {int(best['right_40_50'])}")
    print(f"  Win rate: {best['win_rate_pct']:.1f}%")
    print()
    print("Answer: At the confidence that maximises profit,"
          f" {int(best['right_ge_40'])} signals were 'right' (>= {min_profit_points} pts)"
          f" and {int(best['right_40_50'])} had profit in the {min_profit_points}–{max_profit_points} pts band.")
    return tbl, best


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Confidence vs points profit (NSE 25 pts, BSE 40–50 pts)")
    ap.add_argument("--days", type=int, default=30, help="Days of trade_logs to load")
    ap.add_argument("--unified", action="store_true", help="Use single min/max pts for all exchanges (default: NSE 25, BSE 40–50)")
    ap.add_argument("--min-pts", type=float, default=40, help="Min points (unified mode)")
    ap.add_argument("--max-pts", type=float, default=50, help="Max points (unified mode)")
    ap.add_argument("--rupees", action="store_true", help="Use raw PnL as points (unified mode)")
    args = ap.parse_args()
    if args.unified:
        run_analysis(
            days=args.days,
            min_profit_points=args.min_pts,
            max_profit_points=args.max_pts,
            points_as_rupees=args.rupees,
        )
    else:
        run_analysis_by_exchange(days=args.days)
