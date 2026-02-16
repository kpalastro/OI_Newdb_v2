"""
Data engineering to lower reversal rate and maximise profit on VWAP touch/cross.

Given high reversal rates (~80–87%) on mixed-token runs, this script:
1. Adds features at event time: hour, volume_ratio (vs rolling avg), vwap_distance_pct.
2. Stratifies analysis by: time-of-day, volume strength, VWAP cross strength.
3. Finds segments where reversal rate is LOWER and hit rate is HIGHER.
4. Outputs recommended actions: when to trade (time window), filters (volume, VWAP distance),
   and per-symbol ranking so you trade only the best underlyings.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.vwap_touch_target_nse import (
    load_multi_resolution_bars,
    detect_cross_up,
    detect_cross_down,
    detect_touch,
    scan_forward,
    EventResult,
    run_analysis_per_series,
)


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------

def add_event_features(df: pd.DataFrame, volume_window: int = 20) -> pd.DataFrame:
    """
    Add columns used to stratify events (per token).
    - hour: 9–15 (IST)
    - volume_ratio: volume / rolling_mean(volume, volume_window)
    - vwap_dist_pct: |close - vwap| / vwap * 100 (strength of move from VWAP)
    """
    if df.empty:
        return df
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    if hasattr(ts.dt, "tz_localize"):
        try:
            ts = ts.dt.tz_localize(None)
        except Exception:
            pass
    df["hour"] = ts.dt.hour
    df["volume_ratio"] = np.nan
    df["vwap_dist_pct"] = np.nan
    v = df["volume"].replace(0, np.nan)
    vwap = df["vwap"].replace(0, np.nan)
    close = df["close_price"]
    df["vwap_dist_pct"] = (close - vwap).abs() / vwap * 100.0
    for tok in df["token"].unique():
        mask = df["token"] == tok
        vol = df.loc[mask, "volume"]
        df.loc[mask, "volume_ratio"] = vol / vol.rolling(volume_window, min_periods=1).mean()
    df["volume_ratio"] = df["volume_ratio"].fillna(1.0)
    return df


def run_analysis_with_metadata(
    df: pd.DataFrame,
    token: int,
    symbol: Optional[str],
    target_pts: Tuple[float, float] = (20.0, 25.0),
    reversal_pts: float = 15.0,
    max_forward_bars: int = 24,
    use_touch: bool = True,
) -> List[dict]:
    """
    Run VWAP cross/touch analysis and attach event-level features to each result.
    Returns list of dicts: hit_20, hit_25, rev_before_20, rev_before_25, side, hour, volume_ratio, vwap_dist_pct, token, symbol.
    """
    cross_up = detect_cross_up(df)
    cross_down = detect_cross_down(df)
    if use_touch:
        touch_up, touch_down = detect_touch(df)
        cross_up = cross_up | touch_up
        cross_down = cross_down | touch_down

    out: List[dict] = []
    for i in range(1, len(df)):
        if cross_up.loc[df.index[i]]:
            res = scan_forward(
                df, i, "call",
                target_pts=target_pts,
                reversal_pts=reversal_pts,
                max_bars=max_forward_bars,
            )
            row = {
                "hit_20": res.hit_20, "hit_25": res.hit_25,
                "rev_before_20": res.reversed_before_20, "rev_before_25": res.reversed_before_25,
                "side": "call", "token": token, "symbol": symbol or "",
                "hour": int(df.loc[df.index[i], "hour"]) if "hour" in df.columns else None,
                "volume_ratio": float(df.loc[df.index[i], "volume_ratio"]) if "volume_ratio" in df.columns else None,
                "vwap_dist_pct": float(df.loc[df.index[i], "vwap_dist_pct"]) if "vwap_dist_pct" in df.columns else None,
            }
            out.append(row)
        if cross_down.loc[df.index[i]]:
            res = scan_forward(
                df, i, "put",
                target_pts=target_pts,
                reversal_pts=reversal_pts,
                max_bars=max_forward_bars,
            )
            row = {
                "hit_20": res.hit_20, "hit_25": res.hit_25,
                "rev_before_20": res.reversed_before_20, "rev_before_25": res.reversed_before_25,
                "side": "put", "token": token, "symbol": symbol or "",
                "hour": int(df.loc[df.index[i], "hour"]) if "hour" in df.columns else None,
                "volume_ratio": float(df.loc[df.index[i], "volume_ratio"]) if "volume_ratio" in df.columns else None,
                "vwap_dist_pct": float(df.loc[df.index[i], "vwap_dist_pct"]) if "vwap_dist_pct" in df.columns else None,
            }
            out.append(row)
    return out


def segment_stats(rows: List[dict]) -> dict:
    """Compute hit_20_pct, rev_before_20_pct, expected_pts for a list of event dicts."""
    n = len(rows)
    if n == 0:
        return {"n": 0, "hit_20_pct": 0.0, "rev_20_pct": 0.0, "expected_pts": 0.0, "score": 0.0}
    hit_20 = sum(1 for r in rows if r["hit_20"])
    rev_20 = sum(1 for r in rows if r["rev_before_20"])
    hit_20_pct = 100.0 * hit_20 / n
    rev_20_pct = 100.0 * rev_20 / n
    expected_pts = hit_20 * 20.0 - rev_20 * 15.0
    score = hit_20_pct - rev_20_pct  # higher = better
    return {"n": n, "hit_20_pct": hit_20_pct, "rev_20_pct": rev_20_pct, "expected_pts": expected_pts, "score": score}


# -----------------------------------------------------------------------------
# Stratification and recommendations
# -----------------------------------------------------------------------------

def stratify_and_recommend(all_rows: List[dict]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Stratify by hour, volume_ratio bucket, vwap_dist_pct bucket; find best segments.
    Returns (summary_df, list of recommended action strings).
    """
    actions: List[str] = []
    records: List[dict] = []

    # By hour (9–15)
    for h in range(9, 16):
        sub = [r for r in all_rows if r.get("hour") == h]
        s = segment_stats(sub)
        s["segment_type"] = "hour"
        s["segment_value"] = str(h)
        records.append(s)
    hour_best = max([(r["segment_value"], r) for r in records if r["segment_type"] == "hour" and r["n"] >= 30], key=lambda x: (x[1]["score"], x[1]["n"]), default=(None, None))
    if hour_best[0]:
        actions.append(f"Trade only during hour {hour_best[0]} (IST) — reversal rate {hour_best[1]['rev_20_pct']:.1f}%, hit rate {hour_best[1]['hit_20_pct']:.1f}% (n={hour_best[1]['n']})")

    # By volume_ratio bucket: low <1, med 1–1.5, high >1.5
    records_vol: List[dict] = []
    for label, low, high in [("low_vol", 0.0, 1.0), ("med_vol", 1.0, 1.5), ("high_vol", 1.5, 100.0)]:
        sub = [r for r in all_rows if r.get("volume_ratio") is not None and low <= r["volume_ratio"] < high]
        s = segment_stats(sub)
        s["segment_type"] = "volume_ratio"
        s["segment_value"] = label
        records_vol.append(s)
        records.append(s)
    vol_best = max([r for r in records_vol if r["n"] >= 30], key=lambda r: r["score"], default=None)
    if vol_best:
        actions.append(f"Trade only when volume is {vol_best['segment_value']} (volume_ratio band) — reversal rate {vol_best['rev_20_pct']:.1f}%, hit rate {vol_best['hit_20_pct']:.1f}% (n={vol_best['n']})")

    # By vwap_dist_pct: weak <0.05%, med 0.05–0.15%, strong >=0.15%
    records_vwap: List[dict] = []
    for label, low, high in [("weak_cross", 0.0, 0.05), ("med_cross", 0.05, 0.15), ("strong_cross", 0.15, 100.0)]:
        sub = [r for r in all_rows if r.get("vwap_dist_pct") is not None and low <= r["vwap_dist_pct"] < high]
        s = segment_stats(sub)
        s["segment_type"] = "vwap_dist_pct"
        s["segment_value"] = label
        records_vwap.append(s)
        records.append(s)
    vwap_best = max([r for r in records_vwap if r["n"] >= 30], key=lambda r: r["score"], default=None)
    if vwap_best:
        actions.append(f"Trade only on {vwap_best['segment_value']} (|close-vwap|/vwap band) — reversal rate {vwap_best['rev_20_pct']:.1f}%, hit rate {vwap_best['hit_20_pct']:.1f}% (n={vwap_best['n']})")

    # Combined: best hour + best volume (if we have enough data)
    if hour_best[0] and vol_best:
        sub = [r for r in all_rows if r.get("hour") == int(hour_best[0]) and r.get("volume_ratio") is not None and (
            (vol_best["segment_value"] == "high_vol" and r["volume_ratio"] >= 1.5) or
            (vol_best["segment_value"] == "med_vol" and 1.0 <= r["volume_ratio"] < 1.5) or
            (vol_best["segment_value"] == "low_vol" and r["volume_ratio"] < 1.0)
        )]
        s = segment_stats(sub)
        if s["n"] >= 20:
            actions.append(f"Combined: hour {hour_best[0]} + {vol_best['segment_value']} — reversal {s['rev_20_pct']:.1f}%, hit {s['hit_20_pct']:.1f}% (n={s['n']})")

    summary_df = pd.DataFrame(records)
    return summary_df, actions


def per_symbol_ranking(all_rows: List[dict], min_events: int = 50) -> pd.DataFrame:
    """Rank symbols by score (hit_20_pct - rev_20_pct)."""
    by_sym: dict = defaultdict(list)
    for r in all_rows:
        sym = r.get("symbol") or f"token_{r.get('token')}"
        by_sym[sym].append(r)
    rows = []
    for sym, events in by_sym.items():
        if len(events) < min_events:
            continue
        s = segment_stats(events)
        s["symbol"] = sym
        rows.append(s)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="VWAP data engineering — lower reversal, maximise profit")
    ap.add_argument("--exchange", default="NSE")
    ap.add_argument("--resolution", default="5min", choices=["1min", "5min", "15min"])
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--token", type=int, default=None)
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--limit", type=int, default=25000)
    ap.add_argument("--reversal-pts", type=float, default=15.0)
    ap.add_argument("--max-forward-bars", type=int, default=24)
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-symbols", default=None)
    args = ap.parse_args()

    end_ts = datetime.utcnow()
    start_ts = end_ts - timedelta(days=args.days)
    df = load_multi_resolution_bars(
        exchange=args.exchange,
        resolution=args.resolution,
        start_ts=start_ts,
        end_ts=end_ts,
        token=args.token,
        symbol_contains=args.symbol,
        limit=args.limit,
    )
    if df.empty:
        print("No data.")
        return

    df = add_event_features(df, volume_window=20)
    tokens = df["token"].unique().tolist()
    all_rows: List[dict] = []

    for tok in tokens:
        sub = df[df["token"] == tok].reset_index(drop=True)
        if len(sub) < 10:
            continue
        sym = sub["symbol"].iloc[0] if "symbol" in sub.columns and sub["symbol"].notna().any() else None
        rows = run_analysis_with_metadata(
            sub,
            token=int(tok),
            symbol=sym,
            target_pts=(20.0, 25.0),
            reversal_pts=args.reversal_pts,
            max_forward_bars=args.max_forward_bars,
            use_touch=True,
        )
        all_rows.extend(rows)

    if not all_rows:
        print("No events detected.")
        return

    summary_df, actions = stratify_and_recommend(all_rows)
    symbol_df = per_symbol_ranking(all_rows, min_events=30)

    # Overall baseline
    base = segment_stats(all_rows)
    print("\n" + "=" * 70)
    print("VWAP DATA ENGINEERING — ACTIONS TO MAXIMISE PROFIT (lower reversal rate)")
    print("=" * 70)
    print(f"Exchange: {args.exchange}  Resolution: {args.resolution}  Days: {args.days}  Limit: {args.limit}")
    print(f"Total events: {len(all_rows)}  Tokens: {len(tokens)}")
    print(f"\nBaseline (all events): hit_20_pct={base['hit_20_pct']:.1f}%  rev_before_20_pct={base['rev_20_pct']:.1f}%  score={base['score']:.1f}")
    print("\n--- RECOMMENDED ACTIONS (stratified analysis) ---")
    for a in actions:
        print("  •", a)
    print("\n--- BY HOUR (IST) ---")
    hour_recs = [r for r in summary_df.to_dict("records") if r["segment_type"] == "hour" and r["n"] >= 20]
    for r in sorted(hour_recs, key=lambda x: -x["score"])[:7]:
        print(f"  Hour {r['segment_value']}: n={r['n']}  hit_20={r['hit_20_pct']:.1f}%  rev_20={r['rev_20_pct']:.1f}%  score={r['score']:.1f}")
    print("\n--- BY VOLUME STRENGTH (volume_ratio) ---")
    vol_recs = [r for r in summary_df.to_dict("records") if r["segment_type"] == "volume_ratio" and r["n"] >= 20]
    for r in sorted(vol_recs, key=lambda x: -x["score"]):
        print(f"  {r['segment_value']}: n={r['n']}  hit_20={r['hit_20_pct']:.1f}%  rev_20={r['rev_20_pct']:.1f}%  score={r['score']:.1f}")
    print("\n--- BY VWAP CROSS STRENGTH (vwap_dist_pct) ---")
    vwap_recs = [r for r in summary_df.to_dict("records") if r["segment_type"] == "vwap_dist_pct" and r["n"] >= 20]
    for r in sorted(vwap_recs, key=lambda x: -x["score"]):
        print(f"  {r['segment_value']}: n={r['n']}  hit_20={r['hit_20_pct']:.1f}%  rev_20={r['rev_20_pct']:.1f}%  score={r['score']:.1f}")
    print("\n--- PER-SYMBOL RANKING (trade only top symbols to maximise profit) ---")
    if not symbol_df.empty:
        print(symbol_df.head(15).to_string(index=False))
    else:
        print("  (Insufficient events per symbol; try without --limit or with more days.)")
    print("=" * 70)

    if args.out:
        summary_df.to_csv(args.out, index=False)
        print(f"Segment summary written to {args.out}")
    if args.out_symbols and not symbol_df.empty:
        symbol_df.to_csv(args.out_symbols, index=False)
        print(f"Symbol ranking written to {args.out_symbols}")


if __name__ == "__main__":
    main()
