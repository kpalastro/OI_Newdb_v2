"""
NSE VWAP Touch/Cross — 20–25 Point Target Analysis.

When a bar touches or crosses the VWAP (values stored in multi_resolution_bars):
- BUY CALL when price crosses above VWAP (expect +20 to +25 points).
- BUY PUT when price crosses below VWAP (expect -20 to -25 points).

For each such event we look forward and count:
- How many times the target (20 pts, 25 pts) was hit.
- How many times the trend reversed (price went against the trade) before target.

Goal: Maximize profit by choosing call vs put and target (20 vs 25 pts) based on hit rate
and reversal rate.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database_new import get_db_connection, release_db_connection


def load_multi_resolution_bars(
    exchange: str,
    resolution: str,
    start_ts: Optional[datetime] = None,
    end_ts: Optional[datetime] = None,
    token: Optional[int] = None,
    symbol_contains: Optional[str] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Load multi_resolution_bars (timestamp, OHLC, vwap, token, ...)."""
    conn = get_db_connection()
    try:
        conditions = ["exchange = %s", "resolution = %s"]
        params: List[object] = [exchange, resolution]
        if start_ts is not None:
            conditions.append("timestamp >= %s")
            params.append(start_ts)
        if end_ts is not None:
            conditions.append("timestamp <= %s")
            params.append(end_ts)
        if token is not None:
            conditions.append("token = %s")
            params.append(token)
        if symbol_contains:
            conditions.append("symbol ILIKE %s")
            params.append(f"%{symbol_contains}%")
        where = " AND ".join(conditions)
        limit_clause = f" LIMIT {int(limit)}" if limit else ""
        query = f"""
            SELECT timestamp, exchange, resolution, token, symbol,
                   open_price, high_price, low_price, close_price,
                   volume, vwap
            FROM multi_resolution_bars
            WHERE {where}
            ORDER BY timestamp
            {limit_clause}
        """
        df = pd.read_sql_query(query, conn, params=params)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    finally:
        release_db_connection(conn)


# -----------------------------------------------------------------------------
# VWAP touch/cross detection
# -----------------------------------------------------------------------------

def detect_cross_up(df: pd.DataFrame) -> pd.Series:
    """True where bar crosses above VWAP: prev close <= vwap, curr close > vwap."""
    if df.empty or len(df) < 2:
        return pd.Series(False, index=df.index)
    prev_close = df["close_price"].shift(1)
    prev_vwap = df["vwap"].shift(1)
    curr_close = df["close_price"]
    curr_vwap = df["vwap"]
    return (prev_close <= prev_vwap) & (curr_close > curr_vwap) & curr_vwap.notna() & prev_vwap.notna()


def detect_cross_down(df: pd.DataFrame) -> pd.Series:
    """True where bar crosses below VWAP: prev close >= vwap, curr close < vwap."""
    if df.empty or len(df) < 2:
        return pd.Series(False, index=df.index)
    prev_close = df["close_price"].shift(1)
    prev_vwap = df["vwap"].shift(1)
    curr_close = df["close_price"]
    curr_vwap = df["vwap"]
    return (prev_close >= prev_vwap) & (curr_close < curr_vwap) & curr_vwap.notna() & prev_vwap.notna()


def detect_touch(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Touch: bar's range contains VWAP (low <= vwap <= high).
    cross_up_touch: touch and close > vwap (bullish touch). cross_down_touch: touch and close < vwap (bearish).
    """
    if df.empty:
        return pd.Series(False, index=df.index), pd.Series(False, index=df.index)
    low, high, vwap = df["low_price"], df["high_price"], df["vwap"]
    touch = (low <= vwap) & (vwap <= high) & vwap.notna()
    cross_up_touch = touch & (df["close_price"] > vwap)
    cross_down_touch = touch & (df["close_price"] < vwap)
    return cross_up_touch, cross_down_touch


# -----------------------------------------------------------------------------
# Forward scan: target hit vs reversal
# -----------------------------------------------------------------------------

@dataclass
class EventResult:
    """Result of one VWAP touch/cross event: did we hit target or reverse first?"""
    hit_20: bool
    hit_25: bool
    reversed_before_20: bool
    reversed_before_25: bool
    bars_to_20: Optional[int] = None  # bars until 20 pt hit (or None)
    bars_to_25: Optional[int] = None
    bars_to_reversal: Optional[int] = None
    entry: float = 0.0
    vwap_at: float = 0.0
    side: str = ""  # "call" or "put"


def scan_forward(
    df: pd.DataFrame,
    event_idx: int,
    side: str,
    target_pts: Tuple[float, float] = (20.0, 25.0),
    reversal_pts: float = 15.0,
    max_bars: int = 24,
    use_vwap_reversal: bool = True,
) -> EventResult:
    """
    From event bar (index i), look forward up to max_bars.
    - CALL: entry = close[i]. Hit 20 if any forward high >= entry+20, hit 25 if >= entry+25.
            Reversal: any forward low <= entry - reversal_pts (or <= vwap_at if use_vwap_reversal).
    - PUT:  entry = close[i]. Hit 20 if any forward low <= entry-20, hit 25 if <= entry-25.
            Reversal: any forward high >= entry + reversal_pts (or >= vwap_at).

    Return which happened first: target 20, target 25, or reversal (bar-by-bar: first bar where condition met).
    """
    t20, t25 = target_pts
    entry = float(df.loc[event_idx, "close_price"])
    vwap_at = float(df.loc[event_idx, "vwap"])
    n = len(df)
    hit_20_bar: Optional[int] = None
    hit_25_bar: Optional[int] = None
    reversal_bar: Optional[int] = None

    for j in range(event_idx + 1, min(event_idx + 1 + max_bars, n)):
        high_j = float(df.loc[j, "high_price"])
        low_j = float(df.loc[j, "low_price"])
        vwap_j = float(df.loc[j, "vwap"]) if pd.notna(df.loc[j, "vwap"]) else vwap_at

        if side == "call":
            if high_j >= entry + t20 and hit_20_bar is None:
                hit_20_bar = j - event_idx
            if high_j >= entry + t25 and hit_25_bar is None:
                hit_25_bar = j - event_idx
            rev = (low_j <= entry - reversal_pts) or (use_vwap_reversal and low_j <= vwap_at)
            if rev and reversal_bar is None:
                reversal_bar = j - event_idx
        else:
            if low_j <= entry - t20 and hit_20_bar is None:
                hit_20_bar = j - event_idx
            if low_j <= entry - t25 and hit_25_bar is None:
                hit_25_bar = j - event_idx
            rev = (high_j >= entry + reversal_pts) or (use_vwap_reversal and high_j >= vwap_at)
            if rev and reversal_bar is None:
                reversal_bar = j - event_idx

        # First occurrence wins for "what happened first"
        if hit_20_bar is not None and reversal_bar is not None:
            break
        if hit_25_bar is not None and reversal_bar is not None:
            break

    # Interpret: hit target if we reached target bar and (reversal is None or target bar <= reversal bar)
    hit_20 = hit_20_bar is not None and (reversal_bar is None or hit_20_bar <= reversal_bar)
    hit_25 = hit_25_bar is not None and (reversal_bar is None or hit_25_bar <= reversal_bar)
    rev_before_20 = reversal_bar is not None and (hit_20_bar is None or reversal_bar < hit_20_bar)
    rev_before_25 = reversal_bar is not None and (hit_25_bar is None or reversal_bar < hit_25_bar)

    return EventResult(
        hit_20=hit_20,
        hit_25=hit_25,
        reversed_before_20=rev_before_20,
        reversed_before_25=rev_before_25,
        bars_to_20=hit_20_bar,
        bars_to_25=hit_25_bar,
        bars_to_reversal=reversal_bar,
        entry=entry,
        vwap_at=vwap_at,
        side=side,
    )


def run_analysis_per_series(
    df: pd.DataFrame,
    target_pts: Tuple[float, float] = (20.0, 25.0),
    reversal_pts: float = 15.0,
    max_forward_bars: int = 24,
    use_touch: bool = True,
    use_cross: bool = True,
) -> Tuple[List[EventResult], List[EventResult]]:
    """
    Detect VWAP cross up (call) and cross down (put) events; optionally include touch.
    Returns (call_results, put_results).
    """
    cross_up = detect_cross_up(df)
    cross_down = detect_cross_down(df)
    if use_touch:
        touch_up, touch_down = detect_touch(df)
        cross_up = cross_up | touch_up
        cross_down = cross_down | touch_down

    call_results: List[EventResult] = []
    put_results: List[EventResult] = []

    for i in range(1, len(df)):
        if use_cross or use_touch:
            if cross_up.loc[df.index[i]]:
                res = scan_forward(
                    df, i, "call",
                    target_pts=target_pts,
                    reversal_pts=reversal_pts,
                    max_bars=max_forward_bars,
                )
                call_results.append(res)
            if cross_down.loc[df.index[i]]:
                res = scan_forward(
                    df, i, "put",
                    target_pts=target_pts,
                    reversal_pts=reversal_pts,
                    max_bars=max_forward_bars,
                )
                put_results.append(res)

    return call_results, put_results


def run_analysis_per_series_fade(
    df: pd.DataFrame,
    target_pts: Tuple[float, float] = (20.0, 25.0),
    reversal_pts: float = 15.0,
    max_forward_bars: int = 24,
    use_touch: bool = True,
    use_cross: bool = True,
) -> Tuple[List[EventResult], List[EventResult]]:
    """
    Trade in the direction of reversal (fade the cross):
    - On cross UP → evaluate as PUT (bet price drops 20/25 pts).
    - On cross DOWN → evaluate as CALL (bet price rises 20/25 pts).
    Returns (fade_on_cross_up_results, fade_on_cross_down_results).
    """
    cross_up = detect_cross_up(df)
    cross_down = detect_cross_down(df)
    if use_touch:
        touch_up, touch_down = detect_touch(df)
        cross_up = cross_up | touch_up
        cross_down = cross_down | touch_down

    fade_on_up: List[EventResult] = []   # cross up → we trade PUT (fade)
    fade_on_down: List[EventResult] = []  # cross down → we trade CALL (fade)

    for i in range(1, len(df)):
        if use_cross or use_touch:
            if cross_up.loc[df.index[i]]:
                res = scan_forward(
                    df, i, "put",  # fade: bet down
                    target_pts=target_pts,
                    reversal_pts=reversal_pts,
                    max_bars=max_forward_bars,
                )
                fade_on_up.append(res)
            if cross_down.loc[df.index[i]]:
                res = scan_forward(
                    df, i, "call",  # fade: bet up
                    target_pts=target_pts,
                    reversal_pts=reversal_pts,
                    max_bars=max_forward_bars,
                )
                fade_on_down.append(res)

    return fade_on_up, fade_on_down


def aggregate_results(
    call_results: List[EventResult],
    put_results: List[EventResult],
) -> dict:
    """Aggregate hit/reversal counts and implied profit (points)."""
    def stats(results: List[EventResult], pts_20: float, pts_25: float) -> dict:
        n = len(results)
        if n == 0:
            return {"events": 0, "hit_20": 0, "hit_25": 0, "rev_before_20": 0, "rev_before_25": 0,
                    "hit_20_pct": 0.0, "hit_25_pct": 0.0, "rev_before_20_pct": 0.0, "rev_before_25_pct": 0.0,
                    "expected_pts_20": 0.0, "expected_pts_25": 0.0}
        hit_20 = sum(1 for r in results if r.hit_20)
        hit_25 = sum(1 for r in results if r.hit_25)
        rev_20 = sum(1 for r in results if r.reversed_before_20)
        rev_25 = sum(1 for r in results if r.reversed_before_25)
        # Expected points: hit_20 * 20 - rev_before_20 * reversal_loss (assume -15), etc.
        expected_20 = hit_20 * pts_20 - rev_20 * 15.0  # rough
        expected_25 = hit_25 * pts_25 - rev_25 * 15.0
        return {
            "events": n,
            "hit_20": hit_20,
            "hit_25": hit_25,
            "rev_before_20": rev_20,
            "rev_before_25": rev_25,
            "hit_20_pct": 100.0 * hit_20 / n,
            "hit_25_pct": 100.0 * hit_25 / n,
            "rev_before_20_pct": 100.0 * rev_20 / n,
            "rev_before_25_pct": 100.0 * rev_25 / n,
            "expected_pts_20": expected_20,
            "expected_pts_25": expected_25,
        }

    call_s = stats(call_results, 20.0, 25.0)
    put_s = stats(put_results, 20.0, 25.0)
    call_s["side"] = "call"
    put_s["side"] = "put"
    return {"call": call_s, "put": put_s, "call_results": call_results, "put_results": put_results}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="NSE VWAP touch/cross — 20–25 pt target analysis")
    ap.add_argument("--exchange", default="NSE")
    ap.add_argument("--resolution", default="5min", choices=["1min", "5min", "15min"])
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--token", type=int, default=None, help="Single instrument token")
    ap.add_argument("--symbol", default=None, help="Filter symbol (e.g. NIFTY or NIFTY 50 for index-only)")
    ap.add_argument("--target-low", type=float, default=20.0)
    ap.add_argument("--target-high", type=float, default=25.0)
    ap.add_argument("--reversal-pts", type=float, default=15.0)
    ap.add_argument("--max-forward-bars", type=int, default=24)
    ap.add_argument("--no-touch", action="store_true", help="Only cross, no touch")
    ap.add_argument("--limit", type=int, default=None, help="Max bars to load (for quick runs)")
    ap.add_argument("--out", default=None)
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
        print("No data from multi_resolution_bars. Check exchange, resolution, and date range.")
        return

    # Per-token analysis so we don't mix underlyings
    tokens = df["token"].unique().tolist()
    all_call: List[EventResult] = []
    all_put: List[EventResult] = []

    all_fade_up: List[EventResult] = []
    all_fade_down: List[EventResult] = []

    for tok in tokens:
        sub = df[df["token"] == tok].reset_index(drop=True)
        if len(sub) < 10:
            continue
        call_r, put_r = run_analysis_per_series(
            sub,
            target_pts=(args.target_low, args.target_high),
            reversal_pts=args.reversal_pts,
            max_forward_bars=args.max_forward_bars,
            use_touch=not args.no_touch,
            use_cross=True,
        )
        all_call.extend(call_r)
        all_put.extend(put_r)
        fade_up, fade_down = run_analysis_per_series_fade(
            sub,
            target_pts=(args.target_low, args.target_high),
            reversal_pts=args.reversal_pts,
            max_forward_bars=args.max_forward_bars,
            use_touch=not args.no_touch,
            use_cross=True,
        )
        all_fade_up.extend(fade_up)
        all_fade_down.extend(fade_down)

    agg = aggregate_results(all_call, all_put)
    c, p = agg["call"], agg["put"]

    # Best choice: maximize expected points (hit rate * target - reversal rate * loss)
    print("\n" + "=" * 70)
    print("NSE VWAP TOUCH/CROSS — 20–25 POINT TARGET ANALYSIS")
    print("=" * 70)
    print(f"Exchange: {args.exchange}  Resolution: {args.resolution}  Days: {args.days}")
    print(f"Target: {args.target_low} / {args.target_high} pts   Reversal threshold: {args.reversal_pts} pts   Max forward bars: {args.max_forward_bars}")
    print(f"Bars loaded: {len(df)}   Tokens: {len(tokens)}")
    if len(tokens) > 1:
        print("  (Tip: For Nifty/Bank Nifty only, use --token <instrument_token> so 20–25 pts are comparable.)")
    print()

    print("--- CALL (price crosses above VWAP → buy call, target +20 / +25 pts) ---")
    print(f"  Events: {c['events']}")
    print(f"  Hit 20 pts: {c['hit_20']} ({c['hit_20_pct']:.1f}%)")
    print(f"  Hit 25 pts: {c['hit_25']} ({c['hit_25_pct']:.1f}%)")
    print(f"  Trend reversed before 20 pts: {c['rev_before_20']} ({c['rev_before_20_pct']:.1f}%)")
    print(f"  Trend reversed before 25 pts: {c['rev_before_25']} ({c['rev_before_25_pct']:.1f}%)")
    print(f"  Expected pts (approx, 20 pt target): {c['expected_pts_20']:.0f}")
    print(f"  Expected pts (approx, 25 pt target): {c['expected_pts_25']:.0f}")
    print()

    print("--- PUT (price crosses below VWAP → buy put, target -20 / -25 pts) ---")
    print(f"  Events: {p['events']}")
    print(f"  Hit 20 pts: {p['hit_20']} ({p['hit_20_pct']:.1f}%)")
    print(f"  Hit 25 pts: {p['hit_25']} ({p['hit_25_pct']:.1f}%)")
    print(f"  Trend reversed before 20 pts: {p['rev_before_20']} ({p['rev_before_20_pct']:.1f}%)")
    print(f"  Trend reversed before 25 pts: {p['rev_before_25']} ({p['rev_before_25_pct']:.1f}%)")
    print(f"  Expected pts (approx, 20 pt target): {p['expected_pts_20']:.0f}")
    print(f"  Expected pts (approx, 25 pt target): {p['expected_pts_25']:.0f}")
    print()

    # Reversal duration: how many bars until reversal (price moved reversal_pts against the trade)
    bars_to_rev = [r.bars_to_reversal for r in all_call + all_put if r.bars_to_reversal is not None]
    if bars_to_rev:
        bars_arr = np.array(bars_to_rev, dtype=float)
        mean_b = float(np.mean(bars_arr))
        med_b = float(np.median(bars_arr))
        p25 = float(np.percentile(bars_arr, 25))
        p75 = float(np.percentile(bars_arr, 75))
        p90 = float(np.percentile(bars_arr, 90))
        mins_per_bar = {"1min": 1, "5min": 5, "15min": 15, "1D": 375}.get(args.resolution, 5)
        print("--- REVERSAL DURATION (bars until price moved {} pts against the trade) ---".format(int(args.reversal_pts)))
        print(f"  Events with reversal: {len(bars_to_rev)}")
        print(f"  Mean: {mean_b:.1f} bars   Median: {med_b:.1f} bars")
        print(f"  25th pct: {p25:.0f} bars   75th pct: {p75:.0f} bars   90th pct: {p90:.0f} bars")
        print(f"  At {args.resolution} resolution: median ~{med_b * mins_per_bar:.0f} min, 90% within ~{p90 * mins_per_bar:.0f} min")
        print()

    # FADE (trade in direction of reversal): cross up → PUT, cross down → CALL
    fade_results = all_fade_up + all_fade_down
    fade_hit_20_pct = 0.0
    if fade_results:
        n_f = len(fade_results)
        fade_hit_20 = sum(1 for r in fade_results if r.hit_20)
        fade_hit_25 = sum(1 for r in fade_results if r.hit_25)
        fade_rev_20 = sum(1 for r in fade_results if r.reversed_before_20)
        fade_rev_25 = sum(1 for r in fade_results if r.reversed_before_25)
        fade_hit_20_pct = 100.0 * fade_hit_20 / n_f
        fade_hit_25_pct = 100.0 * fade_hit_25 / n_f
        fade_rev_20_pct = 100.0 * fade_rev_20 / n_f
        fade_rev_25_pct = 100.0 * fade_rev_25 / n_f
        fade_exp_20 = fade_hit_20 * 20.0 - fade_rev_20 * 15.0
        fade_exp_25 = fade_hit_25 * 25.0 - fade_rev_25 * 15.0
        fade_hit_20_pct = 100.0 * fade_hit_20 / n_f
        print("--- FADE (trade in direction of reversal: cross up → BUY PUT, cross down → BUY CALL) ---")
        print(f"  Events: {n_f}  (cross-up→PUT: {len(all_fade_up)}, cross-down→CALL: {len(all_fade_down)})")
        print(f"  Hit 20 pts (win rate): {fade_hit_20} ({fade_hit_20_pct:.1f}%)")
        print(f"  Hit 25 pts: {fade_hit_25} ({fade_hit_25_pct:.1f}%)")
        print(f"  Reversed before 20 pts: {fade_rev_20} ({fade_rev_20_pct:.1f}%)")
        print(f"  Reversed before 25 pts: {fade_rev_25} ({fade_rev_25_pct:.1f}%)")
        print(f"  Expected pts (20 pt target): {fade_exp_20:.0f}  (25 pt target): {fade_exp_25:.0f}")
        follow_win = 100.0 * (c["hit_20"] + p["hit_20"]) / max(c["events"] + p["events"], 1)
        print(f"  >>> FADE win rate (20 pt): {fade_hit_20_pct:.1f}%  vs  FOLLOW win rate (call+put): {follow_win:.1f}%")
        print()

    # Recommendation
    options = [
        ("CALL", "20 pts", c["hit_20_pct"], c["rev_before_20_pct"], c["expected_pts_20"]),
        ("CALL", "25 pts", c["hit_25_pct"], c["rev_before_25_pct"], c["expected_pts_25"]),
        ("PUT", "20 pts", p["hit_20_pct"], p["rev_before_20_pct"], p["expected_pts_20"]),
        ("PUT", "25 pts", p["hit_25_pct"], p["rev_before_25_pct"], p["expected_pts_25"]),
    ]
    # Best: prefer positive expected pts, else higher (hit_pct - rev_pct)
    def score(x):
        return (x[4], x[2] - x[3])
    best = max(options, key=score)
    print(">>> TO MAXIMISE PROFIT (by expected pts, then hit rate minus reversal rate):")
    print(f"    Prefer BUY {best[0]} with {best[1]} target — hit rate {best[2]:.1f}%, reversal before target {best[3]:.1f}%, expected pts ~{best[4]:.0f}")
    if fade_results:
        follow_win = 100.0 * (c["hit_20"] + p["hit_20"]) / max(c["events"] + p["events"], 1)
        if fade_hit_20_pct > follow_win:
            print(f"    Consider FADE (trade reversal): win rate {fade_hit_20_pct:.1f}% vs follow {follow_win:.1f}%.")
    if c["events"] + p["events"] > 0 and (c["expected_pts_20"] <= 0 and p["expected_pts_20"] <= 0):
        print("    (Try --token <Nifty_token> for index-only, or tune --reversal-pts / --max-forward-bars.)")
    print("=" * 70)

    if args.out:
        out_df = pd.DataFrame([
            {"side": "call", "events": c["events"], "hit_20": c["hit_20"], "hit_25": c["hit_25"],
             "rev_before_20": c["rev_before_20"], "rev_before_25": c["rev_before_25"],
             "hit_20_pct": c["hit_20_pct"], "hit_25_pct": c["hit_25_pct"],
             "rev_before_20_pct": c["rev_before_20_pct"], "rev_before_25_pct": c["rev_before_25_pct"]},
            {"side": "put", "events": p["events"], "hit_20": p["hit_20"], "hit_25": p["hit_25"],
             "rev_before_20": p["rev_before_20"], "rev_before_25": p["rev_before_25"],
             "hit_20_pct": p["hit_20_pct"], "hit_25_pct": p["hit_25_pct"],
             "rev_before_20_pct": p["rev_before_20_pct"], "rev_before_25_pct": p["rev_before_25_pct"]},
        ])
        out_df.to_csv(args.out, index=False)
        print(f"Summary written to {args.out}")


if __name__ == "__main__":
    main()
