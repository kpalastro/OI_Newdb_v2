#!/usr/bin/env python3
"""
Backtest ITM CE/PE scalping strategy: 15 points (NSE) / 30 points (BSE).

Uses historical ml_features (ITM OI % and Volume % 3m wavg) and underlying
future price to simulate entries and target/stop exits.

Usage:
  python scripts/backtest_itm_scalping_15_30.py --exchange NSE --days 30
  python scripts/backtest_itm_scalping_15_30.py --exchange BSE --start 2026-01-01 --end 2026-01-30
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import database_new as db


# Default targets and stops (index points)
TARGET_NSE = 15
STOP_NSE = 10
TARGET_BSE = 30
STOP_BSE = 20

# Default params (can be overridden by params dict for sweeps)
DEFAULT_PARAMS = {
    "divergence_min": 5.0,
    "nse_pe_vol_bullish_max": -3.0,
    "bse_pe_vol_bullish_min": 3.8,
    "nse_ce_vol_exhaustion": 16.0,
    "bse_ce_vol_exhaustion": 7.5,
    "min_hour": 11,
    "min_minute": 0,
    "hold_minutes": 30,
}


def _get_param(params: Optional[Dict], key: str, default: Any) -> Any:
    if params is None:
        return default
    return params.get(key, default)


# Re-export for day-specific param finder (after DEFAULT_PARAMS)
DIVERGENCE_MIN = DEFAULT_PARAMS["divergence_min"]
NSE_PE_VOL_BULLISH_MAX = DEFAULT_PARAMS["nse_pe_vol_bullish_max"]
BSE_PE_VOL_BULLISH_MIN = DEFAULT_PARAMS["bse_pe_vol_bullish_min"]
NSE_CE_VOL_EXHAUSTION = DEFAULT_PARAMS["nse_ce_vol_exhaustion"]
BSE_CE_VOL_EXHAUSTION = DEFAULT_PARAMS["bse_ce_vol_exhaustion"]
MIN_HOUR, MIN_MINUTE = DEFAULT_PARAMS["min_hour"], DEFAULT_PARAMS["min_minute"]
HOLD_MINUTES = DEFAULT_PARAMS["hold_minutes"]


def _parse_payload(payload: Any) -> Dict[str, float]:
    if not payload:
        return {}
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return {}
    if not isinstance(payload, dict):
        return {}
    return {
        "itm_volume_ce_pct_change_3m_wavg": payload.get("itm_volume_ce_pct_change_3m_wavg"),
        "itm_volume_pe_pct_change_3m_wavg": payload.get("itm_volume_pe_pct_change_3m_wavg"),
    }


def _is_bullish_signal(
    ce_oi: float,
    pe_oi: float,
    ce_vol: Optional[float],
    pe_vol: Optional[float],
    exchange: str,
    params: Optional[Dict[str, Any]] = None,
) -> bool:
    """True if ITM CE/PE divergence and volume support a long entry."""
    div_min = _get_param(params, "divergence_min", DEFAULT_PARAMS["divergence_min"])
    nse_pe_max = _get_param(params, "nse_pe_vol_bullish_max", DEFAULT_PARAMS["nse_pe_vol_bullish_max"])
    bse_pe_min = _get_param(params, "bse_pe_vol_bullish_min", DEFAULT_PARAMS["bse_pe_vol_bullish_min"])
    nse_ce_exh = _get_param(params, "nse_ce_vol_exhaustion", DEFAULT_PARAMS["nse_ce_vol_exhaustion"])
    bse_ce_exh = _get_param(params, "bse_ce_vol_exhaustion", DEFAULT_PARAMS["bse_ce_vol_exhaustion"])
    divergence = ce_oi - pe_oi
    if divergence < div_min:
        return False
    if exchange == "NSE":
        if pe_vol is not None and pe_vol > nse_pe_max:
            return False
        if ce_vol is not None and ce_vol > nse_ce_exh:
            return False
    else:
        if pe_vol is not None and pe_vol < bse_pe_min:
            return False
        if ce_vol is not None and ce_vol > bse_ce_exh:
            return False
    return True


def _is_bearish_signal(
    ce_oi: float,
    pe_oi: float,
    ce_vol: Optional[float],
    pe_vol: Optional[float],
    exchange: str,
    params: Optional[Dict[str, Any]] = None,
) -> bool:
    """True if PE > CE divergence and volume support a short entry."""
    div_min = _get_param(params, "divergence_min", DEFAULT_PARAMS["divergence_min"])
    divergence = pe_oi - ce_oi
    if divergence < div_min:
        return False
    if exchange == "NSE":
        if pe_vol is not None and pe_vol < 0:
            return False
    return True


def load_ml_features(
    exchange: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """Load ml_features with ITM OI/Vol and underlying future price."""
    conn = db.get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            timestamp,
            COALESCE(underlying_future_price, underlying_price) AS price,
            itm_oi_ce_pct_change_3m_wavg AS ce_oi,
            itm_oi_pe_pct_change_3m_wavg AS pe_oi,
            feature_payload
        FROM ml_features
        WHERE exchange = %s
          AND timestamp >= %s
          AND timestamp < %s
          AND itm_oi_ce_pct_change_3m_wavg IS NOT NULL
          AND itm_oi_pe_pct_change_3m_wavg IS NOT NULL
        ORDER BY timestamp
        """,
        (exchange, start_date, end_date),
    )
    rows = cur.fetchall()
    db.release_db_connection(conn)

    if not rows:
        return pd.DataFrame()

    records = []
    for ts, price, ce_oi, pe_oi, payload in rows:
        vol = _parse_payload(payload)
        records.append(
            {
                "timestamp": ts,
                "price": float(price) if price is not None else None,
                "ce_oi": float(ce_oi) if ce_oi is not None else None,
                "pe_oi": float(pe_oi) if pe_oi is not None else None,
                "ce_vol": vol.get("itm_volume_ce_pct_change_3m_wavg"),
                "pe_vol": vol.get("itm_volume_pe_pct_change_3m_wavg"),
            }
        )
    df = pd.DataFrame(records)
    if df.empty or df["price"].isna().all():
        return pd.DataFrame()
    df = df.dropna(subset=["price", "ce_oi", "pe_oi"])
    df["ts"] = pd.to_datetime(df["timestamp"])
    return df


def run_backtest(
    df: pd.DataFrame,
    exchange: str,
    params: Optional[Dict[str, Any]] = None,
    include_day_fields: bool = False,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """Simulate entries and target/stop exits. Uses future closes as proxy for high/low.
    If params is provided, overrides default divergence, volume thresholds, min_hour, hold_minutes.
    If include_day_fields is True, adds entry_date and day_of_week to each trade for day-specific analysis.
    """
    target = TARGET_NSE if exchange == "NSE" else TARGET_BSE
    stop = STOP_NSE if exchange == "NSE" else STOP_BSE
    min_hour = _get_param(params, "min_hour", DEFAULT_PARAMS["min_hour"])
    min_minute = _get_param(params, "min_minute", DEFAULT_PARAMS["min_minute"])
    hold_minutes = _get_param(params, "hold_minutes", DEFAULT_PARAMS["hold_minutes"])
    results = []
    i = 0
    while i < len(df):
        row = df.iloc[i]
        ts, price = row["ts"], row["price"]
        if pd.isna(price):
            i += 1
            continue
        hour, minute = ts.hour, ts.minute
        if hour < min_hour or (hour == min_hour and minute < min_minute):
            i += 1
            continue
        ce_oi = row["ce_oi"]
        pe_oi = row["pe_oi"]
        ce_vol = row.get("ce_vol")
        pe_vol = row.get("pe_vol")
        if pd.isna(ce_oi) or pd.isna(pe_oi):
            i += 1
            continue

        direction = None
        if _is_bullish_signal(ce_oi, pe_oi, ce_vol, pe_vol, exchange, params):
            direction = 1
        elif _is_bearish_signal(ce_oi, pe_oi, ce_vol, pe_vol, exchange, params):
            direction = -1

        if direction is None:
            i += 1
            continue

        # Look ahead up to hold_minutes
        window = df.iloc[i + 1 : i + 1 + hold_minutes]
        if window.empty:
            i += 1
            continue
        prices = window["price"].values
        max_p = float(prices.max())
        min_p = float(prices.min())
        exit_price = float(prices[-1])
        exit_ts = window["ts"].iloc[-1]

        hit_target = False
        hit_stop = False
        if direction == 1:
            if max_p >= price + target:
                hit_target = True
                exit_price = price + target
            elif min_p <= price - stop:
                hit_stop = True
                exit_price = price - stop
        else:
            if min_p <= price - target:
                hit_target = True
                exit_price = price - target
            elif max_p >= price + stop:
                hit_stop = True
                exit_price = price + stop

        points = (exit_price - price) * direction
        rec = {
            "entry_ts": ts.isoformat(),
            "exit_ts": exit_ts.isoformat(),
            "direction": "long" if direction == 1 else "short",
            "entry_price": price,
            "exit_price": exit_price,
            "points": round(points, 2),
            "hit_target": hit_target,
            "hit_stop": hit_stop,
            "ce_oi": round(ce_oi, 2),
            "pe_oi": round(pe_oi, 2),
        }
        if include_day_fields:
            rec["entry_date"] = ts.date().isoformat()
            rec["day_of_week"] = ts.weekday()  # 0=Monday, 6=Sunday
        results.append(rec)
        # Skip ahead to avoid overlapping trades (same bar = one trade)
        i += 1 + min(hold_minutes, len(window))

    # Summary
    if not results:
        summary = {"trades": 0, "win_rate": 0.0, "avg_points": 0.0, "total_points": 0.0}
    else:
        wins = sum(1 for r in results if r["hit_target"])
        total_pts = sum(r["points"] for r in results)
        summary = {
            "trades": len(results),
            "wins": wins,
            "win_rate": round(100.0 * wins / len(results), 2),
            "avg_points": round(total_pts / len(results), 2),
            "total_points": round(total_pts, 2),
            "target_hits": sum(1 for r in results if r["hit_target"]),
            "stop_hits": sum(1 for r in results if r["hit_stop"]),
        }
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest ITM scalping 15 pts NSE / 30 pts BSE")
    parser.add_argument("--exchange", type=str, default="NSE", choices=["NSE", "BSE"])
    parser.add_argument("--days", type=int, default=30, help="Last N days")
    parser.add_argument("--start", type=str, default="", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="", help="End date YYYY-MM-DD")
    parser.add_argument("--output", type=str, default="", help="Write trades JSON here")
    args = parser.parse_args()
    exchange = args.exchange.upper()

    end_date = datetime.utcnow().date()
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        start_date = end_date - timedelta(days=args.days)

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

    print(f"Backtest ITM scalping: {exchange} from {start_date} to {end_date}")
    print(f"Target: {TARGET_NSE if exchange == 'NSE' else TARGET_BSE} pts, Stop: {STOP_NSE if exchange == 'NSE' else STOP_BSE} pts")
    print()

    df = load_ml_features(exchange, start_dt, end_dt)
    if df.empty:
        print("No ml_features data for this range.")
        sys.exit(1)
    print(f"Loaded {len(df)} minute rows.")

    results, summary = run_backtest(df, exchange)
    print(f"Trades: {summary['trades']}")
    print(f"Win rate (target hit): {summary.get('win_rate', 0)}%")
    print(f"Target hits: {summary.get('target_hits', 0)}, Stop hits: {summary.get('stop_hits', 0)}")
    print(f"Avg points/trade: {summary.get('avg_points', 0)}, Total points: {summary.get('total_points', 0)}")

    if args.output and results:
        out = {"exchange": exchange, "start": str(start_date), "end": str(end_date), "summary": summary, "trades": results}
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
