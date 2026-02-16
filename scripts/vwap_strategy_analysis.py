"""
VWAP Strategy Analysis using multi_resolution_bars.

Loads OHLCV bars with per-bar VWAP from multi_resolution_bars, computes
session (cumulative) VWAP where needed, and backtests multiple VWAP-based
strategies to find the best winning configuration.

Strategies tested:
  - Bar VWAP crossover: long when close > bar VWAP, short when close < bar VWAP.
  - Session VWAP crossover: long when close > session VWAP (cumulative), short when < .
  - Mean reversion (bands): long when close is % below VWAP, short when % above.
  - Strength filter: only trade when |close - vwap| / vwap > threshold.

Metrics: win_rate, total_return_pct, sharpe_approx, max_drawdown_pct, num_trades.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import database_new as db
from database_new import get_db_connection, release_db_connection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_multi_resolution_bars(
    exchange: str,
    resolution: str,
    start_ts: Optional[datetime] = None,
    end_ts: Optional[datetime] = None,
    token: Optional[int] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load multi_resolution_bars from PostgreSQL.
    Columns: timestamp, exchange, resolution, token, symbol, open_price, high_price,
             low_price, close_price, volume, oi, oi_change, vwap, trade_count, ...
    """
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
        where = " AND ".join(conditions)
        limit_clause = f" LIMIT {int(limit)}" if limit else ""
        query = f"""
            SELECT timestamp, exchange, resolution, token, symbol,
                   open_price, high_price, low_price, close_price,
                   volume, oi, oi_change, vwap, trade_count
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


def add_session_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add session VWAP per (date, token): cumulative sum(close*volume) / sum(volume)
    over the session (day). Requires columns: timestamp, token, close_price, volume, vwap.
    """
    if df.empty or "close_price" not in df.columns or "volume" not in df.columns:
        return df
    df = df.copy()
    df["_date"] = df["timestamp"].dt.date
    # For each group (date, token), cumulative typical price * volume and cumulative volume
    df["_tpv"] = (df["close_price"].fillna(0) * df["volume"].fillna(0).clip(lower=0))
    df["_vol"] = df["volume"].fillna(0).clip(lower=0)
    g = df.groupby(["_date", "token"], group_keys=False)
    df["session_vwap"] = g["_tpv"].cumsum() / g["_vol"].cumsum().replace(0, np.nan)
    df.drop(columns=["_date", "_tpv", "_vol"], inplace=True)
    return df


def add_forward_returns(df: pd.DataFrame, periods: int = 1, price_col: str = "close_price") -> pd.DataFrame:
    """Add forward return (next bar return) for backtest PnL."""
    if df.empty or periods < 1:
        return df
    df = df.copy()
    df["forward_return"] = df[price_col].pct_change(periods=periods).shift(-periods)
    return df


# -----------------------------------------------------------------------------
# Strategy signals (vectorized)
# -----------------------------------------------------------------------------

def signal_bar_vwap_crossover(close: pd.Series, vwap: pd.Series) -> pd.Series:
    """+1 long when close > vwap, -1 short when close < vwap, 0 when invalid."""
    s = np.where(close > vwap, 1, np.where(close < vwap, -1, 0))
    invalid = close.isna() | vwap.isna() | (vwap <= 0)
    return pd.Series(np.where(invalid, 0, s), index=close.index)


def signal_session_vwap_crossover(close: pd.Series, session_vwap: pd.Series) -> pd.Series:
    """+1 long when close > session_vwap, -1 short when close < session_vwap."""
    s = np.where(close > session_vwap, 1, np.where(close < session_vwap, -1, 0))
    invalid = close.isna() | session_vwap.isna() | (session_vwap <= 0)
    return pd.Series(np.where(invalid, 0, s), index=close.index)


def signal_mean_reversion_bands(
    close: pd.Series, vwap: pd.Series, pct_low: float, pct_high: float
) -> pd.Series:
    """
    Long when close <= vwap * (1 - pct_low), short when close >= vwap * (1 + pct_high).
    Otherwise 0. (Mean reversion: buy dips below VWAP, sell rallies above.)
    """
    low_bound = vwap * (1 - pct_low)
    high_bound = vwap * (1 + pct_high)
    s = np.where(close <= low_bound, 1, np.where(close >= high_bound, -1, 0))
    invalid = close.isna() | vwap.isna() | (vwap <= 0)
    return pd.Series(np.where(invalid, 0, s), index=close.index)


def signal_strength_filter(close: pd.Series, vwap: pd.Series, min_pct: float) -> pd.Series:
    """
    Use bar VWAP crossover but only when |close - vwap|/vwap >= min_pct (strong move).
    """
    base = signal_bar_vwap_crossover(close, vwap)
    pct_dev = (close - vwap).abs() / vwap.replace(0, np.nan)
    mask = (pct_dev >= min_pct) & (~pct_dev.isna())
    return base.where(mask, 0)


# -----------------------------------------------------------------------------
# Backtest
# -----------------------------------------------------------------------------

@dataclass
class BacktestMetrics:
    win_rate: float
    total_return_pct: float
    num_trades: int
    sharpe_approx: float
    max_drawdown_pct: float
    avg_return_per_trade_pct: float
    strategy_name: str
    resolution: str
    params: dict = field(default_factory=dict)


def backtest_vwap(
    df: pd.DataFrame,
    signal: pd.Series,
    holding_bars: int = 1,
    price_col: str = "close_price",
    forward_return_col: str = "forward_return",
) -> BacktestMetrics:
    """
    Backtest: at each bar, if signal != 0, we assume we enter at close and hold
    for holding_bars. PnL = direction * forward_return (over holding_bars).
    One trade per entry; overlapping entries are treated as separate trades.
    """
    if df.empty or signal is None or forward_return_col not in df.columns:
        return BacktestMetrics(
            win_rate=0.0, total_return_pct=0.0, num_trades=0, sharpe_approx=0.0,
            max_drawdown_pct=0.0, avg_return_per_trade_pct=0.0,
            strategy_name="", resolution="", params={},
        )
    # Entries: bars where we have a non-zero signal and valid forward return
    valid = (signal != 0) & df[forward_return_col].notna()
    returns = (signal.astype(float) * df[forward_return_col]).loc[valid]
    if returns.empty:
        return BacktestMetrics(
            win_rate=0.0, total_return_pct=0.0, num_trades=0, sharpe_approx=0.0,
            max_drawdown_pct=0.0, avg_return_per_trade_pct=0.0,
            strategy_name="", resolution="", params={"holding_bars": holding_bars},
        )
    num_trades = len(returns)
    wins = (returns > 0).sum()
    win_rate = wins / num_trades if num_trades else 0.0
    # Total return: sum of per-trade returns in % (comparable across strategies; multi-token = sum across tokens)
    total_return_pct = float(returns.sum() * 100)
    avg_return_per_trade_pct = float(returns.mean() * 100) if num_trades else 0.0
    # Sharpe: annualized approx (assuming bars are e.g. 1min, 375 bars/day, 252 days)
    std = returns.std()
    if std and std > 0:
        sharpe_approx = float(returns.mean() / std * np.sqrt(252 * 375))  # rough annualization
    else:
        sharpe_approx = 0.0
    # Max drawdown: per-trade returns sequence, peak-to-trough (no cumprod to avoid overflow)
    cum = np.cumsum(returns)
    running_max = np.maximum.accumulate(cum)
    drawdowns = cum - running_max
    max_drawdown_pct = float(np.nanmin(drawdowns) * 100) if len(drawdowns) else 0.0
    return BacktestMetrics(
        win_rate=win_rate,
        total_return_pct=total_return_pct,
        num_trades=num_trades,
        sharpe_approx=sharpe_approx,
        max_drawdown_pct=max_drawdown_pct,
        avg_return_per_trade_pct=avg_return_per_trade_pct,
        strategy_name="",
        resolution="",
        params={"holding_bars": holding_bars},
    )


# -----------------------------------------------------------------------------
# Run all strategy variants
# -----------------------------------------------------------------------------

def run_analysis(
    exchange: str = "NSE",
    resolution: str = "5min",
    days_back: int = 60,
    token: Optional[int] = None,
    limit: Optional[int] = None,
    holding_bars: int = 1,
) -> Tuple[pd.DataFrame, List[BacktestMetrics]]:
    """
    Load bars, add session VWAP and forward returns, run multiple VWAP strategies,
    return combined results and list of metrics.
    """
    end_ts = datetime.utcnow()
    start_ts = end_ts - timedelta(days=days_back)
    df = load_multi_resolution_bars(
        exchange=exchange,
        resolution=resolution,
        start_ts=start_ts,
        end_ts=end_ts,
        token=token,
        limit=limit,
    )
    if df.empty:
        LOGGER.warning("No rows from multi_resolution_bars. Check exchange=%s resolution=%s and data availability.", exchange, resolution)
        return pd.DataFrame(), []

    df = add_session_vwap(df)
    df = add_forward_returns(df, periods=holding_bars)

    results: List[BacktestMetrics] = []
    # 1) Bar VWAP crossover
    sig = signal_bar_vwap_crossover(df["close_price"], df["vwap"])
    m = backtest_vwap(df, sig, holding_bars=holding_bars)
    m.strategy_name = "bar_vwap_crossover"
    m.resolution = resolution
    m.params["holding_bars"] = holding_bars
    results.append(m)

    # 2) Session VWAP crossover
    sig = signal_session_vwap_crossover(df["close_price"], df["session_vwap"])
    m = backtest_vwap(df, sig, holding_bars=holding_bars)
    m.strategy_name = "session_vwap_crossover"
    m.resolution = resolution
    m.params["holding_bars"] = holding_bars
    results.append(m)

    # 3) Mean reversion bands (e.g. 0.2% below = long, 0.2% above = short)
    for pct in (0.001, 0.002, 0.003):
        sig = signal_mean_reversion_bands(df["close_price"], df["vwap"], pct_low=pct, pct_high=pct)
        m = backtest_vwap(df, sig, holding_bars=holding_bars)
        m.strategy_name = "mean_reversion_bands"
        m.resolution = resolution
        m.params = {"pct_band": pct, "holding_bars": holding_bars}
        results.append(m)

    # 4) Strength filter (only trade when |close - vwap|/vwap >= threshold)
    for min_pct in (0.0005, 0.001, 0.002):
        sig = signal_strength_filter(df["close_price"], df["vwap"], min_pct=min_pct)
        m = backtest_vwap(df, sig, holding_bars=holding_bars)
        m.strategy_name = "strength_filter"
        m.resolution = resolution
        m.params = {"min_pct": min_pct, "holding_bars": holding_bars}
        results.append(m)

    summary = pd.DataFrame([
        {
            "strategy": r.strategy_name,
            "resolution": r.resolution,
            "win_rate": r.win_rate,
            "total_return_pct": r.total_return_pct,
            "num_trades": r.num_trades,
            "sharpe_approx": r.sharpe_approx,
            "max_drawdown_pct": r.max_drawdown_pct,
            "params": r.params,
        }
        for r in results
    ])
    return df, results


def run_multi_resolution(
    exchange: str = "NSE",
    days_back: int = 60,
    token: Optional[int] = None,
    limit: Optional[int] = None,
    holding_bars: int = 1,
    resolutions: Optional[List[str]] = None,
) -> List[BacktestMetrics]:
    """Run analysis across multiple resolutions and return all metrics."""
    if resolutions is None:
        resolutions = ["1min", "5min", "15min"]
    all_metrics: List[BacktestMetrics] = []
    for res in resolutions:
        _, results = run_analysis(
            exchange=exchange,
            resolution=res,
            days_back=days_back,
            token=token,
            limit=limit,
            holding_bars=holding_bars,
        )
        all_metrics.extend(results)
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="VWAP strategy analysis on multi_resolution_bars")
    parser.add_argument("--exchange", default="NSE", help="Exchange (e.g. NSE, BSE)")
    parser.add_argument("--resolution", default="5min", choices=["1min", "5min", "15min", "1D"], help="Bar resolution")
    parser.add_argument("--all-resolutions", action="store_true", help="Run 1min, 5min, 15min and compare")
    parser.add_argument("--days", type=int, default=60, help="Days of history")
    parser.add_argument("--token", type=int, default=None, help="Filter by instrument token")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to load (for quick tests)")
    parser.add_argument("--holding-bars", type=int, default=1, help="Hold period in bars for PnL")
    parser.add_argument("--out", default=None, help="Write summary CSV path")
    args = parser.parse_args()

    if args.all_resolutions:
        results = run_multi_resolution(
            exchange=args.exchange,
            days_back=args.days,
            token=args.token,
            limit=args.limit,
            holding_bars=args.holding_bars,
        )
        df = None
    else:
        df, results = run_analysis(
            exchange=args.exchange,
            resolution=args.resolution,
            days_back=args.days,
            token=args.token,
            limit=args.limit,
            holding_bars=args.holding_bars,
        )

    if not results:
        print("No results (no data or no trades). Exiting.")
        return

    # Best by win rate (with minimum trades)
    min_trades = 10
    eligible = [r for r in results if r.num_trades >= min_trades]
    if not eligible:
        eligible = results
    best_win = max(eligible, key=lambda r: r.win_rate)
    best_return = max(eligible, key=lambda r: r.total_return_pct)
    best_sharpe = max(eligible, key=lambda r: r.sharpe_approx)

    print("\n" + "=" * 70)
    print("VWAP STRATEGY ANALYSIS (multi_resolution_bars)")
    print("=" * 70)
    print(f"Exchange: {args.exchange}  Resolution: {args.resolution}  Holding bars: {args.holding_bars}")
    if df is not None:
        print(f"Bars loaded: {len(df)}")
    print()
    print("Best by WIN RATE:", best_win.strategy_name, best_win.params, "->", f"{best_win.win_rate:.2%}", f"({best_win.num_trades} trades)")
    print("Best by TOTAL RETURN %:", best_return.strategy_name, best_return.params, "->", f"{best_return.total_return_pct:.2f}%")
    print("Best by SHARPE (approx):", best_sharpe.strategy_name, best_sharpe.params, "->", f"{best_sharpe.sharpe_approx:.2f}")
    print()
    print("All strategies:")
    for r in results:
        print(f"  {r.strategy_name} {r.params} -> win_rate={r.win_rate:.2%} return_pct={r.total_return_pct:.2f} trades={r.num_trades} sharpe={r.sharpe_approx:.2f} max_dd={r.max_drawdown_pct:.2f}%")
    print("=" * 70)

    summary = pd.DataFrame([
        {
            "strategy": r.strategy_name,
            "resolution": r.resolution,
            "win_rate": r.win_rate,
            "total_return_pct": r.total_return_pct,
            "num_trades": r.num_trades,
            "sharpe_approx": r.sharpe_approx,
            "max_drawdown_pct": r.max_drawdown_pct,
            "params": str(r.params),
        }
        for r in results
    ])
    if args.out:
        summary.to_csv(args.out, index=False)
        print(f"Summary written to {args.out}")
    # Best overall (across resolutions)
    if len(results) > 1:
        best_overall = max(
            [r for r in results if r.num_trades >= 5],
            key=lambda r: (r.win_rate, r.sharpe_approx),
            default=results[0],
        )
        print("\n>>> BEST WINNING STRATEGY (win_rate + sharpe):", best_overall.strategy_name, best_overall.resolution, best_overall.params)


if __name__ == "__main__":
    main()
