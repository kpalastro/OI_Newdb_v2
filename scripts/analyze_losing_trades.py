"""
Analyze losing trades from trade_logs folder.

Aggregates all trades_*.csv files, computes win/loss stats, and breaks down
losing trades by exchange, exit_reason, type (CE/PE), confidence bands, and
entry hour. Optionally joins with ml_features (DB) to analyze feature values
at entry for losing vs winning trades (nse_next_*, oi_next_sentiment, etc.).

Usage:
  python scripts/analyze_losing_trades.py [--trade-logs-dir trade_logs] [--db]
  --db: if set, query ml_features for entry timestamps and compare feature
        distributions (requires DB connection).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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

# Features to emphasize when comparing losing vs winning (nse_next_* and sentiment)
NEXT_EXPIRY_FEATURES = [
    "nse_next_oi_call_total",
    "nse_next_oi_put_total",
    "nse_next_oi_change_call_total",
    "nse_next_oi_change_put_total",
    "nse_next_volume_call_total",
    "nse_next_volume_put_total",
    "nse_next_oi_change_diff_put_call",
    "oi_next_sentiment",
]


def load_all_trades(trade_logs_dir: Path) -> pd.DataFrame:
    """Load and concatenate all trades_*.csv in trade_logs_dir."""
    if not trade_logs_dir.exists():
        LOGGER.warning("Trade logs dir does not exist: %s", trade_logs_dir)
        return pd.DataFrame()

    frames = []
    for f in sorted(trade_logs_dir.glob("trades_*.csv")):
        try:
            df = pd.read_csv(f)
            # Normalize column names
            df.columns = df.columns.str.strip()
            for col in TRADE_LOG_COLUMNS:
                if col not in df.columns and col in ["entry_timestamp", "pnl", "exchange", "exit_reason", "type", "confidence", "entry_reason"]:
                    pass  # optional
            frames.append(df)
        except Exception as e:
            LOGGER.warning("Skip %s: %s", f.name, e)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)

    # Ensure numeric
    if "pnl" in out.columns:
        out["pnl"] = pd.to_numeric(out["pnl"], errors="coerce")
    if "confidence" in out.columns:
        out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")

    # Entry hour from entry_timestamp
    if "entry_timestamp" in out.columns:
        out["entry_ts"] = pd.to_datetime(out["entry_timestamp"], errors="coerce")
        out["entry_hour"] = out["entry_ts"].dt.hour
        out["entry_date"] = out["entry_ts"].dt.date

    return out


def analyze_losing_trades(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary and breakdowns for losing vs winning trades."""
    if df.empty or "pnl" not in df.columns:
        return {"error": "No trades or no pnl column"}

    df = df.copy()
    df = df.dropna(subset=["pnl"])
    total = len(df)
    wins = (df["pnl"] > 0).sum()
    losses = (df["pnl"] < 0).sum()
    flat = (df["pnl"] == 0).sum()

    losing = df[df["pnl"] < 0]
    winning = df[df["pnl"] > 0]

    # By exchange
    by_exchange = df.groupby("exchange", dropna=False).agg(
        total=("pnl", "count"),
        wins=(("pnl", lambda s: (s > 0).sum())),
        losses=(("pnl", lambda s: (s < 0).sum())),
        win_rate=("pnl", lambda s: (s > 0).mean() if len(s) else 0),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
    ).round(4)

    # Losing trades by exit_reason
    exit_reason_counts = losing["exit_reason"].value_counts() if not losing.empty else pd.Series(dtype=int)

    # Losing by type (CE/PE)
    type_counts = losing["type"].value_counts() if not losing.empty else pd.Series(dtype=int)

    # Confidence bands: losing vs winning
    def conf_band(c: float) -> str:
        if pd.isna(c): return "unknown"
        if c < 0.6: return "low_<0.6"
        if c < 0.75: return "mid_0.6_0.75"
        if c < 0.9: return "high_0.75_0.9"
        return "very_high_>=0.9"

    if "confidence" in df.columns:
        df["conf_band"] = df["confidence"].apply(conf_band)
        conf_breakdown = df.groupby("conf_band", dropna=False).agg(
            total=("pnl", "count"),
            losses=(("pnl", lambda s: (s < 0).sum())),
            loss_rate=(("pnl", lambda s: (s < 0).mean() if len(s) else 0)),
        ).round(4)
    else:
        conf_breakdown = None

    # Entry hour distribution for losing trades
    hour_dist = losing["entry_hour"].value_counts().sort_index() if "entry_hour" in losing.columns and not losing.empty else None

    # Avg confidence: losing vs winning
    avg_conf_losing = losing["confidence"].mean() if "confidence" in losing.columns and not losing.empty else None
    avg_conf_winning = winning["confidence"].mean() if "confidence" in winning.columns and not winning.empty else None

    return {
        "total_trades": int(total),
        "wins": int(wins),
        "losses": int(losses),
        "flat": int(flat),
        "win_rate": round((wins / total * 100) if total else 0, 2),
        "total_pnl": float(df["pnl"].sum()),
        "avg_pnl_per_trade": float(df["pnl"].mean()),
        "by_exchange": by_exchange,
        "losing_exit_reason_counts": exit_reason_counts,
        "losing_type_counts": type_counts,
        "conf_breakdown": conf_breakdown,
        "avg_confidence_losing": float(avg_conf_losing) if avg_conf_losing is not None else None,
        "avg_confidence_winning": float(avg_conf_winning) if avg_conf_winning is not None else None,
        "losing_entry_hour_dist": hour_dist,
        "losing_sample": losing.head(20) if not losing.empty else None,
    }


def fetch_features_at_entries(
    entry_timestamps: List[pd.Timestamp],
    exchange: str,
    lookback_minutes: int = 5,
) -> Optional[pd.DataFrame]:
    """
    Query ml_features for rows near entry_timestamps (by exchange).
    Returns DataFrame with timestamp, exchange, and feature columns if DB available.
    """
    try:
        from datetime import timedelta
        from database_new import load_historical_data_for_ml
    except ImportError:
        LOGGER.warning("database_new not available; skipping DB feature fetch.")
        return None

    if not entry_timestamps:
        return None

    min_ts = min(entry_timestamps) - timedelta(minutes=lookback_minutes)
    max_ts = max(entry_timestamps) + timedelta(minutes=lookback_minutes)
    start_date = min_ts.date() if hasattr(min_ts, "date") else pd.Timestamp(min_ts).date()
    end_date = max_ts.date() if hasattr(max_ts, "date") else pd.Timestamp(max_ts).date()

    try:
        df_hist = load_historical_data_for_ml(exchange, start_date, end_date)
        if df_hist is None or df_hist.empty:
            return None
        if "timestamp" not in df_hist.columns and df_hist.index.name is None:
            df_hist = df_hist.reset_index()
        df_hist["timestamp"] = pd.to_datetime(df_hist.get("timestamp", df_hist.index))
        return df_hist
    except Exception as e:
        LOGGER.warning("load_historical_data_for_ml failed: %s", e)
        return None


def compare_features_losing_vs_winning(
    trades_df: pd.DataFrame,
    features_df: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """
    If we have features_df with timestamp and exchange, merge to trades and
    compare mean/std of NEXT_EXPIRY_FEATURES for losing vs winning.
    """
    if features_df is None or features_df.empty or trades_df.empty:
        return None

    # Simple merge by nearest timestamp per (exchange, entry_date, entry_hour)
    trades_df = trades_df.copy()
    trades_df["entry_ts"] = pd.to_datetime(trades_df["entry_timestamp"], errors="coerce")
    trades_df["entry_round"] = trades_df["entry_ts"].dt.floor("5min")

    if "timestamp" not in features_df.columns:
        return None
    features_df["ts_round"] = pd.to_datetime(features_df["timestamp"]).dt.floor("5min")

    merged = trades_df.merge(
        features_df,
        left_on=["exchange", "entry_round"],
        right_on=["exchange", "ts_round"],
        how="left",
        suffixes=("", "_feat"),
    )

    available = [c for c in NEXT_EXPIRY_FEATURES if c in merged.columns]
    if not available:
        return None

    losing = merged[merged["pnl"] < 0]
    winning = merged[merged["pnl"] > 0]

    summary = []
    for col in available:
        summary.append({
            "feature": col,
            "mean_losing": losing[col].mean(),
            "mean_winning": winning[col].mean(),
            "std_losing": losing[col].std(),
            "std_winning": winning[col].std(),
            "diff_mean": (winning[col].mean() - losing[col].mean()) if len(winning) and len(losing) else None,
        })
    return pd.DataFrame(summary).round(6)


def main():
    parser = argparse.ArgumentParser(description="Analyze losing trades from trade_logs")
    parser.add_argument("--trade-logs-dir", type=Path, default=Path("trade_logs"), help="Path to trade_logs folder")
    parser.add_argument("--db", action="store_true", help="Try to join with ml_features (DB) for feature comparison")
    parser.add_argument("--out", type=Path, default=None, help="Write summary to this CSV path")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    trade_logs_dir = args.trade_logs_dir if args.trade_logs_dir.is_absolute() else project_root / args.trade_logs_dir

    LOGGER.info("Loading trades from %s", trade_logs_dir)
    df = load_all_trades(trade_logs_dir)
    if df.empty:
        LOGGER.error("No trades loaded.")
        return 1

    LOGGER.info("Total rows loaded: %d", len(df))

    result = analyze_losing_trades(df)
    if "error" in result:
        LOGGER.error(result["error"])
        return 1

    # Print summary
    LOGGER.info("=== SUMMARY ===")
    LOGGER.info("Total trades: %d | Wins: %d | Losses: %d | Flat: %d", result["total_trades"], result["wins"], result["losses"], result["flat"])
    LOGGER.info("Win rate: %s%% | Total PnL: %s | Avg PnL/trade: %s", result["win_rate"], result["total_pnl"], result["avg_pnl_per_trade"])
    LOGGER.info("Avg confidence (losing): %s | Avg confidence (winning): %s", result["avg_confidence_losing"], result["avg_confidence_winning"])
    LOGGER.info("")
    LOGGER.info("=== BY EXCHANGE ===")
    print(result["by_exchange"].to_string())
    LOGGER.info("")
    LOGGER.info("=== LOSING TRADES BY EXIT REASON ===")
    print(result["losing_exit_reason_counts"].to_string())
    LOGGER.info("")
    LOGGER.info("=== LOSING TRADES BY TYPE (CE/PE) ===")
    print(result["losing_type_counts"].to_string())
    if result.get("conf_breakdown") is not None:
        LOGGER.info("")
        LOGGER.info("=== CONFIDENCE BAND vs LOSS RATE ===")
        print(result["conf_breakdown"].to_string())
    if result.get("losing_entry_hour_dist") is not None:
        LOGGER.info("")
        LOGGER.info("=== LOSING TRADES BY ENTRY HOUR ===")
        print(result["losing_entry_hour_dist"].to_string())

    # Optional: DB feature comparison
    if args.db and result.get("losses", 0) > 0:
        losing_df = df[df["pnl"] < 0]
        exchanges = losing_df["exchange"].unique().tolist()
        for ex in exchanges[:1]:  # one exchange for simplicity
            sub = losing_df[losing_df["exchange"] == ex]
            entry_times = pd.to_datetime(sub["entry_timestamp"].dropna()).tolist()
            feats = fetch_features_at_entries(entry_times, ex)
            cmp = compare_features_losing_vs_winning(df, feats)
            if cmp is not None and not cmp.empty:
                LOGGER.info("")
                LOGGER.info("=== NEXT_EXPIRY FEATURES: MEAN (Losing vs Winning) [sample] ===")
                print(cmp.to_string())

    if args.out:
        out_path = args.out if args.out.is_absolute() else project_root / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result["by_exchange"].to_csv(out_path)
        LOGGER.info("Wrote by_exchange to %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
