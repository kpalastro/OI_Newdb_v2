"""
Analyze 3-minute percentage change in next-expiry OI and volume (CE/PE)
and relate to direction/swing.

- Computes: pct_3m in nse_next_oi_call_total, nse_next_oi_put_total,
  nse_next_volume_call_total, nse_next_volume_put_total (next* CE/PE).
- Direction: oi_next_sentiment = nse_next_oi_change_put_total - nse_next_oi_change_call_total (DB);
  positive = bearish, negative = bullish.
- Optionally joins trade logs to see win rate when 3m % change aligns with trade type (CE/PE).

DB: load_historical_data_for_ml returns table + feature_payload; duplicate column names are
    deduplicated (keep first) so df[col] is always a Series. Required columns: nse_next_oi_call_total,
    nse_next_oi_put_total, nse_next_volume_call_total, nse_next_volume_put_total, oi_next_sentiment.

Usage:
  python scripts/analyze_next_oi_3m_direction.py [--exchange NSE] [--days 30] [--trade-logs-dir trade_logs] [--out reports]
  python scripts/analyze_next_oi_3m_direction.py --validate   # run on synthetic data (no DB)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)

# Next-expiry OI/volume column names (levels for % change)
OI_CE = "nse_next_oi_call_total"
OI_PE = "nse_next_oi_put_total"
VOL_CE = "nse_next_volume_call_total"
VOL_PE = "nse_next_volume_put_total"
SENTIMENT = "oi_next_sentiment"
SENTIMENT_ALT = "nse_next_oi_change_diff_put_call"

LAG_MINUTES = 3

# Columns expected from ml_features (DB schema) for this analysis
REQUIRED_ML_COLUMNS = [
    "timestamp", "exchange",
    OI_CE, OI_PE, VOL_CE, VOL_PE,
    SENTIMENT, SENTIMENT_ALT,
]


def _synthetic_ml_features(n_rows: int = 500, exchange: str = "NSE") -> pd.DataFrame:
    """Build synthetic DataFrame matching ml_features schema for validation without DB."""
    np.random.seed(42)
    base = pd.Timestamp.now().normalize()
    ts = pd.date_range(base, periods=n_rows, freq="1min")
    df = pd.DataFrame({
        "timestamp": ts,
        "exchange": exchange,
        OI_CE: np.abs(np.random.randn(n_rows) * 1e6) + 1e5,
        OI_PE: np.abs(np.random.randn(n_rows) * 1e6) + 1e5,
        VOL_CE: np.abs(np.random.randn(n_rows) * 1e5) + 1e4,
        VOL_PE: np.abs(np.random.randn(n_rows) * 1e5) + 1e4,
    })
    df[SENTIMENT_ALT] = np.random.randn(n_rows) * 1e4
    df[SENTIMENT] = df[SENTIMENT_ALT]
    return df


def load_ml_features(exchange: str, days: int, csv_path: Path | None = None) -> pd.DataFrame:
    """Load ml_features for exchange: from DB (last `days`) or from CSV if csv_path given."""
    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep="first")]
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if "exchange" in df.columns:
            df = df[df["exchange"] == exchange]
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df

    try:
        from datetime import date, timedelta
        from database_new import load_historical_data_for_ml
    except ImportError as e:
        LOGGER.warning("DB not available: %s", e)
        return pd.DataFrame()

    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    df = load_historical_data_for_ml(exchange, start_date, end_date)
    if df is None or df.empty:
        return pd.DataFrame()
    # DB returns table columns + feature_payload deserialized; duplicate names cause df[col] to be DataFrame
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    missing = [c for c in [OI_CE, OI_PE, VOL_CE, VOL_PE] if c not in df.columns]
    if missing:
        LOGGER.warning("ml_features missing columns (check DB): %s", missing)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def add_3m_pct_change(df: pd.DataFrame) -> pd.DataFrame:
    """Add 3-minute percentage change for next* OI and volume CE/PE.
    Uses shift(3) assuming ~1-min bars; fast and correct for regular 1-min data.
    """
    required = [OI_CE, OI_PE, VOL_CE, VOL_PE]
    missing = [c for c in required if c not in df.columns]
    if missing:
        LOGGER.warning("Missing columns for 3m pct change: %s", missing)
        return df

    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    def _as_series(frame, col_name):
        v = frame[col_name]
        if isinstance(v, pd.DataFrame):
            v = v.iloc[:, 0]
        return pd.Series(pd.to_numeric(v, errors="coerce"), index=frame.index)

    for col in required:
        raw = _as_series(df, col)
        lag = raw.shift(LAG_MINUTES)
        pct = np.where(lag != 0, (raw - lag) / lag * 100, np.nan)
        df[f"pct_3m_{col}"] = pct
    return df


# Threshold for "neutral" in CE/PE comparison (|pct| < this = neutral)
NEUTRAL_THRESHOLD_PCT = 1.0


def add_ce_pe_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Add CE vs PE: difference magnitude (CE - PE) and buckets (CE pos/PE neg, etc.)."""
    oi_ce = f"pct_3m_{OI_CE}"
    oi_pe = f"pct_3m_{OI_PE}"
    vol_ce = f"pct_3m_{VOL_CE}"
    vol_pe = f"pct_3m_{VOL_PE}"
    for name, c1, c2 in [
        ("oi", oi_ce, oi_pe),
        ("vol", vol_ce, vol_pe),
    ]:
        if c1 not in df.columns or c2 not in df.columns:
            continue
        a = pd.to_numeric(df[c1], errors="coerce")
        b = pd.to_numeric(df[c2], errors="coerce")
        df[f"pct_3m_{name}_diff_ce_pe"] = a - b  # positive = CE building more than PE
        # Buckets: CE_pos_PE_neg, CE_neg_PE_pos, both_pos, both_neg, CE_pos_PE_neutral, etc.
        pos = a > NEUTRAL_THRESHOLD_PCT
        neg = a < -NEUTRAL_THRESHOLD_PCT
        neu_ce = ~pos & ~neg
        pos_pe = b > NEUTRAL_THRESHOLD_PCT
        neg_pe = b < -NEUTRAL_THRESHOLD_PCT
        neu_pe = ~pos_pe & ~neg_pe
        bucket = np.where(
            pos & (neg_pe | neu_pe), "CE_pos_PE_neg_or_neutral",
            np.where(
                neg & (pos_pe | neu_pe), "CE_neg_PE_pos_or_neutral",
                np.where(pos & pos_pe, "both_pos",
                    np.where(neg & neg_pe, "both_neg",
                        np.where(neg_pe & (pos | neu_ce), "PE_neg_CE_any",
                            np.where(pos_pe & (neg | neu_ce), "PE_pos_CE_any", "neutral_or_mixed"))))))
        df[f"ce_pe_{name}_bucket"] = bucket
    # Combined OI + volume: weighted diff and "both agree" flag
    oi_diff = "pct_3m_oi_diff_ce_pe"
    vol_diff = "pct_3m_vol_diff_ce_pe"
    if oi_diff in df.columns and vol_diff in df.columns:
        oi_d = pd.to_numeric(df[oi_diff], errors="coerce")
        vol_d = pd.to_numeric(df[vol_diff], errors="coerce")
        # Combined: 0.6 * OI diff + 0.4 * volume diff (same scale: CE-PE %)
        df["pct_3m_combined_oi_vol_diff_ce_pe"] = 0.6 * oi_d + 0.4 * vol_d
        # Both agree: same sign (both bullish or both bearish)
        df["oi_vol_agree"] = np.sign(oi_d) == np.sign(vol_d)
        df["oi_vol_agree_strict"] = (oi_d > NEUTRAL_THRESHOLD_PCT) & (vol_d > NEUTRAL_THRESHOLD_PCT) | (oi_d < -NEUTRAL_THRESHOLD_PCT) & (vol_d < -NEUTRAL_THRESHOLD_PCT)
    return df


def add_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Add direction from DB formula: oi_next_sentiment = put_change - call_change.
    Bearish (1) if oi_next_sentiment > 0; bullish (-1) if < 0."""
    col = SENTIMENT if SENTIMENT in df.columns else SENTIMENT_ALT
    if col not in df.columns:
        df["direction"] = np.nan
        return df
    v = df[col]
    if isinstance(v, pd.DataFrame):
        v = v.iloc[:, 0]
    s = pd.to_numeric(v, errors="coerce")
    df["direction"] = np.sign(s)  # 1 = bearish (put build), -1 = bullish (call build), 0 = neutral
    return df


def analyze_direction_alignment(df: pd.DataFrame) -> dict:
    """Analyze how 3m % change in next OI/volume aligns with direction (oi_next_sentiment)."""
    out = {}
    pct_cols = [f"pct_3m_{c}" for c in [OI_CE, OI_PE, VOL_CE, VOL_PE]]
    pct_cols = [c for c in pct_cols if c in df.columns]
    if not pct_cols or "direction" not in df.columns:
        return out

    for col in pct_cols:
        positive = df[col] > 0
        negative = df[col] < 0
        valid = df[col].notna() & (positive | negative)
        if valid.sum() < 10:
            continue
        d = df.loc[valid, "direction"]
        # CE OI/vol up -> expect bullish (direction < 0). PE OI/vol up -> expect bearish (direction > 0).
        if "oi_call" in col or "volume_call" in col:
            align = (positive & (d < 0)) | (negative & (d > 0))  # CE up + bullish, CE down + bearish
        else:
            align = (positive & (d > 0)) | (negative & (d < 0))  # PE up + bearish, PE down + bullish
        out[col] = {
            "n_positive": positive.sum(),
            "n_negative": negative.sum(),
            "align_pct": align.mean() * 100 if valid.any() else 0,
            "when_positive_direction_mean": d[positive].mean() if positive.any() else np.nan,
            "when_negative_direction_mean": d[negative].mean() if negative.any() else np.nan,
        }
    return out


def analyze_ce_vs_pe_direction(df: pd.DataFrame) -> dict:
    """CE vs PE: alignment of diff magnitude and of buckets with direction.
    - OI diff > 0 (CE building more than PE) -> expect bullish (direction < 0).
    - Vol diff > 0 -> expect bullish.
    """
    out = {}
    if "direction" not in df.columns:
        return out
    d = df["direction"]
    for name, diff_col in [
        ("oi", "pct_3m_oi_diff_ce_pe"),
        ("vol", "pct_3m_vol_diff_ce_pe"),
        ("combined_oi_vol", "pct_3m_combined_oi_vol_diff_ce_pe"),
    ]:
        if diff_col not in df.columns:
            continue
        diff = pd.to_numeric(df[diff_col], errors="coerce")
        pos = diff > 0
        neg = diff < 0
        valid = diff.notna() & (pos | neg)
        if valid.sum() < 10:
            continue
        # CE > PE (diff > 0) -> bullish (direction < 0). CE < PE -> bearish (direction > 0).
        align = (pos & (d < 0)) | (neg & (d > 0))
        out[diff_col] = {
            "n_CE_gt_PE": pos.sum(),
            "n_CE_lt_PE": neg.sum(),
            "align_pct": align.mean() * 100 if valid.any() else 0,
            "when_CE_gt_PE_direction_mean": d[pos].mean() if pos.any() else np.nan,
            "when_CE_lt_PE_direction_mean": d[neg].mean() if neg.any() else np.nan,
            "diff_mean_when_CE_gt_PE": diff[pos].mean() if pos.any() else np.nan,
            "diff_mean_when_CE_lt_PE": diff[neg].mean() if neg.any() else np.nan,
        }
    # When OI and volume 3m diff agree (same sign): does alignment improve?
    if "oi_vol_agree" in df.columns and "direction" in df.columns and "pct_3m_oi_diff_ce_pe" in df.columns:
        agree = df["oi_vol_agree"].astype(bool).fillna(False)
        valid = agree & df["pct_3m_oi_diff_ce_pe"].notna()
        if valid.sum() >= 10:
            diff = pd.to_numeric(df.loc[valid, "pct_3m_oi_diff_ce_pe"], errors="coerce")
            d = df.loc[valid, "direction"]
            pos = diff > 0
            neg = diff < 0
            align = (pos & (d < 0)) | (neg & (d > 0))
            out["oi_vol_agree_align"] = {"n": int(valid.sum()), "align_pct": float(align.mean() * 100)}
        if "oi_vol_agree_strict" in df.columns:
            strict = df["oi_vol_agree_strict"].astype(bool).fillna(False)
            v2 = strict & df["pct_3m_oi_diff_ce_pe"].notna()
            if v2.sum() >= 10:
                diff2 = pd.to_numeric(df.loc[v2, "pct_3m_oi_diff_ce_pe"], errors="coerce")
                d2 = df.loc[v2, "direction"]
                pos2 = diff2 > 0
                neg2 = diff2 < 0
                align2 = (pos2 & (d2 < 0)) | (neg2 & (d2 > 0))
                out["oi_vol_agree_strict_align"] = {"n": int(v2.sum()), "align_pct": float(align2.mean() * 100)}
    # Bucket vs direction
    for name in ["oi", "vol"]:
        bucket_col = f"ce_pe_{name}_bucket"
        if bucket_col not in df.columns:
            continue
        bucket_stats = df.groupby(bucket_col, dropna=False).agg(
            count=("direction", "count"),
            direction_mean=("direction", "mean"),
        ).reset_index()
        # alignment: for CE_pos_PE_neg_or_neutral expect bullish (dir<0), for CE_neg_PE_pos expect bearish (dir>0)
        out[f"_bucket_{name}"] = bucket_stats
    return out


def analyze_trades_vs_ce_pe_combo(merged: pd.DataFrame) -> dict:
    """Win rate by CE vs PE: OI diff sign, vol diff sign, combined OI+vol diff, and by bucket (CE pos / PE neg, etc.)."""
    out = {}
    for diff_col, label in [
        ("pct_3m_oi_diff_ce_pe", "oi_diff"),
        ("pct_3m_vol_diff_ce_pe", "vol_diff"),
        ("pct_3m_combined_oi_vol_diff_ce_pe", "combined_oi_vol_diff"),
    ]:
        if diff_col not in merged.columns:
            continue
        diff = pd.to_numeric(merged[diff_col], errors="coerce")
        for name, mask in [
            ("CE_gt_PE", diff > 0),
            ("CE_lt_PE", diff < 0),
        ]:
            m = merged.loc[mask]
            if len(m) < 5:
                continue
            out[f"{label}_{name}"] = {
                "trades": len(m),
                "win_rate_pct": (m["pnl"] > 0).mean() * 100,
                "mean_pnl": m["pnl"].mean(),
            }
    for name in ["oi", "vol"]:
        bucket_col = f"ce_pe_{name}_bucket"
        if bucket_col not in merged.columns:
            continue
        for bucket in merged[bucket_col].dropna().unique():
            m = merged[merged[bucket_col] == bucket]
            if len(m) < 5:
                continue
            safe = str(bucket).replace(" ", "_")
            out[f"bucket_{name}_{safe}"] = {
                "trades": len(m),
                "win_rate_pct": (m["pnl"] > 0).mean() * 100,
                "mean_pnl": m["pnl"].mean(),
            }
    # When OI and volume 3m diff agree: win rate on agreeing subset
    if "oi_vol_agree" in merged.columns and "pnl" in merged.columns:
        agree = merged["oi_vol_agree"].astype(bool).fillna(False)
        m = merged.loc[agree]
        if len(m) >= 5:
            out["oi_vol_agree"] = {"trades": len(m), "win_rate_pct": (m["pnl"] > 0).mean() * 100, "mean_pnl": m["pnl"].mean()}
    if "oi_vol_agree_strict" in merged.columns and "pnl" in merged.columns:
        strict = merged["oi_vol_agree_strict"].astype(bool).fillna(False)
        m = merged.loc[strict]
        if len(m) >= 5:
            out["oi_vol_agree_strict"] = {"trades": len(m), "win_rate_pct": (m["pnl"] > 0).mean() * 100, "mean_pnl": m["pnl"].mean()}
    return out


def load_trades_and_merge(trade_logs_dir: Path, df_ml: pd.DataFrame, exchange: str) -> pd.DataFrame:
    """Load trade logs, keep closed trades with PnL, merge to nearest ml_features row by timestamp."""
    if not trade_logs_dir.exists():
        return pd.DataFrame()
    frames = []
    for f in sorted(trade_logs_dir.glob("trades_*.csv")):
        try:
            frames.append(pd.read_csv(f))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    trades = pd.concat(frames, ignore_index=True)
    trades["pnl"] = pd.to_numeric(trades["pnl"], errors="coerce")
    trades = trades.dropna(subset=["pnl"]).query("exchange == @exchange")
    if trades.empty:
        return pd.DataFrame()
    # Normalize timestamps: make both sides timezone-naive (UTC->naive) for merge_asof
    trades["entry_ts"] = pd.to_datetime(trades["entry_timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    trades = trades.dropna(subset=["entry_ts"])
    if df_ml.empty or "timestamp" not in df_ml.columns:
        return pd.DataFrame()
    df_ml = df_ml.copy()
    df_ml["timestamp"] = pd.to_datetime(df_ml["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    merged = pd.merge_asof(
        trades.sort_values("entry_ts"),
        df_ml.sort_values("timestamp"),
        left_on="entry_ts",
        right_on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=5),
        suffixes=("", "_ml"),
    )
    return merged


def analyze_trades_vs_3m(merged: pd.DataFrame) -> dict:
    """Win rate when 3m % change in next CE/PE OI or volume is positive vs negative (for CE vs PE trades)."""
    out = {}
    for opt_type in ["CE", "PE"]:
        sub = merged[merged["type"] == opt_type] if "type" in merged.columns else pd.DataFrame()
        if sub.empty:
            continue
        oi_col = f"pct_3m_{OI_CE}" if opt_type == "CE" else f"pct_3m_{OI_PE}"
        vol_col = f"pct_3m_{VOL_CE}" if opt_type == "CE" else f"pct_3m_{VOL_PE}"
        for label, col in [("oi", oi_col), ("vol", vol_col)]:
            if col not in sub.columns:
                continue
            pos = sub[col] > 0
            neg = sub[col] < 0
            for name, mask in [("positive", pos), ("negative", neg)]:
                m = sub.loc[mask]
                if len(m) < 5:
                    continue
                win_rate = (m["pnl"] > 0).mean() * 100
                key = f"{opt_type}_{label}_3m_{name}"
                out[key] = {"trades": len(m), "win_rate_pct": win_rate, "mean_pnl": m["pnl"].mean()}
    return out


def main():
    parser = argparse.ArgumentParser(description="Analyze 3m % change in next* OI/volume vs direction/swing")
    parser.add_argument("--exchange", default="NSE", help="Exchange (NSE or BSE)")
    parser.add_argument("--days", type=int, default=30, help="Days of ml_features to load")
    parser.add_argument("--trade-logs-dir", type=Path, default=Path("trade_logs"), help="Trade logs dir for join")
    parser.add_argument("--out", type=Path, default=None, help="Write summary to this dir (e.g. reports/)")
    parser.add_argument("--csv", type=Path, default=None, help="Load ml_features from CSV instead of DB (must have timestamp, exchange, nse_next_* columns)")
    parser.add_argument("--validate", action="store_true", help="Run pipeline on synthetic data (no DB); validates logic and column handling")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    trade_logs_dir = args.trade_logs_dir if args.trade_logs_dir.is_absolute() else project_root / args.trade_logs_dir

    csv_path = None
    if args.csv is not None:
        csv_path = args.csv if args.csv.is_absolute() else project_root / args.csv
    if args.validate:
        LOGGER.info("VALIDATE mode: using synthetic ml_features (no DB)")
        df = _synthetic_ml_features(n_rows=500, exchange=args.exchange)
    else:
        LOGGER.info("Loading ml_features for %s (last %d days)%s...", args.exchange, args.days, " from CSV" if csv_path and csv_path.exists() else "")
        df = load_ml_features(args.exchange, args.days, csv_path=csv_path if (csv_path and csv_path.exists()) else None)
    if df.empty:
        LOGGER.error("No ml_features loaded. Check DB and exchange.")
        return 1

    df = add_3m_pct_change(df)
    df = add_direction(df)
    df = add_ce_pe_comparison(df)

    # --- 3m % change summary ---
    pct_cols = [c for c in df.columns if c.startswith("pct_3m_")]
    LOGGER.info("")
    LOGGER.info("=== 3-minute %% change in next-expiry OI and volume (CE/PE) ===")
    for c in pct_cols:
        s = df[c].dropna()
        if len(s) > 0:
            LOGGER.info("  %s: mean=%.3f%%, std=%.3f%%, count=%d", c, s.mean(), s.std(), len(s))
    LOGGER.info("")

    # --- Direction alignment (3m % change vs oi_next_sentiment) ---
    align = analyze_direction_alignment(df)
    LOGGER.info("=== Direction alignment (3m %% change vs oi_next_sentiment; oi_next_sentiment = put_change - call_change: + = bearish, - = bullish) ===")
    for col, v in align.items():
        LOGGER.info("  %s: when 3m pct positive, direction mean=%.2f; when negative, direction mean=%.2f; align_pct=%.1f%%",
                    col, v["when_positive_direction_mean"], v["when_negative_direction_mean"], v["align_pct"])
    LOGGER.info("")

    # --- Conclusions: direction/swing ---
    LOGGER.info("=== CONCLUSIONS (direction/swing) ===")
    if align:
        for col, v in align.items():
            name = "CE" if "call" in col else "PE"
            metric = "OI" if "oi_" in col else "Volume"
            if v["align_pct"] > 55:
                LOGGER.info("  • When next-expiry %s %s 3m %% change is POSITIVE, direction (sentiment) aligns %.1f%% of the time (support for %s bias).",
                            name, metric, v["align_pct"], "bearish" if name == "PE" else "bullish")
            elif v["align_pct"] < 45:
                LOGGER.info("  • When next-expiry %s %s 3m %% change is POSITIVE, direction aligns only %.1f%% (weak or opposite signal).",
                            name, metric, v["align_pct"])
            else:
                LOGGER.info("  • Next-expiry %s %s 3m %% change has ~neutral alignment (%.1f%%) with sentiment.",
                            name, metric, v["align_pct"])
    else:
        LOGGER.info("  (Insufficient data for alignment stats.)")
    LOGGER.info("")

    # --- CE vs PE: difference magnitude and relative sign ---
    ce_pe_align = {}
    diff_cols = [c for c in df.columns if "diff_ce_pe" in c]
    if diff_cols:
        LOGGER.info("=== CE vs PE: 3m %% difference (CE - PE); positive = CE building more ===")
        for c in diff_cols:
            s = df[c].dropna()
            if len(s) > 0:
                LOGGER.info("  %s: mean=%.3f%%, std=%.3f%%, count=%d", c, s.mean(), s.std(), len(s))
        ce_pe_align.update(analyze_ce_vs_pe_direction(df))
        for col in ["pct_3m_oi_diff_ce_pe", "pct_3m_vol_diff_ce_pe", "pct_3m_combined_oi_vol_diff_ce_pe"]:
            if col not in ce_pe_align:
                continue
            v = ce_pe_align[col]
            LOGGER.info("  %s: when CE>PE align_pct=%.1f%% (expect bullish); when CE<PE direction_mean=%.2f", col, v["align_pct"], v["when_CE_lt_PE_direction_mean"])
        for key in ["oi_vol_agree_align", "oi_vol_agree_strict_align"]:
            if key in ce_pe_align:
                v = ce_pe_align[key]
                LOGGER.info("  %s: n=%d, align_pct=%.1f%%", key, v["n"], v["align_pct"])
        for key in ["_bucket_oi", "_bucket_vol"]:
            if key not in ce_pe_align:
                continue
            tbl = ce_pe_align[key]
            LOGGER.info("  Buckets (%s):", key.replace("_bucket_", ""))
            for _, r in tbl.iterrows():
                LOGGER.info("    %s: count=%d, direction_mean=%.2f", r.iloc[0], int(r["count"]), r["direction_mean"])
    LOGGER.info("")

    # --- Trade-log join: win rate when 3m % change positive vs negative ---
    merged = load_trades_and_merge(trade_logs_dir, df, args.exchange)
    if not merged.empty:
        trade_stats = analyze_trades_vs_3m(merged)
        LOGGER.info("=== Trade outcomes vs 3m %% change at entry (same exchange) ===")
        for key, v in trade_stats.items():
            LOGGER.info("  %s: trades=%d, win_rate=%.1f%%, mean_pnl=%s", key, v["trades"], v["win_rate_pct"], round(v["mean_pnl"], 2))
        combo = analyze_trades_vs_ce_pe_combo(merged)
        LOGGER.info("")
        LOGGER.info("=== Trade outcomes: CE vs PE (diff magnitude and buckets) ===")
        LOGGER.info("  OI/Vol diff = CE 3m%% - PE 3m%%; CE_gt_PE = bullish bias, CE_lt_PE = bearish bias")
        for key, v in combo.items():
            LOGGER.info("  %s: trades=%d, win_rate=%.1f%%, mean_pnl=%s", key, v["trades"], v["win_rate_pct"], round(v["mean_pnl"], 2))
        LOGGER.info("")
        if trade_stats:
            LOGGER.info("  • Prefer CE entries when 3m %% change in next CE OI/volume is positive (if win rate is higher in that bucket).")
            LOGGER.info("  • Prefer PE entries when 3m %% change in next PE OI/volume is positive (if win rate is higher).")
        if combo:
            LOGGER.info("  • Compare CE_gt_PE vs CE_lt_PE: trade when diff aligns with direction (CE>PE for bullish, CE<PE for bearish).")
            if "oi_vol_agree" in combo or "oi_vol_agree_strict" in combo:
                LOGGER.info("  • When OI and volume 3m diff agree (same sign): check oi_vol_agree / oi_vol_agree_strict win_rate vs baseline.")
    else:
        LOGGER.info("(No trade-log merge: dir missing or no closed trades for exchange.)")
    LOGGER.info("")

    if args.out:
        out_dir = args.out if args.out.is_absolute() else project_root / args.out
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / f"next_oi_3m_direction_summary_{args.exchange}.txt"
        with open(summary_path, "w") as f:
            f.write(f"Exchange: {args.exchange}\n")
            f.write("3m %% change in next* OI/volume (CE/PE) vs direction/swing\n")
            f.write("oi_next_sentiment = nse_next_oi_change_put_total - nse_next_oi_change_call_total (DB); + = bearish, - = bullish\n\n")
            for c in pct_cols:
                s = df[c].dropna()
                if len(s) > 0:
                    f.write(f"{c}: mean={s.mean():.3f}%%, std={s.std():.3f}%%, n={len(s)}\n")
            f.write("\nAlignment:\n")
            for col, v in align.items():
                f.write(f"  {col}: align_pct={v['align_pct']:.1f}%%\n")
            if diff_cols:
                f.write("\nCE vs PE diff (CE - PE):\n")
                for c in diff_cols:
                    s = df[c].dropna()
                    if len(s) > 0:
                        f.write(f"  {c}: mean={s.mean():.3f}%%, std={s.std():.3f}%%, n={len(s)}\n")
                if diff_cols and ce_pe_align:
                    for col in ["pct_3m_oi_diff_ce_pe", "pct_3m_vol_diff_ce_pe", "pct_3m_combined_oi_vol_diff_ce_pe"]:
                        if col in ce_pe_align:
                            v = ce_pe_align[col]
                            f.write(f"  {col}: align_pct={v['align_pct']:.1f}%%\n")
                    for key in ["oi_vol_agree_align", "oi_vol_agree_strict_align"]:
                        if key in ce_pe_align:
                            v = ce_pe_align[key]
                            f.write(f"  {key}: n={v['n']}, align_pct={v['align_pct']:.1f}%%\n")
        LOGGER.info("Wrote %s", summary_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
