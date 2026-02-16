#!/usr/bin/env python3
"""
Export a CSV containing only the features used for training (no extra columns).

Uses the same pipeline as train_orchestrator: load_historical_data_for_ml
-> prepare_training_features -> define_triple_barrier_target, then keeps
only columns that are actually used for training (REQUIRED_FEATURE_COLUMNS
that exist in the frame, optionally excluding zero-variance features).

Run from project root. Requires DB access.
  python scripts/export_training_features_csv.py --exchange NSE --days 150
  python scripts/export_training_features_csv.py --exchange NSE --days 30 --output data/NSE_training_features.csv --drop-zero-variance
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database_new as db
from feature_engineering import REQUIRED_FEATURE_COLUMNS, prepare_training_features
from time_utils import today_ist

try:
    from train_model import define_triple_barrier_target
    _TARGET_AVAILABLE = True
except ImportError:
    _TARGET_AVAILABLE = False


def _drop_zero_variance_features(frame: pd.DataFrame, feature_cols: list, min_variance: float = 1e-8) -> list:
    """Keep only columns with variance above threshold (match train_orchestrator)."""
    if not frame.size or not feature_cols:
        return feature_cols
    kept = []
    for col in feature_cols:
        if col not in frame.columns:
            continue
        try:
            var = frame[col].astype(np.float64).var()
            if var is not None and not (np.isnan(var) or var < min_variance):
                kept.append(col)
        except Exception:
            kept.append(col)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export CSV with only the features used for training (no unused columns)."
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default="NSE",
        choices=("NSE", "BSE"),
        help="Exchange to export (default: NSE)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=150,
        help="Number of calendar days of data to load (default: 150)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: data/<exchange>_training_features.csv)",
    )
    parser.add_argument(
        "--drop-zero-variance",
        action="store_true",
        help="Exclude constant/zero-variance features (same as train_orchestrator)",
    )
    parser.add_argument(
        "--no-target",
        action="store_true",
        help="Do not include target column (features only)",
    )
    args = parser.parse_args()

    out_path = args.output
    if not out_path:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{args.exchange}_training_features.csv")

    end_date = today_ist()
    start_date = end_date - timedelta(days=args.days)

    print(f"Loading {args.exchange} from {start_date} to {end_date}...")
    raw = db.load_historical_data_for_ml(args.exchange, start_date, end_date)
    if raw is None or raw.empty:
        print("No data found. Exiting.")
        sys.exit(1)

    print("Preparing features (same as training pipeline)...")
    frame = prepare_training_features(raw, required_columns=REQUIRED_FEATURE_COLUMNS)
    if frame.empty:
        print("Feature preparation yielded no rows. Exiting.")
        sys.exit(1)

    if _TARGET_AVAILABLE and not args.no_target:
        frame = define_triple_barrier_target(frame)
        has_target = "target" in frame.columns
    else:
        has_target = "target" in frame.columns and not args.no_target

    # Only columns that are used for training
    feature_cols = [c for c in REQUIRED_FEATURE_COLUMNS if c in frame.columns]
    if args.drop_zero_variance:
        feature_cols = _drop_zero_variance_features(frame, feature_cols)
        print(f"Dropped zero-variance features; {len(feature_cols)} features kept.")

    # Reset index so timestamp is a column (prepare_training_features uses timestamp as index)
    if "timestamp" not in frame.columns:
        frame = frame.reset_index()
    if "timestamp" not in frame.columns and frame.index.name:
        frame = frame.rename_axis("timestamp").reset_index()

    # Build output: timestamp + exchange (if present) + training features + target (if present)
    out_cols = []
    if "timestamp" in frame.columns:
        out_cols.append("timestamp")
    if "exchange" in frame.columns:
        out_cols.append("exchange")
    out_cols.extend(feature_cols)
    if has_target and "target" in frame.columns:
        out_cols.append("target")

    out_cols = [c for c in out_cols if c in frame.columns]
    export_df = frame[out_cols].copy()

    export_df.to_csv(out_path, index=False)
    print(f"Wrote {len(export_df)} rows, {len(out_cols)} columns -> {out_path}")
    print(f"  Features used for training: {len(feature_cols)}")


if __name__ == "__main__":
    main()
