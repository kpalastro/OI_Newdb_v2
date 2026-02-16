#!/usr/bin/env python3
"""
Dump the last 10 calendar days of prepared training data for BSE and NSE
into data/BSE_last_10_days.csv and data/NSE_last_10_days.csv.

Uses the same pipeline as train_orchestrator: load_historical_data_for_ml
-> prepare_training_features -> define_triple_barrier_target, then keeps
only the last 10 days. Useful to inspect feature columns and sample values
used in training.

Run from project root (same env as train_orchestrator). Requires DB access.
  python scripts/dump_last_10_days_training_data.py
"""
from __future__ import annotations

import os
import sys
from datetime import timedelta

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import database_new as db
from feature_engineering import REQUIRED_FEATURE_COLUMNS, prepare_training_features
from time_utils import today_ist

try:
    from train_model import define_triple_barrier_target
    _TARGET_AVAILABLE = True
except ImportError:
    _TARGET_AVAILABLE = False


def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)

    end_date = today_ist()
    # Load 30 days so rolling stats (e.g. PCR z-score) have enough history
    start_date = end_date - timedelta(days=30)
    last_n_days = 10
    cutoff_date = end_date - timedelta(days=last_n_days - 1)

    for exchange in ("BSE", "NSE"):
        print(f"Loading and preparing {exchange}...")
        raw = db.load_historical_data_for_ml(exchange, start_date, end_date)
        if raw is None or raw.empty:
            print(f"  No data for {exchange}; skipping.")
            continue
        features = prepare_training_features(raw, required_columns=REQUIRED_FEATURE_COLUMNS)
        if features.empty:
            print(f"  Feature preparation yielded no rows for {exchange}; skipping.")
            continue
        if _TARGET_AVAILABLE:
            frame = define_triple_barrier_target(features)
        else:
            frame = features
            print("  (train_model not available; dumping features only, no target column)")
        # Keep only last 10 calendar days (index >= cutoff_date)
        # Index may be tz-aware (e.g. UTC); use same tz for cutoff to avoid comparison error
        cutoff_ts = pd.Timestamp(cutoff_date)
        if frame.index.tz is not None:
            cutoff_ts = cutoff_ts.tz_localize(frame.index.tz)
        subset = frame.loc[frame.index >= cutoff_ts]
        if subset.empty:
            print(f"  No rows in last {last_n_days} days for {exchange}; skipping.")
            continue
        path = os.path.join(out_dir, f"{exchange}_last_10_days.csv")
        subset.to_csv(path, index=True)
        print(f"  Wrote {len(subset)} rows, {len(subset.columns)} columns -> {path}")

    print("Done.")


if __name__ == "__main__":
    main()
