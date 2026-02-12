#!/usr/bin/env python3
"""
Backfill ml_features feature_payload with vix_contango_pct and term_structure_spread
from vix_term_structure where possible.

Run from project root:
  python scripts/backfill_blank_features.py [--exchange NSE|BSE] [--dry-run]

Other blank/zero columns (block_trade_*, oi_velocity_*, sentiment_fii_*, etc.) have
different root causes; see docs/BLANK_ZERO_FEATURES_ANALYSIS.md.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database_new as db
from database_new import _get_placeholder, get_db_connection, release_db_connection, db_lock


def backfill_vix_term_in_payload(exchange: str | None = None, dry_run: bool = False) -> int:
    """
    For each ml_features row with feature_payload, set vix_contango_pct and
    term_structure_spread from the latest vix_term_structure row at or before
    that timestamp (same exchange). Returns number of rows updated.
    """
    ph = _get_placeholder()
    with db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            # Fetch ml_features with feature_payload
            if exchange:
                cursor.execute(
                    f"""
                    SELECT timestamp, exchange, feature_payload
                    FROM ml_features
                    WHERE exchange = {ph} AND feature_payload IS NOT NULL AND feature_payload != ''
                    ORDER BY timestamp
                    """,
                    (exchange,),
                )
            else:
                cursor.execute(
                    """
                    SELECT timestamp, exchange, feature_payload
                    FROM ml_features
                    WHERE feature_payload IS NOT NULL AND feature_payload != ''
                    ORDER BY timestamp
                    """
                )
            rows = cursor.fetchall()
            if not rows:
                logger.info("No ml_features rows with feature_payload found.")
                return 0

            # Get latest vix_term_structure per (exchange) up to each ml_features timestamp
            # We do per-row lookup: for each ml row, get latest vix row where vix.timestamp <= ml.timestamp
            updated = 0
            for (ts, ex, payload_str) in rows:
                cursor.execute(
                    f"""
                    SELECT contango_pct, front_month_price, next_month_price
                    FROM vix_term_structure
                    WHERE exchange = {ph} AND timestamp <= {ph}
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """,
                    (ex, ts),
                )
                vix_row = cursor.fetchone()
                if not vix_row:
                    continue
                contango_pct, front_price, next_price = vix_row
                spread = None
                if front_price is not None and next_price is not None and float(front_price) != 0:
                    spread = float(next_price) - float(front_price)
                try:
                    payload = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                changed = False
                if contango_pct is not None:
                    payload["vix_contango_pct"] = float(contango_pct)
                    changed = True
                if spread is not None:
                    payload["term_structure_spread"] = spread
                    changed = True
                if not changed:
                    continue
                new_payload = json.dumps(payload)
                if dry_run:
                    updated += 1
                    continue
                cursor.execute(
                    f"""
                    UPDATE ml_features
                    SET feature_payload = {ph}
                    WHERE timestamp = {ph} AND exchange = {ph}
                    """,
                    (new_payload, ts, ex),
                )
                if cursor.rowcount:
                    updated += 1
            if not dry_run:
                conn.commit()
            logger.info("Backfill vix term: %s rows updated (dry_run=%s)", updated, dry_run)
            return updated
        finally:
            release_db_connection(conn)


def main():
    ap = argparse.ArgumentParser(description="Backfill blank/zero ML features from supporting tables.")
    ap.add_argument("--exchange", choices=["NSE", "BSE"], default=None, help="Limit to one exchange")
    ap.add_argument("--dry-run", action="store_true", help="Do not write; only report count")
    args = ap.parse_args()
    backfill_vix_term_in_payload(exchange=args.exchange, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
