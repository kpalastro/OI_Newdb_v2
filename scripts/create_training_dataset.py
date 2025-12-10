from __future__ import annotations

import argparse
import logging
from datetime import timedelta

from time_utils import today_ist
import database_new as db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and register a versioned training dataset for an exchange."
    )
    parser.add_argument(
        "--exchange",
        required=True,
        help="Exchange key (e.g. NSE, BSE).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Lookback window in calendar days for ML features.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Optional dataset version tag (default: YYYY-MM-DD for today in IST).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    end_date = today_ist()
    start_date = end_date - timedelta(days=args.days)
    version = args.version or end_date.strftime("%Y-%m-%d")

    logging.info(
        "Creating training dataset for %s from %s to %s (version=%s)",
        args.exchange,
        start_date,
        end_date,
        version,
    )

    summary = db.export_training_window(
        exchange=args.exchange,
        start_timestamp=start_date,
        end_timestamp=end_date,
        file_prefix=f"{args.exchange}_dataset_{version}",
        include_payload=True,
        dataset_version=version,
    )

    if not summary:
        logging.error("No data exported for %s; dataset not created.", args.exchange)
        return

    logging.info("Dataset created: %s", summary)


if __name__ == "__main__":
    main()


