from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def _load_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _save_registry(path: Path, data: Dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")
    except Exception as exc:
        logging.error("Failed to write registry.yml: %s", exc, exc_info=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote a candidate model for an exchange by updating models/registry.yml."
    )
    parser.add_argument(
        "--exchange",
        required=True,
        help="Exchange key (e.g. NSE, BSE).",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Logical model version tag to record in the registry (e.g. v2025-11-28).",
    )
    parser.add_argument(
        "--eval-tag",
        default=None,
        help="Optional eval tag whose eval_<tag>.json will be referenced in deployments.log.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    ex = args.exchange.upper()
    registry_path = Path("models") / "registry.yml"
    registry = _load_registry(registry_path)

    model_dir = f"models/{ex}"
    # Ensure directory exists (this script does not move/copy artifacts yet)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    ex_cfg: Dict[str, Any] = registry.get(ex, {})
    ex_cfg["current"] = {
        "version": args.version,
        "model_dir": model_dir,
    }
    registry[ex] = ex_cfg
    _save_registry(registry_path, registry)

    # Append to deployments.log for audit trail
    deployments_log = Path("models") / "deployments.log"
    deployments_log.parent.mkdir(parents=True, exist_ok=True)

    eval_path = None
    if args.eval_tag:
        candidate = Path("models") / ex / "reports" / f"eval_{args.eval_tag}.json"
        if candidate.exists():
            eval_path = str(candidate)

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "exchange": ex,
        "version": args.version,
        "model_dir": model_dir,
        "eval_path": eval_path,
    }
    with deployments_log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

    logging.info(
        "Updated registry for %s to version %s (model_dir=%s).",
        ex,
        args.version,
        model_dir,
    )
    if eval_path:
        logging.info("Linked evaluation report: %s", eval_path)


if __name__ == "__main__":
    main()


