from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from time_utils import now_ist


def log_recommendation(exchange: str, signal: str, confidence: float, metadata: Dict[str, Any]) -> None:
    """
    Append a single recommendation event to logs/recommendations/YYYY-MM-DD.jsonl.

    Best-effort only: failures are silently ignored to avoid impacting trading.
    """
    try:
        ts = now_ist()
        day_slug = ts.strftime("%Y-%m-%d")
        base_dir = Path("logs") / "recommendations"
        base_dir.mkdir(parents=True, exist_ok=True)
        log_path = base_dir / f"{day_slug}.jsonl"

        payload = {
            "timestamp": ts.isoformat(),
            "exchange": exchange,
            "signal": signal,
            "confidence": float(confidence),
            "regime": metadata.get("regime"),
            "kelly_fraction": metadata.get("kelly_fraction"),
            "recommended_lots": metadata.get("recommended_lots"),
            "model_version": metadata.get("model_version"),
            "signal_id": metadata.get("signal_id"),
        }
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
    except Exception:
        # Recommendation logging must never break the main flow
        return


