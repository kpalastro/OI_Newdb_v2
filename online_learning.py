"""
Online learning service for OI Gemini.

Provides REST-facing helpers to capture feedback labels, maintain rolling
accuracy statistics, and raise degradation alerts that monitoring and ops
teams can act upon.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


STATE_DIR = Path("reports")
STATE_FILE = STATE_DIR / "online_learning_state.json"
RETRAIN_QUEUE_FILE = STATE_DIR / "retrain_queue.json"


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logging.warning("Failed to parse %s; ignoring.", path)
        return None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class FeedbackSummary:
    exchange: str
    signal_id: str
    rolling_accuracy: float
    feedback_count: int
    window_size: int
    degrade_triggered: bool
    last_feedback_at: str
    accuracy_history: list[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "exchange": self.exchange,
            "signal_id": self.signal_id,
            "rolling_accuracy": self.rolling_accuracy,
            "feedback_count": self.feedback_count,
            "window_size": self.window_size,
            "degrade_triggered": self.degrade_triggered,
            "last_feedback_at": self.last_feedback_at,
            "accuracy_history": self.accuracy_history,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


class OnlineLearningService:
    """Centralised helper to persist feedback stats and queue retrain alerts."""

    def __init__(self) -> None:
        self.state: Dict[str, Any] = _load_json(STATE_FILE) or {}

    def update_exchange(self, summary: FeedbackSummary) -> None:
        serialized = summary.to_dict()
        self.state[summary.exchange] = serialized
        _write_json(STATE_FILE, self.state)

        if summary.degrade_triggered:
            self._enqueue_retrain(summary.exchange, serialized)

    def _enqueue_retrain(self, exchange: str, metadata: Dict[str, Any]) -> None:
        queue = _load_json(RETRAIN_QUEUE_FILE) or {"triggers": []}
        queue["triggers"].append(
            {
                "exchange": exchange,
                "triggered_at": metadata.get("last_feedback_at", datetime.utcnow().isoformat()),
                "rolling_accuracy": metadata.get("rolling_accuracy"),
                "window_size": metadata.get("window_size"),
                "signal_id": metadata.get("signal_id"),
            }
        )
        _write_json(RETRAIN_QUEUE_FILE, queue)
        logging.warning("Retrain queued for %s due to rolling accuracy %.2f%%",
                        exchange, metadata.get("rolling_accuracy", 0) * 100)

    def get_state(self) -> Dict[str, Any]:
        return dict(self.state)

