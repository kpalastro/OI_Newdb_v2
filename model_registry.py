"""
Model registry for OI Gemini ML system.
Stores model versions in PostgreSQL (model_versions table) and supports
registration, promotion, rollback, and comparison.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psycopg2
    from psycopg2.extras import DictCursor, Json
    PSYCOPG_AVAILABLE = True
except ImportError:
    PSYCOPG_AVAILABLE = False

import database_new as db

LOG = logging.getLogger(__name__)


def _version_string(exchange: str, model_type: str) -> str:
    """Generate a unique version string for (exchange, model_type)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"v{exchange}_{model_type}_{ts}"


def _compute_file_hash(path: Path) -> Optional[str]:
    """Compute SHA256 hash of file contents for model_path."""
    if not path or not Path(path).exists():
        return None
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:32]
    except Exception:
        return None


class ModelRegistry:
    """Registry for ML model versions using database_new and model_versions table."""

    def __init__(self) -> None:
        if not PSYCOPG_AVAILABLE:
            raise ImportError("psycopg2 is required for model registry. Install: pip install psycopg2-binary")
        self._db = db

    def _conn(self):
        return self._db.get_db_connection()

    def register_model(
        self,
        exchange: str,
        model_type: str,
        model_path: Any,
        validation_metrics: Dict[str, float],
        training_data_start: date,
        training_data_end: date,
        training_samples: int,
        cv_metrics: Optional[List[Dict]] = None,
        artifact_paths: Optional[List[str]] = None,
        notes: Optional[str] = None,
        hyperparameters: Optional[Dict] = None,
    ) -> int:
        """
        Register a new model version. Returns the new row id.
        """
        path = Path(model_path) if model_path else None
        path_str = str(path) if path else ""
        version = _version_string(exchange, model_type)
        model_hash = _compute_file_hash(path) if path else None
        training_date = datetime.now()

        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO model_versions (
                    version, exchange, model_type, model_path, model_hash,
                    training_date, training_data_start_date, training_data_end_date,
                    training_samples, validation_metrics, cv_metrics, artifact_paths,
                    hyperparameters, status, is_production, notes
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'candidate', FALSE, %s
                )
                RETURNING id
                """,
                (
                    version,
                    exchange,
                    model_type,
                    path_str,
                    model_hash,
                    training_date,
                    training_data_start,
                    training_data_end,
                    training_samples,
                    Json(validation_metrics) if validation_metrics else None,
                    Json(cv_metrics) if cv_metrics else None,
                    Json(artifact_paths) if artifact_paths else None,
                    Json(hyperparameters) if hyperparameters else None,
                    notes,
                ),
            )
            row = cur.fetchone()
            conn.commit()
            return row["id"] if hasattr(row, "keys") else row[0]
        except Exception as e:
            conn.rollback()
            LOG.exception("register_model failed: %s", e)
            raise
        finally:
            self._db.release_db_connection(conn)

    def list_models(
        self,
        exchange: Optional[str] = None,
        model_type: Optional[str] = None,
        status: Optional[str] = None,
        is_production: Optional[bool] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """List model versions with optional filters."""
        conn = self._conn()
        try:
            cur = conn.cursor(cursor_factory=DictCursor)
            q = "SELECT * FROM model_versions WHERE 1=1"
            params: List[Any] = []
            if exchange:
                q += " AND exchange = %s"
                params.append(exchange)
            if model_type:
                q += " AND model_type = %s"
                params.append(model_type)
            if status:
                q += " AND status = %s"
                params.append(status)
            if is_production is not None:
                q += " AND is_production = %s"
                params.append(is_production)
            q += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)
            cur.execute(q, params)
            rows = cur.fetchall()
            return [_row_to_model(r) for r in rows]
        finally:
            self._db.release_db_connection(conn)

    def get_model_version(self, version_id: int) -> Optional[Dict]:
        """Get a single model version by id."""
        conn = self._conn()
        try:
            cur = conn.cursor(cursor_factory=DictCursor)
            cur.execute("SELECT * FROM model_versions WHERE id = %s", (version_id,))
            row = cur.fetchone()
            return _row_to_model(row) if row else None
        finally:
            self._db.release_db_connection(conn)

    def get_production_model(self, exchange: str, model_type: str) -> Optional[Dict]:
        """Get current production model for exchange and model_type."""
        conn = self._conn()
        try:
            cur = conn.cursor(cursor_factory=DictCursor)
            cur.execute(
                "SELECT * FROM model_versions WHERE exchange = %s AND model_type = %s AND is_production = TRUE LIMIT 1",
                (exchange, model_type),
            )
            row = cur.fetchone()
            return _row_to_model(row) if row else None
        finally:
            self._db.release_db_connection(conn)

    def promote_to_production(self, version_id: int, reason: Optional[str] = None) -> None:
        """Set this version as production; clear is_production for others of same exchange+model_type."""
        model = self.get_model_version(version_id)
        if not model:
            raise ValueError(f"Model version id {version_id} not found")
        exchange = model["exchange"]
        model_type = model["model_type"]

        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE model_versions SET is_production = FALSE WHERE exchange = %s AND model_type = %s",
                (exchange, model_type),
            )
            cur.execute(
                "UPDATE model_versions SET is_production = TRUE, status = 'active', promoted_at = NOW(), rollback_reason = %s WHERE id = %s",
                (reason, version_id),
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            LOG.exception("promote_to_production failed: %s", e)
            raise
        finally:
            self._db.release_db_connection(conn)

    def compare_models(self, new_id: int, current_id: int) -> Dict:
        """
        Compare two model versions. Returns dict with new_version, current_version,
        improvements (metric -> {new, current, improved, improvement_pct}), should_promote, reason.
        """
        new_m = self.get_model_version(new_id)
        current_m = self.get_model_version(current_id)
        if not new_m:
            raise ValueError(f"New model version id {new_id} not found")
        if not current_m:
            raise ValueError(f"Current model version id {current_id} not found")

        new_metrics = new_m.get("validation_metrics") or {}
        current_metrics = current_m.get("validation_metrics") or {}
        all_keys = sorted(set(new_metrics.keys()) | set(current_metrics.keys()))
        improvements = {}
        for k in all_keys:
            nv = new_metrics.get(k)
            cv = current_metrics.get(k)
            if nv is None and cv is None:
                continue
            nv = float(nv) if nv is not None else 0.0
            cv = float(cv) if cv is not None else 0.0
            delta = nv - cv
            pct = (delta / cv * 100.0) if cv else 0.0
            # Higher is better for f1, precision, recall, accuracy
            higher_better = k.lower() in ("f1_score", "f1", "precision", "recall", "accuracy")
            improved = (delta > 0) if higher_better else (delta < 0)
            improvements[k] = {
                "new": nv,
                "current": cv,
                "improved": improved,
                "improvement_pct": pct,
            }
        should_promote = sum(1 for v in improvements.values() if v["improved"]) > sum(
            1 for v in improvements.values() if not v["improved"]
        ) if improvements else False
        reason = "Metrics improved" if should_promote else "No clear improvement or degradation"
        return {
            "new_version": new_m.get("version", str(new_id)),
            "current_version": current_m.get("version", str(current_id)),
            "improvements": improvements,
            "should_promote": should_promote,
            "reason": reason,
        }

    def rollback_model(
        self,
        exchange: str,
        model_type: str,
        target_version_id: int,
        reason: Optional[str] = None,
    ) -> None:
        """Promote target_version_id to production (rollback from current production)."""
        self.promote_to_production(target_version_id, reason=reason or "Rollback")

    def record_performance(
        self,
        model_version_id: int,
        exchange: str,
        evaluation_date: date,
        signals_generated: Optional[int] = None,
        correct_predictions: Optional[int] = None,
        incorrect_predictions: Optional[int] = None,
        win_rate: Optional[float] = None,
        avg_return: Optional[float] = None,
        sharpe_ratio: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        avg_confidence: Optional[float] = None,
        prediction_distribution: Optional[Dict] = None,
        regime_performance: Optional[Dict] = None,
    ) -> None:
        """Insert or update a row in model_performance_log."""
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO model_performance_log (
                    model_version_id, exchange, evaluation_date,
                    signals_generated, correct_predictions, incorrect_predictions,
                    win_rate, avg_return, sharpe_ratio, max_drawdown, avg_confidence,
                    prediction_distribution, regime_performance
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (model_version_id, evaluation_date)
                DO UPDATE SET
                    signals_generated = EXCLUDED.signals_generated,
                    correct_predictions = EXCLUDED.correct_predictions,
                    incorrect_predictions = EXCLUDED.incorrect_predictions,
                    win_rate = EXCLUDED.win_rate,
                    avg_return = EXCLUDED.avg_return,
                    sharpe_ratio = EXCLUDED.sharpe_ratio,
                    max_drawdown = EXCLUDED.max_drawdown,
                    avg_confidence = EXCLUDED.avg_confidence,
                    prediction_distribution = EXCLUDED.prediction_distribution,
                    regime_performance = EXCLUDED.regime_performance
                """,
                (
                    model_version_id,
                    exchange,
                    evaluation_date,
                    signals_generated,
                    correct_predictions,
                    incorrect_predictions,
                    win_rate,
                    avg_return,
                    sharpe_ratio,
                    max_drawdown,
                    avg_confidence,
                    Json(prediction_distribution) if prediction_distribution else None,
                    Json(regime_performance) if regime_performance else None,
                ),
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            LOG.exception("record_performance failed: %s", e)
            raise
        finally:
            self._db.release_db_connection(conn)


def _row_to_model(row: Any) -> Dict:
    """Convert DB row (dict or Row) to a flat dict with expected keys for CLI/docs."""
    if row is None:
        return {}
    d = dict(row) if hasattr(row, "keys") else {}
    # Normalize keys to match docs (snake_case, and some renames)
    out = {
        "id": d.get("id"),
        "version": d.get("version"),
        "exchange": d.get("exchange"),
        "model_type": d.get("model_type"),
        "model_path": d.get("model_path"),
        "artifact_paths": d.get("artifact_paths"),
        "model_hash": d.get("model_hash"),
        "training_date": d.get("training_date"),
        "training_data_start_date": d.get("training_data_start_date"),
        "training_data_end_date": d.get("training_data_end_date"),
        "training_samples": d.get("training_samples"),
        "hyperparameters": d.get("hyperparameters"),
        "validation_metrics": d.get("validation_metrics"),
        "test_metrics": d.get("test_metrics"),
        "cv_metrics": d.get("cv_metrics"),
        "status": d.get("status"),
        "is_production": d.get("is_production"),
        "promoted_at": d.get("promoted_at"),
        "deprecated_at": d.get("deprecated_at"),
        "previous_version_id": d.get("previous_version_id"),
        "rollback_reason": d.get("rollback_reason"),
        "code_commit_hash": d.get("code_commit_hash"),
        "training_data_hash": d.get("training_data_hash"),
        "created_by": d.get("created_by"),
        "notes": d.get("notes"),
        "created_at": d.get("created_at"),
    }
    return out
