"""
train_orchestrator.py

Walk-forward AutoML orchestrator that wraps the existing training pipeline.
Implements the roadmap requirement for multi-model evaluation (LightGBM,
XGBoost, CatBoost), Optuna tuning per segment, and consolidated reporting.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

import database_new as db
from feature_engineering import FeatureEngineeringError, REQUIRED_FEATURE_COLUMNS, prepare_training_features
from train_model import RegimeHMMTransformer, define_triple_barrier_target
from time_utils import today_ist, now_ist

try:  # Optional dependencies with graceful degradation
    import optuna  # type: ignore
except ImportError:
    optuna = None  # type: ignore

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None  # type: ignore

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None  # type: ignore


LOGGER = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    exchange: str
    days: int = 120
    window_days: int = 45
    step_days: int = 15
    families: Sequence[str] = field(default_factory=lambda: ("lightgbm", "xgboost", "catboost"))
    optuna_trials: int = 10
    output: Optional[Path] = None


@dataclass
class SegmentWindow:
    segment_id: int
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime


@dataclass
class SegmentResult:
    segment: SegmentWindow
    family: str
    metrics: Dict[str, float]
    best_params: Dict[str, Any]
    optuna_score: Optional[float]
    sample_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "segment_id": self.segment.segment_id,
            "family": self.family,
            "train_range": {
                "start": self.segment.train_start.isoformat(),
                "end": self.segment.train_end.isoformat(),
            },
            "validation_range": {
                "start": self.segment.val_start.isoformat(),
                "end": self.segment.val_end.isoformat(),
            },
            "metrics": self.metrics,
            "best_params": self.best_params,
            "optuna_score": self.optuna_score,
            "sample_counts": self.sample_counts,
        }
        return payload


class ModelFamily:
    name: str = "base"
    pretty_name: str = "Base"

    @property
    def available(self) -> bool:
        return True

    def default_params(self) -> Dict[str, Any]:
        raise NotImplementedError

    def build_model(self, params: Dict[str, Any]):
        raise NotImplementedError

    def optuna_space(self, trial: "optuna.trial.Trial") -> Dict[str, Any]:
        return {}


class LightGBMFamily(ModelFamily):
    name = "lightgbm"
    pretty_name = "LightGBM"

    @property
    def available(self) -> bool:
        return lgb is not None

    def default_params(self) -> Dict[str, Any]:
        return {
            "objective": "multiclass",
            "num_class": 3,
            "n_estimators": 500,
            "learning_rate": 0.03,
            "num_leaves": 48,
            "max_depth": -1,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        }

    def build_model(self, params: Dict[str, Any]):
        if lgb is None:
            raise RuntimeError("LightGBM not installed.")
        return lgb.LGBMClassifier(**params)

    def optuna_space(self, trial: "optuna.trial.Trial") -> Dict[str, Any]:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08),
            "num_leaves": trial.suggest_int("num_leaves", 24, 96, step=8),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        }


class XGBoostFamily(ModelFamily):
    name = "xgboost"
    pretty_name = "XGBoost"

    @property
    def available(self) -> bool:
        return XGBClassifier is not None

    def default_params(self) -> Dict[str, Any]:
        return {
            "objective": "multi:softprob",
            "num_class": 3,
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": 42,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
        }

    def build_model(self, params: Dict[str, Any]):
        if XGBClassifier is None:
            raise RuntimeError("XGBoost not installed.")
        params = params.copy()
        params["use_label_encoder"] = False
        return XGBClassifier(**params)

    def optuna_space(self, trial: "optuna.trial.Trial") -> Dict[str, Any]:
        return {
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
        }


class CatBoostFamily(ModelFamily):
    name = "catboost"
    pretty_name = "CatBoost"

    @property
    def available(self) -> bool:
        return CatBoostClassifier is not None

    def default_params(self) -> Dict[str, Any]:
        return {
            "loss_function": "MultiClass",
            "iterations": 600,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "random_seed": 42,
            "verbose": False,
            "allow_writing_files": False,
        }

    def build_model(self, params: Dict[str, Any]):
        if CatBoostClassifier is None:
            raise RuntimeError("CatBoost not installed.")
        return CatBoostClassifier(**params)

    def optuna_space(self, trial: "optuna.trial.Trial") -> Dict[str, Any]:
        return {
            "depth": trial.suggest_int("depth", 4, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 5.0),
            "iterations": trial.suggest_int("iterations", 300, 900, step=100),
        }


FAMILY_REGISTRY: Dict[str, ModelFamily] = {
    "lightgbm": LightGBMFamily(),
    "xgboost": XGBoostFamily(),
    "catboost": CatBoostFamily(),
}


def _select_families(names: Sequence[str]) -> List[ModelFamily]:
    selected: List[ModelFamily] = []
    for name in names:
        family = FAMILY_REGISTRY.get(name.lower())
        if family is None:
            LOGGER.warning("Unknown model family '%s' – skipping.", name)
            continue
        if not family.available:
            LOGGER.warning("Model family '%s' unavailable (missing dependency).", family.pretty_name)
            continue
        selected.append(family)
    return selected


def _load_dataset(exchange: str, days: int) -> pd.DataFrame:
    """
    Load and prepare dataset WITHOUT applying regime features.
    Regime features will be fitted per segment inside the walk-forward loop
    to prevent look-ahead bias.
    """
    end_date = today_ist()
    start_date = end_date - timedelta(days=days)
    raw = db.load_historical_data_for_ml(exchange, start_date, end_date)
    if raw is None or raw.empty:
        raise RuntimeError(f"No data found for {exchange} in the last {days} days.")

    features = prepare_training_features(raw, required_columns=REQUIRED_FEATURE_COLUMNS)
    target_frame = define_triple_barrier_target(features)
    if target_frame.empty:
        raise RuntimeError("Target preparation yielded no rows.")

    # Do NOT add regime features here - they will be fitted per segment
    return target_frame


def _generate_segments(index: pd.DatetimeIndex, window_days: int, step_days: int) -> List[SegmentWindow]:
    segments: List[SegmentWindow] = []
    if index.empty:
        return segments

    window = pd.Timedelta(days=window_days)
    step = pd.Timedelta(days=step_days)
    cursor = index.min()
    end_limit = index.max()
    segment_id = 1

    while cursor + window + step <= end_limit:
        train_start = cursor
        train_end = cursor + window
        val_start = train_end
        val_end = train_end + step

        segments.append(
            SegmentWindow(
                segment_id=segment_id,
                train_start=train_start.to_pydatetime(),
                train_end=train_end.to_pydatetime(),
                val_start=val_start.to_pydatetime(),
                val_end=val_end.to_pydatetime(),
            )
        )
        segment_id += 1
        cursor += step
    return segments


def _slice_frame(frame: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    mask = (frame.index >= start) & (frame.index < end)
    return frame.loc[mask].copy()


def _encode_labels(y: np.ndarray) -> np.ndarray:
    """
    Encode labels from [-1, 0, 1] to [0, 1, 2] for XGBoost/CatBoost compatibility.
    LightGBM can handle [-1, 0, 1] directly, but XGBoost requires labels starting from 0.
    """
    y_encoded = y.copy()
    y_encoded[y == -1] = 0
    y_encoded[y == 0] = 1
    y_encoded[y == 1] = 2
    return y_encoded.astype(int)


def _decode_labels(y: np.ndarray) -> np.ndarray:
    """
    Decode labels from [0, 1, 2] back to [-1, 0, 1] for consistency with original target.
    """
    y_decoded = y.copy()
    y_decoded[y == 0] = -1
    y_decoded[y == 1] = 0
    y_decoded[y == 2] = 1
    return y_decoded.astype(int)


def _prepare_xy(frame: pd.DataFrame, features: Sequence[str], encode_labels: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    feature_cols = [col for col in features if col in frame.columns]
    X = frame[feature_cols].values.astype(np.float32)
    y = frame['target'].values.astype(int)
    if encode_labels:
        y = _encode_labels(y)
    return X, y


def _run_optuna(
    family: ModelFamily,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    base_params: Dict[str, Any],
    trials: int,
) -> Tuple[Dict[str, Any], Optional[float]]:
    if trials <= 0 or optuna is None:
        return base_params, None

    # XGBoost and CatBoost require labels starting from 0
    needs_encoding = family.name in ("xgboost", "catboost")
    y_train_encoded = _encode_labels(y_train) if needs_encoding else y_train
    y_val_encoded = _encode_labels(y_val) if needs_encoding else y_val

    def objective(trial: "optuna.trial.Trial") -> float:
        params = base_params.copy()
        params.update(family.optuna_space(trial))
        model = family.build_model(params)
        model.fit(X_train, y_train_encoded)
        preds = model.predict(X_val)
        # Decode predictions if we encoded labels
        if needs_encoding:
            preds = _decode_labels(preds)
        score = f1_score(y_val, preds, average='macro')
        return float(score)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    tuned_params = base_params.copy()
    tuned_params.update(study.best_params)
    return tuned_params, float(study.best_value)


def run_orchestrator(config: OrchestratorConfig) -> Dict[str, Any]:
    frame = _load_dataset(config.exchange, config.days)
    frame.sort_index(inplace=True)
    segments = _generate_segments(frame.index, config.window_days, config.step_days)

    if not segments:
        raise RuntimeError("Walk-forward segmentation produced zero windows. Increase lookback or adjust window/step.")

    families = _select_families(config.families)
    if not families:
        raise RuntimeError("No model families available. Install LightGBM/XGBoost/CatBoost or adjust flags.")

    feature_cols = [col for col in REQUIRED_FEATURE_COLUMNS if col in frame.columns]
    results: List[SegmentResult] = []

    for segment in segments:
        # 1. Slice Raw Data first (without regime features)
        train_raw = _slice_frame(frame, segment.train_start, segment.train_end)
        val_raw = _slice_frame(frame, segment.val_start, segment.val_end)

        if len(train_raw) < 200 or len(val_raw) < 50:
            LOGGER.warning(
                "Segment %s skipped due to insufficient samples (train=%s, val=%s).",
                segment.segment_id, len(train_raw), len(val_raw)
            )
            continue

        # 2. Fit HMM on Train Raw ONLY (Prevents Leakage)
        hmm_transformer = RegimeHMMTransformer(n_components=4)
        hmm_transformer.fit(train_raw)
        
        # 3. Generate Regimes
        train_regimes = hmm_transformer.transform(train_raw).flatten()
        val_regimes = hmm_transformer.transform(val_raw).flatten()
        
        # 4. Assign regimes to dataframes
        train_df = train_raw.copy()
        val_df = val_raw.copy()
        train_df['regime'] = train_regimes
        val_df['regime'] = val_regimes
        
        # 5. Add regime as a feature for training
        # Include 'regime' in features for this segment
        feature_cols_with_regime = list(feature_cols) + (['regime'] if 'regime' not in feature_cols else [])
        
        X_train, y_train = _prepare_xy(train_df, feature_cols_with_regime)
        X_val, y_val = _prepare_xy(val_df, feature_cols_with_regime)

        for family in families:
            # XGBoost and CatBoost require labels starting from 0
            needs_encoding = family.name in ("xgboost", "catboost")
            y_train_encoded = _encode_labels(y_train) if needs_encoding else y_train
            
            base_params = family.default_params()
            tuned_params, optuna_score = _run_optuna(family, X_train, y_train, X_val, y_val, base_params, config.optuna_trials)

            model = family.build_model(tuned_params)
            model.fit(X_train, y_train_encoded)
            preds = model.predict(X_val)
            
            # Decode predictions if we encoded labels
            if needs_encoding:
                preds = _decode_labels(preds)

            report = classification_report(y_val, preds, zero_division=0, output_dict=True)
            metrics = {
                "accuracy": float(report.get("accuracy", 0.0)),
                "f1_macro": float(report.get("macro avg", {}).get("f1-score", 0.0)),
                "f1_weighted": float(report.get("weighted avg", {}).get("f1-score", 0.0)),
                "precision_macro": float(report.get("macro avg", {}).get("precision", 0.0)),
                "recall_macro": float(report.get("macro avg", {}).get("recall", 0.0)),
            }

            sample_counts = {"train": len(train_df), "validation": len(val_df)}
            result = SegmentResult(segment, family.pretty_name, metrics, tuned_params, optuna_score, sample_counts)
            results.append(result)
            LOGGER.info(
                "Segment %s | %s | Acc %.3f | F1 %.3f",
                segment.segment_id, family.pretty_name, metrics["accuracy"], metrics["f1_macro"]
            )

    if not results:
        raise RuntimeError("No successful segments were evaluated.")

    best_by_family: Dict[str, Dict[str, Any]] = {}
    for family in families:
        family_results = [res for res in results if res.family == family.pretty_name]
        if not family_results:
            continue
        best_segment = max(family_results, key=lambda r: r.metrics.get("f1_macro", 0.0))
        best_by_family[family.pretty_name] = {
            "segment_id": best_segment.segment.segment_id,
            "metrics": best_segment.metrics,
            "params": best_segment.best_params,
            "optuna_score": best_segment.optuna_score,
        }

    summary = {
        "exchange": config.exchange,
        "generated_at": now_ist().isoformat(),
        "dataset": {
            "rows": int(len(frame)),
            "start": frame.index.min().isoformat(),
            "end": frame.index.max().isoformat(),
            "feature_count": len(feature_cols),
        },
        "config": {
            "days": config.days,
            "window_days": config.window_days,
            "step_days": config.step_days,
            "families": [family.pretty_name for family in families],
            "optuna_trials": config.optuna_trials if optuna is not None else 0,
        },
        "segments_evaluated": len(results),
        "segments": [res.to_dict() for res in results],
        "best_by_family": best_by_family,
    }

    report_path = config.output
    if report_path is None:
        model_dir = Path("models") / config.exchange / "reports"
        model_dir.mkdir(parents=True, exist_ok=True)
        report_path = model_dir / "auto_ml_summary.json"
    else:
        report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    LOGGER.info("✓ AutoML summary saved to %s", report_path)

    return summary


def parse_args() -> OrchestratorConfig:
    parser = argparse.ArgumentParser(description="Walk-forward AutoML orchestrator for OI Gemini.")
    parser.add_argument("--exchange", required=True, choices=["NSE", "BSE"])
    parser.add_argument("--days", type=int, default=150, help="Total lookback window in days.")
    parser.add_argument("--window-days", type=int, default=45, help="Training window size for each segment.")
    parser.add_argument("--step-days", type=int, default=15, help="Step size / validation horizon in days.")
    parser.add_argument("--families", nargs="+", default=["lightgbm", "xgboost", "catboost"],
                        help="Model families to evaluate.")
    parser.add_argument("--optuna-trials", type=int, default=10, help="Trials per segment (0 to skip).")
    parser.add_argument("--output", type=Path, default=None, help="Optional override path for the JSON summary.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")
    return OrchestratorConfig(
        exchange=args.exchange,
        days=args.days,
        window_days=args.window_days,
        step_days=args.step_days,
        families=args.families,
        optuna_trials=args.optuna_trials,
        output=args.output,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_orchestrator(cfg)

