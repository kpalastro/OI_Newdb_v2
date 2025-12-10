## OI_Gemini · Phase 2 Final Recommendations & Roadmap

This document consolidates all **not-yet-implemented (or only partially implemented)** items from the Phase 2 blueprints and adds further improvements based on the current OI_Gemini codebase. It is intended as a **practical implementation roadmap**.

---

## 1. Adaptation Loop & MLOps (phase2_adaptation_loop.md)

### 1.1 Daily Retraining & Dataset Versioning

**Current state**
- Models are trained via `train_model.py` and `train_orchestrator.py` using time-window parameters.
- Training data is pulled directly from `ml_features` using `load_historical_data_for_ml`.
- No explicit *dataset versioning* (e.g., `2025-11-08`) or registry beyond `training_batches`.

**Gaps**
1. No dedicated **daily retrain job** that:
   - Consolidates latest `ml_features`/labels at market close.
   - Creates a versioned dataset id (e.g., `2025-11-28`).
2. No dataset registry as described in `phase2_data_foundation.md` (version, paths, record counts, stats).
3. No automated **orchestration script** for end-to-end:
   - Data consolidation → train → evaluate → (maybe) promote.

**Recommended actions**
- Add a lightweight orchestration script, e.g. `scripts/daily_retrain.py`, that:
  - Chooses a rolling lookback window (e.g., last 6–12 months).
  - Calls `export_training_window(...)` for each exchange and registers the batch in `training_batches`.
  - Stores a dataset version tag (e.g., `YYYY-MM-DD`) and reconciles with `training_batches`.
  - Invokes `train_model.py` and optionally `train_orchestrator.py` with consistent parameters.
- Extend `training_batches` or add a small **dataset registry** abstraction on top of it:
  - Link exported CSV/Parquet paths to a version string.
  - Store summary statistics (row count, date range, feature completeness).

### 1.2 Evaluation Gate & Promotion Criteria

**Current state**
- `train_model.py` and `train_orchestrator.py` compute metrics (precision, F1, classification reports).
- Backtesting engine computes trading metrics.
- There is no single “gate” that compares candidate models vs baseline using strict rules.

**Gaps**
1. No codified **acceptance criteria** (e.g., Precision@TopK, calibration error, max drawdown) that must be met before promotion.
2. No process to **compare candidate vs current production model** using standardized backtest windows.

**Recommended actions**
- Implement a small **evaluation module**, e.g. `scripts/evaluate_candidate.py`, that:
  - Loads candidate model metrics from:
    - `models/<EXCHANGE>/reports/training_report.json`.
    - `models/<EXCHANGE>/reports/auto_ml_summary.json`.
    - Backtest JSONs in `reports/backtests` or `models/<EXCHANGE>/reports/backtest_summary.json`.
  - Loads baseline (current production) metrics from archived reports.
  - Applies clearly defined thresholds, such as:
    - Out-of-sample Sharpe ≥ baseline + δ.
    - Max drawdown ≤ baseline + tolerance.
    - Hit rate or F1 ≥ baseline + δ.
  - Writes an **evaluation report** (JSON/markdown) indicating PASS/FAIL per criterion.

### 1.3 Deployment & Model Registry

**Current state**
- Runtime uses the artifacts found under `models/<EXCHANGE>/` without an explicit model version registry.
- No script equivalent to `ml/deploy.py` that atomically updates a registry pointer.

**Gaps**
1. No **registry file** (e.g. `models/registry.yml`) mapping logical “current” models to artifact paths.
2. No **deployment script** that:
   - Copies candidate artifacts into versioned paths.
   - Updates registry atomically.
   - Logs deployments (timestamp, operator, metrics).

**Recommended actions**
- Introduce a simple `models/registry.yml`:
  - For now, one entry per exchange/horizon (even if only one horizon).
  - Track:
    - `model_path`, `selector_path`, `hmm_path`, `version_tag`, and any thresholds.
- Add a `scripts/deploy_model.py` that:
  - Accepts an exchange and a candidate “run id” or path.
  - Copies artifacts to a versioned folder (`models/<EXCHANGE>/<version>/`).
  - Updates `registry.yml` to point “current” to the new version.
  - Appends a line to `models/deployments.log` with metrics and operator name.

### 1.4 Post-Deployment Monitoring & Drift

**Current state**
- Phase 2 metrics collector tracks:
  - Data quality, model performance, system health, and paper trading.
- No explicit PSI/KL drift metrics or latency monitoring.

**Gaps**
1. No explicit **drift detection** comparing training vs live feature distributions.
2. Limited real-time **latency and throughput** metrics for ML inference.
3. No external alert channels (Slack/email) on metric breaches.

**Recommended actions**
- Extend `Phase2MetricsCollector` or a sibling module to:
  - Periodically compute simple drift scores for key features (e.g., PSI or rolling z-score away from training mean).
  - Track inference latency histograms and queue sizes in a lightweight way (even just to JSONL/CSV initially).
- Add an **alerting script** (can run on a schedule) that:
  - Reads the Phase 2 summary.
  - If thresholds are breached (drift, low win rate, high error rate), writes to:
    - A configurable webhook URL (Slack/email) or at least a local alert log for now.

---

## 2. Data Foundation & Versioned Datasets (phase2_data_foundation.md)

### 2.1 Stream Fan-Out & Parquet Micro-Batching

**Current state**
- Live features are engineered inline and stored into `ml_features`.
- There is no separate append-only Parquet stream store (`raw_stream/…`).

**Gaps**
1. Absence of:
   - **Stream tap** that captures raw per-tick/per-5s option leg state into an external bus or micro-batch.
   - **Micro-batch writer** that flushes to `data/raw_stream/{date}/…` in Parquet.

**Recommended actions**
- Implement a minimal **micro-batching pipeline** inside `oi_tracker_kimi_new.py`:
  - After each UI refresh (where data reels are updated), push a normalized snapshot event into an internal queue.
  - A background worker writes batches every N seconds to Parquet, partitioned by `date/exchange`.
- Keep this **optional and non-blocking**:
  - Guard with a config flag to avoid performance issues.
  - Start small (only key fields) and expand later if needed.

### 2.2 Dataset Manifests & Registry

**Current state**
- `export_training_window()` already writes CSV/Parquet and records metadata in `training_batches`.
- There is no manifest file per “version date” as described.

**Gaps**
1. No `dataset.yml` manifest with:
   - Version, exchange list, paths, hash, record count.
2. No explicit `dataset_registry` table; the functionality is partially covered by `training_batches`, but not 1:1.

**Recommended actions**
- On each export window or daily EOD dataset creation:
  - Generate a simple YAML/JSON manifest in `exports/training_batches/` that:
    - Mirrors the concept from the blueprint (version, features_path, labels_path, records).
  - Either:
    - Extend `training_batches.metadata` to include manifest info, or
    - Add a `dataset_registry` table if you want stronger separation of concerns.

### 2.3 Outcome Labels & EOD Labeler

**Current state**
- Labeling is handled via `define_triple_barrier_target` within `train_model.py` using `underlying_price`.
- No generic EOD labeler script that generates multiple horizons or strategy-specific labels as Parquet.

**Gaps**
1. No multi-horizon label pipeline for ΔOI, ΔLTP, or strategy PnL as described.
2. No separate `labels.parquet` for research alongside features.

**Recommended actions**
- Create a **labeling utility** (e.g., `scripts/generate_labels.py`) that:
  - For given `exchange` and date range:
    - Reads `ml_features`.
    - Computes multiple horizons (3/5/15/30 min) forward returns and OI changes.
    - Writes a labels file with IDs linking back to features (e.g., `uid`).
  - Use this as a flexible research tool beyond the triple-barrier target used for production training.

---

## 3. Inference Engine Architecture (phase2_inference_engine.md)

### 3.1 Feature & Model Service Separation

**Current state**
- The Flask app (`oi_tracker_kimi_new.py`) handles:
  - Data ingestion.
  - Feature engineering (`engineer_live_feature_set`).
  - ML model inference (`MLSignalGenerator`).
  - Web UI and Socket.IO.

**Gaps**
1. No dedicated **feature service** worker that takes events and returns normalized vectors via an explicit interface.
2. No standalone **model inference service** (e.g., FastAPI/Falcon/Flask microservice) for scoring.

**Recommended actions**
- Introduce an **internal abstraction** first, without network overhead:
  - A module `ml_inference.py` exposing a clear interface:
    - `prepare_features(handler_snapshot) -> feature_vector`
    - `score_features(feature_vector) -> signal, confidence, metadata`
  - Optionally refactor existing code to call these functions even if they remain in-process.
- Later, if needed, split out to a separate process or service with REST/WebSocket, reusing the same interface.

### 3.2 Model Registry & Versioning

**Current state**
- Artifacts live in `models/<EXCHANGE>/` and are loaded by `MLSignalGenerator`.
- No `registry.yml` mapping versions/horizons to specific paths.

**Gaps**
1. Inability to quickly roll back to a previous version via a registry pointer.
2. No human-readable model version metadata for operators.

**Recommended actions**
- Implement the simple registry proposed in **Section 1.3** and update `MLSignalGenerator` to:
  - Read from the registry.
  - Log the `model_version` in metadata for each signal.

### 3.3 Recommendation Bus & Logging

**Current state**
- Signals are generated and used internally (for banners and auto-execution).
- Trade logs exist in `trade_logs/`, but no structured **recommendation log** as per the blueprint.

**Gaps**
1. No central `recommendations_log` or `recommendation_audit` table.
2. No internal bus abstraction for broadcasting recommendations to multiple consumers (UI, alerts, research).

**Recommended actions**
- Define a minimal **recommendation schema**:
  - `timestamp, exchange, symbol, action, confidence, model_version, regime, kelly_fraction, features_summary`.
- Implement:
  - An in-process “bus” (e.g., a simple `queue.Queue` or de-duplicated logging function) that:
    - Writes JSONL rows to `logs/recommendations/{date}.jsonl`.
    - Optionally inserts into a small `recommendation_audit` table in the DB for structured queries.

---

## 4. Research Strategy & Advanced Models (phase2_research_strategy.md)

### 4.1 Additional Model Families & Explainability

**Current state**
- LightGBM is the main production model (with regime awareness).
- AutoML can explore XGBoost, LightGBM, CatBoost.
- No SHAP-based explainability integrated.

**Gaps**
1. Sequence models (LSTM/TCN), Bayesian models, or stacking ensembles are not implemented.
2. No systematic SHAP/feature-importance export for training runs.

**Recommended actions**
- In the **research-only context** (not immediately in production):
  - Add notebooks or scripts experimenting with:
    - Simple LSTM or TCN on the same features, reusing training windows.
    - SHAP values for LightGBM models:
      - Compute and save summary plots and per-feature importance into `models/<EXCHANGE>/reports/`.
  - Consider a distilled set of features for explainability and risk sign-off (e.g., top N drivers per regime).

### 4.2 Calibration & Metric Expansion

**Current state**
- Training uses F1, precision, and classification reports.
- Backtests use trading metrics.

**Gaps**
1. No explicit **calibration diagnostics** (e.g., reliability diagrams, Brier score).
2. No multi-target models for multiple horizons or strategy PnL targets.

**Recommended actions**
- During evaluation:
  - Compute and plot calibration curves for predicted probabilities.
  - Track Brier score and log-loss per model and per regime.
- Explore:
  - Multi-task models or multi-output predictions (e.g., sign + magnitude).

---

## 5. Risk Governance & Explainability (phase2_risk_governance.md)

### 5.1 Feature Attributions & Rationale in UI

**Current state**
- `MLSignalGenerator` returns:
  - `signal`, `confidence`, `regime`, and rich metadata (Kelly, lots, history).
- UI strings support ML labels (signal, confidence, regime, Kelly, lots).
- No SHAP or feature-level attribution.

**Gaps**
1. No feature attribution logic (e.g., SHAP).
2. No “drivers” panel in UI explaining top contributing features.

**Recommended actions**
- Start with a **simple feature ranking**:
  - Use LightGBM’s built-in feature importance for each model.
  - During inference, show only a small set of features (e.g., PCR, VIX, breadth, OI changes) and their z-scores as rough “drivers”.
- In a second phase:
  - Add an offline SHAP pipeline that:
    - Computes SHAP values on representative samples.
    - Stores global or per-regime summaries for the UI.

### 5.2 Audit Logs & Compliance

**Current state**
- Trade logs and Phase 2 metrics exist.
- No dedicated recommendation audit as JSONL/DB.

**Gaps**
1. No structured “per recommendation” log with model version and feature context.
2. No clear mapping between recommendation → trade → realized outcome in a single table.

**Recommended actions**
- As part of the **recommendation bus** initiative:
  - Log:
    - `id`, `timestamp`, `exchange`, `action`, `confidence`, `model_version`, `regime`, risk flags.
  - For each executed paper trade:
    - Store the originating recommendation id.
  - Later, you can link this to realized PnL and create a “recommendation outcomes” table similar to the blueprint.

### 5.3 External Monitoring & Alerts

**Current state**
- Internal dashboards via Flask `/monitoring` and JSON APIs.
- No integration with external tools (Grafana, Prometheus, Slack).

**Gaps**
1. Lack of external alerting and dashboards.

**Recommended actions**
- As an incremental step:
  - Export key Phase 2 summaries periodically to CSV/JSON for external BI tools (Metabase/Superset).
  - Add a simple script or cron that checks for:
    - Excessive drawdown, low win rate, high error rate.
    - And sends an email or webhook to a configured endpoint.

---

## 6. Additional Codebase Improvements (Own Analysis)

Beyond the Phase 2 documents, the following improvements would increase robustness and maintainability:

### 6.1 Configuration Consolidation & Security

- **Issue**
  - Some exchange configs are duplicated between `config.py` and `oi_tracker_kimi_new.py`.
  - DB credentials and other secrets should consistently come from environment variables.

- **Recommendation**
  - Gradually phase out hard-coded `EXCHANGE_CONFIGS` in `oi_tracker_kimi_new.py` and rely on `AppConfig.exchange_configs`.
  - Ensure all secrets (DB password, API keys) are only loaded from environment variables or `.env` with a clear warning if missing.

### 6.2 Testing & CI

- **Issue**
  - There is no visible automated test suite for:
    - Feature engineering correctness.
    - DB migration safety.
    - Backtest engine.

- **Recommendation**
  - Add unit tests for:
    - `engineer_live_feature_set` and `prepare_training_features` on small fixtures.
    - `define_triple_barrier_target` and `BacktestEngine`.
  - Add a basic CI workflow (even locally) to run tests before deployment.

### 6.3 Documentation Alignment

- **Issue**
  - Some Phase 2 documents still describe future architectures not completely aligned with the current code.

- **Recommendation**
  - Maintain a short “status” section at the top of each Phase 2 doc indicating:
    - What is implemented.
    - What remains conceptual.
  - Keep `GPT_Final_reco.md` as the **source of truth** for implementation priorities.

---

## 7. Suggested Implementation Order

To reduce risk and deliver value incrementally:

1. **Finalize Risk & Monitoring Foundation**
   - Harden `risk_manager` and auto-execution constraints.
   - Ensure `/monitoring` and Phase 2 metrics are stable and well-documented.
2. **Introduce Dataset Versioning & Evaluation Gate**
   - Use existing `training_batches` to anchor dataset versions.
   - Add evaluation scripts and clear promotion criteria.
3. **Add Simple Model Registry & Deployment Script**
   - Implement `registry.yml`, `deploy_model.py`, and `deployments.log`.
4. **Recommendation Logging & Audit**
   - Implement a unified recommendation log and link it to trades and outcomes.
5. **Micro-Batch Data Foundation (Optional but Valuable)**
   - Add Parquet micro-batches for deep research & offline experiments.
6. **Explainability & External Alerts**
   - Add basic feature-importance reporting and minimal alerting.
7. **Longer-Term: Separate Inference Service & Advanced Models**
   - Only after the above foundations are solid and operational processes are in place.

This roadmap balances **practical trading value**, **operational safety**, and **alignment with the Phase 2 vision** without over-engineering the system prematurely.


