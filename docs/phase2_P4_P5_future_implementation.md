### Phase 2 · P4 & P5 Future Implementation Plan

This document captures the **future implementation blueprint** for:
- **P4 – Micro-batch data foundation** (raw Parquet stream + dataset catalog), and
- **P5 – Explainability & external alerts** (feature attributions + alerting around Phase 2 metrics),
so they can be implemented later in a structured way.

---

### 1. P4 – Micro‑Batch Data Foundation

#### 1.1 Objectives

- Capture a **richer raw event stream** (per‑tick / per‑5s snapshots) without blocking the live app.
- Store micro‑batched data in **Parquet** partitions for research and offline model experiments.
- Maintain a lightweight **catalog/registry** that ties Parquet batches back to DB snapshots and training runs.

#### 1.2 Target Architecture

High‑level flow:

`Websocket ticks → ExchangeDataHandler → Stream Tap → In‑process Queue → Micro‑batch Writer → Parquet + ml_feature_batches`

- **Stream tap**:
  - Hook after each UI/feature update in `oi_tracker_kimi_new.py`, where we already build `calls`/`puts` and `ml_features_dict`.
  - Construct lightweight records (dicts) with:
    - Core fields: `timestamp`, `exchange`, `symbol`, `strike`, `option_type`, `ltp`, `oi`, `volume`, `iv`, `vix`, `underlying_price`, `atm_strike`, `time_to_expiry_sec`, PCR snapshot, etc.
  - Push records into an in‑process queue (e.g., `collections.deque` or `queue.Queue`).

- **Micro‑batch writer**:
  - Background thread reading from the queue every N seconds (e.g. 60s):
    - Drain current queue into a pandas `DataFrame`.
    - Write to Parquet at `data/raw_stream/<YYYY-MM-DD>/<exchange>/raw_<HHMMSS>.parquet`.
  - Append a summary row into a **batch registry** table each time (see below).

- **Batch registry (DB table)** – to be added later (conceptually similar to the original Phase 2 spec):
  - Table name: `ml_feature_batches`.
  - Recommended schema (Timescale/SQLite):
    ```sql
    CREATE TABLE ml_feature_batches (
        id SERIAL PRIMARY KEY,
        batch_time TIMESTAMP NOT NULL,
        exchange TEXT NOT NULL,
        parquet_path TEXT NOT NULL,
        record_count INTEGER NOT NULL,
        checksum TEXT,
        created_at TIMESTAMP DEFAULT NOW()
    );
    ```
  - Use it to:
    - Track where each raw batch lives.
    - Optionally store a checksum for integrity verification.

#### 1.3 Implementation Steps

**Step 1: Stream tap & queue**
- Add a small, configurable function in `oi_tracker_kimi_new.py`, e.g. `enqueue_raw_stream_event(exchange, calls, puts, handler_proxy, ml_features_dict)`, guarded by a config flag `ENABLE_RAW_STREAM`.
- Extract only the fields needed for research; avoid heavy nested structures.
- Push to a process‑wide queue (size‑limited; when full, drop the oldest record and log a warning).

**Step 2: Micro‑batch writer**
- Start a new background thread (similar to `_system_health_loop`) that:
  - Sleeps for `RAW_STREAM_FLUSH_SECONDS` (e.g. 60).
  - On wake:
    - Pop all current records from the queue into a list.
    - If the list is non‑empty:
      - Create a `DataFrame`.
      - Write/parquet to `data/raw_stream/<date>/<exchange>/...`.
      - Compute `record_count` and an optional checksum (e.g. SHA256 of the file).
      - Insert a row into `ml_feature_batches`.

**Step 3: DB changes for `ml_feature_batches`**
- Extend `initialize_database()` in `database_new.py`:
  - Add a DDL statement for `ml_feature_batches` (Postgres + SQLite variants).
  - Add an index on `(exchange, batch_time)`.
- Extend `migrate_database()` to add it safely to existing installations.

**Step 4: Configuration & safety**
- Add configuration entries in `AppConfig`:
  - `enable_raw_stream_capture: bool`.
  - `raw_stream_flush_seconds: int`.
  - `raw_stream_root_dir: Path` (default `data/raw_stream`).
- Ensure that:
  - All failures during Parquet write or DB insert are caught and logged.
  - The stream tap does **not** block the main OI/ML pipeline.

#### 1.4 Usage Patterns (Once Implemented)

- **Research**:
  - Analysts load `data/raw_stream/<date>/<exchange>/*.parquet` into notebooks for:
    - New feature engineering.
    - Alternate label definitions.
    - Exploratory microstructure analysis.
- **Traceability**:
  - `ml_feature_batches` will allow:
    - Replaying exactly which raw events fed a given training dataset.
    - Auditing data quality issues down to the batch level.

---

### 2. P5 – Explainability & External Alerts

#### 2.1 Objectives

- Provide traders with **transparent explanations** for signals:
  - Which features most influenced a given BUY/SELL recommendation?
  - What regime/context was active?
- Set up **basic alerts** around Phase 2 metrics:
  - Detect and notify on deteriorating performance, degraded data quality, or system health issues.

#### 2.2 Explainability – Feature Attributions

**Scope:**
- Start with **global / per‑regime** feature importance and optionally extend to sample‑level attributions.

**Planned changes:**

1. **Training‑time SHAP/importance export**
   - Extend `train_model.py`:
     - After final LightGBM model is trained:
       - Compute feature importances via `feature_importances_`.
       - Optionally, for SHAP:
         - If `shap` is installed, compute summary values over a sample of training rows.
     - Save outputs under `models/<EXCHANGE>/reports/`:
       - `feature_importance.json` (mapping feature → importance score).
       - `shap_summary.json` or `shap_top_features.json` (top features globally and by regime).

2. **Runtime metadata enrichment**
   - `MLSignalGenerator.generate_signal` already returns `metadata` with regime, confidence, and model_version.
   - Future extension:
     - Add a compact `top_feature_names` list by:
       - Using the precomputed importance rankings to pick, say, top 3–5 relevant features.
       - Logging their values and normalized magnitudes at inference time (this should be kept light).

3. **Monitoring UI integration**
   - Update `templates/monitoring.html` (future):
     - Add a small “Feature Importance” card per exchange.
     - Use `feature_importance.json` to render a bar chart or a simple table of top features.
   - Optional:
     - Create a dedicated tab or section to display SHAP summaries and narratives.

4. **UI narratives (later phase)**
   - Based on top features and their signs, render short textual rationales, e.g.:
     - “3m OI up 10% with VIX flat; model sees bullish continuation.”
   - These can reuse templates outlined in `phase2_risk_governance.md`.

#### 2.3 External Alerting Around Phase 2 Metrics

**Scope:**
- A simple, script‑based alerting layer on top of existing `Phase2MetricsCollector` summaries.

**Planned components:**

1. **Health check script**
   - New script, e.g. `scripts/check_health_and_alert.py`:
     - For each exchange (NSE/BSE):
       - Load `metrics/phase2_metrics/<EXCHANGE>_summary.json`.
       - Evaluate thresholds, for example:
         - Data quality:
           - `macro_availability_pct` < 90%.
           - `database_write_success_rate` < 99%.
         - Model performance:
           - `total_signals` too low or too high.
           - `avg_confidence` very low or spiking.
         - System health:
           - `memory_usage_mb` > configured cap.
           - `error_rate` > 0 for more than N hours.
         - Paper trading:
           - `total_pnl` < −X over the last N hours.
           - `portfolio_constraint_violations` > 0 frequently.
       - For any breached thresholds, build a human‑readable alert message.

2. **Alert destinations**
   - First stage:
     - Write alerts to `logs/alerts.log` with:
       - Timestamp, exchange, severity (INFO/WARN/CRITICAL), message, and metric snapshot.
   - Second stage (optional future work):
     - Add a minimal HTTP webhook client that:
       - Reads `ALERT_WEBHOOK_URL` and `ALERT_WEBHOOK_TOKEN` from environment variables.
       - Posts JSON alerts to Slack/email/Teams gateway.

3. **Scheduling**
   - Run the script via:
     - Windows Task Scheduler (your environment is Windows).
     - Or a lightweight cron‑style service if you move to Linux.
   - Frequency:
     - Every 5–15 minutes during trading hours.

4. **Runbook integration**
   - Extend the operational runbook (already partially done) to include:
     - What each alert means.
     - How to respond:
       - E.g., halt auto‑execution when certain thresholds are reached.
       - Trigger retraining or investigate data pipelines.

#### 2.4 Longer‑Term Enhancements

- Integrate drift detection libraries (e.g., Evidently) that:
  - Compare live feature distributions to training distributions.
  - Output drift scores that feed into the same alerting mechanism.
- Build a simple dashboard (Grafana/Metabase/Superset) over:
  - `paper_trading_metrics` (already implemented).
  - `ml_features` summaries.
  - Phase 2 summaries.

---

### 3. Summary

- **P4** will add:
  - A non‑blocking raw feature stream, Parquet micro‑batches, and an `ml_feature_batches` registry.
  - This helps research and reproducibility without touching the core live system.
- **P5** will add:
  - Training‑time feature importance and optional SHAP summaries.
  - Runtime metadata hooks for basic explainability.
  - A scriptable alerting layer around Phase 2 metrics, optionally pushing to external channels.

These plans are intentionally modular so each piece can be implemented and tested independently when you are ready to invest in deeper infrastructure. 


