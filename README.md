# OI_Newdb_v2

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![ML](https://img.shields.io/badge/ML-LightGBM%20%7C%20XGBoost%20%7C%20PyTorch-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/status-production--ready-success)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-TimescaleDB-336791?logo=postgresql&logoColor=white)

> An institutional-grade ML system for **NSE/BSE index options trading** — real-time Open Interest (OI) tracking, multi-horizon ensemble models, regime detection, and automated paper-trade execution on the Zerodha Kite platform.

---

## Overview

**OI_Newdb_v2** (a.k.a. "OI Gemini") is an end-to-end quantitative trading pipeline for Indian index options (NIFTY 50 / SENSEX). It ingests live option-chain and market-depth data from Zerodha Kite, engineers a rich microstructure feature set, trains a **multi-horizon model ensemble** (LightGBM + XGBoost + CatBoost + PyTorch LSTM/Transformer + Reinforcement Learning), detects market regimes via Hidden Markov Models, and routes the resulting signals through a portfolio of options strategies with advanced risk management and automated paper-trade execution.

The system is organized into five operational phases:

| Phase | Focus | Key Modules |
|-------|-------|-------------|
| **Data Ingestion** | Live OI/depth/volatility collection from Kite | `connector.py`, `oi_tracker_new.py`, `nse_multi_expiry_collector/`, `data_ingestion/` |
| **Feature Engineering** | Skew, gamma exposure, order-flow, regime metadata, sentiment | `feature_engineering.py`, `sentiment_analyzer.py`, `regime_analysis.py` |
| **Model Training** | Walk-forward AutoML, multi-model, Optuna-tuned | `train_model.py`, `train_orchestrator.py`, `train_all_models.py`, `models/` |
| **Signal Generation** | Runtime ML signal engine + strategy router | `ml_core.py`, `execution/strategy_router.py`, `strategies/` |
| **Execution & Risk** | Auto paper-trade executor, Greeks/VaR risk manager | `execution/auto_executor.py`, `advanced_risk_manager.py`, `risk_manager.py` |

### Key Capabilities

- **Real-time OI tracking** via Zerodha Kite WebSocket (KiteTicker) for NSE & BSE option chains
- **Multi-horizon ensemble** — intraday LSTM, swing LightGBM ensemble, expiry-day Transformer, routed by a `HorizonRouter`
- **HMM-based regime detection** (5 regimes: trending up/down, range-bound, high-vol crash, low-vol compression)
- **6 options strategies** — Gamma Scalping, OI Buildup, Volatility Expansion, Expiry Pin, Next-OI Direction
- **Advanced risk management** — Greeks limits, VaR, correlation checks, Kelly-based position sizing, overconfidence caps
- **Automated paper trading** with confidence thresholds, cooldowns, and real-time LTP-based entry/exit
- **Model registry** with versioning, promotion, rollback, and comparison (PostgreSQL-backed)
- **FastAPI dashboard** + Flask/SocketIO web UI for live visualization
- **Walk-forward backtesting** with Monte Carlo and per-segment Optuna tuning
- **Sentiment layer** — FII/DII flows, market breadth, Perplexity news sentiment, VIX term structure
- **Multi-expiry collection** for NSE weekly/monthly chains

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.11+ |
| **ML / DL** | LightGBM, XGBoost, CatBoost, scikit-learn, PyTorch, stable-baselines3 (PPO/DQN), hmmlearn |
| **Data** | pandas, NumPy, SciPy, pyarrow, Optuna (HPO) |
| **Broker API** | Zerodha Kite Connect (`kiteconnect`), KiteTicker WebSocket, `kite_trade` enctoken auth |
| **Database** | PostgreSQL + TimescaleDB (`psycopg2`) — raw OI tables + ML feature tables |
| **Web / API** | Flask + Flask-SocketIO (main UI), FastAPI + Uvicorn (dashboard server) |
| **Sentiment / Alt Data** | yfinance, BeautifulSoup4, Perplexity API, FII/DII flows, India VIX |
| **Scheduling** | `schedule`, APScheduler-style jobs via `jobs.py` |
| **Infra / Utils** | python-dotenv, joblib, psutil, multiprocessing distributed workers |

---

## Architecture Overview

```
                       ┌─────────────────────────────────────────────┐
                       │            Zerodha Kite (NSE/BSE)            │
                       │   REST API  ·  KiteTicker WebSocket         │
                       └───────────────┬─────────────────────────────┘
                                       │ live option chain + depth
                                       ▼
            ┌──────────────────────────────────────────────────────┐
            │              Data Ingestion Layer                    │
            │  connector.py · oi_tracker_new.py                     │
            │  nse_multi_expiry_collector/ · data_ingestion/        │
            └───────────────┬──────────────────────────────────────┘
                            │ raw OI / LTP / depth ticks
                            ▼
            ┌──────────────────────────────────────────────────────┐
            │         PostgreSQL / TimescaleDB (database_new.py)    │
            │   raw_oi tables  ·  ml_features tables  ·  model_versions│
            └───────────────┬──────────────────────────────────────┘
                            │ historical features
                            ▼
            ┌──────────────────────────────────────────────────────┐
            │            Feature Engineering (feature_engineering.py)│
            │  skew · gamma · order-flow · regime meta · sentiment   │
            └───────────────┬──────────────────────────────────────┘
                            │ feature vectors
              ┌─────────────┴──────────────┐
              ▼                            ▼
   ┌─────────────────────┐      ┌─────────────────────────┐
   │   Training Pipeline  │      │   Runtime ML Engine      │
   │ train_model.py        │      │  ml_core.py              │
   │ train_orchestrator.py │      │  MLSignalGenerator       │
   │ models/ (LSTM, TF,    │      │  → signal + confidence   │
   │  LightGBM, RL)        │      └────────────┬────────────┘
   └──────────┬───────────┘                   │
              │ trained artifacts               ▼
              ▼                   ┌───────────────────────────┐
   ┌─────────────────────┐        │  Strategy Router           │
   │  Model Registry      │◀──────│  execution/strategy_router │
   │  model_registry.py   │        │  → gamma scalping / OI     │
   └─────────────────────┘        │     buildup / vol exp /    │
              │ promotion           │     expiry pin / next-OI   │
              ▼                    └────────────┬──────────────┘
   ┌─────────────────────┐                     │ recommendation
   │  Regime Detector     │─────────────────────┘
   │  regime_analysis.py  │              │
   │  (HMM + heuristics)  │              ▼
   └─────────────────────┘   ┌───────────────────────────┐
                             │ Advanced Risk Manager       │
                             │ advanced_risk_manager.py    │
                             │ Greeks · VaR · Kelly sizing │
                             └────────────┬──────────────┘
                                          │ approved orders
                                          ▼
                             ┌───────────────────────────┐
                             │  Auto Executor (paper)     │
                             │  execution/auto_executor.py│
                             └────────────┬──────────────┘
                                          │ fills
                                          ▼
                             ┌───────────────────────────┐
                             │  Web UI (Flask/SocketIO)    │
                             │  + FastAPI Dashboard        │
                             │  dashboard/server.py        │
                             └───────────────────────────┘
```

---

## Project Structure

```
OI_Newdb_v2/
├── oi_tracker_new.py          # Main Flask/SocketIO app — real-time OI tracking + ML signal UI
├── config.py                  # Centralized config (env-driven), exchange configs (NSE/BSE)
├── connector.py               # Zerodha Kite session + WebSocket connection manager
├── database_new.py            # PostgreSQL/TimescaleDB persistence (raw OI + ML features)
├── feature_engineering.py     # Shared feature set: skew, gamma, order-flow, regime meta
├── ml_core.py                 # Runtime ML signal engine (MLSignalGenerator)
├── regime_analysis.py         # HMM + heuristic regime detection (5 regimes)
├── sentiment_analyzer.py      # FII/DII + breadth + news sentiment
├── risk_manager.py            # Kelly-based position sizing
├── advanced_risk_manager.py   # Greeks limits, VaR, correlation, regime-based sizing
├── model_registry.py          # Model versioning, promotion, rollback (PostgreSQL)
├── app_manager.py             # Application lifecycle manager
├── handlers.py                # Signal/trade event handlers
├── jobs.py                    # Scheduled jobs
├── kite_trade.py              # Kite enctoken auth helper
├── strings.py                 # UI string constants
├── time_utils.py              # IST timezone helpers
├── websocket_manager.py       # WebSocket lifecycle management
├── monitoring.py              # Monitoring blueprint (Flask)
├── monitor_phase2.py          # Phase-2 monitoring metrics
├── online_learning.py         # Online model adaptation
├── recommendation_logging.py  # Signal/recommendation audit logging
│
├── train_model.py             # Core training pipeline (LightGBM + HMM + Expiry Transformer)
├── train_orchestrator.py      # Walk-forward AutoML orchestrator (Optuna per segment)
├── train_all_models.py        # Train all model types for an exchange
├── train_expiry_model.py      # Expiry-day Transformer training
├── train_swing_ensemble.py    # Swing trading ensemble training
├── train_rl.py                # Reinforcement learning (PPO/DQN) training
├── train.py                   # Train wrapper: F1 → val_bpb for autoresearch
├── update_training_params.py  # Hyperparameter update utility
├── create_swing_ensemble.py   # Swing ensemble creation helper
├── analyze_research_results.py# Research experiment results analyzer
├── run_research_experiments.py# Automated research experiment runner
│
├── backfill_nse_features.py   # NSE feature backfill
├── backfill_bse_sentiment.py   # BSE sentiment backfill
├── backfill_itm_volume_features.py
├── update_sentiment_scores.py # Sentiment score updater
├── export_to_csv.py           # DB → CSV export
├── import_csv_to_db.py        # CSV → DB import
├── import_csv_fast.py         # Fast CSV import (bulk)
├── view_database.py           # DB inspection utility
├── delete_records.py          # Record deletion utilities
├── delete_all_records.py
├── delete_today_from_time.py
├── delete_paper_trading_orders.py
├── run_multi_expiry_collector.py
├── reproduce_issue.py
│
├── backtesting/               # Backtesting suite
│   ├── engine.py              # Backtest engine
│   ├── walk_forward.py        # Walk-forward analysis
│   ├── monte_carlo.py         # Monte Carlo simulation
│   └── run.py                 # Backtest runner
│
├── models/                    # Model definitions & artifacts
│   ├── multi_horizon_ensemble.py  # Unified multi-horizon interface
│   ├── horizon_router.py      # Routes predictions by horizon
│   ├── intraday_lstm.py      # Intraday LSTM model
│   ├── swing_ensemble.py     # Swing LightGBM ensemble
│   ├── expiry_transformer.py # Expiry-day Transformer
│   ├── deep_learning.py      # DL predictor
│   ├── reinforcement_learning.py  # RL (PPO/DQN) strategy + executor
│   └── registry.yml          # Model registry config
│
├── strategies/                # Options trading strategies
│   ├── base_strategy.py       # Abstract base + TradeRecommendation
│   ├── gamma_scalping.py     # Gamma scalping
│   ├── oi_buildup.py         # OI buildup
│   ├── vol_expansion.py      # Volatility expansion
│   ├── expiry_pin.py         # Expiry-day pin
│   └── next_oi_direction.py  # Next-OI direction
│
├── execution/                 # Trade execution layer
│   ├── strategy_router.py    # Dynamic model→strategy router
│   └── auto_executor.py     # Automated paper-trade executor
│
├── data_ingestion/            # Alternative & macro data ingestion
│   ├── depth_capture.py      # Market depth capture
│   ├── order_flow_analyzer.py
│   ├── multi_resolution_aggregator.py
│   ├── vix_term_structure.py
│   ├── macro_feeds.py        # Macro economic feeds
│   ├── macro_loader.py
│   └── alternative_data_hooks.py
│
├── distributed/               # Distributed feature computation
│   └── feature_worker.py     # Multiprocessing feature worker
│
├── nse_multi_expiry_collector/# NSE multi-expiry OI collector
│   ├── collector_service.py
│   ├── kite_fetcher.py
│   ├── aggregator.py
│   ├── backfill_multi_expiry.py
│   └── run_collector.py
│
├── dashboard/                 # FastAPI dashboard server
│   └── server.py
│
├── scripts/                   # Operational & analysis scripts
│   ├── create_timescale_db.py
│   ├── create_training_dataset.py
│   ├── deploy_model.py
│   ├── manage_models.py
│   ├── evaluate_candidate.py
│   ├── fetch_india_vix.py
│   ├── analyze_losing_trades.py
│   ├── analyze_trade_timing.py
│   ├── analyze_rl_contribution.py
│   ├── analyze_next_oi_3m_direction.py
│   ├── vwap_strategy_analysis.py
│   ├── generate_realistic_trade_logs.py
│   └── ...
│
├── utils/                     # Shared utilities
│   ├── greeks_calculator.py  # Options Greeks calculator
│   ├── performance.py        # Performance monitoring
│   └── rate_limiter.py       # API rate limiter
│
├── metrics/                   # Phase-2 metrics & RL contribution reports
├── reports/                   # Strategy analysis & backtest reports
├── templates/                 # Flask HTML templates (dashboard, monitoring, etc.)
├── static/                    # CSS assets
├── docs/                      # Operational & design documentation
├── data/                      # Reference CSV datasets (NIFTY/SENSEX)
├── *.sql                      # SQL queries & view definitions
├── *.md                       # Plans, guides, and analysis notes
├── requirements.txt           # Python dependencies
└── .gitignore
```

---

## Setup

### Prerequisites

- Python **3.11+**
- **PostgreSQL 14+** with the **TimescaleDB** extension
- A **Zerodha Kite** account (for live data & paper trading)
- (Optional) Perplexity API key for news sentiment

### 1. Clone

```bash
git clone git@github.com:kpalastro/OI_Newdb_v2.git
cd OI_Newdb_v2
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment

Copy the template and fill in your credentials:

```bash
cp .env.example .env   # if .env.example exists, otherwise create from the template below
```

`.env` must contain (all values are read by `config.py` via `python-dotenv`):

```dotenv
# Zerodha Kite credentials
ZERODHA_USER_ID=your_user_id
ZERODHA_PASSWORD=your_password

# PostgreSQL / TimescaleDB
OI_TRACKER_DB_TYPE=postgresql
OI_TRACKER_DB_HOST=localhost
OI_TRACKER_DB_PORT=5432
OI_TRACKER_DB_NAME=oi_tracker
OI_TRACKER_DB_USER=your_db_user
OI_TRACKER_DB_PASSWORD=your_db_password

# Optional
PERPLEXITY_API_KEY=your_perplexity_key
FLASK_SECRET_KEY=generate_a_random_string
FLASK_HOST=0.0.0.0
FLASK_PORT=5000

# Reinforcement learning execution
OI_TRACKER_RL_EXECUTION_ENABLED=false
OI_TRACKER_RL_ALGORITHM=ppo
OI_TRACKER_RL_PPO_MODEL_PATH=models/rl_ppo.zip
OI_TRACKER_RL_DQN_MODEL_PATH=models/rl_dqn.zip

# Auto-execution risk limits
OI_TRACKER_AUTO_EXEC_MAX_NET_DELTA=0.30
OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS=3
OI_TRACKER_AUTO_EXEC_HIGH_CONFIDENCE_THRESHOLD=0.75
OI_TRACKER_AUTO_EXEC_COOLDOWN_WITH_POSITIONS=300
```

> ⚠️ **Never commit `.env`.** It is in `.gitignore` and must stay local.

### 5. Initialize the database

```bash
# Create the TimescaleDB database and hypertables
python scripts/create_timescale_db.py
```

### 6. Train models (or load pre-trained)

```bash
# Full training pipeline with walk-forward AutoML
python train_orchestrator.py --exchange NSE --days 150

# Or train a single model type
python train_model.py --exchange NSE

# Train all model families (LightGBM, XGBoost, CatBoost, DL, RL)
python train_all_models.py --exchange NSE
```

Trained artifacts are saved under `models/<EXCHANGE>/` and registered in the `model_versions` table via `model_registry.py`.

### 7. Run the application

```bash
# Main web app (Flask + SocketIO) — real-time OI tracking + ML signals
python oi_tracker_new.py

# FastAPI dashboard (optional, advanced visualization)
uvicorn dashboard.server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:5000` in your browser. On first launch you will be prompted for your Kite 2FA code.

---

## Usage

### Live data collection

```bash
# Start the NSE multi-expiry OI collector
python run_multi_expiry_collector.py

# Backfill historical NSE features
python backfill_nse_features.py
```

### Backtesting

```bash
python backtesting/run.py --exchange NSE
python backtesting/walk_forward.py --exchange NSE --window-days 21 --step-days 7
python backtesting/monte_carlo.py --exchange NSE --simulations 10000
```

### Research & hyperparameter experiments

The `autoresearch` workflow runs automated ML hyperparameter experiments:

```bash
python run_research_experiments.py
python analyze_research_results.py
```

---

## Documentation

In-depth operational and design docs live in [`docs/`](docs/):

- [`docs/01_daily_operations.md`](docs/01_daily_operations.md) — daily runbook
- [`docs/02_model_training.md`](docs/02_model_training.md) — training guide
- [`docs/03_model_performance.md`](docs/03_model_performance.md) — performance tracking
- [`docs/BACKTEST_TRAIN_WORKFLOW.md`](docs/BACKTEST_TRAIN_WORKFLOW.md) — backtest/train workflow
- [`docs/PRODUCTION_DEPLOYMENT_GUIDE.md`](docs/PRODUCTION_DEPLOYMENT_GUIDE.md) — deployment
- [`docs/PRODUCTION_MODEL_VERSIONING_DEPLOYMENT.md`](docs/PRODUCTION_MODEL_VERSIONING_DEPLOYMENT.md) — model versioning
- [`docs/risk_management_implementation_summary.md`](docs/risk_management_implementation_summary.md) — risk framework
- [`docs/RL_DQN_PPO_USAGE.md`](docs/RL_DQN_PPO_USAGE.md) — reinforcement learning usage

---

## ⚠️ Disclaimer

This software is for **educational and research purposes only**. It is **not financial advice**. Options trading carries substantial risk of loss. Always do your own research and consult a licensed financial advisor before trading. The authors and contributors are not responsible for any financial losses incurred through the use of this software.

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

Copyright © 2026 **Kul Deep Pal**