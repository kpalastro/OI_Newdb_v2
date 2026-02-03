# Train on Date Range + Backtest on Out-of-Sample Range

This guide shows how to **train the model on a specific date range** (e.g. 2025-12-01 to 2026-01-05) and **backtest on a later, unseen range** (e.g. 2026-01-06 to 2026-01-19) to measure impact without look-ahead bias.

---

## 1. Prerequisites

- **Database** with `ml_features` populated for both the **training** and **backtest** date ranges.
- **Exchange**: NSE or BSE (same exchange for train and backtest).

---

## 2. Workflow

### Step 1: Train on the training window

Use `train_model.py` with **`--start`** and **`--end`** to train only on that range.

```bash
# Train on 2025-12-01 to 2026-01-05 (inclusive)
python train_model.py --exchange NSE --start 2025-12-01 --end 2026-01-05
```

This:

- Loads `ml_features` from the DB for `2025-12-01` through `2026-01-05`
- Runs CV and final training on that data only
- Writes models to `models/NSE/` (e.g. `swing_ensemble.pkl`, `model_features.pkl`, etc.)

**Notes:**

- If you omit `--start`/`--end`, training uses the last `--days` from today:  
  `python train_model.py --exchange NSE --days 90`
- For BSE:  
  `python train_model.py --exchange BSE --start 2025-12-01 --end 2026-01-05`

---

### Step 2: Backtest on the out-of-sample window

Run the backtester on the **later** range. It uses the **currently saved** models (from Step 1).

```bash
# Backtest on 2026-01-06 to 2026-01-19
python -m backtesting.run --exchange NSE --start 2026-01-06 --end 2026-01-19
```

Optional arguments (see `backtesting/run.py`):

- `--min-confidence 0.55` – minimum ML confidence to open a trade
- `--holding-period 15` – holding window in minutes
- `--cost-bps 2.0 --slippage-bps 1.0` – transaction cost and slippage
- `--output path/to/report.json` – custom output path

**Output:**

- By default, the report is written to `reports/backtests/NSE.json` (or `BSE.json`).
- With `--output`, it is written to the path you give.
- The report includes **`daily_summary`**: per-date breakdown of **target hits** (wins), **stop losses** (losses), **breakeven**, and **total net/gross PnL** for each day. The CLI also prints this table to the console.

---

## 3. Example: Full impact check

```bash
# 1) Train on Dec 2025 – early Jan 2026
python train_model.py --exchange NSE --start 2025-12-01 --end 2026-01-05

# 2) Backtest on next two weeks (out-of-sample)
python -m backtesting.run --exchange NSE --start 2026-01-06 --end 2026-01-19

# 3) Inspect results
cat reports/backtests/NSE.json | jq '.metrics'
# Or open the monitoring dashboard; it can load this file.
```

You can compare:

- **Net PnL**, **Sharpe ratio**, **num_trades** from the backtest
- Different training windows (e.g. shorter vs longer) by re-running Step 1 with different `--start`/`--end`, then Step 2 again
- Different `--min-confidence` or other backtest params

---

## 4. Summary

| Step | Command | Purpose |
|------|---------|---------|
| 1. Train | `train_model.py --exchange NSE --start 2025-12-01 --end 2026-01-05` | Fit model on training window only |
| 2. Backtest | `backtesting.run --exchange NSE --start 2026-01-06 --end 2026-01-19` | Evaluate on out-of-sample window |

The backtest uses **whatever is currently in `models/<exchange>/`**, so always run Step 1 before Step 2 when you want to measure the impact of a model trained on a specific date range.
