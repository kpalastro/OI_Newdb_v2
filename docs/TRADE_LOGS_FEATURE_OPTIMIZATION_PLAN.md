# Trade Logs → Feature & Weight Optimization Plan

This document outlines a **reverse-engineering and optimization plan** to improve model performance by analyzing losing trades in `trade_logs/`, then adjusting **features** and **weights**, with **special attention to `nse_next_*` and `bse_next_*`** (next-expiry option chain) features.

---

## 1. Analysis Script: Losing Trades

### 1.1 Script Location and Usage

- **Script:** `scripts/analyze_losing_trades.py`
- **Usage:**
  ```bash
  # Summary + breakdowns (no DB)
  python3 scripts/analyze_losing_trades.py --trade-logs-dir trade_logs

  # With DB join to compare nse_next_* / oi_next_sentiment at entry (losing vs winning)
  python3 scripts/analyze_losing_trades.py --trade-logs-dir trade_logs --db

  # Write by_exchange summary to CSV
  python3 scripts/analyze_losing_trades.py --trade-logs-dir trade_logs --out reports/trade_summary.csv
  ```

### 1.2 What the Script Outputs

- **Summary:** Total trades, wins, losses, flat, win rate, total PnL, avg PnL/trade, avg confidence (losing vs winning).
- **By exchange:** NSE vs BSE – total, wins, losses, win rate, total PnL, avg PnL.
- **Losing trades by exit_reason:** e.g. Stop Loss (-25), Stop Loss (-50), Signal Flip, EOD.
- **Losing trades by type (CE/PE):** Which option type loses more.
- **Confidence band vs loss rate:** low / mid / high / very_high – identifies if high-confidence trades lose more (overconfidence).
- **Losing trades by entry hour:** Time-of-day concentration of losses.
- **Optional (--db):** Mean of `nse_next_*` and `oi_next_sentiment` at entry for losing vs winning trades (if DB and date range available).

### 1.3 Example Findings (from a recent run)

| Metric | Value |
|--------|--------|
| Total trades | 5,266 |
| Wins / Losses / Flat | 2,483 / 2,763 / 20 |
| Win rate | **47.15%** |
| Avg confidence (losing) | 0.785 |
| Avg confidence (winning) | 0.775 |

- **By exchange:** BSE 3,672 trades (46.9% win rate), NSE 1,594 (47.7% win rate).
- **Losing exit reasons:** Stop Loss (-50) 1,877, Stop Loss (-25) 795, EOD 43, Signal Flip 27/18.
- **Losing by type:** CE 1,973, PE 790 → **CE losses dominate**.
- **Confidence vs loss rate:** **very_high (≥0.9) has highest loss rate (~57%)**; high (0.75–0.9) and mid (0.6–0.75) ~51%. Suggests overconfident signals are often wrong.
- **Entry hour:** Losses peak at 10, 13, 14, 9, 12, 15 (IST).

Use these patterns to drive feature and weight changes below.

---

## 2. Current State: `nse_next_*` and `bse_next_*`

### 2.1 `nse_next_*` (NSE Next-Expiry Option Chain)

- **Defined in:** `feature_engineering.py` (REQUIRED_FEATURE_COLUMNS and live computation).
- **Columns:**  
  `nse_next_oi_call_total`, `nse_next_oi_put_total`,  
  `nse_next_oi_change_call_total`, `nse_next_oi_change_put_total`,  
  `nse_next_volume_call_total`, `nse_next_volume_put_total`,  
  `nse_next_oi_change_diff_put_call`, plus `oi_next_sentiment` (same as OI change diff for next expiry).
- **Source:** NSE option chain for **next** expiry (not current weekly). Fetched live in `add_next_expiry_option_chain_features()` and stored in `ml_features` (and `feature_payload`).
- **Training:** Both NSE and BSE models use the same feature list; for BSE rows, these columns are **0.0** when next-expiry data is BSE-only (no NSE cross-feed). So for BSE, `nse_next_*` is effectively “next-expiry” from BSE option chain but **stored under the same names** (`nse_next_*`).

### 2.2 `bse_next_*` (BSE Next-Expiry)

- **Current codebase:** There are **no** `bse_next_*` columns. BSE reuses the **same** keys `nse_next_*` for BSE next-expiry metrics (see `feature_engineering.py` ~932–939: “Add to features with prefix 'nse_next_' (reusing same fields for BSE)”).
- **Implication:**  
  - NSE model: `nse_next_*` = true NSE next-expiry.  
  - BSE model: `nse_next_*` = BSE next-expiry (same names).  
  So the **semantics** differ by exchange; only the **name** is shared.

---

## 3. Feature Adjustments (with focus on next-expiry)

### 3.1 Add Explicit `bse_next_*` (Recommended)

- **Goal:** Clear semantics and room for BSE-specific weighting/normalization.
- **Actions:**
  1. **Schema:** Add columns (e.g. in `ml_features` / feature_payload):  
     `bse_next_oi_call_total`, `bse_next_oi_put_total`,  
     `bse_next_oi_change_call_total`, `bse_next_oi_change_put_total`,  
     `bse_next_volume_call_total`, `bse_next_volume_put_total`,  
     `bse_next_oi_change_diff_put_call`, and optionally `oi_next_sentiment_bse`.
  2. **Live features:** In `feature_engineering.add_next_expiry_option_chain_features()`:  
     - For **BSE:** fill `bse_next_*` from BSE next-expiry; keep filling `nse_next_*` with 0.0 for BSE (or leave as-is for backward compatibility during transition).  
     - For **NSE:** keep filling only `nse_next_*`; `bse_next_*` can be 0.0 or omitted.
  3. **REQUIRED_FEATURE_COLUMNS:** Add the new `bse_next_*` names so BSE training and inference use them. NSE training can keep using only `nse_next_*`.
  4. **Backfill:** Backfill `bse_next_*` from existing BSE next-expiry data where possible (e.g. from current `nse_next_*` values stored for BSE exchange).

This gives:
- Clear separation for research (e.g. “weight BSE next-expiry more for BSE model”).
- Ability to give **different weights/importance** to `nse_next_*` vs `bse_next_*` in feature selection or custom scoring.

### 3.2 Ensure Next-Expiry Features Are Not Trivial

- **Risk:** If next-expiry API often fails, `nse_next_*` / `bse_next_*` are mostly 0.0 → model may underuse them or learn noise.
- **Actions:**
  - Log fill rate of next-expiry features in production (e.g. % of rows with non-zero `nse_next_oi_call_total`).
  - In training, consider a **mask or extra binary feature** “next_expiry_available” so the model can downweight rows where next-expiry was missing.
  - Optionally: **normalize** next-expiry features (e.g. z-score or scale by underlying OI) so magnitude is comparable across days.

### 3.3 Feature Selection and Importance

- **Current:** `train_model.py` uses `SelectFromModel(threshold='median')` per CV fold, so many features (including next-expiry) can be dropped.
- **Actions:**
  - **Force-include** next-expiry: After selection, always add back `nse_next_*` (and `bse_next_*` for BSE) so they are never dropped.
  - Run **feature importance** (e.g. LightGBM) post-training and document importance of `nse_next_*` / `oi_next_sentiment`; if very low, improve data quality or normalization (3.2) before increasing weight.

### 3.4 Use Analysis Script to Correlate Losses with Next-Expiry

- Run `scripts/analyze_losing_trades.py --db` over a period where `ml_features` has next-expiry populated.
- Compare mean/std of `nse_next_*` and `oi_next_sentiment` at **entry** for losing vs winning trades (script already outlines this).
- If losing trades have systematically different next-expiry (e.g. opposite sign of `nse_next_oi_change_diff_put_call`), consider:
  - Adding **interaction** features (e.g. signal × nse_next_oi_change_diff_put_call), or
  - **Rule-based guard:** e.g. reduce confidence or skip BUY when next-expiry OI diff is strongly against the signal.

---

## 4. Weight Adjustments

### 4.1 Regime Strategy Weights

- **Where:** `regime_analysis.py` → `get_strategy_weights(regime)` (risk_scale, trend_weight, mean_reversion_weight).
- **Use:** Losing-trade concentration by regime (if you add regime to trade logs or infer from timestamp + features). If e.g. “RANGE_BOUND” has many more losses, reduce `risk_scale` or increase mean_reversion_weight for that regime.
- **Action:** (Optional) Log regime at entry in trade_logs; re-run analysis by regime; then tune `get_strategy_weights` per regime to reduce size in loss-heavy regimes.

### 4.2 Divergence Boost (ITM OI) in `ml_core.py`

- **Where:** `ml_core.generate_signal()` – bearish/bullish ITM divergence boost (e.g. +0.3 confidence, flip HOLD→SELL/BUY).
- **Risk:** Over-boosting in choppy or EOD-winding conditions can create false signals → more Stop Losses.
- **Action:**  
  - Lower **DIVERGENCE_THRESHOLD** (e.g. from 0.5 to 0.6) so only stronger divergence triggers.  
  - Or add a **regime check:** e.g. no divergence boost in LOW_VOL_COMPRESSION or when `eod_position_winding` > 0.3 (already partially handled by EOD winding suppression).

### 4.3 Confidence Filter (Critical from Current Run)

- **Finding:** **Very high confidence (≥0.9) has the highest loss rate (~57%).**
- **Where:** No hard filter today; position size and execution use the same confidence.
- **Action:**  
  - **Cap or downscale position size** when confidence > 0.88 (e.g. multiply kelly_fraction by 0.7 when confidence ≥ 0.9).  
  - Or **soft filter:** require a second condition (e.g. regime not LOW_VOL_COMPRESSION, or next-expiry OI diff aligned with signal) before acting on very high confidence.  
  - Optionally log “overconfident” trades (e.g. confidence ≥ 0.9) and monitor their win rate in next analysis runs.

### 4.4 Training Weights (Sample / Class)

- **Goal:** Make the model penalize “losing-like” contexts more.
- **Sample weights:** In `train_model.py`, assign **higher weight** to rows that resemble losing trades (e.g. same exchange, same hour band, same option type CE/PE, or regime inferred from features). Requires defining “loss-like” proxy (e.g. past 30-day loss rate by hour/exchange).  
  Example: `sample_weight = 1.0 + 0.5 * is_loss_like_context` so the model fits those contexts better.
- **Class weights:** Already `class_weight='balanced'` in LightGBM; if losses are mostly “wrong direction” (e.g. predicted BUY but market went down), consider **stronger weight for the minority class** in that regime (e.g. custom class_weight dict per fold).

### 4.5 Next-Expiry as Explicit Weight (Custom Ensemble or Rule)

- If you add `bse_next_*`, you can:
  - **BSE model:** In feature importance or a small meta-model, give **extra weight** to `bse_next_*` when deciding signal strength.  
  - **NSE model:** Similarly upweight `nse_next_*` and `oi_next_sentiment` in a post-hoc combiner (e.g. only boost BUY when `nse_next_oi_change_diff_put_call` &lt; 0 and signal is BUY).  
- This can be a **rule layer** in `ml_core.generate_signal()`: e.g. “if signal is BUY but oi_next_sentiment &lt; -K, reduce confidence by 0.1”.

---

## 5. Implementation Order (Recommended)

1. **Run and document analysis**  
   - `python3 scripts/analyze_losing_trades.py --trade-logs-dir trade_logs` (and `--db` if available).  
   - Save outputs (by_exchange, exit_reason, confidence band, entry hour) into `reports/` or docs.

2. **Quick wins (no schema change)**  
   - **Confidence:** Add cap/downscale for very high confidence (e.g. ≥0.9) in `ml_core` or in risk_manager (position size).  
   - **Divergence:** Slightly raise DIVERGENCE_THRESHOLD or add regime check.  
   - **Feature selection:** Force-include `nse_next_*` (and later `bse_next_*`) in selected features in `train_model.py`.

3. **Next-expiry data quality**  
   - Log fill rate of `nse_next_*` in live pipeline; fix API/cache so next-expiry is rarely all zeros.  
   - Optionally add “next_expiry_available” and normalize next-expiry features in training.

4. **Add `bse_next_*`**  
   - Schema + live features + REQUIRED_FEATURE_COLUMNS + backfill; then retrain BSE model with `bse_next_*` and optionally give them higher importance.

5. **Retrain with weights**  
   - Retrain NSE/BSE with force-included next-expiry features; optionally add sample weights for loss-like contexts and re-run analysis to compare win rate and loss rate by confidence/regime.

6. **Optional: Rule layer for next-expiry**  
   - In `ml_core.generate_signal()`, add a small rule that adjusts confidence when `oi_next_sentiment` or `nse_next_oi_change_diff_put_call` contradicts the signal.

---

## 6. Summary Table

| Area | Action | Priority |
|------|--------|----------|
| **nse_next_* / bse_next_* ** | Add explicit `bse_next_*` columns; keep `nse_next_*` for NSE | High |
| **Next-expiry** | Force-include in feature selection; log fill rate; optional mask/normalize | High |
| **Confidence** | Cap or downscale size when confidence ≥ 0.9 (overconfidence) | High |
| **Divergence** | Raise threshold or add regime check to avoid false boosts | Medium |
| **Regime weights** | Tune `get_strategy_weights` using loss concentration by regime | Medium |
| **Training** | Sample weights for loss-like contexts; optional class_weight tweak | Medium |
| **Rule layer** | Reduce confidence when next-expiry OI contradicts signal | Low (after data/features stable) |

This plan ties **trade_logs** analysis directly to **feature and weight** changes, with special attention to **nse_next_*** and **bse_next_*** as requested.
