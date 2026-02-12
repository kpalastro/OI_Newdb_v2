# Blank / Always-Zero ML Features: Root Causes and Fixes

This document explains why certain ML feature columns are always blank or zero, where the issues are, and how to fix or backfill them.

---

## 1. VIX term structure: `vix_contango_pct`, `term_structure_spread`

**Root cause:** Live features read these from `handler.macro_feature_cache`, which is filled from `db.get_latest_vix_term_structure(exchange)`. If the `vix_term_structure` table is never populated (no job or ingestion writing to it), the cache stays empty and both features default to `0.0`.

**Fix (going forward):**
- Ensure VIX term structure is written regularly: call `db.save_vix_term_structure(...)` from a scheduled job or from `data_ingestion/vix_term_structure.py` (or equivalent) so that `vix_term_structure` has rows.
- Then live pipeline will populate `macro_feature_cache` and `vix_contango_pct` / `term_structure_spread` will be non-zero when available.

**Backfill:**
- Run: `python scripts/backfill_blank_features.py [--exchange NSE|BSE]`
- This updates `ml_features.feature_payload` by merging `vix_contango_pct` and `term_structure_spread` from the latest `vix_term_structure` row at or before each `ml_features` timestamp. Only works for rows where `vix_term_structure` has data for that exchange and time.

---

## 2. Block / sweep / smart money: `block_trade_count`, `block_trade_imbalance`, `sweep_order_detected`, `smart_money_flow`

**Root cause:** These come from `handler.flow_cache`, which is updated only when `handler` receives a tick and calls `order_flow_analyzer.process_tick(tick)`. If the live pipeline does not feed trade ticks into the handler (e.g. only option chain snapshots, no tick stream), `flow_cache` stays `{}` and all four features are `0.0`.

**Fix:**
- Wire the trade/tick stream to the handler so that each tick is passed to `process_tick()` and `flow_cache` is updated (see `handlers.py` where `flow_cache.update(metrics)` is called).
- Until ticks are fed, these will remain 0.

**Backfill:** Not feasible from existing DB tables; would require tick-level trade history.

---

## 3. OI concentration (top 3 CE/PE): `oi_concentration_top3_ce`, `oi_concentration_top3_pe`

**Root cause:** These were hardcoded placeholders (`0.0`) in `feature_engineering.calculate_oi_concentration` and in the feature dict.

**Fix:** Implemented. `calculate_oi_concentration` now returns `oi_concentration_top3_ce` and `oi_concentration_top3_pe` (fraction of total CE/PE OI in the top 3 strikes by OI), and the live feature set uses them. New live data will have non-zero values.

**Backfill:** Historical rows were saved with `0.0` in `feature_payload`. To backfill you would need to recompute from historical option snapshots (same timestamp/exchange) and then update `feature_payload` for those rows—non-trivial and not provided here.

---

## 4. OI velocity: `oi_velocity_ce`, `oi_velocity_pe`, `oi_velocity_total`, `oi_velocity_ce_per_minute`, `oi_velocity_pe_per_minute`

**Root cause:** `calculate_oi_velocity` uses `handler.oi_history`: a time series of `{ce_oi, pe_oi, total_oi, timestamp}`. If the handler does not maintain `oi_history` (or it has fewer than `window_minutes` entries), the function returns zeros.

**Fix:**
- Ensure the handler maintains `oi_history` (e.g. append on each option chain update) with at least ~5 minutes of history so that velocity can be computed.
- Then live features will populate these.

**Backfill:** Would require reconstructing OI time series from `option_chain_snapshots` per minute and recomputing velocity; possible but not implemented.

---

## 5. Sentiment FII/DII/inst: `sentiment_fii_net_crores`, `sentiment_dii_net_crores`, `sentiment_inst_net_crores`

**Root cause:** These come from `SentimentAnalyzer.get_sentiment_metrics()` → `fetch_fii_dii()`. If that call fails (e.g. API/scraping unavailable or returns empty), the fallback in `engineer_live_feature_set` sets all three to `0.0`. They are stored only inside `feature_payload` (not as dedicated `ml_features` columns).

**Fix:**
- Ensure FII/DII data source is available and `fetch_fii_dii()` returns valid values (e.g. fix URL, auth, or use an alternative provider).
- Optionally persist FII/DII in `macro_signals` and read from there in the save path so they are consistent with other macro data.

**Backfill:** Only if you have an external time series of FII/DII net crores; then you could write a script to update `feature_payload` for the corresponding timestamps.

---

## 6. CE/PE OI momentum: `ce_oi_momentum_20`, `pe_oi_momentum_20` (and 5, 10)

**Root cause:** Live features compute these with `_average_pct_change(call_options, window)` / `_average_pct_change(put_options, window)`, which expect each option to have `pct_changes` dict with keys like `'5m'`, `'10m'`, `'20m'`. If options from the handler do not have `pct_changes` populated (or those keys are missing), the average is over an empty list and returns `0.0`.

**Fix:**
- Ensure the option chain payload (or handler’s option list) includes `pct_changes` with the required windows (e.g. from option_chain_snapshots or live computation).
- Then live features will produce non-zero momentum when there is enough history.

**Backfill:** Would require recomputing OI % changes from historical snapshots per strike and then averaging; possible but not implemented.

---

## 7. IV / ATM / ITM change and incline: `atm_iv_change_3m`, `atm_iv_change_5m`, `atm_iv_change_15m`, `atm_iv_spike`, `iv_incline_intensity`, `iv_momentum`, `itm_ce_iv_change_3m`, `itm_pe_iv_change_3m`, `iv_spike_strikes_count`, `iv_incline_direction`

**Root cause:** These are computed in `_calculate_iv_change_features`, which uses `handler.option_iv_cache` (current IV per token) and `handler._iv_history_window` (recent IV history per token). If IV is not stored in the option chain or not pushed into the handler’s cache, or if IV history is not maintained, all these features stay `0.0`.

**Fix:**
- Populate `handler.option_iv_cache` from the option chain (e.g. `iv` or equivalent field per option).
- Maintain `handler._iv_history_window` (e.g. append on each update) so that 3m/5m/15m changes and incline metrics can be computed.

**Backfill:** Would require IV history per strike from snapshots; possible in principle but not implemented.

---

## 8. ITM “signal” columns (e.g. `itm_pe_vol_in_optimal_range`, `itm_bullish_signal`, `itm_combined_score`, `itm_peak_bearish_signal`, etc.)

**Root cause:** The codebase does **not** define or compute these exact column names. The implemented ITM-related features are:

- `itm_dominance_signal`, `itm_divergence_strength`, `itm_dominance_ratio`
- `itm_combined_dominance_signal`, `itm_signal_agreement`, `itm_post_1145_enhanced_signal`
- `itm_volume_dominance_signal`, `itm_volume_divergence_strength`
- Plus divergence/winding: `itm_oi_bearish_divergence`, `itm_oi_bullish_divergence`, `eod_position_winding`, `itm_oi_flow_direction`, `is_eod_winding_window`

If your schema or training data expects names like `itm_bullish_signal`, `itm_combined_score`, `itm_pe_vol_in_optimal_range`, etc., they are either:

- Legacy/alternate names: consider mapping them to the implemented features above (e.g. `itm_bullish_signal` → derived from `itm_dominance_signal` or `itm_oi_bullish_divergence`), or
- Not implemented: add new computations and wire them into `engineer_live_feature_set` and `REQUIRED_FEATURE_COLUMNS` if you want them.

**Backfill:** Only after defining how these columns are derived (e.g. from existing ITM features or new logic).

---

## Summary

| Column group | Root cause | Fix | Backfill |
|-------------|------------|-----|----------|
| `vix_contango_pct`, `term_structure_spread` | `vix_term_structure` table empty or not used | Populate `vix_term_structure`; live uses it | Yes: `scripts/backfill_blank_features.py` |
| Block/sweep/smart money | No ticks fed to order flow analyzer | Feed ticks to handler → `process_tick` | No (no tick history) |
| `oi_concentration_top3_ce/pe` | Placeholders | Implemented in `calculate_oi_concentration` | Recompute from snapshots (not scripted) |
| OI velocity | `handler.oi_history` missing or too short | Maintain `oi_history` in handler | Recompute from snapshots (not scripted) |
| FII/DII/inst sentiment | `fetch_fii_dii()` fails or empty | Fix data source / API | Only with external FII/DII series |
| CE/PE OI momentum | `pct_changes` not in options | Add `pct_changes` to option payload | Recompute from snapshots (not scripted) |
| IV change / incline | `option_iv_cache` / `_iv_history_window` empty | Populate IV cache and history | From IV history if available |
| ITM signal names (e.g. `itm_bullish_signal`) | Not implemented (different names in code) | Map to existing ITM features or add new logic | After definition |

Running the backfill script only fixes **vix_contango_pct** and **term_structure_spread** where `vix_term_structure` has data; other columns require the fixes above and, where noted, custom backfill logic if you need history.
