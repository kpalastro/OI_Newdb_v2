# ITM CE/PE Scalping Strategy: 15 Points (NSE) / 30 Points (BSE)

## Summary

Strategy derived from **NSE and BSE chart analysis** (1m candles + ITM CE/PE OI % and Volume % change, 3m weighted average). Targets **15 index points** on NSE (NIFTY futures) and **30 index points** on BSE (SENSEX futures) from historical data.

---

## Chart Observations (Jan 30, 2026)

### NSE (NIFTY26FEBFUT)
- **Strong rally (13:00–14:00):** Price moved ~175 points (25300 → 25475). CE OI % and CE Vol % spiked **positive (>5%)**, PE OI % and PE Vol % went **negative (<-5%)**.
- **Downtrend (11:00–13:00):** CE metrics negative, PE metrics mixed.
- **Conclusion:** Sustained **CE up + PE down** (divergence) aligns with bullish moves; **CE down + PE up** with bearish moves.

### BSE (SENSEX26FEBFUT)
- **Strong rally (13:30–14:00):** ~450 points (82600 → 83050). CE OI % and CE Vol % spiked **~+10%**, PE OI % and PE Vol % dropped **~-10%**.
- **Correction (14:00–14:15):** Price dropped; CE metrics reversed negative, PE metrics recovered toward zero.
- **Conclusion:** Same **CE/PE divergence** confirms direction; magnitude is larger on BSE (hence 30-point target).

---

## Exchange-Specific Logic (from existing analysis)

Your codebase already encodes **NSE vs BSE** differences:

| Indicator        | NSE (interpretation)        | BSE (interpretation)           |
|-----------------|-----------------------------|---------------------------------|
| ITM PE Vol Δ%   | **Lower is better** (-0.30) | **Higher is better** (+0.55)    |
| ITM CE Vol Δ%   | Lower is better (<10% ok)   | Lower is better                 |
| ITM PE Δ%       | Slightly negative best      | Higher is better                |
| ITM CE Δ%       | Moderate positive (0.3–2.5%)| Neutral                         |

- **NSE:** Use **PE Vol Δ% < -3%** as strong bullish filter; avoid when PE Vol Δ% > 0.
- **BSE:** Use **PE Vol Δ% > ~3.8%** (volume increasing) as bullish; opposite of NSE.

---

## Scalping Rules (15 pts NSE / 30 pts BSE)

### 1. Instrument and target
- **NSE:** Trade **NIFTY futures** (or ATM CE/PE); **target = 15 points**, stop = 10 points (or 1:1.5 R:R).
- **BSE:** Trade **SENSEX futures** (or ATM CE/PE); **target = 30 points**, stop = 20 points (or 1:1.5 R:R).

### 2. Entry – Bullish scalp (LONG)
- **ITM CE OI %** (3m wavg): rising or positive.
- **ITM PE OI %** (3m wavg): falling or negative.
- **Divergence:** (CE OI % − PE OI %) > threshold (e.g. **> 5%** for strong move).
- **Volume confirmation (exchange-specific):**
  - **NSE:** ITM PE Vol Δ% < -3% (volume decreasing in puts = put selling).
  - **BSE:** ITM PE Vol Δ% > ~3.8% (volume increasing; BSE-specific).
- **Optional:** CE Vol Δ% not in “exhaustion” (NSE: < 16%; BSE: < 7.5%).
- **Time:** After 11:00 AM; best alignment in your data was **13:00–14:00**.

**Chart trigger:** Enter when CE OI % and CE Vol % turn **clearly positive** and PE OI % and PE Vol % turn **clearly negative** with price breaking out (e.g. above last 5–15m high for long).

### 3. Entry – Bearish scalp (SHORT)
- **ITM PE OI %** rising or positive, **ITM CE OI %** falling or negative.
- **Divergence:** (PE OI % − CE OI %) > threshold (e.g. **> 5%**).
- **Volume confirmation:**
  - **NSE:** ITM PE Vol Δ% > 0% (volume increasing in puts = put buying).
  - **BSE:** Use CE/PE Vol in line with BSE thresholds (PE Vol lower = caution for long; for short, PE Vol higher can support).
- **Chart trigger:** Enter when PE metrics spike and CE metrics drop, with price breaking down (e.g. below last 5–15m low).

### 4. Exit
- **Target:** 15 points (NSE) or 30 points (BSE) in the direction of the trade.
- **Stop:** 10 points (NSE) or 20 points (BSE), or same R:R (e.g. 1:1.5).
- **Time stop:** Max hold e.g. **15–30 minutes**; exit at close if target/stop not hit.
- **Indicator exit:** If CE/PE divergence **reverses** (e.g. CE drops and PE rises on a long), consider early exit even before target.

### 5. Avoid
- **NSE:** ITM PE Vol Δ% > 2%, ITM CE Vol Δ% > 16%.
- **BSE:** ITM PE Vol Δ% strongly negative when looking for long (per BSE logic).
- First 90 minutes (before 11:00 AM).
- When CE and PE metrics are **both** near zero or disagree (OI says one direction, volume says another).

---

## Implementation Hooks

- **ITM evaluation:** Use `utils/itm_feature_evaluator.evaluate_itm_features(features_dict, exchange)` for confidence and skip/position modifiers.
- **Thresholds:** NSE/BSE thresholds are in `utils/itm_feature_evaluator` (e.g. `NSE_THRESHOLDS`, `BSE_THRESHOLDS`).
- **Features:** `itm_oi_ce_pct_change_3m_wavg`, `itm_oi_pe_pct_change_3m_wavg`, and in `feature_payload`: `itm_volume_ce_pct_change_3m_wavg`, `itm_volume_pe_pct_change_3m_wavg`.

---

## Day-specific parameters

Weekday behaviour can differ (e.g. Monday/Friday vs mid-week). To find **day-specific best params** (divergence_min, min_hour, hold_minutes) that maximise total points per weekday:

```bash
python3 scripts/find_day_specific_scalping_params.py --exchange NSE --days 60
python3 scripts/find_day_specific_scalping_params.py --exchange BSE --days 60 --output reports/day_params_bse.json
```

The script sweeps a param grid (divergence 3/5/7%, min_hour 10/11/12, hold 15/30 min), runs the scalping backtest for each combo, groups trades by **day of week**, and for each weekday picks the param set that maximises total points (with a minimum number of trades). Output is printed and optionally written to JSON. Use these day-specific params in live or paper trading (e.g. Monday: stricter divergence + earlier start; Friday: different hold).

---

## Backtesting on Historical Data

1. **Data:** Per-minute `ml_features` (ITM OI/Vol %) + 1m **futures** OHLC from `multi_resolution_bars` (or `underlying_price` / `underlying_future_price` from `ml_features` if aligned to same minute).
2. **Signal:** At each minute, compute entry (long/short) from the rules above.
3. **Simulation:** For each entry, check subsequent 1m bars: if **high ≥ entry + target** (long) or **low ≤ entry − target** (short), count target hit; if **low ≤ entry − stop** (long) or **high ≥ entry + stop** (short), count stop hit; else exit at end of hold window.
4. **Metrics:** Win rate (target hit first), avg points per trade, profit factor, number of trades per day.

Run the script `scripts/backtest_itm_scalping_15_30.py` (see below) with `--exchange NSE` or `--exchange BSE` and date range to validate on history.

---

## Expected Behaviour (from charts)

- **13:00–14:00** window showed the clearest CE-up / PE-down move and large price follow-through; multiple 15 pt (NSE) or 30 pt (BSE) scalps were feasible in that window.
- **Reversals** (e.g. 14:00–14:15 on BSE) are captured by exiting on **divergence reversal** or **stop**.
- Using **3m wavg** keeps signals smooth and avoids one-minute noise; strict divergence and volume filters should reduce false entries.

This gives a single, consistent strategy that respects NSE vs BSE differences and targets **15 points (NSE)** and **30 points (BSE)** from historical data.
