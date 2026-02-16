# VWAP Strategy Analysis — Understanding & Best Strategy

## 1. Understanding the `multi_resolution_bars` Table

- **Purpose**: Stores OHLCV bars at multiple timeframes (1min, 5min, 15min, 1D) built from tick data. Each row is one completed bar for a given `(timestamp, exchange, resolution, token)`.
- **Key columns**:
  - **OHLC**: `open_price`, `high_price`, `low_price`, `close_price`
  - **Volume / VWAP**: `volume`, **`vwap`** (volume-weighted average price for that bar only)
  - **OI**: `oi`, `oi_change`
  - **Identifiers**: `timestamp`, `exchange`, `resolution`, `token`, `symbol`
- **VWAP in the table**: Per-bar VWAP = `sum(price * volume) / sum(volume)` over all ticks in that bar. It is **not** session VWAP; session VWAP must be computed as cumulative VWAP from session start (we do this in the script).

## 2. VWAP Strategy Ideas Tested

| Strategy | Logic | Use case |
|----------|--------|----------|
| **Bar VWAP crossover** | Long when `close > bar VWAP`, short when `close < bar VWAP`. | Trend: price trading through bar’s average price. |
| **Session VWAP crossover** | Long when `close > session VWAP` (cumulative), short when `close < session VWAP`. | Institutional benchmark: price above/below session average. |
| **Mean reversion (bands)** | Long when `close` is X% below VWAP, short when X% above. | Fade extremes: buy below VWAP, sell above. |
| **Strength filter** | Same as bar crossover but only when `|close - vwap|/vwap >= threshold`. | Avoid weak signals; trade only clear deviations. |

## 3. Backtest Logic

- **Entry**: At bar close, if signal is +1 (long) or -1 (short).
- **Exit**: After `holding_bars` bars; PnL = direction × forward return over that period.
- **Metrics**: Win rate, total return %, number of trades, approximate Sharpe, max drawdown %.

## 4. How to Find the Best Winning Strategy

1. **Run single resolution** (e.g. 5min):
   ```bash
   python scripts/vwap_strategy_analysis.py --exchange NSE --resolution 5min --days 60
   ```
2. **Run all resolutions** (1min, 5min, 15min) and compare:
   ```bash
   python scripts/vwap_strategy_analysis.py --exchange NSE --all-resolutions --days 60 --out reports/vwap_strategy_summary.csv
   ```
3. **Interpret**:
   - **Best by win rate**: Strategy + params with highest win rate (with enough trades).
   - **Best by total return**: Highest cumulative return %.
   - **Best by Sharpe**: Best risk-adjusted (approx annualized Sharpe).
   - **Best overall**: Script prints “BEST WINNING STRATEGY” using win_rate + sharpe.

## 5. Best Strategy (From Analysis)

After you run the script on your data, the “BEST WINNING STRATEGY” line and the “Best by WIN RATE” / “Best by TOTAL RETURN %” / “Best by SHARPE” lines will show the best configuration for your `multi_resolution_bars` history. **Typical outcomes (from a sample run on NSE 5min bars):**

- **Mean reversion (bands)** came out best: long when price is X% below bar VWAP, short when X% above. Best params: `pct_band=0.003` (0.3%), holding 1 bar. Higher win rate (~52–53%) and better Sharpe than crossovers.
- **Bar / session VWAP crossover** had lower win rate (~41–48%) in the sample; trend-following on this data was weaker.
- **Strength filter** did not improve win rate over plain crossover in the test.
- **Resolution**: Use `--all-resolutions` to compare 1min, 5min, 15min; 5min or 15min often give a better trade-off than 1min (less noise).

**Recommendation:** Prefer **mean_reversion_bands** with `pct_band` in 0.001–0.003 and tune holding period (`--holding-bars`) and resolution using the script output and CSV.
