# ITM-Only Paper Trading Outcome (Last 3 Days)

**Scope:** Consider only paper_trading_metrics rows where metadata has `itm_bearish_signal=true` OR `itm_bullish_signal=true` (i.e. trade only on these two ITM signals).

**Period:** Last 3 days (2026-01-27, 2026-01-28, 2026-01-29).

---

## Statistics (from DB + trade_logs)

### Paper metrics (DB filter: ITM bearish/bullish only)

| Metric | Value |
|--------|--------|
| Total signals (itm_bearish or itm_bullish) | 2,591 |
| Executed (paper) | 30 |
| Skipped | 2,561 |
| Execution rate | 1.2% |
| Executed with PnL in DB (closed) | 0 (PnL not backfilled in DB for these) |

### By signal type (metadata)

| Signal type | Signals | Executed | Total PnL (DB) |
|------------|---------|----------|------------------|
| itm_bearish_signal | 858 | 15 | ₹0.00 |
| itm_bullish_signal | 1,733 | 15 | ₹0.00 |

### Outcome from trade_logs (closed trades matched by signal_id)

Closed trades in trade_logs whose `signal_id` appears in paper_trading_metrics with ITM bearish/bullish:

| Metric | Value |
|--------|--------|
| Closed trades (ITM signal_id) | 7 |
| **Total PnL** | **₹80.00** |
| Winning trades | 4 |
| Losing trades | 3 |
| **Win rate** | **57.1%** |
| Gross profit | ₹3,364.00 |
| Gross loss | ₹3,284.00 |
| **Profit factor** | **1.02** |

---

## Daily breakdown (DB, ITM-only filter)

| trade_date | exchange | total_signals | executed_trades | skipped_trades | total_pnl |
|------------|----------|---------------|-----------------|----------------|-----------|
| 2026-01-29 | NSE | 128 | 0 | 128 | 0.0 |
| 2026-01-29 | BSE | 1,459 | 7 | 1,452 | 0.0 |
| 2026-01-28 | NSE | 37 | 5 | 32 | 0.0 |
| 2026-01-28 | BSE | 871 | 12 | 859 | 0.0 |
| 2026-01-27 | BSE | 33 | 3 | 30 | 0.0 |
| 2026-01-27 | NSE | 63 | 3 | 60 | 0.0 |

---

## Conclusion

If we **only** traded on **itm_bearish_signal** and **itm_bullish_signal** (last 3 days):

- **7 closed trades** matched (from trade_logs by signal_id).
- **Total PnL: ₹80**, **win rate 57.1%**, **profit factor 1.02** (slightly profitable).
- Sample is small; more days would be needed for robust statistics.

To reproduce:

```bash
python3 scripts/analyze_itm_signal_paper_trading.py --days 3
```

Optional: `--exchange NSE` or `--exchange BSE` to filter by exchange.
