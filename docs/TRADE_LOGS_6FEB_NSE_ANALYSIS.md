# Trade Logs Analysis: 6 Feb NSE – Price Source, Strike Selection & Issues

Analysis of `trade_logs/trades_2026-02-06.csv` (NSE) and the codebase to answer: **where prices are fetched**, **how strike/contract is chosen**, and **what might be wrong**.

---

## 1. Where prices are fetched (source)

**All execution prices (entry and exit) come from Kite (Zerodha), not NSE directly.**

| Use | Source | Code location |
|-----|--------|----------------|
| **Option LTP** (entry/exit, target/stop checks) | `handler.latest_tick_data` → Kite websocket (NFO segment) | `oi_tracker_new.py`: `prepare_web_data()` → `tick = handler.latest_tick_data.get(token, {})`, `ltp = tick.get('last_price')` or `option_price_cache` |
| **Spot (NIFTY)** for ATM strike | Same Kite feed, underlying token (NSE index) | `get_atm_strike()` uses `handler.latest_tick_data.get(handler.underlying_token, {}).get('last_price')` |
| **Config** | NSE: `ltp_exchange: 'NSE'` (spot), `options_exchange: 'NFO'` (options) | `config.py` |

The NSE option-chain API in `feature_engineering.py` (`_fetch_nse_option_chain_data`) is used only for **optional OI/IV-style features** (e.g. when backfilling or enriching features). It is **not** used for live execution or for the prices written to trade logs.

---

## 2. How the strike / contract is chosen

1. **ATM strike**  
   `get_atm_strike(handler)` in `oi_tracker_new.py`:  
   - Uses spot LTP from `handler.latest_tick_data` (Kite).  
   - Rounds to `strike_difference` (50 for NIFTY).

2. **NSE auto-trade contract**  
   `_select_auto_trade_contract(..., exchange='NSE')`:  
   - **Weekly expiry** (not monthly).  
   - **Deep ITM**: 2 strikes away from ATM.  
   - BUY signal → Deep ITM **CALL** (position = -2).  
   - SELL signal → Deep ITM **PUT** (position = +2).  
   - Contract list and **LTP** used for selection come from the **option chain rows** built in `prepare_web_data()` (same Kite LTP as above).

3. **Decision to trade (which strike/side)**  
   - LightGBM (and optionally DL/RL) produces signal (BUY/SELL/HOLD) and confidence from ML features.  
   - Strategy router (`execution/strategy_router.py`) and execution config (e.g. min confidence, kelly) decide whether to act.  
   - Contract selection is then the **first valid** Deep ITM contract from the **calls/puts list** passed in (see next section).

So: **strike and option type are chosen by rule (ATM ± 2, weekly); the exact contract and its price come from the option chain built from Kite data.**

---

## 3. Entry price and exit price in the log

- **Entry price**  
  - Set in `execution/auto_executor.py` → `execute_paper_trade()`.  
  - `current_price` is the **LTP of the selected contract** at the time of selection.  
  - `fill_price` = `current_price` (if confidence > 0.9) or `current_price ± 2 ticks` (LimitChase slippage).  
  - This `fill_price` is stored as `entry_price` in the position and written to the trade log in `_perform_log_trade_entry()`.

- **Exit price**  
  - When a position is closed (target, stop, or signal flip), `monitor_positions()` or `_exit_conflicting_positions_on_signal_flip()` build a **price_map** from **current** `call_options + put_options` (current LTP from handler).  
  - That price is passed to `close_position()` → `schedule_log_trade_exit()` and written as `exit_price` in the CSV.

So: **both entry and exit prices in the 6 Feb NSE log are Kite-derived LTPs** (with optional slippage on entry). The important subtlety is **when** those LTPs are read (see next section).

---

## 4. What is likely wrong: stale snapshot for entry

The pipeline is **asynchronous**:

1. **Data loop** (e.g. every ~5 s) runs `prepare_web_data(handler, ...)` using **current** `handler.latest_tick_data`, then builds a **FeatureJob** with `calls` and `puts` (deep copy of that option chain, including LTPs).
2. The **feature worker** runs in a separate process/thread and can take several seconds (or more under load). When it finishes, it puts a **ResultJob** on `result_queue` with **the same** `result.calls` and `result.puts` (from the time the job was created).
3. **Auto-execution** runs in the **feature result consumer**. It uses `result.calls` and `result.puts` for:
   - `_select_auto_trade_contract(result.calls, result.puts, ...)` → symbol and **current_price**
   - That **current_price** is the LTP from the **snapshot**, not from “right now”.

So:

- **Entry price** in the log can be **many seconds old** (age = time since the data loop last built that snapshot + feature job processing time). If the queue is busy or the worker is slow, entry can be 10–30+ seconds stale.
- **Exit price** is taken when the exit logic runs, from the **current** option chain (same loop or a fresh result), so it is typically **newer** than the entry snapshot.

Consequences:

- Entry and exit are not aligned in time: **asymmetric staleness**.
- You may be logging “entry” at a price that was no longer the market price at decision time.
- In fast markets, this can make PnL and fill assumptions in the log optimistic or pessimistic relative to what would have been achievable with real-time quotes.

---

## 5. kelly_fraction 0.309 vs 0.2163 (not a bug)

In the 6 Feb log, `kelly_fraction` alternates between **0.309** and **0.2163**.  

- These come from `risk_manager.get_optimal_position_size()` → Kelly-style formula and confidence.  
- They are stored in `signal.metadata['kelly_fraction']` and then in the position and trade log.  
- Different signals (different win-rate/confidence inputs) yield different fractions; so 0.309 vs 0.2163 is **expected** and not an error.

---

## 6. Summary table

| Question | Answer |
|----------|--------|
| Where are prices fetched? | **Kite (Zerodha)** – spot and option LTP from `handler.latest_tick_data` (websocket). NSE API is not used for execution prices. |
| How is strike/contract decided? | **NSE:** ATM from spot LTP (Kite); contract = Deep ITM (2 strikes from ATM), weekly expiry; BUY→CALL, SELL→PUT. Selection uses the **option chain snapshot** (calls/puts) passed into the feature result. |
| Entry price in log | Kite LTP of selected contract **at snapshot time** (± simulated slippage). Snapshot can be **stale** (see above). |
| Exit price in log | Kite LTP from **current** option chain when exit is triggered. |
| What’s wrong? | **Stale entry price**: entry uses LTP from an old feature-job snapshot; exit uses current LTP. This can distort PnL and fill assumptions. |

---

## 7. Recommendations

1. **Short term**  
   - When logging or analysing 6 Feb (or any day), treat **entry_price** as “LTP at snapshot time,” not “live price at order time.”  
   - Optionally log `result.timestamp` (or snapshot time) next to each trade to quantify staleness.

2. **Code fix (implemented)**  
   - For auto-execution, **re-fetch current LTP** at the moment of execution (e.g. from `handler.latest_tick_data` or current `handler.latest_oi_data['call_options'/'put_options']`) by symbol, instead of using only `result.calls`/`result.puts`.  
   - Use that fresh price for `current_price` → `fill_price` → `entry_price` and for the trade log. That would align entry with “price at decision time” and reduce asymmetric staleness.

3. **NSE vs NFO**  
   - Execution and trade logs use **NFO (Kite)** option prices. NSE option-chain API remains for features only unless you explicitly add a separate NSE execution path.

If you want, the next step can be a small patch that uses “current” handler option chain by symbol when running `execute_paper_trade()` so that `entry_price` in the log reflects the latest available LTP at execution time.
