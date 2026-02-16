# Why Volume / OI Change Metrics Show 0% on the Dashboard

When **ITM CE VOL Δ%**, **ITM PE VOL Δ%**, and **FUT OI Δ% (3M)** show **0.0%** (and REGIME may show RLOW_VOL_COMPRESSION), the cause is that the **time-series reels** used to compute those changes are empty or too short.

---

## 1. ITM CE VOL Δ% / ITM PE VOL Δ% (0.0%)

**What they are:** Volume-weighted ITM **OI** % change over 3 minutes (from `itm_volume_ce_pct_change_3m_wavg` / `itm_volume_pe_pct_change_3m_wavg`). They use each option’s **OI** % change over 3m (`pct_changes['3m']`), weighted by **volume**.

**Why they are 0:**

- Each option’s `pct_changes['3m']` comes from **OI history** in `handler.data_reels[token]`.
- `pct_diff_3m` is computed only when:
  - `handler.data_reels.get(token)` has a reel, and
  - `len(reel) > 3` (at least 4 points: now and 3 minutes ago).
- If **data_reels** are **empty** or **too short** for the option tokens, `pct_diff_3m` is never set → `pct_changes['3m']` is None → ITM volume % change = 0.

**So “volume not coming” here = option OI reels not populated enough.**

**What to check / fix:**

1. **Cold start:** On startup, **backfill reels from DB** so OI history exists before live ticks:
   - Ensure `backfill_reels_from_db(handlers)` runs after handlers are created (e.g. in your startup/restore flow).
   - It fills `data_reels` from `option_chain_snapshots` (timestamp, token, oi) for the last `DATA_REEL_MAX_LENGTH` minutes.
2. **Live ticks:** Option tokens must receive **ticks that update OI** and those ticks must be passed to `update_data_reel_with_tick(handler, token, tick, now)`. If the WebSocket (or feed) does not send OI for option tokens, or that path doesn’t call `update_data_reel_with_tick`, reels never fill.
3. **DB history:** For backfill to work, `option_chain_snapshots` must have recent rows (timestamp, token, oi) per exchange. If the table is empty or stale, backfill leaves reels empty and 3m change stays N/A → 0 in the UI.

---

## 2. FUT OI Δ% (3M) (0.0%)

**What it is:** Futures **open interest** % change over the last 3 minutes (`percent_oichange_fut_3m`).

**Why it is 0:**

- It is computed from `handler.futures_oi_reels`: a time series of `{timestamp, oi}` for the **futures** contract.
- The code needs **at least 4 entries** (current + 3 earlier minutes) to compute 3m change.
- If **futures_oi_reels** is **empty** or has **fewer than 4 entries**, the function returns 0.0.

**What to check / fix:**

1. **Persistence:** On startup, `futures_oi_reels` is restored from saved state if you use reel persistence (`data.get('futures_oi_reels', [])`). If persistence is not used or the file is missing/empty, the reel starts empty.
2. **Live futures ticks:** `futures_oi_reels` is updated only when a **futures tick** (token == `handler.futures_token`) contains **`'oi'`**. If:
   - The feed doesn’t send futures ticks, or
   - Futures ticks don’t include `oi`,
   the reel never fills → FUT OI Δ% stays 0.
3. **Backfill:** There is no DB backfill for **futures** OI reels (only for option `data_reels`). So without persistence or live futures OI ticks, FUT OI Δ% will remain 0 until enough futures OI updates are received.

---

## 3. REGIME “RLOW_VOL_COMPRESSION”

This is the **regime** label (e.g. low volatility / compression). It is **not** the reason volume is 0; it’s consistent with **flat or zero** OI/volume change when the model sees low activity. Fixing the reels (above) will allow real % changes to show; regime may then change when vol/activity picks up.

---

## Summary

| Metric              | Source                | Why 0%                          | Fix |
|---------------------|-----------------------|----------------------------------|-----|
| ITM CE VOL Δ%       | Option OI reels       | `data_reels` empty or &lt; 4 points | Run `backfill_reels_from_db`; ensure option OI ticks update reels. |
| ITM PE VOL Δ%       | Option OI reels       | Same as above                    | Same. |
| FUT OI Δ% (3M)      | `futures_oi_reels`     | Reel empty or &lt; 4 points        | Restore from persistence; ensure futures ticks include `oi` and update reel. |

**Practical steps:**

1. Ensure **backfill_reels_from_db** runs on startup so option `data_reels` get history from `option_chain_snapshots`.
2. Ensure **option chain snapshots** are being saved (so backfill has data).
3. Ensure **live option ticks** that carry OI call **update_data_reel_with_tick** for the option tokens.
4. For **FUT OI Δ%**, ensure **futures ticks** include `oi` and that the code path that handles them updates `handler.futures_oi_reels` (and/or restore reels from persistence).

Once reels have enough history (e.g. at least 4 minutes), 3m metrics can be non-zero and the dashboard will show real volume/OI changes instead of 0%.
