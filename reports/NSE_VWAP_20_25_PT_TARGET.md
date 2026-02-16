# NSE / BSE VWAP Touch/Cross — 20–25 Point Target (Maximise Profit: Call vs Put)

## What the script does

When a bar **touches or crosses** the VWAP (values stored in `multi_resolution_bars`):

- **BUY CALL** when price crosses **above** VWAP → target **+20 / +25 points**.
- **BUY PUT** when price crosses **below** VWAP → target **-20 / -25 points**.

For each such event it looks **forward** (next N bars) and counts:

1. **How many times the target was hit** (20 pts, 25 pts).
2. **How many times the trend reversed** (price went against the trade) **before** hitting the target.

So you get:

- **Hit rate** = % of events that reached the target (20 or 25 pts).
- **Reversal rate** = % of events where the trend reversed before the target.
- **Expected points** (approx) = (hits × target pts) − (reversals × loss).

## How to maximise profit

1. **Run for a single underlying**  
   Mixing many tokens (different price levels) makes “20–25 points” inconsistent. For NSE index:
   - Use **`--symbol "NIFTY 50"`** for Nifty 50 only, or  
   - **`--token <instrument_token>`** if you know the Nifty/Bank Nifty token.

2. **Compare CALL vs PUT**  
   The script prints:
   - CALL: events, hit 20 pts (%), hit 25 pts (%), reversed before 20 (%), reversed before 25 (%).
   - PUT: same.
   Prefer the **side** (call or put) with **higher hit rate** and **lower reversal rate**, and **higher expected pts**.

3. **Choose target: 20 vs 25 pts**  
   - 20 pts: more hits, smaller reward.  
   - 25 pts: fewer hits, larger reward.  
   Use the reported hit/reversal and expected pts to pick 20 or 25.

4. **Tune parameters**  
   - **`--reversal-pts`** (default 15): what counts as “trend reversed” (points against the trade).  
   - **`--max-forward-bars`** (default 24): how many bars we look ahead for target/reversal.  
   Try different values and re-run to see which gives better hit rate and expected pts.

## Commands

```bash
# NSE, 5min bars, last 60 days, all tokens (mixed; 20–25 pts not comparable)
python3 scripts/vwap_touch_target_nse.py --exchange NSE --resolution 5min --days 60

# Nifty 50 only (if symbol in DB is "NIFTY 50")
python3 scripts/vwap_touch_target_nse.py --exchange NSE --resolution 5min --days 60 --symbol "NIFTY 50"

# Single token (get token from your instruments/DB)
python3 scripts/vwap_touch_target_nse.py --exchange NSE --resolution 5min --days 60 --token 12345

# Tune target and reversal
python3 scripts/vwap_touch_target_nse.py --exchange NSE --resolution 5min --days 60 --target-low 20 --target-high 25 --reversal-pts 15 --max-forward-bars 24

# Write summary CSV
python3 scripts/vwap_touch_target_nse.py --exchange NSE --resolution 5min --days 60 --out reports/vwap_20_25_summary.csv

# BSE (same script, --exchange BSE)
python3 scripts/vwap_touch_target_nse.py --exchange BSE --resolution 5min --days 60
python3 scripts/vwap_data_engineering.py --exchange BSE --resolution 5min --days 60 --out reports/vwap_segments_BSE.csv
```

## BSE vs NSE (sample run: 5min, 30 days, 15k bars)

| Metric | NSE | BSE |
|--------|-----|-----|
| **FOLLOW hit 20 pts (call)** | ~12.5% | **~29.9%** |
| **FOLLOW hit 20 pts (put)** | ~11.6% | **~28.7%** |
| **FOLLOW win rate (combined)** | ~12% | **~29.3%** |
| **Reversal before 20 pts (call)** | ~86% | **~69%** |
| **Reversal before 20 pts (put)** | ~80% | **~66%** |
| **FADE win rate (20 pt)** | ~5% | ~26.6% |
| **Reversal duration** | median 1 bar (~5 min), 90% within 4 bars | same |

**BSE looks better than NSE** for this VWAP touch/cross setup in the sample: higher hit rate (~29% vs ~12%) and lower reversal rate (~66–69% vs ~80–86%). Expected pts are still negative on mixed tokens but less bad on BSE.

**Best filters for BSE (data-engineering run):**
- **Hour 9 (IST):** hit 39.9%, reversal 59.4%.
- **Weak cross (small |close−vwap|/vwap):** hit 39.3%, reversal 60%.
- **Combined hour 9 + low_vol:** hit **41.3%**, reversal **57.6%**.
- **SENSEX index / options:** some symbols showed positive expected pts (e.g. top SENSEX CE: ~65% hit, ~31% reversal).

Use **`--exchange BSE`** and, for SENSEX only, **`--symbol "SENSEX"`** or **`--token <SENSEX_token>`** so 20–25 pts are comparable.

## Interpreting “how many times hit the target” vs “trend changed”

- **Hit target** = within the next `max_forward_bars`, price reached +20/+25 pts (call) or −20/−25 pts (put) **before** a reversal.
- **Trend reversed** = price moved against the trade by `reversal_pts` (or crossed back past VWAP) **before** reaching the target.

So:

- **High hit rate + low reversal rate** → more often you get the 20–25 pt move; better for maximising profit.
- **High reversal rate** → often the move fails before target; consider tighter stop or different entry (e.g. only strong crosses).

Use the printed **expected pts** and the **>>> TO MAXIMISE PROFIT** line to choose **call vs put** and **20 vs 25 pts** for your data.

---

## Data engineering when reversal rate is high (~80–87%)

When a **mixed-token** run shows very high reversal rates, use the data-engineering script to **find segments where reversal is lower and hit rate is higher**, then trade only those segments.

### Script: `scripts/vwap_data_engineering.py`

1. **Adds features** at each VWAP touch/cross event:
   - **hour** (IST)
   - **volume_ratio** = volume / rolling_avg(volume)
   - **vwap_dist_pct** = |close − vwap| / vwap × 100 (strength of cross)

2. **Stratifies** by time-of-day, volume strength, and VWAP cross strength.

3. **Outputs recommended actions**, e.g.:
   - Trade only during **hour 10** (IST) — lower reversal, higher hit rate.
   - Trade only when **volume_ratio** is in a given band (e.g. low_vol or high_vol, depending on data).
   - Trade only on **weak_cross** or **strong_cross** (vwap_dist_pct band) to improve hit vs reversal.
   - **Combined filter**: e.g. hour 10 + low_vol.

4. **Per-symbol ranking**: which symbols (or tokens) have the best score (hit_20_pct − rev_20_pct). Trade only **top-ranked symbols** or a **single underlying** (e.g. NIFTY 50) so 20–25 pts are comparable.

### Commands

```bash
python3 scripts/vwap_data_engineering.py --exchange NSE --resolution 5min --days 60 --limit 25000
python3 scripts/vwap_data_engineering.py --exchange NSE --resolution 5min --days 60 --out reports/vwap_segments.csv --out-symbols reports/vwap_symbol_ranking.csv
```

### Actions to maximise profit (from a sample run)

- **Time:** Trade only **10:00–11:00 IST** (hour 10 had lowest reversal in the sample).
- **Volume:** Prefer **low_vol** (volume below rolling average) in the sample — reversal was lower than med_vol/high_vol.
- **VWAP strength:** **Weak cross** (small |close−vwap|/vwap) had **higher hit rate** (18.6%) in the sample; use this band if your run agrees.
- **Underlying:** Use **single token/symbol** (e.g. `--symbol "NIFTY 50"` or `--token <id>`) so 20–25 points are consistent; avoid mixing many symbols.
- **Combined:** Apply **hour 10 + low_vol** together for a stricter filter and slightly lower reversal (e.g. 77.5% in the sample).

---

## Trading in the direction of reversal (FADE)

If ~77% of the time price “reverses” (moves 15 pts against the initial cross before hitting 20 pts), it is natural to ask: **what if we trade in the direction of the reversal?** (Cross up → BUY PUT, cross down → BUY CALL.)

The script now reports **FADE** stats alongside **FOLLOW**:

- **FADE** = on cross up we buy PUT (target −20 pts); on cross down we buy CALL (target +20 pts).
- **FOLLOW** = on cross up we buy CALL; on cross down we buy PUT (current strategy).

On a sample run (NSE 5min, 15k bars):

- **FADE win rate (20 pt): 5.0%** — price reached our 20 pt target in the reversal direction only 5% of the time.
- **FOLLOW win rate (call+put): 12.0%** — price reached the 20 pt target in the cross direction 12% of the time.
- FADE reversal-before-target was **94.4%** (price moved 15 pts back against our fade before hitting 20 pts).

So **trading the reversal (FADE) does not improve win rate** in this data: the 77% “reversal” means price often retraces 15 pts against the cross, but it does **not** usually then continue 20 pts in that direction before moving 15 pts back again. FOLLOW stays better than FADE on win rate; use the data-engineering filters (time, volume, VWAP strength) to improve FOLLOW rather than switching to FADE.

---

## How long does the reversal last?

The script reports **reversal duration**: how many bars (from the VWAP touch/cross) until price has moved **15 pts against** the trade (reversal threshold).

On a sample run (NSE 5min, 15k bars):

- **Median: 1 bar** (~5 min at 5min resolution).
- **Mean: ~2 bars** (~10 min).
- **90% of reversals occur within ~4 bars** (~20 min).

So **reversals tend to happen quickly** — typically within 5–10 minutes. If you follow the cross and get reversed, the move against you usually shows up in 1–2 bars; by ~20 min, 90% of reversals have already occurred. That supports a **short holding period** or **tight stop** when trading the cross, and explains why FADE rarely wins — the reversal often doesn't extend 20 pts before price moves again.

---

## Can I make money using this information?

**Short answer:** The analysis gives you **information** to improve setups and risk, but in our sample runs the **raw strategy had negative expected points** (high reversal rate, ~12% hit rate). You *can* use this to try to make money only if you **narrow the edge** and **control risk**.

### What the data showed (sample runs)

- **Follow (call on cross up, put on cross down):** ~12% hit 20 pts, ~80–86% reversed before target → **negative expected pts** before costs.
- **Fade:** ~5% win rate → worse.
- **Filters (hour 10, low_vol, weak_cross):** Lower reversal (e.g. ~77%) and higher hit in some segments, but we did not re-run full expected pts on filtered subsets.

So **as-is, unfiltered, mixed tokens:** the numbers did **not** show a positive edge.

### What you’d need for a chance to make money

1. **Positive expected value**  
   Need: (win_rate × target × payoff) − (loss_rate × stop × loss) − costs > 0.  
   That means either:
   - **Higher win rate** and/or **lower loss rate** (use filters: time, volume, VWAP strength, single underlying), or  
   - **Better risk/reward** (e.g. smaller target like 15 pts, or tighter stop so losses are smaller).

2. **Trade only when the edge is strongest**  
   Use the **data-engineering** output:
   - **Time:** e.g. only 10:00–11:00 IST (hour 10 had lower reversal in the sample).
   - **Volume / VWAP strength:** only the bands that showed better hit vs reversal (e.g. low_vol, weak_cross in the sample).
   - **Single underlying:** e.g. Nifty 50 or Bank Nifty only (`--token` or `--symbol "NIFTY 50"`) so 20–25 pts are consistent.

3. **Use reversal duration in risk**  
   Reversals often show in **1–2 bars** (~5–10 min). So:
   - **Short hold:** e.g. exit in 1–2 bars if target not hit.
   - **Tight stop:** e.g. 10–15 pts, and exit as soon as price moves that much against you.

4. **Validate on your own data**  
   Run the scripts on **your** history and **your** symbol:
   - `vwap_touch_target_nse.py` with `--token` or `--symbol "NIFTY 50"`.
   - `vwap_data_engineering.py` to see which segments (hour, volume, VWAP) have best hit vs reversal.
   - Check whether **expected pts** (or a simple P&amp;L simulation) is positive **after** costs and slippage.

5. **Costs and execution**  
   Options (call/put) have spreads, brokerage, and slippage. A small positive edge in points can turn negative after costs. Prefer liquid strikes and factor costs into your expected-value check.

### Practical takeaway

- **Yes, you *can* try to make money** with this information **if** you:
  - Use it to **filter** (time, volume, VWAP, single underlying),
  - **Control risk** (short hold, tight stop, position size),
  - **Verify** on your data that expected value is positive after costs.
- **No guarantee:** Past stats do not guarantee future profits. The analysis helps you **focus on better setups and risk**, not “free money.”
