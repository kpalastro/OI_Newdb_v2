# 3-Minute % Change in Next-Expiry OI/Volume vs Direction/Swing

## What is analyzed

- **Next-expiry (next\*) fields:**  
  `nse_next_oi_call_total`, `nse_next_oi_put_total`,  
  `nse_next_volume_call_total`, `nse_next_volume_put_total`  
  (CE/PE OI and volume for the next-expiry option chain.)

- **3-minute percentage change:**  
  For each minute bar at time `t`, the script uses the value at `t - 3` minutes and computes:
  - `pct_3m_nse_next_oi_call_total` = (OI_CE_t − OI_CE_{t−3}) / OI_CE_{t−3} × 100
  - Same for PE OI and for CE/PE volume.

- **Direction:**  
  `oi_next_sentiment` (or `nse_next_oi_change_diff_put_call`):  
  - **Positive** = put OI change > call OI change → **bearish** bias  
  - **Negative** = call OI change > put OI change → **bullish** bias  

- **Swing / alignment:**  
  Whether the 3m % change in next\* OI/volume **aligns** with this direction (e.g. CE OI 3m % up + sentiment negative = bullish alignment).

## Script

```bash
# With DB (ml_features)
python scripts/analyze_next_oi_3m_direction.py --exchange NSE --days 30 --trade-logs-dir trade_logs --out reports

# With exported ml_features CSV (no DB)
python scripts/analyze_next_oi_3m_direction.py --exchange NSE --days 30 --csv path/to/ml_features_export.csv --out reports
```

## Outputs

1. **3m % change summary**  
   Mean, std, count for each of:  
   `pct_3m_nse_next_oi_call_total`, `pct_3m_nse_next_oi_put_total`,  
   `pct_3m_nse_next_volume_call_total`, `pct_3m_nse_next_volume_put_total`.

2. **Direction alignment**  
   For each 3m % change series:
   - When 3m % is **positive**, mean direction (sentiment) and when **negative**.
   - **align_pct**: % of rows where 3m % change and direction agree:
     - CE OI/volume: CE up + bullish (sentiment &lt; 0) or CE down + bearish (sentiment &gt; 0).
     - PE OI/volume: PE up + bearish (sentiment &gt; 0) or PE down + bullish (sentiment &lt; 0).

3. **Conclusions (direction/swing)**  
   - If **align_pct &gt; 55%**: “When next-expiry CE/PE OI (or volume) 3m % change is positive, direction aligns X% of the time” → supports using 3m % change for direction/swing.
   - If **align_pct &lt; 45%**: Weak or opposite signal.
   - If **~45–55%**: Roughly neutral.

4. **Trade outcomes (if trade logs provided)**  
   Win rate and mean PnL when 3m % change at **entry** is positive vs negative, by trade type (CE/PE) and by OI vs volume.  
   Use this to prefer CE entries when 3m % change in next CE OI/volume is positive (if that bucket has higher win rate), and similarly for PE.

## Summary table (conclusions)

| 3m % change series              | Alignment with direction | Interpretation for direction/swing |
|--------------------------------|--------------------------|------------------------------------|
| Next CE OI 3m % up              | High align → bullish    | CE OI building in 3m supports bullish swing. |
| Next PE OI 3m % up              | High align → bearish    | PE OI building in 3m supports bearish swing. |
| Next CE volume 3m % up          | High align → bullish    | Next-expiry call volume surge supports bullish. |
| Next PE volume 3m % up         | High align → bearish    | Next-expiry put volume surge supports bearish. |

Run the script with your data (DB or CSV) to get your own numbers and conclusions.
