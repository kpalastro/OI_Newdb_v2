# Realistic Trade Logs

This folder contains **fresh trade logs** with **realistic PnL** and **exit from target/stop**, recomputed from `option_chain_snapshots` (DB).

## How they were generated

- **Script:** `scripts/generate_realistic_trade_logs.py`
- **Input:** Existing logs in `trade_logs/`
- **Time filter:** Only trades with **entry between 9:16 and 15:20 IST** are included.
- **Entry:** LTP at `entry_timestamp` (nearest snapshot at or before entry).
- **Exit:** For each trade, the script scans DB snapshots *after* entry and finds the **first minute** where **target** or **stop** is hit (NSE: 25 pts, BSE: 50 pts). That minute’s timestamp and LTP are written as `exit_timestamp`, `exit_price`, and `exit_reason` (e.g. `Target Hit (+25)` or `Stop Loss (-25)`). So exit reflects the model decision outcome (did the trade hit target or stop, and when), not the old log’s exit. If neither target nor stop is hit in the snapshot range, the script falls back to the original exit time and LTP at that time.

## Columns

Same as original trade logs, plus:

- **pnl_source:**  
  - `target_stop` = exit time and price from first target/stop hit in DB snapshots.  
  - `db_snapshot` = exit from original exit time, LTP from DB at that time.  
  - `original` = no snapshot at entry, so original values kept.

## Limitations

- Snapshots are **minute** granularity. Exit is the first *minute* when target or stop is hit, not tick-level.
- BSE (and symbols without snapshots in the DB for that time) keep original values and `pnl_source=original`.

## Compare original vs realistic PnL

```bash
# Original logs
python3 scripts/analyze_losing_trades.py --trade-logs-dir trade_logs --out reports/original_summary.csv

# Realistic logs
python3 scripts/analyze_losing_trades.py --trade-logs-dir trade_logs_realistic --out reports/realistic_summary.csv
```

Total PnL in realistic logs reflects model decisions with **snapshot-based** entry/exit prices (closer to what the fix aims for: price at execution time).
