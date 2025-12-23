# Running Multi-Expiry Collector During Live Market

This guide explains how to collect multi-expiry option chain data during live market hours.

## Quick Start

### Method 1: Standalone Script (Recommended)

Run the collector service as a standalone process:

```bash
python run_multi_expiry_collector.py --exchange NSE
```

**Or using the module script:**

```bash
python nse_multi_expiry_collector/run_collector.py --exchange NSE
```

### What Happens:

1. **At Market Open (9:15 AM IST)**:
   - Service automatically fetches today's open price
   - Calculates base strike (nearest ATM strike)
   - Identifies next 5 expiries
   - Initializes daily parameters

2. **Every Minute (9:15 AM - 3:30 PM IST)**:
   - Fetches option chain data for base strike across all 5 expiries
   - Aggregates OI, Change in OI, Volume, and IV
   - Saves to `nse_multi_expiry_minute_data` table
   - Skips if data already exists for that timestamp

3. **After Market Close (3:30 PM IST)**:
   - Service stops automatically
   - Can be left running, it will resume next market day

## Prerequisites

1. **Zerodha Kite Credentials**:
   - User ID and Password (from config or environment variables)
   - 2FA code (entered at startup)

2. **Database**:
   - PostgreSQL database must be accessible
   - `nse_multi_expiry_minute_data` table must exist

3. **Market Hours**:
   - Service only collects data during market hours (9:15 AM - 3:30 PM IST)
   - Outside market hours, it waits until next market open

## Running Options

### Option 1: With Command Line Arguments

```bash
python run_multi_expiry_collector.py \
    --exchange NSE \
    --user-id YOUR_USER_ID \
    --password YOUR_PASSWORD
```

### Option 2: Using Config File

If credentials are in `config.py` or environment variables:

```bash
python run_multi_expiry_collector.py --exchange NSE
```

Then enter 2FA code when prompted.

## Running in Background

### Linux/Mac (using nohup):

```bash
nohup python run_multi_expiry_collector.py --exchange NSE > collector.log 2>&1 &
```

### Linux/Mac (using screen):

```bash
screen -S multi_expiry_collector
python run_multi_expiry_collector.py --exchange NSE
# Press Ctrl+A then D to detach
# Reattach later with: screen -r multi_expiry_collector
```

### Linux/Mac (using tmux):

```bash
tmux new -s multi_expiry_collector
python run_multi_expiry_collector.py --exchange NSE
# Press Ctrl+B then D to detach
# Reattach later with: tmux attach -t multi_expiry_collector
```

### Windows (as a service or scheduled task):

1. Create a batch file `start_collector.bat`:
```batch
@echo off
cd /d C:\path\to\OI_Newdb_v2
python run_multi_expiry_collector.py --exchange NSE
pause
```

2. Run it or schedule it as a Windows Task Scheduler job

## Monitoring

### Check if Service is Running:

```bash
ps aux | grep run_multi_expiry_collector
```

### View Logs:

If running with nohup:
```bash
tail -f collector.log
```

### Check Database:

```sql
-- Check latest records
SELECT 
    timestamp,
    exchange,
    base_strike,
    total_oi_call_all_expiries,
    total_oi_put_all_expiries
FROM nse_multi_expiry_minute_data
WHERE exchange = 'NSE'
ORDER BY timestamp DESC
LIMIT 10;

-- Check today's record count
SELECT COUNT(*) 
FROM nse_multi_expiry_minute_data
WHERE exchange = 'NSE'
  AND DATE(timestamp) = CURRENT_DATE;
```

## Stopping the Service

- Press `Ctrl+C` if running in foreground
- If running in background, find the process and kill it:
  ```bash
  ps aux | grep run_multi_expiry_collector
  kill <PID>
  ```

## Troubleshooting

### Service Not Starting:

1. Check Kite credentials are correct
2. Verify 2FA code is valid
3. Check database connection
4. Ensure market is open (or service will wait)

### No Data Being Collected:

1. Check logs for errors
2. Verify base strike and expiries are found:
   ```sql
   SELECT DISTINCT base_strike, expiry_1_date 
   FROM nse_multi_expiry_minute_data 
   WHERE exchange = 'NSE' 
   ORDER BY timestamp DESC LIMIT 1;
   ```
3. Check if data already exists (service skips duplicates)

### Data Quality Issues:

- If all metrics are zero, check:
  - Option instruments exist for the strike/expiry combination
  - Kite API is returning data
  - Market is actively trading

## Integration with Main App

If you're running the main `oi_tracker_new.py` application, you can integrate the collector:

```python
from nse_multi_expiry_collector import MultiExpiryCollectorService

# After Kite connection is established
collector_service = MultiExpiryCollectorService(
    kite_obj=app_manager.connector.kite,
    exchange="NSE"
)

# Optionally update with websocket data for faster access
collector_service.update_websocket_data(websocket_tick_data)

# Start service
collector_service.start()
```

This way, it runs alongside your main application and shares the Kite connection.

## Notes

- **Open Price & Strike**: Calculated once at market open and remains constant all day
- **Next 5 Expiries**: Determined at market open, includes weekly and monthly expiries
- **Data Deduplication**: Service automatically skips saving if data already exists
- **Timezone**: All timestamps are in IST (Indian Standard Time)
- **Change in OI**: Calculated by comparing current minute to previous minute
