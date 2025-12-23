# Quick Start: Fetch Multi-Expiry Data During Running Market

## Step-by-Step Instructions

### Step 1: Start the Collector Service

Open a terminal and run:

```bash
cd /Users/kpal/projects/dilip/OI_Newdb_v2
python run_multi_expiry_collector.py --exchange NSE
```

### Step 2: Enter 2FA Code

When prompted, enter your Zerodha 2FA (Two-Factor Authentication) code:

```
Enter 2FA code: [Enter your 2FA code here]
```

### Step 3: Service Starts Automatically

The service will:
- Connect to Kite API
- Wait if market is closed (it will start automatically at 9:15 AM IST)
- At market open, fetch open price and calculate base strike
- Collect data every minute from 9:15 AM to 3:30 PM IST

### Step 4: Monitor Collection

You'll see logs like:
```
INFO - Connected as: YOUR_USER_ID
INFO - Starting multi-expiry collector service for NSE...
INFO - Market is closed. Waiting for market open...
INFO - Initializing daily parameters...
INFO - Open price: 25766.0, Base strike: 25750
INFO - Found 5 next expiries: [2025-12-23, 2025-12-30, ...]
INFO - Saved multi-expiry data for NSE at 2025-12-17 09:15:00
INFO - Saved multi-expiry data for NSE at 2025-12-17 09:16:00
...
```

## Running in Background (Recommended)

Since you'll want this to run continuously, run it in the background:

### On Mac/Linux:

**Option A: Using screen (Recommended)**
```bash
screen -S collector
python run_multi_expiry_collector.py --exchange NSE
# Enter 2FA code
# Press Ctrl+A then D to detach
```

To check on it later:
```bash
screen -r collector
```

**Option B: Using nohup**
```bash
nohup python run_multi_expiry_collector.py --exchange NSE > collector.log 2>&1 &
# Enter 2FA code when prompted
# View logs with: tail -f collector.log
```

## Verify Data is Being Collected

### Check Database Directly:

```sql
-- See latest records
SELECT 
    timestamp,
    base_strike,
    total_oi_call_all_expiries,
    total_oi_put_all_expiries,
    total_oi_change_call_all_expiries,
    total_oi_change_put_all_expiries
FROM nse_multi_expiry_minute_data
WHERE exchange = 'NSE'
ORDER BY timestamp DESC
LIMIT 10;
```

### Check from Python:

```python
from database_new import get_db_connection, release_db_connection
from time_utils import now_ist

conn = get_db_connection()
cursor = conn.cursor()
cursor.execute("""
    SELECT COUNT(*), MAX(timestamp) 
    FROM nse_multi_expiry_minute_data 
    WHERE exchange = 'NSE' 
    AND DATE(timestamp) = CURRENT_DATE
""")
count, latest = cursor.fetchone()
print(f"Today's records: {count}, Latest: {latest}")
release_db_connection(conn)
```

## Common Scenarios

### Scenario 1: Market is Currently Open

```bash
python run_multi_expiry_collector.py --exchange NSE
# Enter 2FA code
# Service will immediately:
# 1. Get current open price (from first trade of the day)
# 2. Calculate base strike
# 3. Start collecting data every minute
```

### Scenario 2: Market is Closed (Before 9:15 AM or After 3:30 PM)

```bash
python run_multi_expiry_collector.py --exchange NSE
# Enter 2FA code
# Service will wait and start automatically at 9:15 AM IST next day
# You can leave it running - it will handle market hours automatically
```

### Scenario 3: You Want to Run Multiple Exchanges

Run in separate terminals/screens:
```bash
# Terminal 1
screen -S collector_nse
python run_multi_expiry_collector.py --exchange NSE

# Terminal 2
screen -S collector_bse
python run_multi_expiry_collector.py --exchange BSE
```

## Important Notes

1. **Service Needs to Run Continuously**: The service must be running during market hours to collect data. It won't collect data if it's not running.

2. **Data Collection Frequency**: Data is collected every minute (at :00 seconds of each minute)

3. **Automatic Deduplication**: If data already exists for a timestamp, it skips saving (no duplicates)

4. **No Data if Market Closed**: Service waits during non-market hours - this is normal behavior

5. **Base Strike Calculated Once**: The base strike is calculated at market open and stays the same all day

## Troubleshooting

### "No data being collected"

1. Check if market is open (9:15 AM - 3:30 PM IST)
2. Check logs for errors
3. Verify Kite connection is active
4. Check if service is actually running: `ps aux | grep run_multi_expiry_collector`

### "Service stopped unexpectedly"

1. Check logs for error messages
2. Verify database connection
3. Check Kite API limits (if you're making too many requests)
4. Restart the service

### "All metrics are zero"

1. Wait a few minutes - data might not be available immediately
2. Check if option instruments exist for the strike/expiry
3. Verify market is actively trading

## Stop the Service

- **If running in foreground**: Press `Ctrl+C`
- **If running in screen**: Reattach (`screen -r collector`) then press `Ctrl+C`
- **If running with nohup**: Find process and kill:
  ```bash
  ps aux | grep run_multi_expiry_collector
  kill <PID>
  ```
