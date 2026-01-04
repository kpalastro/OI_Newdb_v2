# How to Keep Filling Data During Running Market

## Important Understanding

**`nse_multi_expiry_analytics` is a VIEW (not a table)** - it's automatically computed from the base table `nse_multi_expiry_minute_data`.

- ✅ **Data is saved to**: `nse_multi_expiry_minute_data` table
- ✅ **View is automatically updated**: `nse_multi_expiry_analytics` view shows data from the table
- ❌ **Don't try to insert into the view** - it's read-only

## Solution: Run the Collector Service Continuously

To keep filling data during running market, you need to run the collector service **continuously**. Here's how:

### Step 1: Start the Collector Service

```bash
python run_multi_expiry_collector.py --exchange NSE
```

Enter your 2FA code when prompted.

### Step 2: Run it in Background (So it Keeps Running)

The service needs to run continuously. Here are the best options:

#### Option A: Using Screen (Recommended)

```bash
# Start a screen session
screen -S multi_expiry_collector

# Run the collector
python run_multi_expiry_collector.py --exchange NSE
# Enter 2FA code

# Detach from screen (service keeps running)
# Press: Ctrl+A, then press D

# To check on it later:
screen -r multi_expiry_collector

# To stop it:
screen -r multi_expiry_collector
# Then press Ctrl+C
```

#### Option B: Using nohup (Alternative)

```bash
nohup python run_multi_expiry_collector.py --exchange NSE > collector.log 2>&1 &
# Enter 2FA code when prompted

# View logs:
tail -f collector.log

# Stop it:
ps aux | grep run_multi_expiry_collector
kill <PID>
```

#### Option C: Using tmux

```bash
tmux new -s collector
python run_multi_expiry_collector.py --exchange NSE
# Enter 2FA code
# Press Ctrl+B then D to detach

# Reattach:
tmux attach -t collector
```

### Step 3: Verify Data is Being Collected

Check the database to confirm data is being saved:

```sql
-- Check latest records (should update every minute during market hours)
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

-- Check today's record count (should increase every minute)
SELECT COUNT(*) as today_records
FROM nse_multi_expiry_minute_data
WHERE exchange = 'NSE'
  AND DATE(timestamp) = CURRENT_DATE;

-- The view will automatically show this data
SELECT COUNT(*) as analytics_records
FROM nse_multi_expiry_analytics
WHERE exchange = 'NSE'
  AND DATE(timestamp) = CURRENT_DATE;
```

## How It Works

### During Market Hours (9:15 AM - 3:30 PM IST):

1. **Every minute**, the service:
   - Fetches option chain data from Kite API
   - Aggregates across 5 expiries
   - Saves to `nse_multi_expiry_minute_data` table
   - **The view automatically updates** (no action needed)

2. **Data flows like this**:
   ```
   Kite API → Collector Service → nse_multi_expiry_minute_data (table)
                                              ↓
                                  nse_multi_expiry_analytics (view) ← You query this
   ```

### Outside Market Hours:

- Service waits automatically
- Will resume at next market open (9:15 AM IST)
- No manual intervention needed

## Keep It Running 24/7

To ensure continuous data collection:

### 1. Run in Background Session (as shown above)

Use `screen` or `tmux` so it survives terminal disconnections.

### 2. Auto-Start on System Boot (Optional)

#### Linux (using systemd):

Create `/etc/systemd/system/multi-expiry-collector.service`:

```ini
[Unit]
Description=Multi-Expiry Option Chain Collector
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/Users/kpal/projects/dilip/OI_Newdb_v2
ExecStart=/usr/bin/python3 run_multi_expiry_collector.py --exchange NSE
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable multi-expiry-collector.service
sudo systemctl start multi-expiry-collector.service
```

#### Mac (using launchd):

Create `~/Library/LaunchAgents/com.multi-expiry.collector.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.multi-expiry.collector</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>/Users/kpal/projects/dilip/OI_Newdb_v2/run_multi_expiry_collector.py</string>
        <string>--exchange</string>
        <string>NSE</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/kpal/projects/dilip/OI_Newdb_v2</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

Then:
```bash
launchctl load ~/Library/LaunchAgents/com.multi-expiry.collector.plist
```

## Monitoring & Troubleshooting

### Check if Service is Running:

```bash
ps aux | grep run_multi_expiry_collector
```

### Check Recent Data:

```sql
-- Latest timestamp should be within last few minutes during market hours
SELECT 
    MAX(timestamp) as latest_timestamp,
    COUNT(*) as total_records_today
FROM nse_multi_expiry_minute_data
WHERE exchange = 'NSE'
  AND DATE(timestamp) = CURRENT_DATE;
```

### If Data Stops Coming:

1. **Check service is running**: `ps aux | grep run_multi_expiry_collector`
2. **Check logs**: `tail -f collector.log` (if using nohup)
3. **Check database connection**: Service might have lost DB connection
4. **Check Kite connection**: 2FA might have expired
5. **Restart service**: Stop and start again

## Key Points

✅ **Run the collector service continuously** - this is what fills the data  
✅ **Data goes into `nse_multi_expiry_minute_data` table**  
✅ **View `nse_multi_expiry_analytics` updates automatically**  
✅ **Service handles market hours automatically**  
✅ **Use screen/tmux to keep it running**  
❌ **Don't try to insert directly into the view**  

## Quick Reference

```bash
# Start collecting (foreground)
python run_multi_expiry_collector.py --exchange NSE

# Start collecting (background with screen)
screen -S collector
python run_multi_expiry_collector.py --exchange NSE
# Ctrl+A then D to detach

# Check if running
ps aux | grep run_multi_expiry_collector

# View in screen session
screen -r collector

# Stop service
# If in screen: screen -r collector then Ctrl+C
# If with nohup: kill <PID>
```
