# NSE Multi-Expiry Option Chain Collector

This module collects and aggregates option chain data across the next 5 expiries for a base strike calculated from the market open price. Data is stored minute-by-minute in the `nse_multi_expiry_minute_data` table.

## Overview

The collector:
- Calculates the base strike from market open price (rounded to nearest strike difference)
- Identifies the next 5 expiries (can be weekly, monthly, or any combination)
- Fetches option chain data (OI, Change in OI, Volume, IV) for the base strike across all 5 expiries
- Aggregates metrics (sums for OI/Volume/Change in OI, averages for IV)
- Stores data every minute during market hours (9:15 AM - 3:30 PM IST)

## Features

- **IV Handling**: Uses IV from API if available, otherwise calculates using Black-Scholes
- **Change in OI**: Calculated by comparing current minute to previous minute
- **Deduplication**: Skips saving if data already exists for a timestamp
- **Websocket Support**: Can use websocket tick data for faster access (optional)

## Database Schema

Table: `nse_multi_expiry_minute_data`

| Column | Type | Description |
|--------|------|-------------|
| timestamp | TIMESTAMP | Minute-level timestamp |
| exchange | TEXT | Exchange name (e.g., "NSE") |
| open_price | DOUBLE PRECISION | Market open price (same all day) |
| base_strike | DOUBLE PRECISION | Calculated strike (same all day) |
| expiry_1_date to expiry_5_date | DATE | Next 5 expiry dates |
| total_oi_call_all_expiries | DOUBLE PRECISION | Sum OI Call (all 5 expiries) |
| total_oi_put_all_expiries | DOUBLE PRECISION | Sum OI Put (all 5 expiries) |
| total_oi_change_call_all_expiries | DOUBLE PRECISION | Sum Change OI Call |
| total_oi_change_put_all_expiries | DOUBLE PRECISION | Sum Change OI Put |
| total_volume_call_all_expiries | DOUBLE PRECISION | Sum Volume Call |
| total_volume_put_all_expiries | DOUBLE PRECISION | Sum Volume Put |
| avg_iv_call_all_expiries | DOUBLE PRECISION | Average IV Call |
| avg_iv_put_all_expiries | DOUBLE PRECISION | Average IV Put |

**Primary Key**: `(timestamp, exchange, base_strike)`

## Usage

### 1. Run Background Collector Service

Start the service that collects data every minute during market hours:

```bash
python nse_multi_expiry_collector/run_collector.py --exchange NSE
```

**Options:**
- ``: Exchange name (default: NSE)
- `--user-id`: Zerodha user ID (optional, uses config/env if not provided)
- `--password`: Zerodha password (optional, uses config/env if not provided)

The service will:
- Initialize daily parameters at market open (9:15 AM)
- Collect data every minute until market close (3:30 PM)
- Stop automatically after market close

### 2. Backfill Historical Data

Backfill data for past dates:

```bash
python nse_multi_expiry_collector/backfill_multi_expiry.py \
    --exchange NSE \
    --start-date 2026-01-02 \
    --end-date 2026-01-02
```

**Options:**
- `--exchange`: Exchange name (default: NSE)
- `--start-date`: Start date in YYYY-MM-DD format (required)
- `--end-date`: End date in YYYY-MM-DD format (required)
- `--user-id`: Zerodha user ID (optional)
- `--password`: Zerodha password (optional)

**Note**: The script will skip weekends automatically.

### 3. Use as Library

```python
from nse_multi_expiry_collector import (
    MultiExpiryCollectorService,
    get_next_5_expiries,
    aggregate_multi_expiry_data,
    calculate_base_strike
)

# Initialize service
service = MultiExpiryCollectorService(kite_obj=kite, exchange="NSE")

# Start collection
service.start()

# Stop when done
service.stop()
```

## Integration with Main App

To integrate with the main `oi_tracker_new.py` application, you can initialize the service during app startup:

```python
from nse_multi_expiry_collector import MultiExpiryCollectorService

# After Kite connection is established
collector_service = MultiExpiryCollectorService(
    kite_obj=app_manager.connector.kite,
    exchange="NSE"
)

# Optionally update with websocket data
collector_service.update_websocket_data(websocket_tick_data)

# Start service
collector_service.start()
```

## Module Structure

```
nse_multi_expiry_collector/
├── __init__.py              # Module exports
├── aggregator.py            # Strike calculation and aggregation logic
├── kite_fetcher.py          # Kite API functions for fetching option chain data
├── collector_service.py     # Background service class
├── backfill_multi_expiry.py # Historical backfill script
├── run_collector.py         # Standalone service runner
└── README.md                # This file
```

## Data Aggregation

The collector aggregates data across 5 expiries as follows:

- **OI (Open Interest)**: Sum across all expiries (separate for Calls and Puts)
- **Change in OI**: Sum across all expiries (current minute - previous minute)
- **Volume**: Sum across all expiries
- **IV (Implied Volatility)**: Average across all expiries (only includes valid IV values)

## Error Handling

- Missing instruments: Returns zeros for that expiry
- API errors: Logs error and continues
- Duplicate records: Skips saving if record already exists
- Market closed: Service stops automatically

## Requirements

- Kite API connection (Zerodha credentials)
- PostgreSQL database with TimescaleDB extension (optional but recommended)
- Python packages: `kite_trade`, `psycopg2-binary`

## Notes

- Open price and base strike are calculated once at market open and remain the same all day
- The next 5 expiries are determined based on available option contracts
- IV calculation requires option prices and spot price; falls back to 0 if unavailable
- Change in OI requires previous minute's data; returns 0 if not available