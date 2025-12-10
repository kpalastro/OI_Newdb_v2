# Data Maintenance Guide

## Overview

This guide covers periodic database maintenance, data cleanup, and backup procedures to ensure optimal system performance.

---

## 1. Database Overview

### Tables & Storage

| Table | Purpose | Growth Rate | Retention |
|-------|---------|-------------|-----------|
| `option_chain_snapshots` | Raw option data | ~50MB/day | 90 days |
| `ml_features` | Engineered features | ~30MB/day | 120 days |
| `multi_resolution_bars` | OHLCV bars | ~10MB/day | 60 days |
| `macro_signals` | Macro data | ~1MB/day | 180 days |
| `trade_history` | Paper trades | ~100KB/day | Permanent |
| `vix_term_structure` | VIX data | ~500KB/day | 90 days |

### TimescaleDB Hypertables

The system uses TimescaleDB for efficient time-series storage:
- `option_chain_snapshots` - Auto-compressed chunks
- `ml_features` - 1-day chunk interval
- `multi_resolution_bars` - 1-day chunk interval

---

## 2. Daily Maintenance (Automatic)

These are handled automatically by the system:

| Task | Frequency | Handler |
|------|-----------|---------|
| Tick data aggregation | Every minute | `MultiResolutionAggregator` |
| Feature persistence | Every 30 seconds | `database_new.py` |
| Reel state checkpoint | On shutdown | `oi_tracker_new.py` |

**No action required** - just ensure the application runs during market hours.

---

## 3. Weekly Maintenance

### 3.1 Database Vacuum & Analyze

Run every weekend to reclaim storage and update statistics:

```bash
# Connect to PostgreSQL
psql -d oi_db_new -U dilip

# Run maintenance commands
VACUUM ANALYZE option_chain_snapshots;
VACUUM ANALYZE ml_features;
VACUUM ANALYZE multi_resolution_bars;

# Exit
\q
```

### 3.2 Check Table Sizes

```sql
-- Run in psql
SELECT 
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) as total_size,
    pg_size_pretty(pg_relation_size(schemaname || '.' || tablename)) as data_size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname || '.' || tablename) DESC;
```

---

## 4. Monthly Maintenance

### 4.1 Data Retention Cleanup

Delete old data beyond retention period:

```sql
-- Option chain snapshots (keep 90 days)
DELETE FROM option_chain_snapshots 
WHERE timestamp < NOW() - INTERVAL '90 days';

-- ML features (keep 120 days for training)
DELETE FROM ml_features 
WHERE timestamp < NOW() - INTERVAL '120 days';

-- Multi-resolution bars (keep 60 days)
DELETE FROM multi_resolution_bars 
WHERE bar_time < NOW() - INTERVAL '60 days';

-- VIX term structure (keep 90 days)
DELETE FROM vix_term_structure 
WHERE timestamp < NOW() - INTERVAL '90 days';

-- Run vacuum after deletions
VACUUM ANALYZE;
```

### 4.2 TimescaleDB Chunk Management

```sql
-- View chunk status
SELECT chunk_name, range_start, range_end, 
       pg_size_pretty(before_compression_total_bytes) as before,
       pg_size_pretty(after_compression_total_bytes) as after
FROM timescaledb_information.chunk_compression_stats
WHERE hypertable_name = 'option_chain_snapshots'
ORDER BY range_end DESC
LIMIT 10;

-- Compress old chunks manually (if needed)
SELECT compress_chunk(chunk) 
FROM show_chunks('option_chain_snapshots', older_than => INTERVAL '7 days') chunk;
```

### 4.3 Index Maintenance

```sql
-- Rebuild indexes (run during off-hours)
REINDEX TABLE option_chain_snapshots;
REINDEX TABLE ml_features;
```

---

## 5. Backup Procedures

### 5.1 Database Backup

**Daily Backup (Recommended)**

```bash
#!/bin/bash
# backup_database.sh - Run daily via cron

BACKUP_DIR="/path/to/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/oi_db_new_${TIMESTAMP}.sql.gz"

# Create backup
pg_dump -h localhost -U dilip -d oi_db_new | gzip > "${BACKUP_FILE}"

# Keep only last 7 daily backups
find "${BACKUP_DIR}" -name "oi_db_new_*.sql.gz" -mtime +7 -delete

echo "Backup created: ${BACKUP_FILE}"
```

**Schedule with Cron**
```bash
# Edit crontab
crontab -e

# Add line (runs at 4 AM daily)
0 4 * * * /path/to/backup_database.sh >> /var/log/db_backup.log 2>&1
```

### 5.2 Model Backup

Models are backed up during archiving (see `03_model_performance.md`).

Additional backup:
```bash
# Weekly model backup
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/
```

### 5.3 Trade Logs Backup

```bash
# Backup trade logs monthly
tar -czf trade_logs_$(date +%Y%m).tar.gz trade_logs/
```

---

## 6. Storage Optimization

### 6.1 Check Disk Usage

```bash
# Check database size
psql -d oi_db_new -c "SELECT pg_size_pretty(pg_database_size('oi_db_new'));"

# Check disk space
df -h
```

### 6.2 Enable Compression (TimescaleDB)

```sql
-- Enable compression on hypertables (if not already)
ALTER TABLE option_chain_snapshots 
SET (timescaledb.compress = true, 
     timescaledb.compress_segmentby = 'exchange');

ALTER TABLE ml_features 
SET (timescaledb.compress = true,
     timescaledb.compress_segmentby = 'exchange');

-- Set compression policy (compress after 7 days)
SELECT add_compression_policy('option_chain_snapshots', INTERVAL '7 days');
SELECT add_compression_policy('ml_features', INTERVAL '7 days');
```

### 6.3 Expected Storage Usage

| Duration | Estimated Size | Action |
|----------|----------------|--------|
| 1 month | ~3 GB | Normal |
| 3 months | ~8 GB | Consider cleanup |
| 6 months | ~15 GB | Cleanup required |
| 1 year | ~25 GB | Archive old data |

---

## 7. Maintenance Calendar

### Weekly Tasks (Every Sunday)

- [ ] Run VACUUM ANALYZE
- [ ] Check table sizes
- [ ] Verify backup completed
- [ ] Review error logs

### Monthly Tasks (1st Weekend)

- [ ] Run data retention cleanup
- [ ] Check TimescaleDB compression
- [ ] Archive old trade logs
- [ ] Rebuild indexes (if needed)
- [ ] Retrain models (see `02_model_training.md`)

### Quarterly Tasks

- [ ] Review retention policies
- [ ] Archive old backups to cold storage
- [ ] Performance audit
- [ ] Storage capacity planning

---

## 8. Maintenance Scripts

### Complete Weekly Script

Save as `weekly_maintenance.sh`:

```bash
#!/bin/bash
# Weekly Maintenance Script
# Run every Sunday after market hours

echo "=========================================="
echo "OI Trading System - Weekly Maintenance"
echo "Date: $(date)"
echo "=========================================="

# 1. Database maintenance
echo "1. Running VACUUM ANALYZE..."
psql -d oi_db_new -U dilip -c "VACUUM ANALYZE;"

# 2. Check table sizes
echo "2. Table sizes:"
psql -d oi_db_new -U dilip -c "
SELECT tablename, 
       pg_size_pretty(pg_total_relation_size('public.' || tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public' 
ORDER BY pg_total_relation_size('public.' || tablename) DESC 
LIMIT 10;"

# 3. Database backup
echo "3. Creating backup..."
./backup_database.sh

# 4. Check disk space
echo "4. Disk usage:"
df -h | grep -E "^/dev|Filesystem"

echo "=========================================="
echo "Weekly maintenance complete!"
echo "=========================================="
```

### Complete Monthly Script

Save as `monthly_maintenance.sh`:

```bash
#!/bin/bash
# Monthly Maintenance Script
# Run 1st weekend of each month

echo "=========================================="
echo "OI Trading System - Monthly Maintenance"
echo "=========================================="

# 1. Data cleanup
echo "1. Cleaning old data..."
psql -d oi_db_new -U dilip << EOF
DELETE FROM option_chain_snapshots WHERE timestamp < NOW() - INTERVAL '90 days';
DELETE FROM ml_features WHERE timestamp < NOW() - INTERVAL '120 days';
DELETE FROM multi_resolution_bars WHERE bar_time < NOW() - INTERVAL '60 days';
VACUUM ANALYZE;
EOF

# 2. Compress old chunks
echo "2. Compressing old data..."
psql -d oi_db_new -U dilip -c "
SELECT compress_chunk(chunk) 
FROM show_chunks('option_chain_snapshots', older_than => INTERVAL '7 days') chunk;"

# 3. Archive old trade logs
echo "3. Archiving trade logs..."
MONTH=$(date -d "last month" +%Y%m)
tar -czf "trade_logs_archive_${MONTH}.tar.gz" trade_logs/*${MONTH:0:7}* 2>/dev/null

echo "Monthly maintenance complete!"
```

---

## 9. Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Database slow | Needs vacuum | Run `VACUUM ANALYZE` |
| Disk full | Data accumulation | Run cleanup script |
| Connection errors | Too many connections | Restart PostgreSQL |
| Chunks not compressing | Policy not set | Add compression policy |
| Backup failing | Disk space | Clean old backups |
