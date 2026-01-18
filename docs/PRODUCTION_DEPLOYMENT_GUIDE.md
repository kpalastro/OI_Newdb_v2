# Complete Production Deployment Guide

This is the master guide for deploying the entire OI Tracker system to production, including all components, features, and services.

## Table of Contents

1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Infrastructure Setup](#infrastructure-setup)
4. [Database Deployment](#database-deployment)
5. [Application Deployment](#application-deployment)
6. [Model Deployment](#model-deployment)
7. [Feature Backfilling](#feature-backfilling)
8. [Monitoring & Maintenance](#monitoring--maintenance)
9. [Rollback Procedures](#rollback-procedures)
10. [Production Checklist](#production-checklist)

## System Overview

### Architecture Components

1. **Web Application** (`oi_tracker_new.py`)
   - Flask web server with WebSocket support
   - Real-time data collection and processing
   - Dashboard UI (port 5000)
   - Monitoring dashboard (port 8000)

2. **Database** (PostgreSQL/TimescaleDB)
   - Option chain snapshots
   - ML features storage
   - Model registry
   - Performance logs

3. **ML Models**
   - LightGBM regime models
   - HMM regime detection
   - Model registry for versioning

4. **Data Collection**
   - Real-time option chain data
   - Multi-expiry data collection
   - Sentiment analysis

5. **Background Services**
   - Feature engineering workers
   - Model training pipeline
   - Performance monitoring

### System Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended) or macOS
- **Python**: 3.10+
- **Database**: PostgreSQL 12+ with TimescaleDB extension
- **Memory**: 8GB+ RAM (16GB recommended)
- **Disk**: 100GB+ free space
- **Network**: Stable internet connection for data feeds

## Prerequisites

### 1. Server Setup

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python 3.10+
sudo apt install python3.10 python3.10-venv python3-pip -y

# Install PostgreSQL
sudo apt install postgresql postgresql-contrib -y

# Install TimescaleDB (optional but recommended)
# Follow: https://docs.timescale.com/install/latest/self-hosted/
```

### 2. Database Setup

```bash
# Create database user
sudo -u postgres psql
CREATE USER oi_tracker WITH PASSWORD 'your_secure_password';
CREATE DATABASE oi_tracker_db OWNER oi_tracker;
GRANT ALL PRIVILEGES ON DATABASE oi_tracker_db TO oi_tracker;
\q

# Install TimescaleDB extension (if using TimescaleDB)
sudo -u postgres psql -d oi_tracker_db
CREATE EXTENSION IF NOT EXISTS timescaledb;
\q
```

### 3. Environment Variables

Create `.env` file in project root:

```bash
# Database Configuration
OI_TRACKER_DB_TYPE=postgres
OI_TRACKER_DB_HOST=localhost
OI_TRACKER_DB_PORT=5432
OI_TRACKER_DB_NAME=oi_tracker_db
OI_TRACKER_DB_USER=oi_tracker
OI_TRACKER_DB_PASSWORD=your_secure_password

# Zerodha/Kite API Credentials
ZERODHA_USER_ID=your_user_id
ZERODHA_PASSWORD=your_password
ZERODHA_API_KEY=your_api_key
ZERODHA_API_SECRET=your_api_secret

# Flask Configuration
OI_TRACKER_FLASK_HOST=0.0.0.0
OI_TRACKER_FLASK_PORT=5000

# Optional: Logging
OI_TRACKER_LOG_LEVEL=INFO
```

### 4. Python Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Infrastructure Setup

### 1. Directory Structure

```bash
# Create necessary directories
mkdir -p models/NSE models/BSE
mkdir -p logs
mkdir -p reports/backtests
mkdir -p exports/training_batches
mkdir -p state
mkdir -p static templates
```

### 2. File Permissions

```bash
# Ensure proper permissions
chmod +x oi_tracker_new.py
chmod +x train_model.py
chmod +x scripts/*.py
chmod 755 models/
chmod 755 logs/
```

### 3. Systemd Service (Linux)

Create `/etc/systemd/system/oi-tracker.service`:

```ini
[Unit]
Description=OI Tracker Application
After=network.target postgresql.service

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/OI_Newdb_v2
Environment="PATH=/path/to/OI_Newdb_v2/.venv/bin"
ExecStart=/path/to/OI_Newdb_v2/.venv/bin/python oi_tracker_new.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable oi-tracker
sudo systemctl start oi-tracker
sudo systemctl status oi-tracker
```

## Database Deployment

### Step 1: Initialize Database Schema

```bash
# Run database initialization
python -c "from database_new import initialize_database; initialize_database()"
```

This creates:
- All required tables
- Model registry tables (`model_versions`, `model_performance_log`)
- Indexes for performance
- TimescaleDB hypertables (if extension installed)

### Step 2: Verify Database Setup

```bash
# Check tables exist
python -c "
from database_new import get_db_connection, release_db_connection
conn = get_db_connection()
cursor = conn.cursor()
cursor.execute(\"\"\"
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public'
    ORDER BY table_name
\"\"\")
tables = [row[0] for row in cursor.fetchall()]
print('Database tables:', len(tables))
for table in tables[:10]:
    print(f'  - {table}')
release_db_connection(conn)
"
```

### Step 3: Database Backup Setup

```bash
# Create backup script
cat > /usr/local/bin/backup_oi_tracker_db.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backups/oi_tracker"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR
pg_dump -h localhost -U oi_tracker oi_tracker_db | gzip > $BACKUP_DIR/backup_$DATE.sql.gz
# Keep only last 30 days
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +30 -delete
EOF

chmod +x /usr/local/bin/backup_oi_tracker_db.sh

# Add to crontab (daily at 2 AM)
(crontab -l 2>/dev/null; echo "0 2 * * * /usr/local/bin/backup_oi_tracker_db.sh") | crontab -
```

## Application Deployment

### Step 1: Code Deployment

```bash
# Clone or pull latest code
git clone https://github.com/your-repo/OI_Newdb_v2.git
# or
cd OI_Newdb_v2
git pull origin main  # or feature/model-registry

# Activate virtual environment
source .venv/bin/activate

# Install/update dependencies
pip install -r requirements.txt
```

### Step 2: Configuration

```bash
# Copy and edit environment file
cp .env.example .env
nano .env  # Edit with production values

# Verify configuration
python -c "from config import get_config; c = get_config(); print(f'DB: {c.db_host}:{c.db_port}/{c.db_name}')"
```

### Step 3: Test Application

```bash
# Test database connection
python -c "from database_new import get_db_connection, release_db_connection; conn = get_db_connection(); print('✓ DB connected'); release_db_connection(conn)"

# Test application startup (dry run)
python -c "from oi_tracker_new import app; print('✓ App imports successfully')"
```

### Step 4: Start Application

**Development/Testing:**
```bash
python oi_tracker_new.py
```

**Production (with systemd):**
```bash
sudo systemctl start oi-tracker
sudo systemctl status oi-tracker
```

**Production (with screen/tmux):**
```bash
screen -S oi-tracker
source .venv/bin/activate
python oi_tracker_new.py
# Press Ctrl+A then D to detach
```

### Step 5: Verify Application

```bash
# Check if application is running
curl http://localhost:5000/health  # If health endpoint exists
curl http://localhost:5000/  # Main dashboard

# Check logs
tail -f logs/oi_tracker.log
# or
journalctl -u oi-tracker -f  # If using systemd
```

## Model Deployment

### Step 1: Train Initial Models

```bash
# Train NSE model
python train_model.py --exchange NSE --days 90

# Train BSE model (if needed)
python train_model.py --exchange BSE --days 90
```

**Note:** Models are automatically registered in the model registry during training.

### Step 2: Verify Model Registration

```bash
# List registered models
python scripts/manage_models.py list --exchange NSE

# Check production models
python scripts/manage_models.py list --production
```

### Step 3: Promote Models to Production

```bash
# Get latest model ID
LATEST_MODEL_ID=$(python scripts/manage_models.py list --exchange NSE --latest | grep "Model ID" | awk '{print $3}')

# Promote to production
python scripts/manage_models.py promote --id $LATEST_MODEL_ID --reason "Initial production deployment"
```

### Step 4: Verify Model Loading

```python
# Test model loading
from model_registry import ModelRegistry
from pathlib import Path
import pickle

registry = ModelRegistry()
prod_model = registry.get_production_model('NSE', 'lightgbm_regime')

if prod_model:
    model_path = Path(prod_model['model_path'])
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Production model loaded: {prod_model['version']}")
else:
    print("✗ No production model found")
```

**See:** `docs/PRODUCTION_MODEL_VERSIONING_DEPLOYMENT.md` for detailed model management.

## Feature Backfilling

### Step 1: Backfill ITM Volume Features

```bash
# Backfill NSE (90 days)
python backfill_itm_volume_features.py --exchange NSE --days 90

# Backfill BSE (if needed)
python backfill_itm_volume_features.py --exchange BSE --days 90
```

**Expected time:** 1-2 minutes per day of data (90 days ≈ 2-3 hours)

### Step 2: Verify Backfill

```bash
# Check backfill success rate
python -c "
from database_new import get_db_connection, release_db_connection
conn = get_db_connection()
cursor = conn.cursor()
cursor.execute(\"\"\"
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN feature_payload->>'itm_volume_ce_pct_change_3m_wavg' IS NOT NULL THEN 1 END) as has_ce_vol,
        COUNT(CASE WHEN feature_payload->>'itm_volume_pe_pct_change_3m_wavg' IS NOT NULL THEN 1 END) as has_pe_vol,
        COUNT(CASE WHEN pcr_total_volume IS NOT NULL THEN 1 END) as has_pcrv
    FROM ml_features
    WHERE exchange = 'NSE'
    AND timestamp >= NOW() - INTERVAL '90 days'
\"\"\")
row = cursor.fetchone()
print(f'Total: {row[0]}, CE Vol: {row[1]} ({row[1]/row[0]*100:.1f}%), PE Vol: {row[2]} ({row[2]/row[0]*100:.1f}%), PCRV: {row[3]} ({row[3]/row[0]*100:.1f}%)')
release_db_connection(conn)
"
```

**See:** `docs/PRODUCTION_BACKFILL_GUIDE.md` for detailed backfill procedures.

## Monitoring & Maintenance

### 1. Application Health Monitoring

```bash
# Check application status
sudo systemctl status oi-tracker

# View recent logs
journalctl -u oi-tracker -n 100

# Monitor real-time logs
tail -f logs/oi_tracker.log
```

### 2. Database Monitoring

```bash
# Check database connections
sudo -u postgres psql -d oi_tracker_db -c "SELECT count(*) FROM pg_stat_activity WHERE datname = 'oi_tracker_db';"

# Check table sizes
sudo -u postgres psql -d oi_tracker_db -c "
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 10;
"
```

### 3. Model Performance Monitoring

```bash
# View production model performance
python scripts/manage_models.py show --exchange NSE --type lightgbm_regime --latest --performance

# Check for performance degradation
python -c "
from model_registry import ModelRegistry
from datetime import date, timedelta
registry = ModelRegistry()
prod = registry.get_production_model('NSE', 'lightgbm_regime')
if prod:
    # Get last 7 days performance
    print(f'Production model: {prod[\"version\"]}')
    # Add performance query here
"
```

### 4. Dashboard Access

- **Main Dashboard**: `http://your-server:5000`
- **Monitoring Dashboard**: `http://your-server:8000/monitoring`
- **Multi-Expiry Analytics**: `http://your-server:5000/multi-expiry-analytics`

### 5. Scheduled Tasks

**Daily Model Retraining (Optional):**

```bash
# Add to crontab (train weekly on Sunday at 2 AM)
(crontab -l 2>/dev/null; echo "0 2 * * 0 cd /path/to/OI_Newdb_v2 && .venv/bin/python train_model.py --exchange NSE --days 90") | crontab -
```

**Daily Performance Recording:**

```bash
# Create performance recording script
cat > /path/to/OI_Newdb_v2/scripts/record_daily_performance.py << 'EOF'
#!/usr/bin/env python3
from model_registry import ModelRegistry
from datetime import date
import sys
sys.path.insert(0, '/path/to/OI_Newdb_v2')

registry = ModelRegistry()
prod = registry.get_production_model('NSE', 'lightgbm_regime')
if prod:
    # Record performance metrics (implement based on your metrics collection)
    # registry.record_performance(...)
    pass
EOF

# Add to crontab (daily at 11 PM)
(crontab -l 2>/dev/null; echo "0 23 * * * cd /path/to/OI_Newdb_v2 && .venv/bin/python scripts/record_daily_performance.py") | crontab -
```

## Rollback Procedures

### 1. Application Rollback

```bash
# Stop application
sudo systemctl stop oi-tracker

# Revert code
cd /path/to/OI_Newdb_v2
git checkout <previous-commit-hash>
# or
git checkout main  # If deploying from feature branch

# Restart
sudo systemctl start oi-tracker
```

### 2. Model Rollback

```bash
# List previous production models
python scripts/manage_models.py list --exchange NSE --status deprecated

# Rollback to previous version
python scripts/manage_models.py rollback \
    --exchange NSE \
    --type lightgbm_regime \
    --target <previous_version_id> \
    --reason "Performance degradation - reverting"
```

### 3. Database Rollback

```bash
# Restore from backup
gunzip < /backups/oi_tracker/backup_YYYYMMDD_HHMMSS.sql.gz | \
    psql -h localhost -U oi_tracker oi_tracker_db
```

### 4. Feature Rollback

If backfill causes issues:

```bash
# Stop any running backfill processes
pkill -f backfill_itm_volume_features

# Restore database from backup (if needed)
# See Database Rollback above
```

## Production Checklist

### Pre-Deployment

- [ ] Server infrastructure ready (CPU, RAM, disk)
- [ ] PostgreSQL installed and configured
- [ ] Database created and user configured
- [ ] Environment variables set in `.env`
- [ ] Python virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Code deployed to production server
- [ ] File permissions set correctly
- [ ] Systemd service configured (if using)
- [ ] Backup scripts configured

### Database Setup

- [ ] Database initialized (`initialize_database()`)
- [ ] All tables created successfully
- [ ] Indexes created
- [ ] TimescaleDB extension installed (if using)
- [ ] Database backup configured
- [ ] Connection tested

### Application Deployment

- [ ] Application starts without errors
- [ ] Database connection successful
- [ ] Web interface accessible (port 5000)
- [ ] Monitoring dashboard accessible (port 8000)
- [ ] WebSocket connections working
- [ ] Logs being written correctly
- [ ] Systemd service running (if using)

### Model Deployment

- [ ] Models trained successfully
- [ ] Models registered in registry
- [ ] Production models promoted
- [ ] Model loading tested
- [ ] Model inference working

### Feature Backfilling

- [ ] ITM volume features backfilled
- [ ] PCRV backfilled
- [ ] Backfill success rate >95%
- [ ] Features verified in database

### Post-Deployment

- [ ] Application monitoring configured
- [ ] Database monitoring configured
- [ ] Model performance tracking active
- [ ] Scheduled tasks configured
- [ ] Backup verification successful
- [ ] Team trained on operations
- [ ] Documentation accessible
- [ ] Rollback procedures tested

## Troubleshooting

### Application Won't Start

```bash
# Check logs
journalctl -u oi-tracker -n 50
# or
tail -n 50 logs/oi_tracker.log

# Common issues:
# 1. Database connection failed - check .env file
# 2. Port already in use - check with: netstat -tulpn | grep 5000
# 3. Missing dependencies - run: pip install -r requirements.txt
```

### Database Connection Issues

```bash
# Test connection
psql -h localhost -U oi_tracker -d oi_tracker_db

# Check PostgreSQL status
sudo systemctl status postgresql

# Check firewall
sudo ufw status
```

### Model Loading Fails

```bash
# Verify model files exist
ls -la models/NSE/

# Check model registry
python scripts/manage_models.py list --exchange NSE

# Test model loading
python -c "from model_registry import ModelRegistry; ..."
```

### Performance Issues

```bash
# Check system resources
htop
df -h
free -h

# Check database performance
sudo -u postgres psql -d oi_tracker_db -c "SELECT * FROM pg_stat_activity;"

# Check application logs for errors
grep -i error logs/oi_tracker.log | tail -20
```

## Security Considerations

### 1. Database Security

- Use strong passwords
- Limit database access to application server only
- Use SSL connections if database is remote
- Regular security updates

### 2. Application Security

- Keep `.env` file secure (chmod 600)
- Use firewall to restrict access
- Enable HTTPS if exposing to internet
- Regular security updates

### 3. API Credentials

- Store credentials in `.env` file (not in code)
- Use environment variables
- Rotate credentials regularly
- Limit API key permissions

## Support & Documentation

### Related Guides

- **Model Versioning**: `docs/PRODUCTION_MODEL_VERSIONING_DEPLOYMENT.md`
- **Feature Backfilling**: `docs/PRODUCTION_BACKFILL_GUIDE.md`
- **Model Registry Guide**: `MODEL_REGISTRY_GUIDE.md`
- **Access Model Registry**: `ACCESS_MODEL_REGISTRY.md`

### Getting Help

1. Check application logs: `logs/oi_tracker.log`
2. Check system logs: `journalctl -u oi-tracker`
3. Review documentation in `docs/` directory
4. Check database for data issues
5. Verify configuration in `.env`

---

**Last Updated:** January 2026  
**Version:** 1.0
