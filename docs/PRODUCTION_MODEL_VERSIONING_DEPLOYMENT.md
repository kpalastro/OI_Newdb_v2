# Production Deployment Guide: Model Versioning System

This guide provides step-by-step instructions for deploying the Model Versioning/Registry system to production.

## Overview

The Model Registry provides:
- ✅ **Automatic Version Tracking**: Every trained model is automatically registered
- ✅ **Performance Comparison**: Compare new models with production models
- ✅ **Safe Promotion**: Promote models to production with automatic rollback capability
- ✅ **Performance Monitoring**: Track production model performance over time
- ✅ **Audit Trail**: Complete history of all model versions and deployments

## Prerequisites

### 1. Database Setup

The model registry requires two database tables that are automatically created:

- `model_versions` - Stores all model versions with metadata and metrics
- `model_performance_log` - Tracks production performance metrics

**Verify tables exist:**

```bash
python -c "
from database_new import get_db_connection, release_db_connection
conn = get_db_connection()
cursor = conn.cursor()
cursor.execute(\"\"\"
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
    AND table_name IN ('model_versions', 'model_performance_log')
\"\"\")
tables = [row[0] for row in cursor.fetchall()]
print('Found tables:', tables)
if len(tables) == 2:
    print('✓ Model registry tables exist')
else:
    print('⚠ Missing tables. Run: python -c \"from database_new import initialize_database; initialize_database()\"')
release_db_connection(conn)
"
```

**If tables don't exist, create them:**

```bash
python -c "from database_new import initialize_database; initialize_database()"
```

### 2. Model Storage

Ensure model files are stored in a persistent location accessible to production:

```bash
# Verify model directory structure
ls -la models/NSE/
ls -la models/BSE/
```

**Expected structure:**
```
models/
├── NSE/
│   ├── regime_models.pkl
│   ├── hmm_model.pkl
│   ├── feature_selector.pkl
│   └── ...
└── BSE/
    └── ...
```

### 3. Code Deployment

Ensure the following files are deployed:
- `model_registry.py` - Core registry implementation
- `database_new.py` - Database connection (with model registry tables)
- `train_model.py` - Training script (with auto-registration)
- `scripts/manage_models.py` - CLI tool for model management

## Deployment Steps

### Step 1: Database Migration

**Run database initialization** (creates tables if they don't exist):

```bash
python -c "from database_new import initialize_database; initialize_database()"
```

**Verify migration:**

```bash
python -c "
from database_new import get_db_connection, release_db_connection
conn = get_db_connection()
cursor = conn.cursor()
cursor.execute(\"\"\"
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name = 'model_versions'
    ORDER BY ordinal_position
\"\"\")
print('model_versions columns:')
for row in cursor.fetchall():
    print(f'  {row[0]}: {row[1]}')
release_db_connection(conn)
"
```

### Step 2: Initial Model Registration

If you have existing models in production, register them manually:

```python
from model_registry import ModelRegistry
from pathlib import Path
from datetime import date

registry = ModelRegistry()

# Register existing production model
version = registry.register_model(
    exchange='NSE',
    model_type='lightgbm_regime',
    model_path=Path('models/NSE/regime_models.pkl'),
    validation_metrics={
        'f1_score': 0.65,  # Replace with actual metrics
        'precision': 0.62,
        'recall': 0.68,
        'accuracy': 0.64
    },
    training_data_start=date(2025, 1, 1),  # Replace with actual dates
    training_data_end=date(2025, 3, 31),
    training_samples=50000,  # Replace with actual count
    notes="Initial production model - pre-registry deployment"
)

# Promote to production
registry.promote_to_production(
    version_id=version,
    reason="Initial production deployment"
)

print(f"✓ Registered and promoted model version {version}")
```

### Step 3: Verify Auto-Registration

**Test that training automatically registers models:**

```bash
# Run a test training (use small dataset for testing)
python train_model.py --exchange NSE --days 7

# Verify model was registered
python scripts/manage_models.py list --exchange NSE --latest
```

**Expected output:**
```
Model ID: 1
Version: v1.0.0
Exchange: NSE
Model Type: lightgbm_regime
Status: candidate
F1 Score: 0.65
...
```

### Step 4: Configure Production Workflow

The model registry is **automatically integrated** into the training pipeline. No additional configuration needed.

**Training workflow:**
1. Run training: `python train_model.py --exchange NSE --days 90`
2. Model is automatically registered as `candidate`
3. Review and compare with production
4. Promote if performance is better

## Production Workflow

### Workflow 1: Training and Promotion

```bash
# 1. Train new model (auto-registers as candidate)
python train_model.py --exchange NSE --days 90

# 2. List new candidates
python scripts/manage_models.py list --exchange NSE --status candidate

# 3. Compare with production
python scripts/manage_models.py compare --new <candidate_id> --current <production_id>

# 4. If performance is better, promote
python scripts/manage_models.py promote --id <candidate_id> --reason "F1 improved by 5%"
```

### Workflow 2: Monitoring Production Performance

**Record daily performance metrics:**

```python
from model_registry import ModelRegistry
from datetime import date

registry = ModelRegistry()

# Get current production model
prod_model = registry.get_production_model('NSE', 'lightgbm_regime')
if prod_model:
    # Record performance (e.g., from backtesting or live trading)
    registry.record_performance(
        model_version_id=prod_model['id'],
        exchange='NSE',
        evaluation_date=date.today(),
        signals_generated=150,
        correct_predictions=95,
        incorrect_predictions=55,
        win_rate=0.633,
        avg_return=0.025,
        sharpe_ratio=1.8,
        max_drawdown=0.05,
        avg_confidence=0.72
    )
    print("✓ Performance recorded")
```

**View performance history:**

```bash
python scripts/manage_models.py show --id <model_id> --performance
```

### Workflow 3: Rollback Procedure

If production model performance degrades:

```bash
# 1. List previous production models
python scripts/manage_models.py list --exchange NSE --status deprecated

# 2. Rollback to previous version
python scripts/manage_models.py rollback \
    --exchange NSE \
    --type lightgbm_regime \
    --target <previous_version_id> \
    --reason "Performance degradation detected - reverting to v1.2.0"
```

**Or using Python:**

```python
from model_registry import ModelRegistry

registry = ModelRegistry()

registry.rollback_model(
    exchange='NSE',
    model_type='lightgbm_regime',
    target_version_id=3,
    reason="Performance degradation detected"
)
```

## CLI Reference

### List Models

```bash
# List all models
python scripts/manage_models.py list

# List by exchange
python scripts/manage_models.py list --exchange NSE

# List production models only
python scripts/manage_models.py list --production

# List candidates only
python scripts/manage_models.py list --status candidate

# List latest model
python scripts/manage_models.py list --exchange NSE --latest
```

### Show Model Details

```bash
# Show by ID
python scripts/manage_models.py show --id 5

# Show latest model
python scripts/manage_models.py show --exchange NSE --type lightgbm_regime --latest

# Show with performance history
python scripts/manage_models.py show --id 5 --performance
```

### Compare Models

```bash
# Compare two models
python scripts/manage_models.py compare --new 5 --current 3

# Compare latest candidate with production
python scripts/manage_models.py compare --new latest --current production --exchange NSE
```

### Promote Model

```bash
# Promote to production
python scripts/manage_models.py promote --id 5 --reason "F1 improved by 5%"
```

### Rollback Model

```bash
# Rollback to previous version
python scripts/manage_models.py rollback \
    --exchange NSE \
    --type lightgbm_regime \
    --target 3 \
    --reason "Performance issue"
```

## Integration with Production Systems

### 1. Live Trading System

**Load production model:**

```python
from model_registry import ModelRegistry
from pathlib import Path
import pickle

registry = ModelRegistry()

# Get current production model
prod_model = registry.get_production_model('NSE', 'lightgbm_regime')
if prod_model:
    model_path = Path(prod_model['model_path'])
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✓ Loaded production model: {prod_model['version']}")
    print(f"  F1 Score: {prod_model['validation_metrics']['f1_score']:.3f}")
```

### 2. Monitoring Dashboard

The Model Monitoring Dashboard (`/monitoring`) automatically displays:
- Current production models
- Latest training metrics
- Backtest results
- Performance trends

**Access:** `http://your-server:5000/monitoring`

### 3. Automated Deployment Script

Create a deployment script for CI/CD:

```bash
#!/bin/bash
# deploy_model.sh

EXCHANGE=$1
MODEL_ID=$2
REASON=$3

if [ -z "$EXCHANGE" ] || [ -z "$MODEL_ID" ] || [ -z "$REASON" ]; then
    echo "Usage: ./deploy_model.sh <EXCHANGE> <MODEL_ID> <REASON>"
    exit 1
fi

# Promote model
python scripts/manage_models.py promote --id "$MODEL_ID" --reason "$REASON"

# Verify promotion
python scripts/manage_models.py show --id "$MODEL_ID" | grep -q "is_production.*True"
if [ $? -eq 0 ]; then
    echo "✓ Model $MODEL_ID promoted to production for $EXCHANGE"
    exit 0
else
    echo "✗ Promotion failed"
    exit 1
fi
```

## Production Checklist

### Initial Deployment

- [ ] Database tables created (`model_versions`, `model_performance_log`)
- [ ] Existing production models registered
- [ ] Training script tested (auto-registration works)
- [ ] CLI tool verified (`scripts/manage_models.py`)
- [ ] Model storage paths verified
- [ ] Production model loaded successfully in live system

### Regular Operations

- [ ] Models auto-registered after each training run
- [ ] New candidates reviewed before promotion
- [ ] Performance metrics recorded daily
- [ ] Production model performance monitored
- [ ] Rollback procedure tested and documented

### Before Promotion

- [ ] Compare new model with production
- [ ] Verify F1 score improvement ≥ 3% (or manual override)
- [ ] Check precision/recall not degraded significantly
- [ ] Review training data quality
- [ ] Test model loading in staging environment
- [ ] Document promotion reason

### After Promotion

- [ ] Verify model is marked as production
- [ ] Old production model marked as deprecated
- [ ] Production system reloaded with new model
- [ ] Monitor initial performance closely
- [ ] Record first day performance metrics

## Troubleshooting

### Issue: Models not auto-registering

**Symptoms:** Training completes but no model in registry

**Diagnosis:**
```bash
# Check training logs for registration errors
grep -i "register" logs/training.log

# Verify model files exist
ls -la models/NSE/regime_models.pkl
```

**Solution:**
- Check database connection
- Verify `ModelRegistry` import in `train_model.py`
- Check file permissions on model directory
- Review training logs for exceptions

### Issue: Promotion fails

**Symptoms:** `promote_to_production()` returns error

**Diagnosis:**
```python
from model_registry import ModelRegistry
registry = ModelRegistry()

# Check if model exists
model = registry.get_model_by_id(model_id)
print(model)
```

**Solution:**
- Verify model ID exists
- Check database permissions
- Ensure model file exists at specified path
- Review error logs

### Issue: Production model not loading

**Symptoms:** `get_production_model()` returns None

**Diagnosis:**
```bash
# Check if any production models exist
python scripts/manage_models.py list --production
```

**Solution:**
- Verify at least one model is marked `is_production=True`
- Check exchange and model_type match
- Review database for production models

### Issue: Performance metrics not recording

**Symptoms:** `record_performance()` succeeds but metrics not visible

**Diagnosis:**
```python
from model_registry import ModelRegistry
registry = ModelRegistry()

# Check performance log
model = registry.get_model_by_id(model_id)
print(model.get('performance_history', []))
```

**Solution:**
- Verify `model_version_id` exists
- Check date format (must be `date` object, not `datetime`)
- Review database constraints
- Check for duplicate entries (same date + model_version_id)

## Security Considerations

### Database Access

- **Read-only access** for monitoring/querying
- **Write access** only for training and deployment scripts
- Use connection pooling (already implemented)
- Consider read replicas for monitoring queries

### Model Files

- Store models in secure, backed-up location
- Use version control for model metadata (not model files)
- Implement access controls on model directory
- Regular backups of model files

### Audit Trail

All model operations are logged:
- Model registration (with git commit hash)
- Promotion/deprecation (with reason)
- Rollback (with reason)
- Performance recording (with date)

**Query audit trail:**

```sql
SELECT 
    id,
    version,
    exchange,
    model_type,
    status,
    is_production,
    created_at,
    promoted_at,
    notes
FROM model_versions
WHERE exchange = 'NSE'
ORDER BY created_at DESC;
```

## Performance Optimization

### Database Indexes

Indexes are automatically created:
- `idx_model_versions_exchange_type` - Fast lookup by exchange/type
- `idx_model_versions_status` - Filter by status
- `idx_model_versions_production` - Fast production model lookup
- `idx_model_performance_version` - Performance history queries

### Query Optimization

**For high-frequency queries (e.g., loading production model):**

```python
# Cache production model in application
# Refresh cache on promotion/rollback events
```

**For monitoring dashboards:**

```python
# Use read replicas
# Cache recent performance metrics
# Batch queries where possible
```

## Backup and Recovery

### Database Backup

```bash
# Backup model registry tables
pg_dump -h localhost -U user -d database_name \
    -t model_versions -t model_performance_log \
    > model_registry_backup.sql
```

### Model Files Backup

```bash
# Backup model directory
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/
```

### Recovery Procedure

1. **Restore database tables:**
   ```bash
   psql -h localhost -U user -d database_name < model_registry_backup.sql
   ```

2. **Restore model files:**
   ```bash
   tar -xzf models_backup_YYYYMMDD.tar.gz
   ```

3. **Verify production models:**
   ```bash
   python scripts/manage_models.py list --production
   ```

## Monitoring and Alerts

### Key Metrics to Monitor

1. **Model Performance:**
   - Daily win rate
   - Sharpe ratio trends
   - Max drawdown
   - Signal accuracy

2. **Registry Health:**
   - Number of candidate models
   - Time since last training
   - Production model age
   - Failed registrations

### Alert Conditions

Set up alerts for:
- Production model performance degradation (>10% drop in win rate)
- No new models trained in 7+ days
- Failed model registrations
- Promotion failures

### Example Monitoring Query

```sql
-- Check production model performance trend
SELECT 
    mv.version,
    mv.created_at,
    mpl.evaluation_date,
    mpl.win_rate,
    mpl.sharpe_ratio,
    mpl.max_drawdown
FROM model_versions mv
JOIN model_performance_log mpl ON mv.id = mpl.model_version_id
WHERE mv.is_production = TRUE
  AND mv.exchange = 'NSE'
ORDER BY mpl.evaluation_date DESC
LIMIT 30;
```

## Best Practices

1. **Always review before promoting:** Use `compare_models()` to verify improvements
2. **Document promotions:** Add meaningful reasons when promoting
3. **Monitor closely after promotion:** Track first 24-48 hours of performance
4. **Regular performance recording:** Record metrics daily for trend analysis
5. **Test rollback procedure:** Ensure you can quickly rollback if needed
6. **Keep model files backed up:** Models are large, ensure backups
7. **Version control code:** Model registry tracks git commit hash
8. **Regular training:** Retrain models weekly/monthly with fresh data

## Next Steps

After successful deployment:

1. **Set up automated training:** Schedule weekly/monthly retraining
2. **Implement performance monitoring:** Daily performance recording
3. **Create deployment dashboard:** Visualize model versions and performance
4. **Document team procedures:** Train team on promotion/rollback
5. **Set up alerts:** Monitor for performance degradation

---

**Last Updated:** January 2026  
**Version:** 1.0
