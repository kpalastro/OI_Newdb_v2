# Tracking RL Model Contribution

This guide explains how to monitor and verify the contribution of RL models to trading decisions.

## Overview

RL models contribute in two ways:
1. **Signal Generation** (`RLStrategy`) - Provides trading signals (BUY/SELL/HOLD)
2. **Execution Optimization** (`RLExecutor`) - Optimizes order placement (price offset, aggression)

## Methods to Track RL Contribution

### 1. Log Analysis

RL models log their predictions with detailed information:

#### Signal Generation Logs
```
[NSE] RL Strategy (PPO): Signal=1, PositionSize=0.750
[NSE] RL Signal Generated: BUY (confidence=0.750, algorithm=PPO)
```

#### Execution Logs
```
[NSE] RL Executor (PPO): PriceOffset=0.150, Aggression=1, FillProb=0.80
[NSE] RL Execution Used: Symbol=NIFTY25DEC23400CE, Offset=0.150, Aggression=1, FillProb=0.80
```

### 2. Using the Analysis Script

Run the analysis script to get a comprehensive report:

```bash
python scripts/analyze_rl_contribution.py
```

This will:
- Analyze recommendation logs (`logs/recommendations/`)
- Analyze execution logs (`trade_logs/`)
- Generate a report showing:
  - How many recommendations used RL
  - How many executions used RL
  - Breakdown by exchange
  - Contribution percentage

**Example Output:**
```
============================================================
RL MODEL CONTRIBUTION ANALYSIS
============================================================
Generated: 2025-12-10T10:30:00

RECOMMENDATIONS (Signal Generation):
  Total Recommendations: 150
  RL as Primary Source: 25 (16.7%)
  RL in Ensemble: 45 (30.0%)
  RL Used (Any): 70 (46.7%)

  By Exchange:
    NSE:
      Primary: 15, Ensemble: 30, Any: 45
    BSE:
      Primary: 10, Ensemble: 15, Any: 25

EXECUTIONS (Order Placement):
  Total Executions: 50
  RL Execution Used: 30 (60.0%)

SUMMARY:
  RL Contribution Level: MEDIUM
  RL Active: True
============================================================
```

### 3. Check Logs Directly

#### Recommendation Logs
```bash
# View recent recommendations
tail -f logs/recommendations/$(date +%Y-%m-%d).jsonl | grep -i "rl"

# Count RL usage
grep -c '"signal_source":"rl"' logs/recommendations/*.jsonl
```

#### Application Logs
```bash
# View RL-related logs
tail -f oi_tracker.log | grep -i "rl"

# Count RL predictions
grep -c "RL Strategy" oi_tracker.log
grep -c "RL Executor" oi_tracker.log
```

### 4. Check Configuration

Verify RL is enabled:

```python
from config import get_config

config = get_config()
print(f"RL Execution Enabled: {config.rl_execution_enabled}")
print(f"RL Algorithm: {config.rl_algorithm}")
print(f"RL Use Ensemble: {config.rl_use_ensemble}")
print(f"RL PPO Model: {config.rl_ppo_model_path}")
print(f"RL DQN Model: {config.rl_dqn_model_path}")
```

### 5. Check Model Loading Status

At application startup, check logs for:

```
[NSE] RL Strategy: Model not loaded
[NSE] RL Executor initialized: False (algorithm: PPO)
```

Or if models loaded successfully:
```
[NSE] PPO model loaded: models/rl_ppo_model.zip
[NSE] DQN model loaded: models/rl_dqn_model.zip
[NSE] RL Executor initialized: True (algorithm: ENSEMBLE)
```

### 6. Real-time Monitoring

#### In Strategy Router
The `generate_signal()` method logs when RL is used:
- Check if `signal.source == 'rl'` or `signal.source == 'ensemble'`
- Check `signal.metadata` for RL details:
  - `rl_algorithm`: Which algorithm was used
  - `rl_ppo_loaded`: Whether PPO model is loaded
  - `rl_dqn_loaded`: Whether DQN model is loaded
  - `position_size`: RL's position size recommendation

#### In Auto Executor
The `execute_paper_trade()` method logs when RL execution is used:
- Look for "RL Execution Used" in logs
- Check if `rl_placement` is not None

### 7. Metrics Collection

The system tracks RL usage in metrics:

```python
from metrics.phase2_metrics import get_metrics_collector

collector = get_metrics_collector('NSE')
# Metrics include RL contribution tracking
```

## Interpreting Results

### High Contribution (>50%)
- RL is actively used in most decisions
- Models are loaded and working
- Consider reviewing RL performance vs other models

### Medium Contribution (20-50%)
- RL is used but not dominant
- May be in ensemble mode
- Check if routing mode is set to 'ensemble' or 'adaptive'

### Low Contribution (<20%)
- RL is rarely used
- Possible issues:
  - Models not loaded
  - Routing mode set to 'lightgbm' or 'dl'
  - RL models returning HOLD signals
  - State not provided to RL strategy

### Zero Contribution (0%)
- RL is not being used at all
- Check:
  1. Is `rl_execution_enabled=True`?
  2. Are model files present?
  3. Are models loading successfully?
  4. Is routing mode excluding RL?

## Troubleshooting

### RL Not Contributing

1. **Check Model Files:**
   ```bash
   ls -la models/rl_*.zip
   ```

2. **Check Configuration:**
   ```bash
   python -c "from config import get_config; c = get_config(); print(f'RL Enabled: {c.rl_execution_enabled}, Algorithm: {c.rl_algorithm}')"
   ```

3. **Check Logs for Errors:**
   ```bash
   grep -i "rl.*error\|rl.*failed\|rl.*not" oi_tracker.log
   ```

4. **Verify Routing Mode:**
   - Check `strategy_router.routing_mode`
   - Set to 'ensemble' or 'adaptive' to include RL

5. **Check State Provision:**
   - RL Strategy needs `RLState` object
   - Verify state is being created and passed

### RL Contributing But Low Quality

1. **Check Model Training:**
   - Review training logs
   - Verify models were trained on recent data
   - Check training metrics (reward, loss)

2. **Compare RL vs Other Models:**
   - Use analysis script to compare performance
   - Check if RL signals align with market conditions

3. **Retrain Models:**
   ```bash
   python train_rl.py --algorithm BOTH --exchange NSE --days 30
   ```

## Best Practices

1. **Regular Monitoring:**
   - Run analysis script daily
   - Review logs weekly
   - Track contribution trends

2. **A/B Testing:**
   - Compare performance with/without RL
   - Test different algorithms (PPO vs DQN vs Ensemble)

3. **Model Updates:**
   - Retrain models monthly
   - Update when market conditions change
   - Monitor model performance metrics

4. **Documentation:**
   - Keep notes on RL contribution levels
   - Document when RL performs well/poorly
   - Track correlation with market regimes

