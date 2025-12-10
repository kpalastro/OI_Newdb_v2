# Using Both DQN and PPO in Reinforcement Learning

This guide explains how to use both DQN and PPO algorithms in the reinforcement learning implementation.

## Overview

The RL system now supports three modes:
1. **PPO Only** - Uses Proximal Policy Optimization
2. **DQN Only** - Uses Deep Q-Network
3. **Ensemble Mode** - Uses both PPO and DQN together (combines predictions)

## Training Models

### Train Both Models

To train both PPO and DQN models for ensemble use:

```bash
python train_rl.py --algorithm BOTH --exchange NSE --days 30
```

This will create:
- `models/rl_ppo_model.zip` - PPO model
- `models/rl_dqn_model.zip` - DQN model

### Train Individual Models

```bash
# Train PPO only
python train_rl.py --algorithm PPO --exchange NSE --days 30

# Train DQN only
python train_rl.py --algorithm DQN --exchange NSE --days 30
```

## Configuration

### Environment Variables

Set these environment variables to configure RL behavior:

```bash
# Enable RL execution
export OI_TRACKER_RL_EXECUTION_ENABLED=true

# Choose algorithm: PPO, DQN, or ENSEMBLE
export OI_TRACKER_RL_ALGORITHM=ENSEMBLE

# Enable ensemble mode (alternative to setting algorithm=ENSEMBLE)
export OI_TRACKER_RL_USE_ENSEMBLE=true

# Model paths (for ensemble mode)
export OI_TRACKER_RL_PPO_MODEL_PATH=models/rl_ppo_model.zip
export OI_TRACKER_RL_DQN_MODEL_PATH=models/rl_dqn_model.zip

# Single model path (for PPO or DQN only mode)
export OI_TRACKER_RL_MODEL_PATH=models/rl_execution_model.zip
```

### Configuration File

Alternatively, you can modify `config.py` directly:

```python
# RL Execution (Phase 2)
rl_execution_enabled: bool = True
rl_algorithm: str = "ENSEMBLE"  # Options: "PPO", "DQN", "ENSEMBLE"
rl_use_ensemble: bool = True
rl_ppo_model_path: str = "models/rl_ppo_model.zip"
rl_dqn_model_path: str = "models/rl_dqn_model.zip"
rl_model_path: str = "models/rl_execution_model.zip"  # For single model mode
```

## Usage Modes

### 1. Single Algorithm Mode (PPO or DQN)

**Configuration:**
```bash
export OI_TRACKER_RL_ALGORITHM=PPO
export OI_TRACKER_RL_MODEL_PATH=models/rl_execution_model.zip
```

**How it works:**
- Uses only one algorithm (PPO or DQN)
- Simpler and faster
- Good for testing individual algorithm performance

### 2. Ensemble Mode (Both PPO and DQN)

**Configuration:**
```bash
export OI_TRACKER_RL_ALGORITHM=ENSEMBLE
# OR
export OI_TRACKER_RL_USE_ENSEMBLE=true

export OI_TRACKER_RL_PPO_MODEL_PATH=models/rl_ppo_model.zip
export OI_TRACKER_RL_DQN_MODEL_PATH=models/rl_dqn_model.zip
```

**How it works:**
- Both PPO and DQN models make predictions
- Predictions are combined using ensemble voting:
  - **Signal (BUY/SELL/HOLD)**: Majority vote
  - **Position Size**: Averaged
  - **Execution Placement**: Averaged price offset, majority vote on aggression

**Benefits:**
- More robust predictions
- Reduces overfitting to single algorithm
- Better generalization

## Code Usage

### In Strategy Router

The `RLStrategy` class automatically uses ensemble mode if configured:

```python
from models.reinforcement_learning import RLStrategy

# Ensemble mode
rl_strategy = RLStrategy(
    exchange="NSE",
    use_ensemble=True,
    ppo_model_path="models/rl_ppo_model.zip",
    dqn_model_path="models/rl_dqn_model.zip"
)

# Single algorithm mode
rl_strategy = RLStrategy(
    exchange="NSE",
    model_path="models/rl_execution_model.zip",
    algorithm="PPO"  # or "DQN"
)
```

### In Auto Executor

The `RLExecutor` class also supports both modes:

```python
from models.reinforcement_learning import RLExecutor

# Ensemble mode
rl_executor = RLExecutor(
    exchange="NSE",
    use_ensemble=True,
    ppo_model_path="models/rl_ppo_model.zip",
    dqn_model_path="models/rl_dqn_model.zip"
)

# Single algorithm mode
rl_executor = RLExecutor(
    exchange="NSE",
    model_path="models/rl_execution_model.zip",
    algorithm="PPO"  # or "DQN"
)
```

## Ensemble Prediction Logic

### For Trading Signals (RLStrategy)

1. Both models predict independently
2. **Signal**: Majority vote (BUY if both say BUY, HOLD on tie)
3. **Position Size**: Average of both predictions

### For Execution (RLExecutor)

1. Both models predict placement details
2. **Price Offset**: Average of both predictions
3. **Aggression**: Majority vote (or average if tied)
4. **Fill Probability**: Average of both estimates

## Performance Considerations

- **Ensemble Mode**: 
  - Slower (2x predictions)
  - More memory usage
  - More robust predictions
  
- **Single Mode**:
  - Faster
  - Less memory
  - Simpler debugging

## Troubleshooting

### Models Not Loading

Check that model files exist:
```bash
ls -la models/rl_*.zip
```

### Ensemble Not Working

Ensure both models are trained and paths are correct:
```bash
# Check config
python -c "from config import get_config; c = get_config(); print(f'PPO: {c.rl_ppo_model_path}, DQN: {c.rl_dqn_model_path}, Ensemble: {c.rl_use_ensemble}')"
```

### Fallback Behavior

If ensemble is enabled but one model fails to load:
- System will use the available model(s)
- Logs will show which models loaded successfully
- No errors will be raised (graceful degradation)

## Example Workflow

1. **Train both models:**
   ```bash
   python train_rl.py --algorithm BOTH
   ```

2. **Enable ensemble mode:**
   ```bash
   export OI_TRACKER_RL_EXECUTION_ENABLED=true
   export OI_TRACKER_RL_USE_ENSEMBLE=true
   export OI_TRACKER_RL_PPO_MODEL_PATH=models/rl_ppo_model.zip
   export OI_TRACKER_RL_DQN_MODEL_PATH=models/rl_dqn_model.zip
   ```

3. **Run the application:**
   ```bash
   python oi_tracker_new.py
   ```

4. **Check logs** for RL initialization:
   ```
   [NSE] PPO model loaded: models/rl_ppo_model.zip
   [NSE] DQN model loaded: models/rl_dqn_model.zip
   [NSE] RL Executor initialized: True (algorithm: ENSEMBLE)
   ```

