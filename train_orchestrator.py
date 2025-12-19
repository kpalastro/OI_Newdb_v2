"""
train_orchestrator.py

Walk-forward AutoML orchestrator that wraps the existing training pipeline.
Implements the roadmap requirement for multi-model evaluation (LightGBM,
XGBoost, CatBoost, RL), Optuna tuning per segment, and consolidated reporting.

Supports:
- Supervised learning: LightGBM, XGBoost, CatBoost
- Reinforcement learning: PPO, DQN (via RL family)
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

# Configure multiprocessing for macOS compatibility (must be done early, before any multiprocessing imports)
# macOS (especially Apple Silicon M1/M2) requires 'spawn' instead of 'fork' to avoid segfaults
try:
    mp.set_start_method('spawn', force=False)
except RuntimeError:
    # Already set by another module (e.g., oi_tracker_new.py), ignore
    pass

# Disable stable-baselines3 multiprocessing to avoid segfaults on macOS
# This uses single-threaded training which is slower but more stable
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import database_new as db
from feature_engineering import FeatureEngineeringError, REQUIRED_FEATURE_COLUMNS, prepare_training_features
from train_model import RegimeHMMTransformer, define_triple_barrier_target
from time_utils import today_ist, now_ist

try:  # Optional dependencies with graceful degradation
    import optuna  # type: ignore
except ImportError:
    optuna = None  # type: ignore

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None  # type: ignore

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None  # type: ignore

try:
    from stable_baselines3 import PPO, DQN
    from models.reinforcement_learning import TradingEnvironment
    # Import gymnasium/gym for environment wrapper
    # stable-baselines3 requires gymnasium, but we fallback to gym for compatibility
    try:
        import gymnasium
        from gymnasium import Env, spaces
        _GymEnv = Env
        gym = gymnasium  # For compatibility with code that uses 'gym'
    except ImportError:
        try:
            import gym
            from gym import Env, spaces
            _GymEnv = Env
        except ImportError:
            gym = None
            _GymEnv = None
            spaces = None
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    PPO = None
    DQN = None
    TradingEnvironment = None
    gym = None
    _GymEnv = None
    spaces = None


try:
    import joblib
except ImportError:
    joblib = None

# Check if progress bar dependencies are available for RL training
try:
    import tqdm
    import rich
    PROGRESS_BAR_AVAILABLE = True
except ImportError:
    PROGRESS_BAR_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    exchange: str
    days: int = 120
    window_days: int = 45
    step_days: int = 15
    families: Sequence[str] = field(default_factory=lambda: ("lightgbm", "xgboost", "catboost"))
    optuna_trials: int = 10
    output: Optional[Path] = None


@dataclass
class SegmentWindow:
    segment_id: int
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime


@dataclass
class SegmentResult:
    segment: SegmentWindow
    family: str
    metrics: Dict[str, float]
    best_params: Dict[str, Any]
    optuna_score: Optional[float]
    sample_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "segment_id": self.segment.segment_id,
            "family": self.family,
            "train_range": {
                "start": self.segment.train_start.isoformat(),
                "end": self.segment.train_end.isoformat(),
            },
            "validation_range": {
                "start": self.segment.val_start.isoformat(),
                "end": self.segment.val_end.isoformat(),
            },
            "metrics": self.metrics,
            "best_params": self.best_params,
            "optuna_score": self.optuna_score,
            "sample_counts": self.sample_counts,
        }
        return payload


class ModelFamily:
    name: str = "base"
    pretty_name: str = "Base"

    @property
    def available(self) -> bool:
        return True

    def default_params(self) -> Dict[str, Any]:
        raise NotImplementedError

    def build_model(self, params: Dict[str, Any]):
        raise NotImplementedError

    def optuna_space(self, trial: "optuna.trial.Trial") -> Dict[str, Any]:
        return {}


class LightGBMFamily(ModelFamily):
    name = "lightgbm"
    pretty_name = "LightGBM"

    @property
    def available(self) -> bool:
        return lgb is not None

    def default_params(self) -> Dict[str, Any]:
        return {
            "objective": "multiclass",
            "num_class": 3,
            "n_estimators": 500,
            "learning_rate": 0.03,
            "num_leaves": 48,
            "max_depth": -1,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        }

    def build_model(self, params: Dict[str, Any]):
        if lgb is None:
            raise RuntimeError("LightGBM not installed.")
        return lgb.LGBMClassifier(**params)

    def optuna_space(self, trial: "optuna.trial.Trial") -> Dict[str, Any]:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08),
            "num_leaves": trial.suggest_int("num_leaves", 24, 96, step=8),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        }


class XGBoostFamily(ModelFamily):
    name = "xgboost"
    pretty_name = "XGBoost"

    @property
    def available(self) -> bool:
        return XGBClassifier is not None

    def default_params(self) -> Dict[str, Any]:
        return {
            "objective": "multi:softprob",
            "num_class": 3,
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": 42,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
        }

    def build_model(self, params: Dict[str, Any]):
        if XGBClassifier is None:
            raise RuntimeError("XGBoost not installed.")
        params = params.copy()
        params["use_label_encoder"] = False
        return XGBClassifier(**params)

    def optuna_space(self, trial: "optuna.trial.Trial") -> Dict[str, Any]:
        return {
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
        }


class CatBoostFamily(ModelFamily):
    name = "catboost"
    pretty_name = "CatBoost"

    @property
    def available(self) -> bool:
        return CatBoostClassifier is not None

    def default_params(self) -> Dict[str, Any]:
        return {
            "loss_function": "MultiClass",
            "iterations": 600,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "random_seed": 42,
            "verbose": False,
            "allow_writing_files": False,
        }

    def build_model(self, params: Dict[str, Any]):
        if CatBoostClassifier is None:
            raise RuntimeError("CatBoost not installed.")
        return CatBoostClassifier(**params)

    def optuna_space(self, trial: "optuna.trial.Trial") -> Dict[str, Any]:
        return {
            "depth": trial.suggest_int("depth", 4, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 5.0),
            "iterations": trial.suggest_int("iterations", 300, 900, step=100),
        }


class RLFamily(ModelFamily):
    """Reinforcement Learning family (PPO/DQN) for signal generation."""
    name = "rl"
    pretty_name = "RL"
    
    def __init__(self, algorithm: str = "PPO"):
        self.algorithm = algorithm.upper()
        self.pretty_name = f"RL-{self.algorithm}"
    
    @property
    def available(self) -> bool:
        return SB3_AVAILABLE and PPO is not None and DQN is not None
    
    def default_params(self) -> Dict[str, Any]:
        if self.algorithm == "PPO":
            return {
                "learning_rate": 0.0003,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "total_timesteps": 10000,
            }
        else:  # DQN
            return {
                "learning_rate": 0.0001,
                "buffer_size": 10000,
                "learning_starts": 1000,
                "batch_size": 32,
                "gamma": 0.99,
                "total_timesteps": 10000,
            }
    
    def build_model(self, params: Dict[str, Any]):
        # This will be called with environment, not built here
        raise NotImplementedError("RL models are built with environment")
    
    def optuna_space(self, trial: "optuna.trial.Trial") -> Dict[str, Any]:
        if self.algorithm == "PPO":
            return {
                "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.001, log=True),
                "n_steps": trial.suggest_int("n_steps", 1024, 4096, step=512),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                "n_epochs": trial.suggest_int("n_epochs", 5, 20),
            }
        else:  # DQN
            return {
                "learning_rate": trial.suggest_float("learning_rate", 0.00005, 0.0005, log=True),
                "buffer_size": trial.suggest_int("buffer_size", 5000, 20000, step=5000),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            }


FAMILY_REGISTRY: Dict[str, ModelFamily] = {
    "lightgbm": LightGBMFamily(),
    "xgboost": XGBoostFamily(),
    "catboost": CatBoostFamily(),
    "rl": RLFamily("PPO"),
    "rl-ppo": RLFamily("PPO"),
    "rl-dqn": RLFamily("DQN"),
}


def _select_families(names: Sequence[str]) -> List[ModelFamily]:
    selected: List[ModelFamily] = []
    for name in names:
        family = FAMILY_REGISTRY.get(name.lower())
        if family is None:
            LOGGER.warning("Unknown model family '%s' – skipping.", name)
            continue
        if not family.available:
            LOGGER.warning("Model family '%s' unavailable (missing dependency).", family.pretty_name)
            continue
        selected.append(family)
    return selected


def _load_dataset(exchange: str, days: int) -> pd.DataFrame:
    """
    Load and prepare dataset WITHOUT applying regime features.
    Regime features will be fitted per segment inside the walk-forward loop
    to prevent look-ahead bias.
    """
    end_date = today_ist()
    start_date = end_date - timedelta(days=days)
    raw = db.load_historical_data_for_ml(exchange, start_date, end_date)
    if raw is None or raw.empty:
        raise RuntimeError(f"No data found for {exchange} in the last {days} days.")

    features = prepare_training_features(raw, required_columns=REQUIRED_FEATURE_COLUMNS)
    target_frame = define_triple_barrier_target(features)
    if target_frame.empty:
        raise RuntimeError("Target preparation yielded no rows.")

    # Do NOT add regime features here - they will be fitted per segment
    return target_frame


def _generate_segments(index: pd.DatetimeIndex, window_days: int, step_days: int) -> List[SegmentWindow]:
    segments: List[SegmentWindow] = []
    if index.empty:
        LOGGER.warning("Index is empty, cannot generate segments")
        return segments

    window = pd.Timedelta(days=window_days)
    step = pd.Timedelta(days=step_days)
    cursor = index.min()
    end_limit = index.max()
    segment_id = 1
    
    LOGGER.debug(f"Generating segments: cursor={cursor}, end_limit={end_limit}, window={window}, step={step}")

    while cursor + window + step <= end_limit:
        train_start = cursor
        train_end = cursor + window
        val_start = train_end
        val_end = train_end + step

        segments.append(
            SegmentWindow(
                segment_id=segment_id,
                train_start=train_start.to_pydatetime(),
                train_end=train_end.to_pydatetime(),
                val_start=val_start.to_pydatetime(),
                val_end=val_end.to_pydatetime(),
            )
        )
        segment_id += 1
        cursor += step
    return segments


def _slice_frame(frame: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    mask = (frame.index >= start) & (frame.index < end)
    return frame.loc[mask].copy()


def _encode_labels(y: np.ndarray) -> np.ndarray:
    """
    Encode labels from [-1, 0, 1] to [0, 1, 2] for XGBoost/CatBoost compatibility.
    LightGBM can handle [-1, 0, 1] directly, but XGBoost requires labels starting from 0.
    """
    y_encoded = y.copy()
    y_encoded[y == -1] = 0
    y_encoded[y == 0] = 1
    y_encoded[y == 1] = 2
    return y_encoded.astype(int)


def _decode_labels(y: np.ndarray) -> np.ndarray:
    """
    Decode labels from [0, 1, 2] back to [-1, 0, 1] for consistency with original target.
    """
    y_decoded = y.copy()
    y_decoded[y == 0] = -1
    y_decoded[y == 1] = 0
    y_decoded[y == 2] = 1
    return y_decoded.astype(int)


def _prepare_xy(frame: pd.DataFrame, features: Sequence[str], encode_labels: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    feature_cols = [col for col in features if col in frame.columns]
    X = frame[feature_cols].values.astype(np.float32)
    y = frame['target'].values.astype(int)
    if encode_labels:
        y = _encode_labels(y)
    return X, y


class GymTradingEnvironmentWrapper(_GymEnv if _GymEnv else object):
    """
    Wrapper to make TradingEnvironment compatible with stable-baselines3 Gym interface.
    Inherits from gymnasium.Env or gym.Env to be recognized by stable-baselines3.
    Uses the same pattern as ExecutionEnvironment in reinforcement_learning.py.
    """
    def __init__(self, trading_env: TradingEnvironment):
        if _GymEnv is None:
            raise ImportError("gymnasium or gym must be installed for RL training")
        
        # Initialize base class
        super().__init__()
        
        self.trading_env = trading_env
        # Define action space: MultiDiscrete for signal (-1,0,1) and position_size (discretized 0-10)
        # Flattened to Discrete(33): 3 signals * 11 position levels
        self.action_space = spaces.Discrete(33)  # 3 signals * 11 position levels
        # Observation space matches TradingEnvironment state
        state_dim = len(trading_env._get_state())
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        # Set metadata for Gymnasium compatibility
        self.metadata = {"render_modes": []}
        self.render_mode = None
    
    def reset(self, seed=None, options=None):
        obs = self.trading_env.reset()
        return obs, {}
    
    def step(self, action):
        # Convert discrete action to RLAction
        # action: 0-32
        # signal: action // 11 -> 0,1,2 -> -1,0,1
        # position_size: (action % 11) / 10.0 -> 0.0 to 1.0
        signal_map = {0: -1, 1: 0, 2: 1}
        signal_idx = action // 11
        position_level = action % 11
        signal = signal_map.get(signal_idx, 0)
        position_size = position_level / 10.0
        
        from models.reinforcement_learning import RLAction
        rl_action = RLAction(signal=signal, position_size=position_size)
        
        obs, reward, done, info = self.trading_env.step(rl_action)
        # Gymnasium format: (obs, reward, terminated, truncated, info)
        # Gym format: (obs, reward, done, info)
        # stable-baselines3 handles both, but prefers Gymnasium format
        return obs, reward, done, False, info
    
    def render(self):
        """Render method required by Gym interface (no-op for training)."""
        return None


def _train_rl_segment(
    family: RLFamily,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    optuna_trials: int,
) -> Tuple[Dict[str, float], Dict[str, Any], Optional[float]]:
    """
    Train RL model on a segment and evaluate on validation data.
    
    Returns:
        Tuple of (metrics, params, optuna_score)
    """
    LOGGER.info(f"Entering _train_rl_segment for {family.algorithm}")
    
    if not SB3_AVAILABLE or TradingEnvironment is None:
        LOGGER.warning("Stable Baselines3 or TradingEnvironment not available, skipping RL training")
        return {"mean_reward": 0.0, "episodes": 0, "std_reward": 0.0, "accuracy": 0.0, "f1_macro": 0.0}, {}, None
    
    LOGGER.info(f"Preparing data for RL training: train={len(train_df)}, val={len(val_df)}, features={len(feature_cols)}")
    
    # Skip RL training if data is too small (RL needs more data and is very slow)
    # With window-days=4, datasets are typically too small for effective RL training
    if len(train_df) < 500:
        LOGGER.warning(
            f"Insufficient training data for RL ({len(train_df)} rows). "
            f"RL needs at least 500 rows for effective training. "
            f"With window-days=4, consider using window-days>=30 for RL training. "
            f"Skipping RL training for this segment."
        )
        return {"mean_reward": 0.0, "episodes": 0, "std_reward": 0.0, "accuracy": 0.0, "f1_macro": 0.0}, {}, None
    
    if len(val_df) < 50:
        LOGGER.warning(
            f"Insufficient validation data for RL ({len(val_df)} rows). "
            f"RL needs at least 50 rows. Skipping RL training for this segment."
        )
        return {"mean_reward": 0.0, "episodes": 0, "std_reward": 0.0, "accuracy": 0.0, "f1_macro": 0.0}, {}, None
    
    # Prepare training data with future returns for reward calculation
    train_df_with_future = train_df.copy()
    if 'future_return' not in train_df_with_future.columns:
        # Calculate future return from target (simplified)
        train_df_with_future['future_return'] = train_df_with_future['target'].astype(float) * 0.01
        LOGGER.info("Calculated future_return from target column")
    
    LOGGER.info(f"Creating TradingEnvironment with {len(train_df_with_future)} rows...")
    # Create base environment
    try:
        base_env = TradingEnvironment(
            exchange=family.name,
            features_df=train_df_with_future[feature_cols + ['future_return']],
            initial_capital=1_000_000.0
        )
        LOGGER.info("TradingEnvironment created successfully")
    except Exception as e:
        LOGGER.error(f"Failed to create TradingEnvironment: {e}", exc_info=True)
        return {"mean_reward": 0.0, "episodes": 0, "std_reward": 0.0, "accuracy": 0.0, "f1_macro": 0.0}, {}, None
    
    # Wrap for Gym compatibility
    LOGGER.info("Wrapping environment for Gym compatibility...")
    try:
        env = GymTradingEnvironmentWrapper(base_env)
        LOGGER.info("Environment wrapped successfully")
    except Exception as e:
        LOGGER.error(f"Failed to wrap environment: {e}", exc_info=True)
        return {"mean_reward": 0.0, "episodes": 0, "std_reward": 0.0, "accuracy": 0.0, "f1_macro": 0.0}, {}, None
    
    base_params = family.default_params()
    algorithm = family.algorithm
    LOGGER.info(f"RL algorithm: {algorithm}, base_params: {base_params}")
    
    # Reduce timesteps for orchestrator (faster training)
    total_timesteps = base_params.pop("total_timesteps", 5000)  # Reduced from 10000
    LOGGER.info(f"Using {total_timesteps} timesteps for RL training")
    
    # Optuna tuning for RL
    tuned_params = base_params.copy()
    optuna_score = None
    
    if optuna_trials > 0 and optuna is not None:
        LOGGER.info("=" * 80)
        LOGGER.info(f"Starting Optuna hyperparameter tuning for {family.algorithm} ({min(optuna_trials, 5)} trials)")
        LOGGER.info(f"RL training will use {total_timesteps} timesteps per trial (reduced for speed)")
        LOGGER.info("=" * 80)
        
        # Flush logs
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        
        def objective(trial: "optuna.trial.Trial") -> float:
            try:
                LOGGER.info("=" * 80)
                LOGGER.info(f"RL Optuna trial {trial.number}: Starting...")
                LOGGER.info("=" * 80)
                sys.stdout.flush()
                sys.stderr.flush()
                
                params = base_params.copy()
                params.update(family.optuna_space(trial))
                timesteps = params.pop("total_timesteps", total_timesteps)
                LOGGER.info(f"RL Optuna trial {trial.number}: Parameters: {params}, timesteps: {timesteps}")
                sys.stdout.flush()
                
                # Create fresh environment for each trial
                LOGGER.info(f"RL Optuna trial {trial.number}: Creating environment...")
                sys.stdout.flush()
                trial_base_env = TradingEnvironment(
                    exchange=family.name,
                    features_df=train_df_with_future[feature_cols + ['future_return']],
                    initial_capital=1_000_000.0
                )
                trial_env = GymTradingEnvironmentWrapper(trial_base_env)
                LOGGER.info(f"RL Optuna trial {trial.number}: Environment created")
                sys.stdout.flush()
                
                LOGGER.info(f"RL Optuna trial {trial.number}: Creating {algorithm} model...")
                sys.stdout.flush()
                
                if algorithm == "PPO":
                    # Disable verbose and use single process to avoid multiprocessing issues
                    model = PPO("MlpPolicy", trial_env, verbose=0, **params)
                else:  # DQN
                    # Disable verbose and use single process to avoid multiprocessing issues
                    model = DQN("MlpPolicy", trial_env, verbose=0, **params)
                
                LOGGER.info(f"RL Optuna trial {trial.number}: Model created. Starting training for {timesteps} timesteps...")
                LOGGER.info(f"RL Optuna trial {trial.number}: This may take 1-5 minutes. Please wait...")
                sys.stdout.flush()
                sys.stderr.flush()
                
                import time
                start_time = time.time()
                
                # Train the model (disable progress bar to reduce multiprocessing overhead)
                try:
                    model.learn(total_timesteps=timesteps, progress_bar=False, log_interval=100)
                except Exception as e:
                    LOGGER.error(f"RL Optuna trial {trial.number}: model.learn() failed: {e}", exc_info=True)
                    # Clean up training resources on training failure
                    try:
                        del model
                        del trial_env
                        del trial_base_env
                        import gc
                        gc.collect()
                    except:
                        pass
                    raise
                
                elapsed = time.time() - start_time
                LOGGER.info(f"RL Optuna trial {trial.number}: Training completed in {elapsed:.1f} seconds")
                sys.stdout.flush()
                
                # Evaluate on validation
                LOGGER.info(f"RL Optuna trial {trial.number}: Starting evaluation...")
                val_df_with_future = val_df.copy()
                if 'future_return' not in val_df_with_future.columns:
                    val_df_with_future['future_return'] = val_df_with_future['target'].astype(float) * 0.01
                
                val_base_env = TradingEnvironment(
                    exchange=family.name,
                    features_df=val_df_with_future[feature_cols + ['future_return']],
                    initial_capital=1_000_000.0
                )
                val_env = GymTradingEnvironmentWrapper(val_base_env)
                
                # Run evaluation episodes
                total_reward = 0.0
                episodes = 0
                max_steps_per_episode = min(len(val_df), 1000)  # Safety limit
                
                try:
                    for episode_num in range(5):  # 5 evaluation episodes
                        obs, _ = val_env.reset()
                        episode_reward = 0.0
                        done = False
                        truncated = False
                        steps = 0
                        
                        while not (done or truncated) and steps < max_steps_per_episode:
                            action, _ = model.predict(obs, deterministic=True)
                            obs, reward, done, truncated, _ = val_env.step(action)
                            episode_reward += reward
                            steps += 1
                        
                        if steps >= max_steps_per_episode:
                            LOGGER.warning(f"RL Optuna trial episode {episode_num} hit step limit ({max_steps_per_episode})")
                        
                        total_reward += episode_reward
                        episodes += 1
                    
                    avg_reward = total_reward / max(episodes, 1)
                    LOGGER.info(f"RL Optuna trial {trial.number}: Completed with avg reward: {avg_reward:.3f}")
                    return avg_reward
                finally:
                    # Clean up resources after evaluation is complete
                    try:
                        del model
                        del trial_env
                        del trial_base_env
                        del val_env
                        del val_base_env
                        import gc
                        gc.collect()
                    except:
                        pass
            except Exception as e:
                LOGGER.error(f"RL Optuna trial {trial.number} failed: {e}", exc_info=True)
                return -1.0  # Return negative reward for failed trials
        
        study = optuna.create_study(direction='maximize')
        try:
            # Timeout: 10 minutes per trial * 5 trials = 50 minutes max
            # But we'll set a per-trial timeout of 15 minutes
            LOGGER.info(f"Starting RL Optuna optimization with {min(optuna_trials, 5)} trials...")
            LOGGER.info("NOTE: RL training can be slow. Each trial may take 1-5 minutes.")
            
            # Use a shorter timeout for faster feedback
            study.optimize(
                objective, 
                n_trials=min(optuna_trials, 5), 
                show_progress_bar=False, 
                timeout=600  # 10 min total timeout (reduced from 15)
            )
            LOGGER.info("RL Optuna optimization completed")
        except KeyboardInterrupt:
            LOGGER.warning("RL Optuna optimization interrupted by user")
            raise
        except Exception as e:
            LOGGER.error(f"RL Optuna optimization failed: {e}", exc_info=True)
        
        if study.best_params:
            tuned_params.update(study.best_params)
            optuna_score = float(study.best_value) if study.best_value is not None else None
            LOGGER.info(f"RL Optuna best score: {optuna_score:.3f}")
        else:
            LOGGER.warning("RL Optuna optimization produced no results, using default params")
    
    # Train final model with tuned params
    timesteps = tuned_params.pop("total_timesteps", total_timesteps) if "total_timesteps" in tuned_params else total_timesteps
    
    LOGGER.info(f"Training final {family.algorithm} model for {timesteps} timesteps...")
    
    try:
        LOGGER.info(f"Creating final {algorithm} model with params: {tuned_params}")
        if algorithm == "PPO":
            model = PPO("MlpPolicy", env, verbose=0, **tuned_params)
        else:  # DQN
            model = DQN("MlpPolicy", env, verbose=0, **tuned_params)
        
        import time
        start_time = time.time()
        LOGGER.info(f"Starting final model training (this may take several minutes)...")
        model.learn(total_timesteps=timesteps, progress_bar=False, log_interval=100)
        elapsed = time.time() - start_time
        LOGGER.info(f"Final {family.algorithm} model training complete in {elapsed:.1f} seconds")
        
        # Clean up after training
        import gc
        gc.collect()
    except Exception as e:
        LOGGER.error(f"RL model training failed: {e}")
        # Return default metrics on failure
        return {"mean_reward": 0.0, "episodes": 0, "std_reward": 0.0, "accuracy": 0.0, "f1_macro": 0.0}, tuned_params, optuna_score
    
    # Evaluate on validation
    val_df_with_future = val_df.copy()
    if 'future_return' not in val_df_with_future.columns:
        val_df_with_future['future_return'] = val_df_with_future['target'].astype(float) * 0.01
    
    val_base_env = TradingEnvironment(
        exchange=family.name,
        features_df=val_df_with_future[feature_cols + ['future_return']],
        initial_capital=1_000_000.0
    )
    val_env = GymTradingEnvironmentWrapper(val_base_env)
    
    # Run evaluation
    total_reward = 0.0
    episodes = 0
    episode_rewards = []
    max_steps_per_episode = min(len(val_df), 1000)  # Safety limit
    
    LOGGER.info(f"Starting RL evaluation: {family.algorithm} on {len(val_df)} validation samples")
    
    for episode_num in range(10):  # 10 evaluation episodes
        obs, _ = val_env.reset()
        episode_reward = 0.0
        done = False
        truncated = False
        steps = 0
        
        while not (done or truncated) and steps < max_steps_per_episode:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = val_env.step(action)
            episode_reward += reward
            steps += 1
            
            # Log progress every 100 steps
            if steps % 100 == 0:
                LOGGER.debug(f"RL eval episode {episode_num}: step {steps}/{max_steps_per_episode}, reward={episode_reward:.2f}")
        
        if steps >= max_steps_per_episode:
            LOGGER.warning(f"RL evaluation episode {episode_num} hit step limit ({max_steps_per_episode})")
        
        episode_rewards.append(episode_reward)
        total_reward += episode_reward
        episodes += 1
        
        if (episode_num + 1) % 5 == 0:
            LOGGER.info(f"RL evaluation: completed {episode_num + 1}/10 episodes, avg reward so far: {total_reward / episodes:.3f}")
    
    LOGGER.info(f"RL evaluation complete: {episodes} episodes, mean reward: {total_reward / max(episodes, 1):.3f}")
    
    mean_reward = total_reward / max(episodes, 1)
    std_reward = np.std(episode_rewards) if episode_rewards else 0.0
    
    # Convert to classification-like metrics for consistency
    # Use reward as proxy for performance
    metrics = {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "episodes": episodes,
        "accuracy": float(np.clip((mean_reward + 1) / 2, 0, 1)),  # Normalize reward to [0,1]
        "f1_macro": float(np.clip(mean_reward, 0, 1)),  # Use reward as proxy
    }
    
    return metrics, tuned_params, optuna_score


def _run_optuna(
    family: ModelFamily,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    base_params: Dict[str, Any],
    trials: int,
) -> Tuple[Dict[str, Any], Optional[float]]:
    if trials <= 0 or optuna is None:
        return base_params, None

    # XGBoost and CatBoost require labels starting from 0
    needs_encoding = family.name in ("xgboost", "catboost")
    y_train_encoded = _encode_labels(y_train) if needs_encoding else y_train
    y_val_encoded = _encode_labels(y_val) if needs_encoding else y_val

    def objective(trial: "optuna.trial.Trial") -> float:
        params = base_params.copy()
        params.update(family.optuna_space(trial))
        model = family.build_model(params)
        model.fit(X_train, y_train_encoded)
        preds = model.predict(X_val)
        # Decode predictions if we encoded labels
        if needs_encoding:
            preds = _decode_labels(preds)
        score = f1_score(y_val, preds, average='macro')
        return float(score)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    tuned_params = base_params.copy()
    tuned_params.update(study.best_params)
    return tuned_params, float(study.best_value)


def run_orchestrator(config: OrchestratorConfig) -> Dict[str, Any]:
    frame = _load_dataset(config.exchange, config.days)
    frame.sort_index(inplace=True)
    
    # Diagnostic info
    if frame.empty:
        raise RuntimeError(f"No data loaded for {config.exchange}")
    
    date_range = frame.index.max() - frame.index.min()
    required_range = pd.Timedelta(days=config.window_days + config.step_days)
    
    LOGGER.info(
        f"Dataset: {len(frame)} rows, date range: {date_range.days} days "
        f"(from {frame.index.min()} to {frame.index.max()})"
    )
    LOGGER.info(
        f"Required range for segments: {required_range.days} days "
        f"(window={config.window_days}, step={config.step_days})"
    )
    
    segments = _generate_segments(frame.index, config.window_days, config.step_days)

    if not segments:
        raise RuntimeError(
            f"Walk-forward segmentation produced zero windows. "
            f"Data range: {date_range.days} days, Required: {required_range.days} days. "
            f"Increase --days (current: {config.days}) or reduce --window-days (current: {config.window_days}) "
            f"or --step-days (current: {config.step_days})."
        )

    families = _select_families(config.families)
    if not families:
        raise RuntimeError("No model families available. Install LightGBM/XGBoost/CatBoost or adjust flags.")

    feature_cols = [col for col in REQUIRED_FEATURE_COLUMNS if col in frame.columns]
    results: List[SegmentResult] = []

    for segment in segments:
        # 1. Slice Raw Data first (without regime features)
        train_raw = _slice_frame(frame, segment.train_start, segment.train_end)
        val_raw = _slice_frame(frame, segment.val_start, segment.val_end)

        if len(train_raw) < 200 or len(val_raw) < 50:
            LOGGER.warning(
                "Segment %s skipped due to insufficient samples (train=%s, val=%s).",
                segment.segment_id, len(train_raw), len(val_raw)
            )
            continue

        # 2. Fit HMM on Train Raw ONLY (Prevents Leakage)
        hmm_transformer = RegimeHMMTransformer(n_components=4)
        hmm_transformer.fit(train_raw)
        
        # 3. Generate Regimes
        train_regimes = hmm_transformer.transform(train_raw).flatten()
        val_regimes = hmm_transformer.transform(val_raw).flatten()
        
        # 4. Assign regimes to dataframes
        train_df = train_raw.copy()
        val_df = val_raw.copy()
        train_df['regime'] = train_regimes
        val_df['regime'] = val_regimes
        
        # 5. Add regime as a feature for training
        # Include 'regime' in features for this segment
        feature_cols_with_regime = list(feature_cols) + (['regime'] if 'regime' not in feature_cols else [])
        
        X_train, y_train = _prepare_xy(train_df, feature_cols_with_regime)
        X_val, y_val = _prepare_xy(val_df, feature_cols_with_regime)

        for family in families:
            if family.name == "rl":
                # RL training is different - uses environment
                LOGGER.info("=" * 80)
                LOGGER.info(
                    "Segment %s | %s | Starting RL training (train=%d, val=%d samples)...",
                    segment.segment_id, family.pretty_name, len(train_df), len(val_df)
                )
                LOGGER.info("=" * 80)
                
                # Flush logs to ensure they're visible
                import sys
                sys.stdout.flush()
                sys.stderr.flush()
                
                try:
                    metrics, tuned_params, optuna_score = _train_rl_segment(
                        family, train_df, val_df, feature_cols_with_regime, config.optuna_trials
                    )
                    LOGGER.info(
                        "Segment %s | %s | RL training completed: reward=%.3f",
                        segment.segment_id, family.pretty_name, metrics.get("mean_reward", 0.0)
                    )
                except KeyboardInterrupt:
                    LOGGER.warning(f"Segment {segment.segment_id} | {family.pretty_name} | RL training interrupted by user")
                    raise
                except Exception as e:
                    LOGGER.error(f"Segment {segment.segment_id} | {family.pretty_name} | RL training failed: {e}", exc_info=True)
                    # Return default metrics on failure
                    metrics = {"mean_reward": 0.0, "episodes": 0, "std_reward": 0.0, "accuracy": 0.0, "f1_macro": 0.0}
                    tuned_params = {}
                    optuna_score = None
                sample_counts = {"train": len(train_df), "validation": len(val_df)}
                result = SegmentResult(segment, family.pretty_name, metrics, tuned_params, optuna_score, sample_counts)
                results.append(result)
                LOGGER.info(
                    "Segment %s | %s | Reward %.3f | Episodes %d",
                    segment.segment_id, family.pretty_name, 
                    metrics.get("mean_reward", 0.0), metrics.get("episodes", 0)
                )
            else:
                # XGBoost and CatBoost require labels starting from 0
                needs_encoding = family.name in ("xgboost", "catboost")
                y_train_encoded = _encode_labels(y_train) if needs_encoding else y_train
                
                base_params = family.default_params()
                tuned_params, optuna_score = _run_optuna(family, X_train, y_train, X_val, y_val, base_params, config.optuna_trials)

                model = family.build_model(tuned_params)
                model.fit(X_train, y_train_encoded)
                preds = model.predict(X_val)
                
                # Decode predictions if we encoded labels
                if needs_encoding:
                    preds = _decode_labels(preds)

                report = classification_report(y_val, preds, zero_division=0, output_dict=True)
                metrics = {
                    "accuracy": float(report.get("accuracy", 0.0)),
                    "f1_macro": float(report.get("macro avg", {}).get("f1-score", 0.0)),
                    "f1_weighted": float(report.get("weighted avg", {}).get("f1-score", 0.0)),
                    "precision_macro": float(report.get("macro avg", {}).get("precision", 0.0)),
                    "recall_macro": float(report.get("macro avg", {}).get("recall", 0.0)),
                }

                sample_counts = {"train": len(train_df), "validation": len(val_df)}
                result = SegmentResult(segment, family.pretty_name, metrics, tuned_params, optuna_score, sample_counts)
                results.append(result)
                LOGGER.info(
                    "Segment %s | %s | Acc %.3f | F1 %.3f",
                    segment.segment_id, family.pretty_name, metrics["accuracy"], metrics["f1_macro"]
                )

    if not results:
        raise RuntimeError("No successful segments were evaluated.")

    best_by_family: Dict[str, Dict[str, Any]] = {}
    for family in families:
        family_results = [res for res in results if res.family == family.pretty_name]
        if not family_results:
            continue
        # RL uses mean_reward, others use f1_macro
        if family.name == "rl":
            best_segment = max(family_results, key=lambda r: r.metrics.get("mean_reward", 0.0))
        else:
            best_segment = max(family_results, key=lambda r: r.metrics.get("f1_macro", 0.0))
        best_by_family[family.pretty_name] = {
            "segment_id": best_segment.segment.segment_id,
            "metrics": best_segment.metrics,
            "params": best_segment.best_params,
            "optuna_score": best_segment.optuna_score,
        }

    summary = {
        "exchange": config.exchange,
        "generated_at": now_ist().isoformat(),
        "dataset": {
            "rows": int(len(frame)),
            "start": frame.index.min().isoformat(),
            "end": frame.index.max().isoformat(),
            "feature_count": len(feature_cols),
        },
        "config": {
            "days": config.days,
            "window_days": config.window_days,
            "step_days": config.step_days,
            "families": [family.pretty_name for family in families],
            "optuna_trials": config.optuna_trials if optuna is not None else 0,
        },
        "segments_evaluated": len(results),
        "segments": [res.to_dict() for res in results],
        "best_by_family": best_by_family,
    }

    report_path = config.output
    if report_path is None:
        model_dir = Path("models") / config.exchange / "reports"
        model_dir.mkdir(parents=True, exist_ok=True)
        report_path = model_dir / "auto_ml_summary.json"
    else:
        report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    LOGGER.info("✓ AutoML summary saved to %s", report_path)

    # Save the best trained model
    LOGGER.info("=" * 80)
    LOGGER.info("Initiating model save process after walk-forward evaluation...")
    LOGGER.info("=" * 80)
    try:
        _save_best_model(config, frame, feature_cols, best_by_family, families)
    except Exception as e:
        LOGGER.error(f"CRITICAL: Model save process failed with exception: {e}", exc_info=True)
        LOGGER.warning("Walk-forward evaluation completed, but model save failed. Check logs above for details.")

    return summary


def _save_best_model(
    config: OrchestratorConfig,
    frame: pd.DataFrame,
    feature_cols: List[str],
    best_by_family: Dict[str, Dict[str, Any]],
    families: List[ModelFamily],
):
    """
    Train and save the best model on all available data.
    
    Args:
        config: Orchestrator configuration
        frame: Full dataset DataFrame
        feature_cols: List of feature column names
        best_by_family: Dictionary with best model info per family
        families: List of model families that were evaluated
    """
    LOGGER.info("=" * 80)
    LOGGER.info("Starting model save process...")
    LOGGER.info(f"best_by_family keys: {list(best_by_family.keys())}")
    LOGGER.info(f"Available families: {[f.pretty_name for f in families]}")
    
    if not best_by_family:
        LOGGER.warning("No best models found in best_by_family, skipping model save")
        return
    
    # Determine overall best family (by f1_macro or mean_reward)
    best_family_name = None
    best_score = -float('inf')
    
    for family_name, info in best_by_family.items():
        metrics = info.get("metrics", {})
        # RL uses mean_reward, others use f1_macro
        if family_name.startswith("RL") or "RL" in family_name:
            score = metrics.get("mean_reward", 0.0)
            LOGGER.debug(f"Family {family_name}: score={score} (mean_reward)")
        else:
            score = metrics.get("f1_macro", 0.0)
            LOGGER.debug(f"Family {family_name}: score={score} (f1_macro)")
        
        if score > best_score:
            best_score = score
            best_family_name = family_name
    
    if not best_family_name:
        LOGGER.warning("Could not determine best family, skipping model save")
        LOGGER.warning(f"best_by_family contents: {best_by_family}")
        return
    
    LOGGER.info(f"Selected best family: {best_family_name} with score: {best_score:.4f}")
    
    # Find the corresponding ModelFamily object
    best_family = None
    for family in families:
        if family.pretty_name == best_family_name:
            best_family = family
            break
    
    if not best_family:
        LOGGER.warning(f"Could not find ModelFamily for {best_family_name}, skipping model save")
        return
    
    best_info = best_by_family[best_family_name]
    best_params = best_info.get("params", {})
    
    LOGGER.info("=" * 80)
    LOGGER.info(f"Training final production model: {best_family_name}")
    LOGGER.info(f"Best score: {best_score:.4f}")
    LOGGER.info(f"Best parameters: {best_params}")
    LOGGER.info(f"Dataframe shape: {frame.shape}")
    LOGGER.info(f"Feature columns count: {len(feature_cols)}")
    LOGGER.info("=" * 80)
    
    try:
        if frame.empty:
            LOGGER.error("Dataframe is empty, cannot train final model")
            return
        # Prepare data for final training (use all available data)
        # Apply HMM regime transformation
        hmm_transformer = RegimeHMMTransformer(n_components=4)
        hmm_transformer.fit(frame)
        frame_with_regime = frame.copy()
        regimes = hmm_transformer.transform(frame).flatten()
        frame_with_regime['regime'] = regimes
        
        # Include regime in features
        feature_cols_with_regime = list(feature_cols) + (['regime'] if 'regime' not in feature_cols else [])
        
        # Verify we have data
        if len(frame_with_regime) == 0:
            LOGGER.error("No data available after regime transformation")
            return
        
        X_all, y_all = _prepare_xy(frame_with_regime, feature_cols_with_regime)
        
        if len(X_all) == 0 or len(y_all) == 0:
            LOGGER.error(f"No training data available (X_all={len(X_all)}, y_all={len(y_all)})")
            return
        
        LOGGER.info(f"Prepared training data: X shape={X_all.shape}, y shape={y_all.shape}")
        
        # Encode labels for XGBoost/CatBoost
        needs_encoding = best_family.name in ("xgboost", "catboost")
        y_all_encoded = _encode_labels(y_all) if needs_encoding else y_all
        
        # Train final model
        if best_family.name == "rl":
            # For RL models, save the model object if available
            # Note: RL models are typically saved by train_rl.py, but we can save parameters
            LOGGER.warning("RL models are saved separately by train_rl.py. Cannot save RL model here.")
            LOGGER.info("RL model parameters are saved in the auto_ml_summary.json report.")
            return
        else:
            LOGGER.info(f"Training {best_family_name} model on all {len(X_all)} samples...")
            # Train tree-based model on all data
            final_model = best_family.build_model(best_params)
            final_model.fit(X_all, y_all_encoded)
            
            # Save model artifacts
            model_dir = Path("models") / config.exchange
            model_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"Model directory: {model_dir.absolute()}")
            
            # Save the trained model
            if joblib is None:
                LOGGER.error("joblib not available, cannot save model")
                LOGGER.error("Please install joblib: pip install joblib")
                return
            
            # Save model with family-specific filename
            model_filename_map = {
                "lightgbm": "lightgbm_classifier.pkl",
                "xgboost": "xgboost_classifier.pkl",
                "catboost": "catboost_classifier.pkl",
            }
            model_filename = model_filename_map.get(best_family.name, f"{best_family.name}_classifier.pkl")
            model_path = model_dir / model_filename
            
            # Save the trained model
            try:
                joblib.dump(final_model, model_path)
                if model_path.exists():
                    file_size = model_path.stat().st_size / 1024  # KB
                    LOGGER.info(f"✓ Saved {best_family_name} model to {model_path} ({file_size:.1f} KB)")
                else:
                    LOGGER.error(f"✗ Model file was not created: {model_path}")
            except Exception as e:
                LOGGER.error(f"✗ Failed to save model to {model_path}: {e}", exc_info=True)
                raise
            
            # Save feature columns
            feature_path = model_dir / "model_features.pkl"
            try:
                joblib.dump(feature_cols_with_regime, feature_path)
                LOGGER.info(f"✓ Saved feature columns to {feature_path}")
            except Exception as e:
                LOGGER.error(f"✗ Failed to save feature columns: {e}", exc_info=True)
                raise
            
            # Save HMM transformer for regime detection
            hmm_path = model_dir / "hmm_regime_model.pkl"
            try:
                joblib.dump(hmm_transformer.model, hmm_path)
                LOGGER.info(f"✓ Saved HMM regime model to {hmm_path}")
            except Exception as e:
                LOGGER.error(f"✗ Failed to save HMM model: {e}", exc_info=True)
                raise
            
            # Save swing_ensemble.pkl for multi_horizon_ensemble.py compatibility
            swing_ensemble_data = {
                'models': {best_family.name: final_model},
                'weights': {best_family.name: 1.0},
                '_is_fitted': True
            }
            swing_path = model_dir / "swing_ensemble.pkl"
            try:
                joblib.dump(swing_ensemble_data, swing_path)
                LOGGER.info(f"✓ Saved swing_ensemble.pkl to {swing_path}")
            except Exception as e:
                LOGGER.error(f"✗ Failed to save swing_ensemble: {e}", exc_info=True)
                raise
            
            # Save training metadata
            metadata = {
                "exchange": config.exchange,
                "model_family": best_family_name,
                "best_score": best_score,
                "best_params": best_params,
                "training_date": now_ist().isoformat(),
                "feature_count": len(feature_cols_with_regime),
                "total_samples": len(X_all),
            }
            metadata_path = model_dir / "training_metadata.json"
            try:
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                LOGGER.info(f"✓ Saved training metadata to {metadata_path}")
            except Exception as e:
                LOGGER.error(f"✗ Failed to save metadata: {e}", exc_info=True)
                raise
            
            # Verify all files were saved
            saved_files = [
                model_path,
                feature_path,
                hmm_path,
                swing_path,
                metadata_path
            ]
            missing_files = [f for f in saved_files if not f.exists()]
            if missing_files:
                LOGGER.error(f"✗ Some files were not saved: {missing_files}")
            else:
                LOGGER.info("=" * 80)
                LOGGER.info(f"✓ All model artifacts successfully saved to {model_dir.absolute()}")
                LOGGER.info(f"  - Model: {model_path.name}")
                LOGGER.info(f"  - Features: {feature_path.name}")
                LOGGER.info(f"  - HMM: {hmm_path.name}")
                LOGGER.info(f"  - Swing Ensemble: {swing_path.name}")
                LOGGER.info(f"  - Metadata: {metadata_path.name}")
                LOGGER.info("=" * 80)
        
    except Exception as e:
        LOGGER.error(f"Error saving best model: {e}", exc_info=True)
        LOGGER.warning("Walk-forward evaluation completed, but model save failed")


def parse_args() -> OrchestratorConfig:
    parser = argparse.ArgumentParser(description="Walk-forward AutoML orchestrator for OI Gemini.")
    parser.add_argument("--exchange", required=True, choices=["NSE", "BSE"])
    parser.add_argument("--days", type=int, default=150, help="Total lookback window in days.")
    parser.add_argument("--window-days", type=int, default=30, help="Training window size for each segment.")
    parser.add_argument("--step-days", type=int, default=7, help="Step size / validation horizon in days.")
    parser.add_argument("--families", nargs="+", default=["lightgbm", "xgboost", "catboost"],
                        help="Model families to evaluate. Options: lightgbm, xgboost, catboost, rl, rl-ppo, rl-dqn")
    parser.add_argument("--optuna-trials", type=int, default=10, help="Trials per segment (0 to skip).")
    parser.add_argument("--output", type=Path, default=None, help="Optional override path for the JSON summary.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")
    return OrchestratorConfig(
        exchange=args.exchange,
        days=args.days,
        window_days=args.window_days,
        step_days=args.step_days,
        families=args.families,
        optuna_trials=args.optuna_trials,
        output=args.output,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_orchestrator(cfg)

