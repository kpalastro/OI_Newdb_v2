"""
Reinforcement learning framework for OI Gemini (Phase 2).

Provides infrastructure for RL-based trading strategies and execution optimization.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.env_util import make_vec_env
    from gymnasium import spaces  # Use gymnasium if available, else gym
    import gymnasium as gym
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    PPO = None
    DQN = None
    make_vec_env = None
    spaces = None
    gym = None

LOGGER = logging.getLogger(__name__)


@dataclass
class RLState:
    """State representation for RL agent."""
    features: np.ndarray
    current_position: float  # Current position size (-1 to 1)
    portfolio_value: float
    timestamp: str


@dataclass
class RLAction:
    """Action space for RL agent."""
    signal: int  # -1=SELL, 0=HOLD, 1=BUY
    position_size: float  # Fraction of capital (0 to 1)


class TradingEnvironment:
    """
    Gym-style environment for RL training using backtest engine.
    """
    
    def __init__(
        self,
        exchange: str,
        features_df: Any,  # pd.DataFrame
        initial_capital: float = 1_000_000.0,
    ):
        self.exchange = exchange
        self.features_df = features_df
        self.initial_capital = initial_capital
        self.current_step = 0
        self.current_position = 0.0
        self.portfolio_value = initial_capital
        self.done = False
        
        if features_df is None or len(features_df) == 0:
            raise ValueError("Features dataframe is empty")
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = 0
        self.current_position = 0.0
        self.portfolio_value = self.initial_capital
        self.done = False
        return self._get_state()
    
    def step(self, action: RLAction) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return next state, reward, done, info.
        
        Args:
            action: RLAction with signal and position_size
        
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.done:
            return self._get_state(), 0.0, True, {}
        
        # Get current price and future return
        current_row = self.features_df.iloc[self.current_step]
        # future_return = current_row.get('future_return', 0.0) # Unused var
        # current_price = current_row.get('underlying_price', 0.0) # Unused var
        
        # Calculate PnL from action
        position_change = action.position_size - self.current_position
        if action.signal != 0 and position_change != 0:
            # Transaction cost (assume 0.02% per trade)
            transaction_cost = abs(position_change) * self.portfolio_value * 0.0002
            self.portfolio_value -= transaction_cost
        
        # Update position
        self.current_position = action.position_size
        
        # Calculate reward (PnL normalized by capital)
        if self.current_step + 1 < len(self.features_df):
            next_return = self.features_df.iloc[self.current_step + 1].get('future_return', 0.0)
            pnl = self.current_position * next_return * self.portfolio_value
            reward = pnl / self.initial_capital  # Normalized reward
        else:
            reward = 0.0
        
        self.current_step += 1
        self.done = self.current_step >= len(self.features_df) - 1
        
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.current_position,
            'step': self.current_step,
        }
        
        return self._get_state(), reward, self.done, info
    
    def _get_state(self) -> np.ndarray:
        """Extract state vector from current features."""
        if self.current_step >= len(self.features_df):
            return np.zeros(50)  # Default state size
        
        row = self.features_df.iloc[self.current_step]
        # Extract numeric features (exclude timestamp, target, etc.)
        feature_cols = [col for col in self.features_df.columns 
                       if col not in ['timestamp', 'target', 'future_return']]
        state = row[feature_cols].values.astype(float)
        
        # Add position and portfolio info
        state = np.append(state, [self.current_position, self.portfolio_value / self.initial_capital])
        
        return state


class RLStrategy:
    """Wrapper for RL-based trading strategy. Supports both PPO and DQN algorithms."""
    
    def __init__(
        self,
        exchange: str,
        model_path: Optional[str] = None,
        algorithm: str = "PPO",
        ppo_model_path: Optional[str] = None,
        dqn_model_path: Optional[str] = None,
        use_ensemble: bool = False,
    ):
        self.exchange = exchange
        self.algorithm = algorithm.upper()
        self.use_ensemble = use_ensemble
        self.model = None
        self.ppo_model = None
        self.dqn_model = None
        self.model_loaded = False
        self.ppo_loaded = False
        self.dqn_loaded = False
        
        if not SB3_AVAILABLE:
            LOGGER.warning(f"[{exchange}] Stable Baselines3 not available, RL disabled")
            return
        
        # Load models based on configuration
        if use_ensemble:
            # Load both PPO and DQN for ensemble
            if ppo_model_path:
                self._load_model(ppo_model_path, "PPO")
            if dqn_model_path:
                self._load_model(dqn_model_path, "DQN")
            self.model_loaded = self.ppo_loaded or self.dqn_loaded
        elif model_path:
            # Single model mode (backward compatible)
            self._load_model(model_path, self.algorithm)
            self.model_loaded = True
    
    def _load_model(self, model_path: str, algorithm: str) -> None:
        """Load trained RL model."""
        try:
            if algorithm.upper() == "PPO":
                self.ppo_model = PPO.load(model_path)
                self.ppo_loaded = True
                if not self.use_ensemble:
                    self.model = self.ppo_model
                LOGGER.info(f"[{self.exchange}] PPO model loaded: {model_path}")
            elif algorithm.upper() == "DQN":
                self.dqn_model = DQN.load(model_path)
                self.dqn_loaded = True
                if not self.use_ensemble:
                    self.model = self.dqn_model
                LOGGER.info(f"[{self.exchange}] DQN model loaded: {model_path}")
            else:
                LOGGER.error(f"[{self.exchange}] Unknown RL algorithm: {algorithm}")
                return
        except Exception as e:
            LOGGER.error(f"[{self.exchange}] Failed to load {algorithm} model: {e}")
    
    def predict(self, state: np.ndarray) -> RLAction:
        """
        Generate action from state. Uses ensemble if both models are loaded.
        
        Args:
            state: State vector
        
        Returns:
            RLAction with signal and position_size
        """
        if not self.model_loaded:
            return RLAction(signal=0, position_size=0.0)
        
        # Ensemble mode: combine predictions from both models
        if self.use_ensemble and (self.ppo_loaded or self.dqn_loaded):
            return self._ensemble_predict(state)
        
        # Single model mode
        if self.model is None:
            return RLAction(signal=0, position_size=0.0)
        
        try:
            action, _ = self.model.predict(state, deterministic=True)
            return self._parse_action(action)
        except Exception as e:
            LOGGER.error(f"[{self.exchange}] RL prediction error: {e}")
            return RLAction(signal=0, position_size=0.0)
    
    def _ensemble_predict(self, state: np.ndarray) -> RLAction:
        """Combine predictions from both PPO and DQN models."""
        actions = []
        
        # Get PPO prediction
        if self.ppo_loaded:
            try:
                action, _ = self.ppo_model.predict(state, deterministic=True)
                actions.append(self._parse_action(action))
            except Exception as e:
                LOGGER.debug(f"[{self.exchange}] PPO prediction failed: {e}")
        
        # Get DQN prediction
        if self.dqn_loaded:
            try:
                action, _ = self.dqn_model.predict(state, deterministic=True)
                actions.append(self._parse_action(action))
            except Exception as e:
                LOGGER.debug(f"[{self.exchange}] DQN prediction failed: {e}")
        
        if not actions:
            return RLAction(signal=0, position_size=0.0)
        
        # Ensemble voting: average position size, majority vote on signal
        if len(actions) == 1:
            return actions[0]
        
        # Combine signals (majority vote)
        signals = [a.signal for a in actions]
        buy_votes = sum(1 for s in signals if s == 1)
        sell_votes = sum(1 for s in signals if s == -1)
        
        if buy_votes > sell_votes:
            ensemble_signal = 1
        elif sell_votes > buy_votes:
            ensemble_signal = -1
        else:
            ensemble_signal = 0  # HOLD on tie
        
        # Average position sizes
        ensemble_position_size = np.mean([a.position_size for a in actions])
        
        return RLAction(signal=ensemble_signal, position_size=ensemble_position_size)
    
    def _parse_action(self, action: Any) -> RLAction:
        """Parse action from model output to RLAction."""
        if isinstance(action, (list, np.ndarray)):
            signal = int(action[0]) if len(action) > 0 else 0
            position_size = float(action[1]) if len(action) > 1 else 0.0
        else:
            signal = int(action)
            position_size = 0.5  # Default position size
        
        return RLAction(signal=signal, position_size=position_size)


# --- Phase 2: Execution Optimization ---

@dataclass
class PlacementDetails:
    """Output from RL Executor."""
    price_offset: float # Ticks from mid-price
    aggression: int # 0=Passive, 1=Aggressive
    fill_probability_est: float


class ExecutionEnvironment(gym.Env if gym else object):
    """
    RL Environment for Order Execution optimization.
    Optimizes placement price and timing to minimize slippage.
    """
    def __init__(self, tick_data: List[Dict[str, Any]], target_quantity: int = 1):
        if not SB3_AVAILABLE:
            return
            
        self.tick_data = tick_data
        self.target_quantity = target_quantity
        self.current_step = 0
        
        # Action Space: [Price Offset (Continuous), Aggression (Discrete)]
        # Price Offset: -2.0 to +2.0 ticks
        # Aggression: 0 or 1
        self.action_space = spaces.Dict({
            "price_offset": spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32),
            "aggression": spaces.Discrete(2)
        })
        
        # State Space: [Spread, Imbalance, Volatility, Time Remaining]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        self.current_step = 0
        return self._get_obs(), {}
        
    def step(self, action):
        # Unpack action
        price_offset = float(action["price_offset"][0])
        aggression = int(action["aggression"])
        
        # Simulate Fill
        tick = self.tick_data[self.current_step]
        mid_price = (tick['bid'] + tick['ask']) / 2.0
        spread = tick['ask'] - tick['bid']
        
        # Simplified simulation logic
        fill_price = mid_price + (price_offset * 0.05) # Assuming 0.05 tick
        
        # Calculate Reward (Implementation Shortfall)
        # Benchmark = Mid Price at arrival
        benchmark_price = (self.tick_data[0]['bid'] + self.tick_data[0]['ask']) / 2.0
        slippage = benchmark_price - fill_price # For buy (want lower)
        
        reward = slippage 
        if aggression == 1:
            # Pay spread but guaranteed fill (mostly)
            reward -= (spread / 2.0)
            
        self.current_step += 1
        done = True # Single step execution for now (simulating placement decision)
        
        return self._get_obs(), reward, done, False, {}
        
    def _get_obs(self):
        if self.current_step >= len(self.tick_data):
            return np.zeros(4)
            
        tick = self.tick_data[self.current_step]
        # Calculate features
        spread = tick['ask'] - tick['bid']
        imbalance = tick['bid_size'] / (tick['ask_size'] + 1e-9)
        vol = 0.0 # Placeholder
        time_left = 1.0 # Placeholder
        
        return np.array([spread, imbalance, vol, time_left], dtype=np.float32)


class RLExecutor:
    """
    RL Agent for trade execution.
    Decides how to place orders (Limit price, Aggression) based on market microstructure.
    Supports both PPO and DQN algorithms.
    """
    def __init__(
        self, 
        exchange: str, 
        model_path: Optional[str] = None,
        algorithm: str = "PPO",
        ppo_model_path: Optional[str] = None,
        dqn_model_path: Optional[str] = None,
        use_ensemble: bool = False,
    ):
        self.exchange = exchange
        self.algorithm = algorithm.upper()
        self.use_ensemble = use_ensemble
        self.model = None
        self.ppo_model = None
        self.dqn_model = None
        self.is_ready = False
        self.ppo_ready = False
        self.dqn_ready = False
        
        if not SB3_AVAILABLE:
            return
        
        # Load models based on configuration
        if use_ensemble:
            # Load both PPO and DQN for ensemble
            if ppo_model_path:
                try:
                    self.ppo_model = PPO.load(ppo_model_path)
                    self.ppo_ready = True
                    LOGGER.info(f"[{exchange}] RLExecutor PPO model loaded: {ppo_model_path}")
                except Exception as e:
                    LOGGER.warning(f"[{exchange}] Failed to load RLExecutor PPO model: {e}")
            
            if dqn_model_path:
                try:
                    self.dqn_model = DQN.load(dqn_model_path)
                    self.dqn_ready = True
                    LOGGER.info(f"[{exchange}] RLExecutor DQN model loaded: {dqn_model_path}")
                except Exception as e:
                    LOGGER.warning(f"[{exchange}] Failed to load RLExecutor DQN model: {e}")
            
            self.is_ready = self.ppo_ready or self.dqn_ready
        elif model_path:
            # Single model mode (backward compatible)
            try:
                if self.algorithm == "PPO":
                    self.model = PPO.load(model_path)
                elif self.algorithm == "DQN":
                    self.model = DQN.load(model_path)
                else:
                    LOGGER.warning(f"[{exchange}] Unknown algorithm {self.algorithm}, defaulting to PPO")
                    self.model = PPO.load(model_path)
                
                self.is_ready = True
                LOGGER.info(f"[{exchange}] RLExecutor {self.algorithm} model loaded: {model_path}")
            except Exception as e:
                LOGGER.warning(f"[{exchange}] Failed to load RLExecutor model: {e}")

    def decide_placement(
        self, 
        symbol: str, 
        current_price: float, 
        spread: float, 
        imbalance: float
    ) -> PlacementDetails:
        """
        Decide how to place the order. Uses ensemble if both models are loaded.
        """
        if not self.is_ready:
            # Fallback: Passive Limit at Best Bid/Ask
            return PlacementDetails(price_offset=0.0, aggression=0, fill_probability_est=0.5)
            
        # Construct state vector
        # [Spread, Imbalance, Volatility, Time Remaining]
        # Using placeholders for missing data
        obs = np.array([spread, imbalance, 0.0, 1.0], dtype=np.float32)
        
        # Ensemble mode: combine predictions from both models
        if self.use_ensemble and (self.ppo_ready or self.dqn_ready):
            return self._ensemble_placement(obs)
        
        # Single model mode
        if self.model is None:
            return PlacementDetails(price_offset=0.0, aggression=0, fill_probability_est=0.5)
        
        try:
            action, _ = self.model.predict(obs, deterministic=True)
            price_offset = float(action["price_offset"][0]) if isinstance(action, dict) else float(action[0])
            aggression = int(action["aggression"]) if isinstance(action, dict) else int(action[1]) if len(action)>1 else 0
            
            return PlacementDetails(
                price_offset=price_offset,
                aggression=aggression,
                fill_probability_est=0.8 if aggression == 1 else 0.4
            )
        except Exception as e:
            LOGGER.error(f"[{self.exchange}] RLExecutor prediction error: {e}")
            return PlacementDetails(price_offset=0.0, aggression=0, fill_probability_est=0.5)
    
    def _ensemble_placement(self, obs: np.ndarray) -> PlacementDetails:
        """Combine placement decisions from both PPO and DQN models."""
        placements = []
        
        # Get PPO prediction
        if self.ppo_ready:
            try:
                action, _ = self.ppo_model.predict(obs, deterministic=True)
                price_offset = float(action["price_offset"][0]) if isinstance(action, dict) else float(action[0])
                aggression = int(action["aggression"]) if isinstance(action, dict) else int(action[1]) if len(action)>1 else 0
                placements.append(PlacementDetails(
                    price_offset=price_offset,
                    aggression=aggression,
                    fill_probability_est=0.8 if aggression == 1 else 0.4
                ))
            except Exception as e:
                LOGGER.debug(f"[{self.exchange}] PPO placement prediction failed: {e}")
        
        # Get DQN prediction
        if self.dqn_ready:
            try:
                action, _ = self.dqn_model.predict(obs, deterministic=True)
                price_offset = float(action["price_offset"][0]) if isinstance(action, dict) else float(action[0])
                aggression = int(action["aggression"]) if isinstance(action, dict) else int(action[1]) if len(action)>1 else 0
                placements.append(PlacementDetails(
                    price_offset=price_offset,
                    aggression=aggression,
                    fill_probability_est=0.8 if aggression == 1 else 0.4
                ))
            except Exception as e:
                LOGGER.debug(f"[{self.exchange}] DQN placement prediction failed: {e}")
        
        if not placements:
            return PlacementDetails(price_offset=0.0, aggression=0, fill_probability_est=0.5)
        
        if len(placements) == 1:
            return placements[0]
        
        # Ensemble: average price offset, majority vote on aggression
        avg_price_offset = np.mean([p.price_offset for p in placements])
        # Majority vote on aggression (or average if tied)
        aggression_votes = [p.aggression for p in placements]
        avg_aggression = int(round(np.mean(aggression_votes)))
        avg_fill_prob = np.mean([p.fill_probability_est for p in placements])
        
        return PlacementDetails(
            price_offset=avg_price_offset,
            aggression=avg_aggression,
            fill_probability_est=avg_fill_prob
        )
