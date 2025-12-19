"""
Train RL Execution Model (Phase 2).

This script trains PPO and/or DQN agents for the RLExecutor using synthetic tick data derived 
from historical OHLC candles. It generates model artifacts required to enable RL-based execution 
in the main application.

Usage:
    # Train PPO only
    python train_rl.py --algorithm PPO
    
    # Train DQN only
    python train_rl.py --algorithm DQN
    
    # Train both (for ensemble)
    python train_rl.py --algorithm BOTH
"""
import logging
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.reinforcement_learning import ExecutionEnvironment
import database_new as db
from time_utils import today_ist, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def generate_synthetic_ticks(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate synthetic tick data from OHLC candles.
    Simulates bid/ask spread and depth based on volatility (high-low range).
    """
    ticks = []
    
    LOGGER.info("Generating synthetic ticks from OHLC data...")
    for _, row in df.iterrows():
        close_price = row['close']
        
        # Estimate volatility proxy
        volatility = (row['high'] - row['low']) / close_price if close_price > 0 else 0.001
        
        # Simulate spread (higher volatility -> wider spread)
        # Base spread 0.05, max spread 1.0
        spread = max(0.05, min(1.0, volatility * 100))
        
        # Randomize imbalance slightly
        imbalance_bias = np.random.uniform(0.1, 10.0)
        
        # Create synthetic tick
        tick = {
            'last_price': close_price,
            'bid': close_price - (spread / 2),
            'ask': close_price + (spread / 2),
            'bid_size': 1000 * imbalance_bias,
            'ask_size': 1000 * (1 / imbalance_bias),
            'volume': row['volume']
        }
        ticks.append(tick)
    
    return ticks

def train_rl_model(exchange: str = 'NSE', days: int = 30, algorithm: str = "PPO"):
    """
    Train the RL execution model.
    
    Args:
        exchange: Exchange name (NSE, BSE, etc.)
        days: Number of days of historical data to use
        algorithm: Algorithm to train - "PPO", "DQN", or "BOTH"
    """
    
    # 1. Load Data
    start_date = today_ist() - timedelta(days=days)
    end_date = today_ist() + timedelta(days=1)
    
    LOGGER.info(f"Loading historical data for {exchange} from {start_date} to {end_date}...")
    
    df = db.load_historical_data_for_ml(exchange, start_date, end_date)
    
    if df.empty:
        LOGGER.error("No data found for training.")
        return

    LOGGER.info(f"Loaded {len(df)} rows with columns: {list(df.columns)[:10]}...")  # Show first 10 columns
    
    # Ensure required columns exist
    if 'close' not in df.columns:
        # Fallback if names differ (e.g. underlying_price)
        if 'underlying_price' in df.columns:
            # Ensure underlying_price is a Series, not DataFrame
            underlying_price_series = df['underlying_price']
            if isinstance(underlying_price_series, pd.DataFrame):
                # If it's a DataFrame, take the first column or first row
                underlying_price_series = underlying_price_series.iloc[:, 0] if underlying_price_series.shape[1] > 0 else underlying_price_series.iloc[0]
            
            df['close'] = underlying_price_series
            df['high'] = underlying_price_series * 1.001  # Dummy
            df['low'] = underlying_price_series * 0.999   # Dummy
            df['volume'] = 1000
            LOGGER.info("Created OHLC columns from underlying_price")
        else:
            LOGGER.error(f"Dataframe missing 'close' or 'underlying_price' column. Available columns: {list(df.columns)}")
            return
    
    # Ensure we have the required columns for tick generation
    required_cols = ['close', 'high', 'low', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        LOGGER.error(f"Missing required columns: {missing_cols}")
        return

    # 2. Generate Synthetic Ticks from Candles
    # Since we don't have high-freq tick data stored, we simulate it.
    ticks = generate_synthetic_ticks(df)
    
    # 3. Initialize Environment
    LOGGER.info("Initializing Execution Environment...")
    env = ExecutionEnvironment(tick_data=ticks)
    
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    ppo_model = None
    dqn_model = None
    
    # 4. Train PPO
    if algorithm.upper() in ["PPO", "BOTH"]:
        LOGGER.info("Starting PPO Training...")
        try:
            ppo_model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048)
            ppo_model.learn(total_timesteps=10000)
            LOGGER.info("PPO training complete.")
            
            ppo_model_path = output_dir / "rl_ppo_model.zip"
            ppo_model.save(str(ppo_model_path))
            LOGGER.info(f"PPO model saved to {ppo_model_path}")
        except Exception as e:
            LOGGER.error(f"PPO training failed: {e}")
    
    # 5. Train DQN
    if algorithm.upper() in ["DQN", "BOTH"]:
        LOGGER.info("Starting DQN Training...")
        try:
            dqn_model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.0001, buffer_size=10000)
            dqn_model.learn(total_timesteps=10000)
            LOGGER.info("DQN training complete.")
            
            dqn_model_path = output_dir / "rl_dqn_model.zip"
            dqn_model.save(str(dqn_model_path))
            LOGGER.info(f"DQN model saved to {dqn_model_path}")
        except Exception as e:
            LOGGER.error(f"DQN training failed: {e}")
    
    # 6. Save default model (for backward compatibility)
    if algorithm.upper() == "PPO" and ppo_model is not None:
        default_model_path = output_dir / "rl_execution_model.zip"
        ppo_model.save(str(default_model_path))
        LOGGER.info(f"Default model saved to {default_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL execution models")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        choices=["PPO", "DQN", "BOTH"],
        help="Algorithm to train: PPO, DQN, or BOTH (for ensemble)"
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default="NSE",
        help="Exchange name (default: NSE)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of historical data (default: 30)"
    )
    
    args = parser.parse_args()
    train_rl_model(exchange=args.exchange, days=args.days, algorithm=args.algorithm)
