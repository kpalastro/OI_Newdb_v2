"""
Train RL Execution Model (Phase 2).

This script trains a PPO agent for the RLExecutor using synthetic tick data derived 
from historical OHLC candles. It generates the 'models/rl_execution_model.zip' artifact
required to enable RL-based execution in the main application.
"""
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.reinforcement_learning import ExecutionEnvironment
from database_new import get_db
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

def train_rl_model(exchange: str = 'NSE', days: int = 30):
    """Train the RL execution model."""
    
    # 1. Load Data
    db = get_db()
    start_date = today_ist() - timedelta(days=days)
    end_date = today_ist() + timedelta(days=1)
    
    LOGGER.info(f"Loading historical data for {exchange} from {start_date} to {end_date}...")
    
    # We use 'load_historical_data_for_ml' or similar. 
    # Since we need generic OHLC, we can query recent active tokens or a specific index.
    # For simplicity, let's try to load NIFTY 50 futures data if possible, or just raw option data.
    # Actually, let's just use whatever `load_historical_data_for_ml` gives us, usually it returns the processed features
    # but we need raw candles. Let's use `db.fetch_market_data_calib` or similar if available, 
    # or just use the `raw_data` from ML load.
    
    df = db.load_historical_data_for_ml(exchange, start_date, end_date)
    
    if df.empty:
        LOGGER.error("No data found for training.")
        return

    # Ensure required columns exist
    if 'close' not in df.columns:
        # Fallback if names differ (e.g. underlying_price)
        if 'underlying_price' in df.columns:
            df['close'] = df['underlying_price']
            df['high'] = df['underlying_price'] * 1.001 # Dummy
            df['low'] = df['underlying_price'] * 0.999 # Dummy
            df['volume'] = 1000
        else:
             LOGGER.error("Dataframe missing 'close' or 'underlying_price' column.")
             return

    # 2. Generate Synthetic Ticks from Candles
    # Since we don't have high-freq tick data stored, we simulate it.
    ticks = generate_synthetic_ticks(df)
    
    # 3. Initialize Environment
    LOGGER.info("Initializing Execution Environment...")
    env = ExecutionEnvironment(tick_data=ticks)
    
    # 4. Train Agent
    LOGGER.info("Starting PPO Training...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048)
    
    try:
        model.learn(total_timesteps=10000)
        LOGGER.info("Training complete.")
        
        # 5. Save Model
        output_dir = Path("models")
        output_dir.mkdir(exist_ok=True)
        model_path = output_dir / "rl_execution_model.zip"
        
        model.save(str(model_path))
        LOGGER.info(f"Model saved to {model_path}")
        
    except Exception as e:
        LOGGER.error(f"Training failed: {e}")

if __name__ == "__main__":
    train_rl_model()
