
"""
train_expiry_model.py

Specialized training script for the ExpiryDayTransformer (Deep Learning/PyTorch).
Updates the model to align with the new 117-feature set.
"""
import logging
import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import database_new as db
from time_utils import today_ist
from feature_engineering import REQUIRED_FEATURE_COLUMNS, prepare_training_features
from train_model import define_triple_barrier_target
from models.expiry_transformer import ExpiryDayTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_expiry_model(exchange: str, days: int = 365, epochs: int = 50, lr: float = 0.001):
    # 1. Load Data (Longer lookback for expiry since it's only once a week/month)
    end_date = today_ist() + timedelta(days=1)
    start_date = today_ist() - timedelta(days=days)
    
    logging.info(f"Loading data from {start_date.date()} to {end_date.date()}...")
    raw_data = db.load_historical_data_for_ml(exchange, start_date, end_date)
    
    if raw_data.empty:
        logging.error("No data found.")
        return

    # 2. Prepare Features
    logging.info("Preparing features...")
    df = prepare_training_features(raw_data, REQUIRED_FEATURE_COLUMNS)
    
    # 3. Filter for Expiry Context (0-DTE and 1-DTE)
    # ExpiryDayTransformer is designed for "end game" dynamics
    # We filter where time_to_expiry_hours <= 25 (approx 1 day + 1 hour)
    if 'time_to_expiry_hours' not in df.columns:
        logging.error("Missing 'time_to_expiry_hours' feature.")
        return
        
    df_expiry = df[df['time_to_expiry_hours'] <= 25].copy()
    logging.info(f"Filtered for expiry context: {len(df_expiry)} rows (from {len(df)})")
    
    if len(df_expiry) < 500:
        logging.warning("Insufficient data for deep learning training. Exiting.")
        return

    # 4. Define Targets
    # Direction Target (Triple Barrier)
    df_expiry = define_triple_barrier_target(df_expiry, look_forward=15) # 15 min horizon
    
    # Pin Risk Target: Is Close within 0.1% of a strike? 
    # Strikes are usually integers or multiples of 50/100.
    # Heuristic: Closest 50 level.
    # NIFTY strikes are 50 steps.
    strike_step = 50.0 if exchange == 'NSE' else 100.0 # Approx
    
    # Distance to nearest strike
    df_expiry['dist_to_strike'] = df_expiry['underlying_price'] % strike_step
    df_expiry['dist_to_strike'] = np.minimum(df_expiry['dist_to_strike'], strike_step - df_expiry['dist_to_strike'])
    
    # Pin Risk = 1.0 if within 10 points of strike, else 0.0
    df_expiry['pin_risk_target'] = (df_expiry['dist_to_strike'] < 10.0).astype(float)
    
    # Gamma Flip Target (Placeholder 0.0 as we don't have perfect label)
    df_expiry['gamma_dist_target'] = 0.0
    
    # Encode Direction (-1, 0, 1) -> (0, 1, 2)
    # -1 -> 0 (Sell)
    #  0 -> 1 (Hold)
    #  1 -> 2 (Buy)
    df_expiry['direction_target'] = df_expiry['target'].astype(int) + 1
    
    # Drop NaNs
    df_expiry.dropna(subset=REQUIRED_FEATURE_COLUMNS + ['direction_target', 'pin_risk_target'], inplace=True)
    
    # 5. Prepare Tensors
    feature_cols = [c for c in REQUIRED_FEATURE_COLUMNS if c in df_expiry.columns]
    X_np = df_expiry[feature_cols].values.astype(np.float32)
    y_dir_np = df_expiry['direction_target'].values.astype(np.int64)
    y_pin_np = df_expiry['pin_risk_target'].values.astype(np.float32)
    y_gamma_np = df_expiry['gamma_dist_target'].values.astype(np.float32)
    
    dataset = TensorDataset(
        torch.from_numpy(X_np),
        torch.from_numpy(y_dir_np),
        torch.from_numpy(y_pin_np),
        torch.from_numpy(y_gamma_np)
    )
    
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 6. Initialize Model
    # Feature dim corresponds to NEW feature count (e.g. 117)
    feature_dim = len(feature_cols)
    logging.info(f"Initializing ExpiryDayTransformer with feature_dim={feature_dim}")
    
    model = ExpiryDayTransformer(feature_dim=feature_dim)
    
    criteria_dir = nn.CrossEntropyLoss()
    criteria_pin = nn.BCEWithLogitsLoss() # Pin head output is linear, BCEWithLogits applies Sigmoid
    # Note: ExpiryDayTransformer forward() applies Sigmoid to pin_risk?
    # Let's check model definition.
    # ExpiryDayTransformer.forward sends pin output through torch.sigmoid.
    # So we should use BCELoss.
    criteria_pin = nn.BCELoss()
    criteria_gamma = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 7. Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        acc_correct = 0
        total_samples = 0
        
        for X_batch, y_dir_batch, y_pin_batch, y_gamma_batch in loader:
            optimizer.zero_grad()
            
            # Forward
            out = model(X_batch) # Returns dict {'pin_risk', 'gamma_dist', 'direction_probs'}
            # direction_probs is Softmaxed. CrossEntropyLoss expects Logits usually.
            # But ExpiryDayTransformer code: `direction_logits = self.direction_head(pooled)` -> `F.softmax`.
            # If we want to safely train, we should modify the model to return logits OR use NLLLoss with log-probs.
            # OR we can just use the logits if accessible.
            # The `forward` returns `direction_probs` (softmaxed). 
            # CrossEntropyLoss expects logits. 
            # NLLLoss expects Log_Softmax.
            # Training on Softmaxed output with CrossEntropyLoss is mathematically wrong/unstable.
            
            # Hack: `torch.log(prob + epsilon)` -> NLLLoss
            log_probs = torch.log(out['direction_probs'] + 1e-9)
            loss_dir = nn.NLLLoss()(log_probs, y_dir_batch)
            
            loss_pin = criteria_pin(out['pin_risk'].squeeze(), y_pin_batch)
            loss_gamma = criteria_gamma(out['gamma_dist'].squeeze(), y_gamma_batch)
            
            # Weighted sum (Prioritize Direction)
            loss = loss_dir + 0.2 * loss_pin + 0.0 * loss_gamma
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Metrics
            pred_cls = out['direction_probs'].argmax(dim=1)
            acc_correct += (pred_cls == y_dir_batch).sum().item()
            total_samples += y_dir_batch.size(0)
            
        avg_loss = total_loss / len(loader)
        acc = acc_correct / total_samples
        if (epoch + 1) % 5 == 0:
            logging.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

    # 8. Save Model
    model_dir = Path("models") / exchange
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / "expiry_transformer.pt"
    
    torch.save(model.state_dict(), save_path)
    logging.info(f"✓ Trained model saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exchange', required=True, default='NSE')
    parser.add_argument('--days', type=int, default=180)
    args = parser.parse_args()
    
    train_expiry_model(args.exchange, args.days)
