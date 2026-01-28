#!/usr/bin/env python3
"""
Diagnose why training has insufficient samples.

This script checks:
1. How many rows have valid price change data
2. How many have non-zero price changes
3. Why samples might be filtered out
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import database_new as db
from feature_engineering import prepare_training_features, REQUIRED_FEATURE_COLUMNS

def diagnose_samples(exchange: str, start_date: datetime, end_date: datetime):
    """Diagnose why training samples are insufficient."""
    print(f"\n{'='*80}")
    print(f"DIAGNOSING TRAINING SAMPLES")
    print(f"{'='*80}")
    print(f"Exchange: {exchange}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    
    # Load data (same as training script)
    from train_option_return_models import load_option_price_data, apply_quality_filters
    
    print("\n1. Loading option price data...")
    df = load_option_price_data(exchange, start_date, end_date)
    print(f"   Initial rows: {len(df)}")
    
    if df.empty:
        print("   ❌ No data loaded!")
        return
    
    # Check price change columns
    print("\n2. Checking price change columns...")
    horizons = ['3m', '5m', '10m', '15m']
    for horizon in horizons:
        ce_col = f'ce_price_change_{horizon}'
        pe_col = f'pe_price_change_{horizon}'
        
        if ce_col in df.columns and pe_col in df.columns:
            ce_total = len(df)
            ce_not_null = df[ce_col].notna().sum()
            ce_not_zero = ((df[ce_col] != 0) & df[ce_col].notna()).sum()
            ce_zero = ((df[ce_col] == 0) & df[ce_col].notna()).sum()
            
            pe_total = len(df)
            pe_not_null = df[pe_col].notna().sum()
            pe_not_zero = ((df[pe_col] != 0) & df[pe_col].notna()).sum()
            pe_zero = ((df[pe_col] == 0) & df[pe_col].notna()).sum()
            
            print(f"\n   [{horizon}]")
            print(f"     CE: Total={ce_total}, Non-null={ce_not_null}, "
                  f"Non-zero={ce_not_zero}, Zero={ce_zero}, "
                  f"Null={ce_total - ce_not_null}")
            print(f"     PE: Total={pe_total}, Non-null={pe_not_null}, "
                  f"Non-zero={pe_not_zero}, Zero={pe_zero}, "
                  f"Null={pe_total - pe_not_null}")
            
            if ce_not_null > 0:
                ce_mean = df[ce_col][df[ce_col].notna()].mean()
                ce_std = df[ce_col][df[ce_col].notna()].std()
                print(f"     CE Stats: Mean={ce_mean:.2f}%, Std={ce_std:.2f}%")
            
            if pe_not_null > 0:
                pe_mean = df[pe_col][df[pe_col].notna()].mean()
                pe_std = df[pe_col][df[pe_col].notna()].std()
                print(f"     PE Stats: Mean={pe_mean:.2f}%, Std={pe_std:.2f}%")
        else:
            print(f"   [{horizon}] ❌ Columns missing: {ce_col}, {pe_col}")
    
    # Apply quality filters
    print("\n3. Applying quality filters...")
    df_filtered = apply_quality_filters(df, min_option_price=1.0)
    print(f"   After filters: {len(df_filtered)} rows ({len(df_filtered)/len(df)*100:.1f}% retained)")
    
    # Check again after filtering
    print("\n4. Checking price changes after filtering...")
    for horizon in horizons:
        ce_col = f'ce_price_change_{horizon}'
        pe_col = f'pe_price_change_{horizon}'
        
        if ce_col in df_filtered.columns and pe_col in df_filtered.columns:
            ce_not_null = df_filtered[ce_col].notna().sum()
            ce_not_zero = ((df_filtered[ce_col] != 0) & df_filtered[ce_col].notna()).sum()
            
            pe_not_null = df_filtered[pe_col].notna().sum()
            pe_not_zero = ((df_filtered[pe_col] != 0) & df_filtered[pe_col].notna()).sum()
            
            print(f"   [{horizon}]")
            print(f"     CE: Non-null={ce_not_null}, Non-zero={ce_not_zero}")
            print(f"     PE: Non-null={pe_not_null}, Non-zero={pe_not_zero}")
            
            # Check if enough samples
            if ce_not_null > 50:
                print(f"     ✅ CE has enough samples ({ce_not_null} > 50)")
            else:
                print(f"     ❌ CE insufficient samples ({ce_not_null} <= 50)")
            
            if pe_not_null > 50:
                print(f"     ✅ PE has enough samples ({pe_not_null} > 50)")
            else:
                print(f"     ❌ PE insufficient samples ({pe_not_null} <= 50)")
    
    # Check option prices
    print("\n5. Checking option prices...")
    if 'ce_price' in df_filtered.columns and 'pe_price' in df_filtered.columns:
        ce_price_stats = df_filtered['ce_price'].describe()
        pe_price_stats = df_filtered['pe_price'].describe()
        print(f"   CE Price: Mean={ce_price_stats['mean']:.2f}, "
              f"Min={ce_price_stats['min']:.2f}, Max={ce_price_stats['max']:.2f}")
        print(f"   PE Price: Mean={pe_price_stats['mean']:.2f}, "
              f"Min={pe_price_stats['min']:.2f}, Max={pe_price_stats['max']:.2f}")
    
    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exchange', default='NSE', help='Exchange name')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    diagnose_samples(args.exchange, start_date, end_date)
