#!/usr/bin/env python3
"""
Research script to find best performing models and parameters for NSE and BSE.

This script runs train_orchestrator.py with research-focused settings:
- Multiple model families (LightGBM, XGBoost, CatBoost)
- Optuna hyperparameter tuning
- Walk-forward validation across multiple time segments
- Comprehensive metrics and parameter reports

Usage:
    python run_research_experiments.py --exchange NSE
    python run_research_experiments.py --exchange BSE
    python run_research_experiments.py --all  # Run both exchanges sequentially
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_orchestrator(exchange: str, days: int = 180, window_days: int = 45, 
                     step_days: int = 15, optuna_trials: int = 20, 
                     families: list = None, output_dir: Path = None):
    """
    Run train_orchestrator.py with research settings.
    
    Args:
        exchange: 'NSE' or 'BSE'
        days: Total lookback window (default 180 days for robust research)
        window_days: Training window size per segment (default 45 days)
        step_days: Step size between segments (default 15 days)
        optuna_trials: Number of Optuna hyperparameter tuning trials (default 20)
        families: List of model families to test (default: all tree-based)
        output_dir: Directory to save results (default: reports/research/)
    """
    if families is None:
        families = ["lightgbm", "xgboost", "catboost"]
    
    if output_dir is None:
        output_dir = Path("reports/research")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{exchange}_research_{timestamp}.json"
    
    cmd = [
        sys.executable, "train_orchestrator.py",
        "--exchange", exchange,
        "--days", str(days),
        "--window-days", str(window_days),
        "--step-days", str(step_days),
        "--optuna-trials", str(optuna_trials),
        "--output", str(output_file),
        "--families"
    ] + families
    
    print("=" * 80)
    print(f"Starting research experiment for {exchange}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")
    print(f"Output will be saved to: {output_file}")
    print("=" * 80)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ {exchange} research experiment completed successfully!")
        print(f"  Results saved to: {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {exchange} research experiment failed with exit code {e.returncode}")
        return None
    except KeyboardInterrupt:
        print(f"\n⚠ {exchange} research experiment interrupted by user")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run research experiments to find best models and parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run research for NSE only
  python run_research_experiments.py --exchange NSE
  
  # Run research for BSE only
  python run_research_experiments.py --exchange BSE
  
  # Run both exchanges sequentially
  python run_research_experiments.py --all
  
  # Custom settings: more data, more tuning trials
  python run_research_experiments.py --exchange NSE --days 240 --optuna-trials 50
  
  # Test only LightGBM and XGBoost (faster)
  python run_research_experiments.py --exchange NSE --families lightgbm xgboost
        """
    )
    
    parser.add_argument(
        "--exchange",
        choices=["NSE", "BSE"],
        help="Exchange to run research on (use --all to run both)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run research for both NSE and BSE sequentially"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Total lookback window in days (default: 180)"
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=45,
        help="Training window size per segment (default: 45)"
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=15,
        help="Step size between segments (default: 15)"
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=20,
        help="Number of Optuna hyperparameter tuning trials per segment (default: 20)"
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=["lightgbm", "xgboost", "catboost"],
        choices=["lightgbm", "xgboost", "catboost", "rl", "rl-ppo", "rl-dqn"],
        help="Model families to evaluate (default: lightgbm xgboost catboost)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/research"),
        help="Directory to save research results (default: reports/research/)"
    )
    
    args = parser.parse_args()
    
    if not args.exchange and not args.all:
        parser.error("Must specify either --exchange or --all")
    
    exchanges = []
    if args.all:
        exchanges = ["NSE", "BSE"]
    elif args.exchange:
        exchanges = [args.exchange]
    
    results = {}
    for exchange in exchanges:
        print(f"\n{'='*80}")
        print(f"Starting research for {exchange}")
        print(f"{'='*80}\n")
        
        output_file = run_orchestrator(
            exchange=exchange,
            days=args.days,
            window_days=args.window_days,
            step_days=args.step_days,
            optuna_trials=args.optuna_trials,
            families=args.families,
            output_dir=args.output_dir
        )
        
        results[exchange] = output_file
    
    # Summary
    print("\n" + "=" * 80)
    print("RESEARCH EXPERIMENTS SUMMARY")
    print("=" * 80)
    for exchange, output_file in results.items():
        if output_file:
            print(f"✓ {exchange}: {output_file}")
        else:
            print(f"✗ {exchange}: Failed")
    print("=" * 80)
    
    if all(results.values()):
        print("\n✓ All research experiments completed successfully!")
        print("\nNext steps:")
        print("1. Review the JSON reports in reports/research/")
        print("2. Compare metrics across model families and segments")
        print("3. Identify best performing model family and parameters")
        print("4. Train production model using train_model.py with best parameters")
    else:
        print("\n⚠ Some experiments failed. Check logs above for details.")


if __name__ == "__main__":
    main()
