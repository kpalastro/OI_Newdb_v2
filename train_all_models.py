"""
Master Script: Retrain All Models From Scratch

This script orchestrates training of all models in the system:
1. Main Trading Models (LightGBM + HMM regimes) - train_model.py
2. Swing Ensemble Models (XGBoost + LightGBM) - train_swing_ensemble.py
3. Expiry Models (Deep Learning) - train_expiry_model.py
4. RL Models (PPO + DQN) - train_rl.py
5. Walk-Forward Orchestrator (optional) - train_orchestrator.py

Usage:
    # Train all models for all exchanges
    python train_all_models.py --all
    
    # Train only specific model types
    python train_all_models.py --main --swing --expiry --rl
    
    # Train for specific exchanges only
    python train_all_models.py --all --exchanges NSE BSE
    
    # Skip RL training (faster)
    python train_all_models.py --all --skip-rl
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
LOGGER = logging.getLogger(__name__)

# Default exchanges to train
DEFAULT_EXCHANGES = ['NSE', 'BSE']
ALL_EXCHANGES = ['NSE', 'BSE', 'NSE_MONTHLY', 'BANKNIFTY_MONTHLY']

# Training scripts
TRAIN_MAIN = 'train_model.py'
TRAIN_SWING = 'train_swing_ensemble.py'
TRAIN_EXPIRY = 'train_expiry_model.py'
TRAIN_RL = 'train_rl.py'
TRAIN_ORCHESTRATOR = 'train_orchestrator.py'


def run_command(cmd: List[str], description: str) -> bool:
    """
    Run a training command and return success status.
    
    Args:
        cmd: Command to run as list of strings
        description: Human-readable description of what's being trained
        
    Returns:
        True if successful, False otherwise
    """
    LOGGER.info(f"\n{'='*80}")
    LOGGER.info(f"Starting: {description}")
    LOGGER.info(f"Command: {' '.join(cmd)}")
    LOGGER.info(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        LOGGER.info(f"\n✓ Successfully completed: {description}\n")
        return True
    except subprocess.CalledProcessError as e:
        LOGGER.error(f"\n✗ Failed: {description}")
        LOGGER.error(f"Exit code: {e.returncode}\n")
        return False
    except Exception as e:
        LOGGER.exception(f"\n✗ Error running {description}: {e}\n")
        return False


def train_main_models(exchanges: List[str], days: int = 90) -> dict:
    """Train main trading models (LightGBM + HMM regimes)."""
    results = {}
    
    for exchange in exchanges:
        if exchange not in ['NSE', 'BSE']:
            LOGGER.warning(f"Skipping {exchange} - train_model.py only supports NSE and BSE")
            continue
            
        cmd = [sys.executable, TRAIN_MAIN, '--exchange', exchange, '--days', str(days)]
        success = run_command(
            cmd,
            f"Main Trading Model ({exchange})"
        )
        results[f'main_{exchange}'] = success
        
    return results


def train_swing_models(exchanges: List[str], days: int = 90) -> dict:
    """Train swing ensemble models (XGBoost + LightGBM)."""
    results = {}
    
    for exchange in exchanges:
        if exchange not in ['NSE', 'BSE']:
            LOGGER.warning(f"Skipping {exchange} - train_swing_ensemble.py only supports NSE and BSE")
            continue
            
        cmd = [sys.executable, TRAIN_SWING, '--exchange', exchange, '--days', str(days)]
        success = run_command(
            cmd,
            f"Swing Ensemble Model ({exchange})"
        )
        results[f'swing_{exchange}'] = success
        
    return results


def train_expiry_models(exchanges: List[str], days: int = 365) -> dict:
    """Train expiry models (Deep Learning)."""
    results = {}
    
    for exchange in exchanges:
        if exchange not in ['NSE', 'BSE']:
            LOGGER.warning(f"Skipping {exchange} - train_expiry_model.py only supports NSE and BSE")
            continue
            
        cmd = [sys.executable, TRAIN_EXPIRY, '--exchange', exchange, '--days', str(days)]
        success = run_command(
            cmd,
            f"Expiry Model ({exchange})"
        )
        results[f'expiry_{exchange}'] = success
        
    return results


def train_rl_models(exchanges: List[str], days: int = 30, algorithm: str = 'BOTH') -> dict:
    """Train RL models (PPO + DQN) for execution optimization."""
    results = {}
    
    # RL models are exchange-agnostic (they work with any exchange)
    # We train once and use for all exchanges
    cmd = [
        sys.executable, TRAIN_RL,
        '--algorithm', algorithm,
        '--exchange', exchanges[0] if exchanges else 'NSE',  # Use first exchange for data
        '--days', str(days)
    ]
    
    success = run_command(
        cmd,
        f"RL Models (PPO + DQN)"
    )
    results['rl_models'] = success
    
    return results


def train_orchestrator(exchanges: List[str], days: int = 120, optuna_trials: int = 10) -> dict:
    """Train models using walk-forward orchestrator (includes hyperparameter tuning)."""
    results = {}
    
    for exchange in exchanges:
        cmd = [
            sys.executable, TRAIN_ORCHESTRATOR,
            '--exchange', exchange,
            '--days', str(days),
            '--optuna-trials', str(optuna_trials)
        ]
        
        success = run_command(
            cmd,
            f"Walk-Forward Orchestrator ({exchange})"
        )
        results[f'orchestrator_{exchange}'] = success
        
    return results


def print_summary(results: dict):
    """Print training summary."""
    LOGGER.info("\n" + "="*80)
    LOGGER.info("TRAINING SUMMARY")
    LOGGER.info("="*80)
    
    total = len(results)
    successful = sum(1 for v in results.values() if v)
    failed = total - successful
    
    LOGGER.info(f"Total models trained: {total}")
    LOGGER.info(f"Successful: {successful}")
    LOGGER.info(f"Failed: {failed}")
    
    if failed > 0:
        LOGGER.info("\nFailed models:")
        for name, success in results.items():
            if not success:
                LOGGER.info(f"  ✗ {name}")
    
    LOGGER.info("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Retrain all models from scratch',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train everything
  python train_all_models.py --all
  
  # Train only main and swing models
  python train_all_models.py --main --swing
  
  # Train for specific exchanges
  python train_all_models.py --all --exchanges NSE BSE
  
  # Skip RL training (faster)
  python train_all_models.py --all --skip-rl
  
  # Use orchestrator instead of individual models (includes hyperparameter tuning)
  python train_all_models.py --orchestrator-only
        """
    )
    
    # Model type flags
    parser.add_argument(
        '--all',
        action='store_true',
        help='Train all model types (main, swing, expiry, RL)'
    )
    parser.add_argument(
        '--main',
        action='store_true',
        help='Train main trading models (LightGBM + HMM)'
    )
    parser.add_argument(
        '--swing',
        action='store_true',
        help='Train swing ensemble models (XGBoost + LightGBM)'
    )
    parser.add_argument(
        '--expiry',
        action='store_true',
        help='Train expiry models (Deep Learning)'
    )
    parser.add_argument(
        '--rl',
        action='store_true',
        help='Train RL models (PPO + DQN)'
    )
    parser.add_argument(
        '--orchestrator',
        action='store_true',
        help='Train using walk-forward orchestrator (includes hyperparameter tuning)'
    )
    parser.add_argument(
        '--orchestrator-only',
        action='store_true',
        help='Only use orchestrator (skip individual model training)'
    )
    
    # Options
    parser.add_argument(
        '--exchanges',
        nargs='+',
        choices=ALL_EXCHANGES,
        default=DEFAULT_EXCHANGES,
        help=f'Exchanges to train (default: {DEFAULT_EXCHANGES})'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=90,
        help='Days of historical data for main/swing models (default: 90)'
    )
    parser.add_argument(
        '--expiry-days',
        type=int,
        default=365,
        help='Days of historical data for expiry models (default: 365)'
    )
    parser.add_argument(
        '--rl-days',
        type=int,
        default=30,
        help='Days of historical data for RL models (default: 30)'
    )
    parser.add_argument(
        '--orchestrator-days',
        type=int,
        default=120,
        help='Days of historical data for orchestrator (default: 120)'
    )
    parser.add_argument(
        '--optuna-trials',
        type=int,
        default=10,
        help='Number of Optuna trials for orchestrator (default: 10)'
    )
    parser.add_argument(
        '--skip-rl',
        action='store_true',
        help='Skip RL model training (faster)'
    )
    parser.add_argument(
        '--rl-algorithm',
        choices=['PPO', 'DQN', 'BOTH'],
        default='BOTH',
        help='RL algorithm to train (default: BOTH)'
    )
    
    args = parser.parse_args()
    
    # Determine what to train
    if args.orchestrator_only:
        train_main = False
        train_swing = False
        train_expiry = False
        train_rl = False
        train_orch = True
    elif args.all:
        train_main = True
        train_swing = True
        train_expiry = True
        train_rl = not args.skip_rl
        train_orch = args.orchestrator
    else:
        train_main = args.main
        train_swing = args.swing
        train_expiry = args.expiry
        train_rl = args.rl and not args.skip_rl
        train_orch = args.orchestrator
    
    # If nothing selected, show help
    if not any([train_main, train_swing, train_expiry, train_rl, train_orch]):
        parser.print_help()
        LOGGER.error("\nError: No models selected for training. Use --all or specify individual model types.")
        sys.exit(1)
    
    # Start training
    LOGGER.info("="*80)
    LOGGER.info("STARTING COMPLETE MODEL RETRAINING FROM SCRATCH")
    LOGGER.info("="*80)
    LOGGER.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    LOGGER.info(f"Exchanges: {args.exchanges}")
    LOGGER.info(f"Training:")
    LOGGER.info(f"  - Main models: {train_main}")
    LOGGER.info(f"  - Swing models: {train_swing}")
    LOGGER.info(f"  - Expiry models: {train_expiry}")
    LOGGER.info(f"  - RL models: {train_rl}")
    LOGGER.info(f"  - Orchestrator: {train_orch}")
    LOGGER.info("="*80 + "\n")
    
    all_results = {}
    
    # Train models
    if train_orch:
        LOGGER.info("Using orchestrator for training (includes hyperparameter tuning)...")
        all_results.update(
            train_orchestrator(args.exchanges, args.orchestrator_days, args.optuna_trials)
        )
    else:
        # Individual model training
        if train_main:
            all_results.update(train_main_models(args.exchanges, args.days))
        
        if train_swing:
            all_results.update(train_swing_models(args.exchanges, args.days))
        
        if train_expiry:
            all_results.update(train_expiry_models(args.exchanges, args.expiry_days))
        
        if train_rl:
            all_results.update(train_rl_models(args.exchanges, args.rl_days, args.rl_algorithm))
    
    # Print summary
    print_summary(all_results)
    
    # Final status
    all_success = all(all_results.values())
    if all_success:
        LOGGER.info("\n✓ All models trained successfully!")
        LOGGER.info("You can now restart the OI Tracker to use the new models.")
        sys.exit(0)
    else:
        LOGGER.warning("\n⚠ Some models failed to train. Check logs above for details.")
        sys.exit(1)


if __name__ == '__main__':
    main()

