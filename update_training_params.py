#!/usr/bin/env python3
"""
Automatically update DEFAULT_MODEL_PARAMS in train_model.py with best parameters
from research experiment results.

Usage:
    python3 update_training_params.py <research_json_file> [--exchange NSE|BSE] [--family lightgbm|xgboost|catboost]
    
Example:
    python3 update_training_params.py reports/research/NSE_research_20260121_204200.json --exchange NSE --family lightgbm
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TRAIN_MODEL_FILE = Path('train_model.py')

# Required parameters that should always be present (even if not in research)
REQUIRED_PARAMS = {
    'objective': 'multiclass',
    'num_class': 3,
    'verbosity': -1,
    'n_jobs': -1,
    'random_state': 42,
    'class_weight': 'balanced',  # Important for imbalanced classes
}

# Parameters that should be preserved from current defaults if not in research
DEFAULT_FALLBACKS = {
    'min_child_samples': 20,
    'min_split_gain': 0.0,
}


def load_research_results(file_path: Path) -> Dict[str, Any]:
    """Load and parse research JSON file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Research file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def extract_best_params(data: Dict[str, Any], family: str = 'lightgbm') -> Dict[str, Any]:
    """
    Extract best parameters for the specified model family.
    
    Args:
        data: Research JSON data
        family: Model family name (lightgbm, xgboost, catboost)
    
    Returns:
        Dictionary of best parameters
    """
    # Normalize family name (case-insensitive)
    family_lower = family.lower()
    family_map = {
        'lightgbm': 'LightGBM',
        'xgboost': 'XGBoost',
        'catboost': 'CatBoost',
    }
    family_key = family_map.get(family_lower, family)
    
    best_by_family = data.get('best_by_family', {})
    
    if family_key not in best_by_family:
        available = ', '.join(best_by_family.keys())
        raise ValueError(
            f"Family '{family_key}' not found in research results. "
            f"Available families: {available}"
        )
    
    best_info = best_by_family[family_key]
    params = best_info.get('params', {})
    metrics = best_info.get('metrics', {})
    
    logger.info(f"Found best {family_key} parameters:")
    logger.info(f"  Segment ID: {best_info.get('segment_id', 'N/A')}")
    logger.info(f"  F1 Macro: {metrics.get('f1_macro', 0):.4f}")
    logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
    
    return params


def normalize_params(params: Dict[str, Any], family: str = 'lightgbm') -> Dict[str, Any]:
    """
    Normalize and merge parameters with required defaults.
    
    Args:
        params: Parameters from research
        family: Model family name
    
    Returns:
        Normalized parameter dictionary ready for train_model.py
    """
    normalized = {}
    
    # Start with required parameters
    normalized.update(REQUIRED_PARAMS)
    
    # Add parameters from research (will override required if present)
    for key, value in params.items():
        # Convert parameter names if needed
        if key == 'max_depth' and value == -1:
            # LightGBM uses -1 for unlimited depth, but we might want to cap it
            # Keep -1 as it means "no limit" in LightGBM
            normalized[key] = value
        else:
            normalized[key] = value
    
    # Add fallback defaults for parameters not in research
    for key, default_value in DEFAULT_FALLBACKS.items():
        if key not in normalized:
            normalized[key] = default_value
    
    # Ensure class_weight is set (important for imbalanced data)
    if 'class_weight' not in normalized:
        normalized['class_weight'] = 'balanced'
    
    return normalized


def format_param_value(value: Any) -> str:
    """Format a parameter value for Python code."""
    if isinstance(value, str):
        return f"'{value}'"
    elif isinstance(value, bool):
        return str(value)
    elif isinstance(value, (int, float)):
        return str(value)
    elif value is None:
        return 'None'
    else:
        return repr(value)


def update_train_model_file(params: Dict[str, Any], exchange: str = 'NSE', backup: bool = True) -> None:
    """
    Update exchange-specific DEFAULT_MODEL_PARAMS in train_model.py.
    
    Args:
        params: Dictionary of parameters to set
        exchange: Exchange name ('NSE' or 'BSE')
        backup: Whether to create a backup file
    """
    if not TRAIN_MODEL_FILE.exists():
        raise FileNotFoundError(f"train_model.py not found at {TRAIN_MODEL_FILE}")
    
    # Read current file
    with open(TRAIN_MODEL_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create backup if requested
    if backup:
        backup_path = TRAIN_MODEL_FILE.with_suffix('.py.backup')
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Created backup: {backup_path}")
    
    # Determine which parameter dictionary to update
    exchange_upper = exchange.upper()
    if exchange_upper == 'BSE':
        param_dict_name = 'DEFAULT_MODEL_PARAMS_BSE'
    else:
        param_dict_name = 'DEFAULT_MODEL_PARAMS_NSE'
    
    # Find the exchange-specific DEFAULT_MODEL_PARAMS block
    # Pattern: DEFAULT_MODEL_PARAMS_NSE: Dict[str, object] = { ... } or DEFAULT_MODEL_PARAMS_BSE: ...
    pattern = rf'({param_dict_name}:\s*Dict\[str,\s*object\]\s*=\s*\{{)(.*?)(\}})'
    
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        # Fallback: try to find DEFAULT_MODEL_PARAMS (for backward compatibility)
        fallback_pattern = r'(DEFAULT_MODEL_PARAMS:\s*Dict\[str,\s*object\]\s*=\s*\{)(.*?)(\})'
        match = re.search(fallback_pattern, content, re.DOTALL)
        if not match:
            raise ValueError(
                f"Could not find {param_dict_name} or DEFAULT_MODEL_PARAMS in train_model.py. "
                f"Make sure the file has exchange-specific parameter dictionaries."
            )
        else:
            logger.warning(f"Found DEFAULT_MODEL_PARAMS (legacy). Consider updating to {param_dict_name}.")
    
    # Build new parameter dictionary string
    param_lines = []
    param_lines.append("    'objective': 'multiclass',")
    param_lines.append("    'num_class': 3,")
    
    # Sort other parameters for consistency
    other_params = {k: v for k, v in params.items() 
                   if k not in ['objective', 'num_class']}
    
    # Add parameters in a logical order
    param_order = [
        'n_estimators',
        'learning_rate',
        'num_leaves',
        'max_depth',
        'class_weight',
        'n_jobs',
        'random_state',
        'colsample_bytree',
        'subsample',
        'verbosity',
        'min_child_samples',
        'min_split_gain',
    ]
    
    # Add ordered parameters first
    for key in param_order:
        if key in other_params:
            value = other_params[key]
            # Add comment for some parameters
            comment = ""
            if key == 'n_estimators' and value != 500:
                comment = f"  # Updated from research (was 500)"
            elif key == 'learning_rate' and value != 0.02:
                comment = f"  # Updated from research (was 0.02)"
            elif key == 'num_leaves' and value != 32:
                comment = f"  # Updated from research (was 32)"
            elif key == 'max_depth' and value != 6:
                comment = f"  # Updated from research (was 6)"
            
            param_lines.append(f"    '{key}': {format_param_value(value)},{comment}")
            del other_params[key]
    
    # Add any remaining parameters
    for key, value in sorted(other_params.items()):
        param_lines.append(f"    '{key}': {format_param_value(value)},")
    
    new_params_block = '\n'.join(param_lines)
    
    # Replace the parameters block
    new_content = (
        content[:match.start(1)] +
        match.group(1) + '\n' +
        new_params_block + '\n' +
        match.group(3) +
        content[match.end(3):]
    )
    
    # Write updated file
    with open(TRAIN_MODEL_FILE, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    logger.info(f"✓ Updated {TRAIN_MODEL_FILE}")
    logger.info(f"  Updated {param_dict_name} with {len(params)} parameters")


def main():
    parser = argparse.ArgumentParser(
        description="Update DEFAULT_MODEL_PARAMS in train_model.py with best parameters from research"
    )
    parser.add_argument(
        'research_file',
        type=Path,
        help='Path to research JSON file (e.g., reports/research/NSE_research_*.json)'
    )
    parser.add_argument(
        '--exchange',
        choices=['NSE', 'BSE'],
        help='Exchange name (optional, will be inferred from research file if not provided)'
    )
    parser.add_argument(
        '--family',
        choices=['lightgbm', 'xgboost', 'catboost'],
        default='lightgbm',
        help='Model family to use (default: lightgbm). Note: train_model.py only supports LightGBM currently'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create a backup file'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be updated without making changes'
    )
    
    args = parser.parse_args()
    
    try:
        # Load research results
        logger.info(f"Loading research results from: {args.research_file}")
        data = load_research_results(args.research_file)
        
        # Determine exchange (from args, or from research file, or default to NSE)
        exchange = args.exchange
        if not exchange:
            exchange = data.get('exchange', 'NSE')
            logger.info(f"Exchange not specified, using '{exchange}' from research file")
        
        # Validate exchange if both provided
        if args.exchange and data.get('exchange') and data.get('exchange') != args.exchange:
            logger.warning(
                f"Exchange mismatch: research file has '{data.get('exchange')}', "
                f"but --exchange specified '{args.exchange}'. Using '{args.exchange}'."
            )
        
        # Extract best parameters
        logger.info(f"Extracting best parameters for {args.family}...")
        research_params = extract_best_params(data, family=args.family)
        
        # Normalize parameters
        normalized_params = normalize_params(research_params, family=args.family)
        
        # Show what will be updated
        logger.info(f"\nParameters to update for {exchange}:")
        for key, value in sorted(normalized_params.items()):
            logger.info(f"  {key}: {value}")
        
        if args.dry_run:
            logger.info("\n[DRY RUN] No changes made. Use without --dry-run to apply updates.")
            return
        
        # Update train_model.py
        logger.info(f"\nUpdating {TRAIN_MODEL_FILE} for {exchange}...")
        update_train_model_file(normalized_params, exchange=exchange, backup=not args.no_backup)
        
        logger.info(f"\n✓ Successfully updated train_model.py for {exchange}!")
        logger.info("  Next steps:")
        logger.info(f"  1. Review the changes in train_model.py ({exchange} parameters)")
        logger.info(f"  2. Run training: python3 train_model.py --exchange {exchange} --days 90")
        logger.info("  3. Validate model performance")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
