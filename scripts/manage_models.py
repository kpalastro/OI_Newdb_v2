#!/usr/bin/env python3
"""
CLI tool for managing model registry.

Usage:
    python scripts/manage_models.py list [--exchange NSE] [--type lightgbm_regime] [--production]
    python scripts/manage_models.py show <version_id>
    python scripts/manage_models.py compare --new <id> --current <id>
    python scripts/manage_models.py promote --id <id> [--reason "reason"]
    python scripts/manage_models.py rollback --exchange <exchange> --type <type> --target <id> [--reason "reason"]
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_registry import ModelRegistry


def format_metrics(metrics_dict):
    """Format metrics dictionary for display."""
    if not metrics_dict:
        return "N/A"
    return ", ".join([f"{k}: {v:.3f}" for k, v in metrics_dict.items() if isinstance(v, (int, float))])


def list_models(args):
    """List models with optional filters."""
    registry = ModelRegistry()
    
    filters = {}
    if args.exchange:
        filters['exchange'] = args.exchange
    if args.type:
        filters['model_type'] = args.type
    if args.production:
        filters['status'] = 'active'
        filters['is_production'] = True
    else:
        # Don't filter by is_production if --production flag not set
        pass
    
    models = registry.list_models(**filters, limit=args.limit)
    
    if not models:
        print("No models found.")
        return
    
    print(f"\n{'ID':<6} {'Version':<20} {'Exchange':<8} {'Type':<20} {'Status':<12} {'F1':<8} {'Created':<20}")
    print("-" * 100)
    
    for model in models:
        metrics = model.get('validation_metrics', {})
        f1 = metrics.get('f1_score', 0.0) if metrics else 0.0
        status = model['status']
        if model.get('is_production'):
            status += " (PROD)"
        
        print(f"{model['id']:<6} {model['version']:<20} {model['exchange']:<8} "
              f"{model['model_type']:<20} {status:<12} {f1:<8.3f} {str(model['created_at']):<20}")
    
    print(f"\nTotal: {len(models)} models")


def show_model(args):
    """Show detailed information about a model."""
    registry = ModelRegistry()
    model = registry.get_model_version(args.version_id)
    
    if not model:
        print(f"Model version {args.version_id} not found.")
        return
    
    print(f"\n{'='*60}")
    print(f"Model Version: {model['version']}")
    print(f"{'='*60}")
    print(f"ID: {model['id']}")
    print(f"Exchange: {model['exchange']}")
    print(f"Type: {model['model_type']}")
    print(f"Status: {model['status']}")
    print(f"Production: {'Yes' if model['is_production'] else 'No'}")
    print(f"\nModel Path: {model['model_path']}")
    
    if model['artifact_paths']:
        print(f"\nArtifacts:")
        for artifact in model['artifact_paths']:
            print(f"  - {artifact}")
    
    print(f"\nTraining Data:")
    print(f"  Start: {model['training_data_start_date']}")
    print(f"  End: {model['training_data_end_date']}")
    print(f"  Samples: {model['training_samples']}")
    print(f"  Date: {model['training_date']}")
    
    if model['validation_metrics']:
        print(f"\nValidation Metrics:")
        for key, value in model['validation_metrics'].items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
    
    if model['test_metrics']:
        print(f"\nTest Metrics:")
        for key, value in model['test_metrics'].items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
    
    if model['hyperparameters']:
        print(f"\nHyperparameters:")
        for key, value in model['hyperparameters'].items():
            print(f"  {key}: {value}")
    
    if model['code_commit_hash']:
        print(f"\nCode Commit: {model['code_commit_hash'][:8]}")
    
    if model['notes']:
        print(f"\nNotes: {model['notes']}")
    
    print(f"\nCreated: {model['created_at']}")
    if model['promoted_at']:
        print(f"Promoted: {model['promoted_at']}")
    if model['deprecated_at']:
        print(f"Deprecated: {model['deprecated_at']}")


def compare_models(args):
    """Compare two model versions."""
    registry = ModelRegistry()
    
    try:
        comparison = registry.compare_models(args.new, args.current)
        
        print(f"\n{'='*60}")
        print(f"Model Comparison")
        print(f"{'='*60}")
        print(f"New Model: {comparison['new_version']} (ID: {args.new})")
        print(f"Current Production: {comparison['current_version']} (ID: {args.current})")
        print(f"\n{'Metric':<20} {'New':<12} {'Current':<12} {'Improvement':<15} {'Status':<10}")
        print("-" * 70)
        
        for metric, data in comparison['improvements'].items():
            status = "✓ Improved" if data['improved'] else "⊘ Degraded"
            improvement = f"{data['improvement_pct']:+.2f}%"
            print(f"{metric:<20} {data['new']:<12.4f} {data['current']:<12.4f} {improvement:<15} {status:<10}")
        
        print(f"\nRecommendation: {'PROMOTE' if comparison['should_promote'] else 'DO NOT PROMOTE'}")
        print(f"Reason: {comparison['reason']}")
        
    except Exception as e:
        print(f"Error comparing models: {e}")
        sys.exit(1)


def promote_model(args):
    """Promote a model to production."""
    registry = ModelRegistry()
    
    try:
        registry.promote_to_production(args.id, args.reason)
        model = registry.get_model_version(args.id)
        print(f"✓ Successfully promoted model {model['version']} to production")
        print(f"  Exchange: {model['exchange']}")
        print(f"  Type: {model['model_type']}")
    except Exception as e:
        print(f"Error promoting model: {e}")
        sys.exit(1)


def rollback_model(args):
    """Rollback to a previous model version."""
    registry = ModelRegistry()
    
    try:
        registry.rollback_model(
            exchange=args.exchange,
            model_type=args.type,
            target_version_id=args.target,
            reason=args.reason
        )
        model = registry.get_model_version(args.target)
        print(f"✓ Successfully rolled back to model {model['version']}")
        print(f"  Exchange: {model['exchange']}")
        print(f"  Type: {model['model_type']}")
    except Exception as e:
        print(f"Error rolling back model: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Model Registry Management CLI')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List models')
    list_parser.add_argument('--exchange', help='Filter by exchange')
    list_parser.add_argument('--type', help='Filter by model type')
    list_parser.add_argument('--production', action='store_true', help='Show only production models')
    list_parser.add_argument('--limit', type=int, default=50, help='Maximum number of results')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show model details')
    show_parser.add_argument('version_id', type=int, help='Model version ID')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two models')
    compare_parser.add_argument('--new', type=int, required=True, help='New model version ID')
    compare_parser.add_argument('--current', type=int, required=True, help='Current production model version ID')
    
    # Promote command
    promote_parser = subparsers.add_parser('promote', help='Promote model to production')
    promote_parser.add_argument('--id', type=int, required=True, help='Model version ID to promote')
    promote_parser.add_argument('--reason', help='Reason for promotion')
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback to previous model')
    rollback_parser.add_argument('--exchange', required=True, help='Exchange name')
    rollback_parser.add_argument('--type', required=True, help='Model type')
    rollback_parser.add_argument('--target', type=int, required=True, help='Target model version ID')
    rollback_parser.add_argument('--reason', help='Reason for rollback')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'list':
        list_models(args)
    elif args.command == 'show':
        show_model(args)
    elif args.command == 'compare':
        compare_models(args)
    elif args.command == 'promote':
        promote_model(args)
    elif args.command == 'rollback':
        rollback_model(args)


if __name__ == '__main__':
    main()
