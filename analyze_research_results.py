#!/usr/bin/env python3
"""
Analyze research results from train_orchestrator.py JSON output.

Usage:
    python analyze_research_results.py reports/research/NSE_research_20260121_204200.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

def analyze_research_file(filepath: Path) -> Dict[str, Any]:
    """Analyze a research JSON file and return comprehensive insights."""
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    exchange = data['exchange']
    config = data['config']
    segments = data['segments']
    best_by_family = data.get('best_by_family', {})
    
    # Aggregate metrics by family
    family_metrics = defaultdict(list)
    family_params = defaultdict(list)
    
    for segment in segments:
        family = segment['family']
        if segment['metrics'].get('f1_macro', 0) > 0:  # Skip failed runs
            family_metrics[family].append({
                'segment_id': segment['segment_id'],
                'f1_macro': segment['metrics']['f1_macro'],
                'f1_weighted': segment['metrics'].get('f1_weighted', 0),
                'accuracy': segment['metrics'].get('accuracy', 0),
                'precision_macro': segment['metrics'].get('precision_macro', 0),
                'recall_macro': segment['metrics'].get('recall_macro', 0),
            })
            if segment.get('best_params'):
                family_params[family].append(segment['best_params'])
    
    # Calculate statistics per family
    family_stats = {}
    for family, metrics_list in family_metrics.items():
        if not metrics_list:
            continue
        
        f1_scores = [m['f1_macro'] for m in metrics_list]
        acc_scores = [m['accuracy'] for m in metrics_list]
        
        family_stats[family] = {
            'count': len(metrics_list),
            'f1_macro': {
                'mean': sum(f1_scores) / len(f1_scores),
                'max': max(f1_scores),
                'min': min(f1_scores),
                'std': (sum((x - sum(f1_scores)/len(f1_scores))**2 for x in f1_scores) / len(f1_scores))**0.5
            },
            'accuracy': {
                'mean': sum(acc_scores) / len(acc_scores),
                'max': max(acc_scores),
                'min': min(acc_scores),
            },
            'best_segment': max(metrics_list, key=lambda x: x['f1_macro']),
        }
    
    # Find overall best
    best_family = max(family_stats.keys(), 
                     key=lambda k: family_stats[k]['f1_macro']['mean'])
    
    return {
        'exchange': exchange,
        'config': config,
        'family_stats': family_stats,
        'best_by_family': best_by_family,
        'overall_best_family': best_family,
        'total_segments': len(segments),
    }


def print_analysis(analysis: Dict[str, Any]):
    """Print formatted analysis results."""
    
    print("=" * 80)
    print(f"RESEARCH ANALYSIS: {analysis['exchange']}")
    print("=" * 80)
    print()
    
    # Configuration
    config = analysis['config']
    print("Configuration:")
    print(f"  - Lookback: {config['days']} days")
    print(f"  - Window: {config['window_days']} days")
    print(f"  - Step: {config['step_days']} days")
    print(f"  - Optuna Trials: {config['optuna_trials']}")
    print(f"  - Families Tested: {', '.join(config['families'])}")
    print(f"  - Total Segments: {analysis['total_segments']}")
    print()
    
    # Family comparison
    print("=" * 80)
    print("MODEL FAMILY COMPARISON")
    print("=" * 80)
    print()
    
    family_stats = analysis['family_stats']
    best_family = analysis['overall_best_family']
    
    # Sort by mean F1
    sorted_families = sorted(family_stats.items(), 
                            key=lambda x: x[1]['f1_macro']['mean'], 
                            reverse=True)
    
    for rank, (family, stats) in enumerate(sorted_families, 1):
        marker = "🏆" if family == best_family else "  "
        print(f"{marker} {rank}. {family}")
        print(f"   Segments: {stats['count']}")
        print(f"   F1 Macro: {stats['f1_macro']['mean']:.4f} (mean) | "
              f"{stats['f1_macro']['max']:.4f} (max) | "
              f"{stats['f1_macro']['min']:.4f} (min) | "
              f"±{stats['f1_macro']['std']:.4f} (std)")
        print(f"   Accuracy: {stats['accuracy']['mean']:.4f} (mean) | "
              f"{stats['accuracy']['max']:.4f} (max) | "
              f"{stats['accuracy']['min']:.4f} (min)")
        print(f"   Best Segment: {stats['best_segment']['segment_id']} "
              f"(F1={stats['best_segment']['f1_macro']:.4f}, "
              f"Acc={stats['best_segment']['accuracy']:.4f})")
        print()
    
    # Best parameters per family
    print("=" * 80)
    print("BEST PARAMETERS BY FAMILY")
    print("=" * 80)
    print()
    
    best_by_family = analysis['best_by_family']
    for family, best_result in best_by_family.items():
        if not best_result.get('params'):
            continue
        
        print(f"{family}:")
        print(f"  Segment: {best_result['segment_id']}")
        print(f"  F1 Macro: {best_result['metrics']['f1_macro']:.4f}")
        print(f"  Accuracy: {best_result['metrics']['accuracy']:.4f}")
        print("  Parameters:")
        for key, value in best_result['params'].items():
            if isinstance(value, float):
                print(f"    {key}: {value:.6f}")
            else:
                print(f"    {key}: {value}")
        print()
    
    # Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    best_result = best_by_family.get(best_family, {})
    print(f"✓ Best Model Family: {best_family}")
    print(f"  - Mean F1 Macro: {family_stats[best_family]['f1_macro']['mean']:.4f}")
    print(f"  - Best Segment F1: {best_result.get('metrics', {}).get('f1_macro', 0):.4f}")
    print()
    
    if best_family == "XGBoost":
        print("  → Use XGBoost for production training")
        print("  → Parameters from best segment:")
        params = best_result.get('params', {})
        print(f"     - n_estimators: {params.get('n_estimators', 'N/A')}")
        print(f"     - learning_rate: {params.get('learning_rate', 'N/A'):.6f}")
        print(f"     - max_depth: {params.get('max_depth', 'N/A')}")
        print(f"     - subsample: {params.get('subsample', 'N/A'):.6f}")
        print(f"     - colsample_bytree: {params.get('colsample_bytree', 'N/A'):.6f}")
        print(f"     - reg_lambda: {params.get('reg_lambda', 'N/A'):.6f}")
    elif best_family == "LightGBM":
        print("  → Use LightGBM for production training")
        print("  → Parameters from best segment:")
        params = best_result.get('params', {})
        print(f"     - n_estimators: {params.get('n_estimators', 'N/A')}")
        print(f"     - learning_rate: {params.get('learning_rate', 'N/A'):.6f}")
        print(f"     - num_leaves: {params.get('num_leaves', 'N/A')}")
        print(f"     - subsample: {params.get('subsample', 'N/A'):.6f}")
        print(f"     - colsample_bytree: {params.get('colsample_bytree', 'N/A'):.6f}")
    elif best_family == "CatBoost":
        print("  → Use CatBoost for production training")
        print("  → Parameters from best segment:")
        params = best_result.get('params', {})
        print(f"     - iterations: {params.get('iterations', 'N/A')}")
        print(f"     - learning_rate: {params.get('learning_rate', 'N/A'):.6f}")
        print(f"     - depth: {params.get('depth', 'N/A')}")
        print(f"     - l2_leaf_reg: {params.get('l2_leaf_reg', 'N/A'):.6f}")
    
    print()
    print("  Next Steps:")
    print("  1. Update train_model.py DEFAULT_MODEL_PARAMS with best parameters")
    print("  2. Run full production training: python train_model.py --exchange NSE --days 90")
    print("  3. Validate on out-of-sample data")
    print("  4. Deploy to live trading system")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_research_results.py <research_json_file>")
        sys.exit(1)
    
    filepath = Path(sys.argv[1])
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    try:
        analysis = analyze_research_file(filepath)
        print_analysis(analysis)
    except Exception as e:
        print(f"Error analyzing file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
