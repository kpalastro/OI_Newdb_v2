"""
Analyze RL Model Contribution to Trading Decisions

This script analyzes logs and metrics to determine how much RL models
contribute to trading decisions vs other models (LightGBM, DL).
"""
import json
import logging
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any

import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def analyze_recommendation_logs(log_dir: str = "logs/recommendations") -> pd.DataFrame:
    """
    Analyze recommendation logs to see RL contribution.
    
    Returns DataFrame with columns:
    - timestamp
    - exchange
    - signal_source (lightgbm, dl, rl, ensemble)
    - signal
    - confidence
    - rl_used (bool)
    - rl_contribution (if ensemble, what was RL's vote)
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        LOGGER.warning(f"Log directory {log_dir} does not exist")
        return pd.DataFrame()
    
    records = []
    
    # Read all JSONL files
    for log_file in sorted(log_path.glob("*.jsonl")):
        LOGGER.info(f"Reading {log_file}")
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except json.JSONDecodeError:
                    continue
    
    if not records:
        LOGGER.warning("No records found in logs")
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    # Extract RL contribution
    if 'signal_source' in df.columns:
        df['rl_used'] = df['signal_source'].str.contains('rl', case=False, na=False)
        df['rl_primary'] = df['signal_source'] == 'rl'
        df['rl_in_ensemble'] = df['signal_source'] == 'ensemble'
    else:
        df['rl_used'] = False
        df['rl_primary'] = False
        df['rl_in_ensemble'] = False
    
    # Extract metadata for RL details
    if 'metadata' in df.columns:
        df['rl_position_size'] = df['metadata'].apply(
            lambda x: x.get('position_size', None) if isinstance(x, dict) else None
        )
        df['rl_algorithm'] = df['metadata'].apply(
            lambda x: x.get('rl_algorithm', None) if isinstance(x, dict) else None
        )
    
    return df


def analyze_execution_logs() -> pd.DataFrame:
    """
    Analyze execution logs to see RL executor usage.
    """
    # Check for execution logs in trade_logs
    trade_logs_dir = Path("trade_logs")
    if not trade_logs_dir.exists():
        return pd.DataFrame()
    
    records = []
    for log_file in sorted(trade_logs_dir.glob("*.csv")):
        try:
            df_trades = pd.read_csv(log_file)
            if 'entry_reason' in df_trades.columns:
                df_trades['rl_execution_used'] = df_trades['entry_reason'].str.contains('RL', case=False, na=False)
                records.append(df_trades)
        except Exception as e:
            LOGGER.debug(f"Error reading {log_file}: {e}")
    
    if records:
        return pd.concat(records, ignore_index=True)
    return pd.DataFrame()


def generate_rl_contribution_report(df_recommendations: pd.DataFrame, df_executions: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive report on RL contribution.
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'recommendations': {},
        'executions': {},
        'summary': {}
    }
    
    # Analyze Recommendations
    if not df_recommendations.empty:
        total_recommendations = len(df_recommendations)
        rl_primary = df_recommendations['rl_primary'].sum() if 'rl_primary' in df_recommendations.columns else 0
        rl_in_ensemble = df_recommendations['rl_in_ensemble'].sum() if 'rl_in_ensemble' in df_recommendations.columns else 0
        rl_used_any = df_recommendations['rl_used'].sum() if 'rl_used' in df_recommendations.columns else 0
        
        report['recommendations'] = {
            'total': total_recommendations,
            'rl_primary_count': int(rl_primary),
            'rl_primary_pct': float(rl_primary / total_recommendations * 100) if total_recommendations > 0 else 0.0,
            'rl_in_ensemble_count': int(rl_in_ensemble),
            'rl_in_ensemble_pct': float(rl_in_ensemble / total_recommendations * 100) if total_recommendations > 0 else 0.0,
            'rl_used_any_count': int(rl_used_any),
            'rl_used_any_pct': float(rl_used_any / total_recommendations * 100) if total_recommendations > 0 else 0.0,
        }
        
        # By exchange
        if 'exchange' in df_recommendations.columns:
            by_exchange = df_recommendations.groupby('exchange').agg({
                'rl_primary': 'sum',
                'rl_in_ensemble': 'sum',
                'rl_used': 'sum'
            }).to_dict('index')
            report['recommendations']['by_exchange'] = {
                k: {
                    'rl_primary': int(v.get('rl_primary', 0)),
                    'rl_ensemble': int(v.get('rl_in_ensemble', 0)),
                    'rl_any': int(v.get('rl_used', 0))
                }
                for k, v in by_exchange.items()
            }
    
    # Analyze Executions
    if not df_executions.empty and 'rl_execution_used' in df_executions.columns:
        total_executions = len(df_executions)
        rl_executions = df_executions['rl_execution_used'].sum()
        
        report['executions'] = {
            'total': total_executions,
            'rl_execution_count': int(rl_executions),
            'rl_execution_pct': float(rl_executions / total_executions * 100) if total_executions > 0 else 0.0,
        }
    
    # Summary
    report['summary'] = {
        'rl_contribution_level': 'high' if report['recommendations'].get('rl_used_any_pct', 0) > 50 else 
                                 'medium' if report['recommendations'].get('rl_used_any_pct', 0) > 20 else 'low',
        'rl_active': report['recommendations'].get('rl_used_any_count', 0) > 0 or report['executions'].get('rl_execution_count', 0) > 0
    }
    
    return report


def print_report(report: Dict[str, Any]):
    """Print a formatted report."""
    print("\n" + "="*60)
    print("RL MODEL CONTRIBUTION ANALYSIS")
    print("="*60)
    print(f"Generated: {report['timestamp']}\n")
    
    # Recommendations
    rec = report.get('recommendations', {})
    if rec:
        print("RECOMMENDATIONS (Signal Generation):")
        print(f"  Total Recommendations: {rec.get('total', 0)}")
        print(f"  RL as Primary Source: {rec.get('rl_primary_count', 0)} ({rec.get('rl_primary_pct', 0):.1f}%)")
        print(f"  RL in Ensemble: {rec.get('rl_in_ensemble_count', 0)} ({rec.get('rl_in_ensemble_pct', 0):.1f}%)")
        print(f"  RL Used (Any): {rec.get('rl_used_any_count', 0)} ({rec.get('rl_used_any_pct', 0):.1f}%)")
        
        if 'by_exchange' in rec:
            print("\n  By Exchange:")
            for exchange, stats in rec['by_exchange'].items():
                print(f"    {exchange}:")
                print(f"      Primary: {stats['rl_primary']}, Ensemble: {stats['rl_ensemble']}, Any: {stats['rl_any']}")
    
    # Executions
    exec_data = report.get('executions', {})
    if exec_data:
        print("\nEXECUTIONS (Order Placement):")
        print(f"  Total Executions: {exec_data.get('total', 0)}")
        print(f"  RL Execution Used: {exec_data.get('rl_execution_count', 0)} ({exec_data.get('rl_execution_pct', 0):.1f}%)")
    
    # Summary
    summary = report.get('summary', {})
    print("\nSUMMARY:")
    print(f"  RL Contribution Level: {summary.get('rl_contribution_level', 'unknown').upper()}")
    print(f"  RL Active: {summary.get('rl_active', False)}")
    print("\n" + "="*60 + "\n")


def main():
    """Main analysis function."""
    LOGGER.info("Analyzing RL contribution...")
    
    # Analyze recommendation logs
    df_rec = analyze_recommendation_logs()
    LOGGER.info(f"Found {len(df_rec)} recommendation records")
    
    # Analyze execution logs
    df_exec = analyze_execution_logs()
    LOGGER.info(f"Found {len(df_exec)} execution records")
    
    # Generate report
    report = generate_rl_contribution_report(df_rec, df_exec)
    
    # Print report
    print_report(report)
    
    # Save report
    output_file = Path("metrics/rl_contribution_report.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    LOGGER.info(f"Report saved to {output_file}")
    
    return report


if __name__ == "__main__":
    main()

