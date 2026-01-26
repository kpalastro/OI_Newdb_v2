#!/usr/bin/env python3
"""
Reverse Engineering Analysis: ITM Feature Importance

This script analyzes the role and importance of ITM features in trading decisions:
- ITM CE Δ% (itm_oi_ce_pct_change_3m_wavg)
- ITM PE Δ% (itm_oi_pe_pct_change_3m_wavg)
- ITM CE Vol Δ% (itm_volume_ce_pct_change_3m_wavg)
- ITM PE Vol Δ% (itm_volume_pe_pct_change_3m_wavg)

Usage:
    python scripts/analyze_itm_features_importance.py --days 30
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.analyze_trade_logs import load_trade_logs

try:
    import database_new as db
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("Warning: database_new not available. Will use trade logs only.")


def load_features_for_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Load feature data from database for each trade entry time."""
    
    if not DB_AVAILABLE:
        print("Database not available. Cannot load features.")
        return trades
    
    print(f"\nLoading features for {len(trades)} trades...")
    
    # Filter to closed trades with valid timestamps
    closed_trades = trades[trades.get('status', pd.Series()) == 'CLOSED'].copy()
    closed_trades = closed_trades[closed_trades['entry_timestamp'].notna()].copy()
    
    if len(closed_trades) == 0:
        print("No closed trades with timestamps found.")
        return trades
    
    # Convert timestamps - assume IST if no timezone
    closed_trades['entry_timestamp'] = pd.to_datetime(closed_trades['entry_timestamp'])
    
    # If no timezone, assume IST (UTC+5:30)
    if closed_trades['entry_timestamp'].dt.tz is None:
        closed_trades['entry_timestamp'] = closed_trades['entry_timestamp'].dt.tz_localize('Asia/Kolkata')
    
    # Convert to UTC for database query
    closed_trades['entry_timestamp_utc'] = closed_trades['entry_timestamp'].dt.tz_convert('UTC')
    
    # Get unique exchanges
    exchanges = closed_trades['exchange'].unique()
    
    enriched_trades = []
    
    # Query all features at once for efficiency
    try:
        conn = db.get_db_connection()
        if not conn:
            print("Could not get database connection.")
            return trades
        
        cursor = conn.cursor()
        
        for exchange in exchanges:
            exchange_trades = closed_trades[closed_trades['exchange'] == exchange].copy()
            
            if len(exchange_trades) == 0:
                continue
            
            # Get time range (use UTC timestamps)
            min_time = exchange_trades['entry_timestamp_utc'].min() - timedelta(minutes=10)
            max_time = exchange_trades['entry_timestamp_utc'].max() + timedelta(minutes=10)
            
            # Query all features in this time range - use direct columns
            query = """
                SELECT 
                    timestamp,
                    itm_oi_ce_pct_change_3m_wavg,
                    itm_oi_pe_pct_change_3m_wavg,
                    feature_payload
                FROM ml_features
                WHERE exchange = %s
                  AND timestamp BETWEEN %s AND %s
                ORDER BY timestamp
            """
            
            cursor.execute(query, (exchange, min_time, max_time))
            feature_rows = cursor.fetchall()
            
            if not feature_rows:
                continue
            
            # Create a lookup by timestamp (rounded to nearest minute)
            feature_lookup = {}
            for row in feature_rows:
                feat_time = row[0]
                itm_ce_delta = row[1]
                itm_pe_delta = row[2]
                feat_payload_str = row[3] if len(row) > 3 else None
                
                # Try to parse feature_payload if it's JSON string
                feat_payload = {}
                if feat_payload_str:
                    try:
                        import json
                        feat_payload = json.loads(feat_payload_str) if isinstance(feat_payload_str, str) else feat_payload_str
                    except:
                        pass
                
                time_key = feat_time.replace(second=0, microsecond=0)
                feature_lookup[time_key] = {
                    'itm_ce_delta_pct': itm_ce_delta,
                    'itm_pe_delta_pct': itm_pe_delta,
                    'itm_ce_vol_delta_pct': feat_payload.get('itm_volume_ce_pct_change_3m_wavg') if isinstance(feat_payload, dict) else None,
                    'itm_pe_vol_delta_pct': feat_payload.get('itm_volume_pe_pct_change_3m_wavg') if isinstance(feat_payload, dict) else None,
                }
            
            # Match trades to features
            matched_count = 0
            for idx, trade in exchange_trades.iterrows():
                entry_time_utc = trade['entry_timestamp_utc']
                
                # Try exact match, then ±1, ±2, ±3 minutes
                features = None
                for offset in [0, -1, 1, -2, 2, -3, 3]:
                    check_time = entry_time_utc + timedelta(minutes=offset)
                    check_time_rounded = check_time.replace(second=0, microsecond=0)
                    if check_time_rounded in feature_lookup:
                        features = feature_lookup[check_time_rounded]
                        matched_count += 1
                        break
                
                if features:
                    trade_dict = trade.to_dict()
                    trade_dict['itm_ce_delta_pct'] = features.get('itm_ce_delta_pct')
                    trade_dict['itm_pe_delta_pct'] = features.get('itm_pe_delta_pct')
                    trade_dict['itm_ce_vol_delta_pct'] = features.get('itm_ce_vol_delta_pct')
                    trade_dict['itm_pe_vol_delta_pct'] = features.get('itm_pe_vol_delta_pct')
                    
                    enriched_trades.append(trade_dict)
        
        cursor.close()
        db.release_db_connection(conn)
        
        if matched_count > 0:
            print(f"  Matched {matched_count} trades to features")
    
    except Exception as e:
        print(f"Error loading features: {e}")
        import traceback
        traceback.print_exc()
        return trades
    
    if not enriched_trades:
        print("No features found in database. Using trade logs only.")
        return trades
    
    enriched_df = pd.DataFrame(enriched_trades)
    print(f"Successfully loaded features for {len(enriched_df)} trades ({len(enriched_df)/len(closed_trades)*100:.1f}% of closed trades)")
    
    return enriched_df


def analyze_feature_distributions(trades: pd.DataFrame) -> Dict[str, Any]:
    """Analyze distribution of ITM features for winners vs losers."""
    
    closed = trades[trades.get('status', pd.Series()) == 'CLOSED'].copy()
    
    if len(closed) == 0:
        return {}
    
    # Check which features are available
    itm_features = {
        'itm_ce_delta_pct': 'ITM CE Δ%',
        'itm_pe_delta_pct': 'ITM PE Δ%',
        'itm_ce_vol_delta_pct': 'ITM CE Vol Δ%',
        'itm_pe_vol_delta_pct': 'ITM PE Vol Δ%'
    }
    
    available_features = {k: v for k, v in itm_features.items() if k in closed.columns}
    
    if not available_features:
        print("No ITM features found in trade data.")
        return {}
    
    print("\n" + "="*60)
    print("ITM FEATURE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    winners = closed[closed.get('is_winner', pd.Series()) == True]
    losers = closed[closed.get('is_winner', pd.Series()) == False]
    
    results = {}
    
    for feature_key, feature_name in available_features.items():
        if feature_key not in closed.columns:
            continue
        
        # Remove NaN values
        feature_data = closed[feature_key].dropna()
        if len(feature_data) == 0:
            continue
        
        winner_data = winners[feature_key].dropna()
        loser_data = losers[feature_key].dropna()
        
        if len(winner_data) == 0 or len(loser_data) == 0:
            continue
        
        print(f"\n{feature_name} ({feature_key}):")
        print("-" * 60)
        
        # Overall statistics
        print(f"  Overall:")
        print(f"    Mean: {feature_data.mean():.4f}")
        print(f"    Median: {feature_data.median():.4f}")
        print(f"    Std: {feature_data.std():.4f}")
        print(f"    Range: [{feature_data.min():.4f}, {feature_data.max():.4f}]")
        
        # Winner statistics
        print(f"  Winners ({len(winner_data)} trades):")
        print(f"    Mean: {winner_data.mean():.4f}")
        print(f"    Median: {winner_data.median():.4f}")
        print(f"    Std: {winner_data.std():.4f}")
        
        # Loser statistics
        print(f"  Losers ({len(loser_data)} trades):")
        print(f"    Mean: {loser_data.mean():.4f}")
        print(f"    Median: {loser_data.median():.4f}")
        print(f"    Std: {loser_data.std():.4f}")
        
        # Difference
        mean_diff = winner_data.mean() - loser_data.mean()
        median_diff = winner_data.median() - loser_data.median()
        
        print(f"  Difference (Winner - Loser):")
        print(f"    Mean Diff: {mean_diff:+.4f}")
        print(f"    Median Diff: {median_diff:+.4f}")
        
        # Statistical significance (simple t-test approximation)
        if len(winner_data) > 10 and len(loser_data) > 10:
            from scipy import stats
            try:
                t_stat, p_value = stats.ttest_ind(winner_data, loser_data)
                print(f"    T-statistic: {t_stat:.4f}")
                print(f"    P-value: {p_value:.4f}")
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"    Significance: {significance}")
            except:
                pass
        
        results[feature_key] = {
            'feature_name': feature_name,
            'overall': {
                'mean': float(feature_data.mean()),
                'median': float(feature_data.median()),
                'std': float(feature_data.std()),
                'min': float(feature_data.min()),
                'max': float(feature_data.max())
            },
            'winners': {
                'mean': float(winner_data.mean()),
                'median': float(winner_data.median()),
                'std': float(winner_data.std()),
                'count': len(winner_data)
            },
            'losers': {
                'mean': float(loser_data.mean()),
                'median': float(loser_data.median()),
                'std': float(loser_data.std()),
                'count': len(loser_data)
            },
            'difference': {
                'mean_diff': float(mean_diff),
                'median_diff': float(median_diff)
            }
        }
    
    return results


def analyze_feature_correlation_with_pnl(trades: pd.DataFrame) -> Dict[str, Any]:
    """Analyze correlation between ITM features and PnL."""
    
    closed = trades[trades.get('status', pd.Series()) == 'CLOSED'].copy()
    
    if len(closed) == 0 or 'pnl' not in closed.columns:
        return {}
    
    itm_features = ['itm_ce_delta_pct', 'itm_pe_delta_pct', 
                    'itm_ce_vol_delta_pct', 'itm_pe_vol_delta_pct']
    
    available_features = [f for f in itm_features if f in closed.columns]
    
    if not available_features:
        return {}
    
    print("\n" + "="*60)
    print("ITM FEATURE CORRELATION WITH PnL")
    print("="*60)
    
    correlations = {}
    
    for feature in available_features:
        # Remove NaN
        data = closed[[feature, 'pnl']].dropna()
        
        if len(data) < 10:
            continue
        
        corr = data[feature].corr(data['pnl'])
        correlations[feature] = float(corr)
        
        feature_name = {
            'itm_ce_delta_pct': 'ITM CE Δ%',
            'itm_pe_delta_pct': 'ITM PE Δ%',
            'itm_ce_vol_delta_pct': 'ITM CE Vol Δ%',
            'itm_pe_vol_delta_pct': 'ITM PE Vol Δ%'
        }.get(feature, feature)
        
        print(f"{feature_name:20s}: Correlation = {corr:+.4f}")
    
    return correlations


def find_optimal_feature_ranges(trades: pd.DataFrame) -> Dict[str, Any]:
    """Find optimal ranges for ITM features that maximize win rate."""
    
    closed = trades[trades.get('status', pd.Series()) == 'CLOSED'].copy()
    
    if len(closed) == 0:
        return {}
    
    itm_features = {
        'itm_ce_delta_pct': 'ITM CE Δ%',
        'itm_pe_delta_pct': 'ITM PE Δ%',
        'itm_ce_vol_delta_pct': 'ITM CE Vol Δ%',
        'itm_pe_vol_delta_pct': 'ITM PE Vol Δ%'
    }
    
    available_features = {k: v for k, v in itm_features.items() if k in closed.columns}
    
    if not available_features:
        return {}
    
    print("\n" + "="*60)
    print("OPTIMAL FEATURE RANGES (Maximizing Win Rate)")
    print("="*60)
    
    optimal_ranges = {}
    
    for feature_key, feature_name in available_features.items():
        if feature_key not in closed.columns:
            continue
        
        feature_data = closed[[feature_key, 'is_winner']].dropna()
        
        if len(feature_data) < 20:
            continue
        
        # Create quantile-based buckets
        feature_data['bucket'] = pd.qcut(
            feature_data[feature_key],
            q=5,
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
            duplicates='drop'
        )
        
        bucket_analysis = feature_data.groupby('bucket').agg({
            'is_winner': ['mean', 'count'],
            feature_key: ['mean', 'min', 'max']
        })
        
        print(f"\n{feature_name}:")
        print(bucket_analysis)
        
        # Find best bucket
        best_bucket = bucket_analysis[('is_winner', 'mean')].idxmax()
        best_win_rate = bucket_analysis.loc[best_bucket, ('is_winner', 'mean')]
        best_range = (
            bucket_analysis.loc[best_bucket, (feature_key, 'min')],
            bucket_analysis.loc[best_bucket, (feature_key, 'max')]
        )
        
        optimal_ranges[feature_key] = {
            'feature_name': feature_name,
            'best_bucket': str(best_bucket),
            'best_win_rate': float(best_win_rate),
            'optimal_range': (float(best_range[0]), float(best_range[1])),
            'bucket_analysis': bucket_analysis.to_dict()
        }
        
        print(f"  Best Range: {best_range[0]:.4f} to {best_range[1]:.4f} (Win Rate: {best_win_rate:.2%})")
    
    return optimal_ranges


def analyze_feature_combinations(trades: pd.DataFrame) -> Dict[str, Any]:
    """Analyze combinations of ITM features (e.g., CE vs PE divergence)."""
    
    closed = trades[trades.get('status', pd.Series()) == 'CLOSED'].copy()
    
    if len(closed) == 0:
        return {}
    
    print("\n" + "="*60)
    print("ITM FEATURE COMBINATIONS ANALYSIS")
    print("="*60)
    
    combinations = {}
    
    # 1. OI Delta Divergence (PE - CE)
    if 'itm_pe_delta_pct' in closed.columns and 'itm_ce_delta_pct' in closed.columns:
        closed['itm_oi_divergence'] = (
            closed['itm_pe_delta_pct'] - closed['itm_ce_delta_pct']
        ).fillna(0)
        
        divergence_data = closed[['itm_oi_divergence', 'is_winner', 'pnl']].dropna()
        
        if len(divergence_data) > 10:
            # Positive divergence = PE > CE (bearish)
            # Negative divergence = CE > PE (bullish)
            positive_div = divergence_data[divergence_data['itm_oi_divergence'] > 0]
            negative_div = divergence_data[divergence_data['itm_oi_divergence'] < 0]
            
            print("\nOI Delta Divergence (PE Δ% - CE Δ%):")
            print(f"  Positive Divergence (PE > CE, Bearish): {len(positive_div)} trades")
            if len(positive_div) > 0:
                print(f"    Win Rate: {positive_div['is_winner'].mean():.2%}")
                print(f"    Avg PnL: {positive_div['pnl'].mean():.2f}")
            
            print(f"  Negative Divergence (CE > PE, Bullish): {len(negative_div)} trades")
            if len(negative_div) > 0:
                print(f"    Win Rate: {negative_div['is_winner'].mean():.2%}")
                print(f"    Avg PnL: {negative_div['pnl'].mean():.2f}")
            
            combinations['oi_divergence'] = {
                'positive_divergence': {
                    'count': len(positive_div),
                    'win_rate': float(positive_div['is_winner'].mean()) if len(positive_div) > 0 else 0,
                    'avg_pnl': float(positive_div['pnl'].mean()) if len(positive_div) > 0 else 0
                },
                'negative_divergence': {
                    'count': len(negative_div),
                    'win_rate': float(negative_div['is_winner'].mean()) if len(negative_div) > 0 else 0,
                    'avg_pnl': float(negative_div['pnl'].mean()) if len(negative_div) > 0 else 0
                }
            }
    
    # 2. Volume Delta Divergence (PE - CE)
    if 'itm_pe_vol_delta_pct' in closed.columns and 'itm_ce_vol_delta_pct' in closed.columns:
        closed['itm_vol_divergence'] = (
            closed['itm_pe_vol_delta_pct'] - closed['itm_ce_vol_delta_pct']
        ).fillna(0)
        
        vol_divergence_data = closed[['itm_vol_divergence', 'is_winner', 'pnl']].dropna()
        
        if len(vol_divergence_data) > 10:
            positive_vol_div = vol_divergence_data[vol_divergence_data['itm_vol_divergence'] > 0]
            negative_vol_div = vol_divergence_data[vol_divergence_data['itm_vol_divergence'] < 0]
            
            print("\nVolume Delta Divergence (PE Vol Δ% - CE Vol Δ%):")
            print(f"  Positive Divergence (PE Vol > CE Vol): {len(positive_vol_div)} trades")
            if len(positive_vol_div) > 0:
                print(f"    Win Rate: {positive_vol_div['is_winner'].mean():.2%}")
                print(f"    Avg PnL: {positive_vol_div['pnl'].mean():.2f}")
            
            print(f"  Negative Divergence (CE Vol > PE Vol): {len(negative_vol_div)} trades")
            if len(negative_vol_div) > 0:
                print(f"    Win Rate: {negative_vol_div['is_winner'].mean():.2%}")
                print(f"    Avg PnL: {negative_vol_div['pnl'].mean():.2f}")
            
            combinations['volume_divergence'] = {
                'positive_divergence': {
                    'count': len(positive_vol_div),
                    'win_rate': float(positive_vol_div['is_winner'].mean()) if len(positive_vol_div) > 0 else 0,
                    'avg_pnl': float(positive_vol_div['pnl'].mean()) if len(positive_vol_div) > 0 else 0
                },
                'negative_divergence': {
                    'count': len(negative_vol_div),
                    'win_rate': float(negative_vol_div['is_winner'].mean()) if len(negative_vol_div) > 0 else 0,
                    'avg_pnl': float(negative_vol_div['pnl'].mean()) if len(negative_vol_div) > 0 else 0
                }
            }
    
    # 3. Signal Agreement (OI and Volume agree)
    if 'itm_oi_divergence' in closed.columns and 'itm_vol_divergence' in closed.columns:
        closed['signal_agreement'] = (
            (closed['itm_oi_divergence'] > 0) == (closed['itm_vol_divergence'] > 0)
        )
        
        agreement_data = closed[['signal_agreement', 'is_winner', 'pnl']].dropna()
        
        if len(agreement_data) > 10:
            agree = agreement_data[agreement_data['signal_agreement'] == True]
            disagree = agreement_data[agreement_data['signal_agreement'] == False]
            
            print("\nSignal Agreement (OI and Volume signals agree):")
            print(f"  Agreement: {len(agree)} trades")
            if len(agree) > 0:
                print(f"    Win Rate: {agree['is_winner'].mean():.2%}")
                print(f"    Avg PnL: {agree['pnl'].mean():.2f}")
            
            print(f"  Disagreement: {len(disagree)} trades")
            if len(disagree) > 0:
                print(f"    Win Rate: {disagree['is_winner'].mean():.2%}")
                print(f"    Avg PnL: {disagree['pnl'].mean():.2f}")
            
            combinations['signal_agreement'] = {
                'agreement': {
                    'count': len(agree),
                    'win_rate': float(agree['is_winner'].mean()) if len(agree) > 0 else 0,
                    'avg_pnl': float(agree['pnl'].mean()) if len(agree) > 0 else 0
                },
                'disagreement': {
                    'count': len(disagree),
                    'win_rate': float(disagree['is_winner'].mean()) if len(disagree) > 0 else 0,
                    'avg_pnl': float(disagree['pnl'].mean()) if len(disagree) > 0 else 0
                }
            }
    
    return combinations


def generate_feature_importance_summary(analyses: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary of feature importance and recommendations."""
    
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE SUMMARY & RECOMMENDATIONS")
    print("="*60)
    
    summary = {}
    
    # Feature distributions
    if 'distributions' in analyses:
        print("\n1. FEATURE IMPORTANCE (Based on Winner-Loser Difference):")
        print("-" * 60)
        
        feature_importance = []
        for feature_key, data in analyses['distributions'].items():
            importance_score = abs(data['difference']['mean_diff']) / (data['overall']['std'] + 1e-6)
            feature_importance.append({
                'feature': data['feature_name'],
                'importance_score': importance_score,
                'mean_diff': data['difference']['mean_diff'],
                'interpretation': 'Higher in winners' if data['difference']['mean_diff'] > 0 else 'Lower in winners'
            })
        
        feature_importance.sort(key=lambda x: x['importance_score'], reverse=True)
        
        for i, feat in enumerate(feature_importance, 1):
            print(f"{i}. {feat['feature']:20s}")
            print(f"   Importance Score: {feat['importance_score']:.4f}")
            print(f"   {feat['interpretation']} (Diff: {feat['mean_diff']:+.4f})")
        
        summary['feature_importance_ranking'] = feature_importance
    
    # Optimal ranges
    if 'optimal_ranges' in analyses:
        print("\n2. OPTIMAL FEATURE RANGES:")
        print("-" * 60)
        for feature_key, data in analyses['optimal_ranges'].items():
            print(f"{data['feature_name']:20s}:")
            print(f"  Optimal Range: [{data['optimal_range'][0]:.4f}, {data['optimal_range'][1]:.4f}]")
            print(f"  Win Rate in Range: {data['best_win_rate']:.2%}")
        
        summary['optimal_ranges'] = analyses['optimal_ranges']
    
    # Correlations
    if 'correlations' in analyses:
        print("\n3. CORRELATION WITH PnL:")
        print("-" * 60)
        sorted_corr = sorted(analyses['correlations'].items(), key=lambda x: abs(x[1]), reverse=True)
        for feature_key, corr in sorted_corr:
            feature_name = {
                'itm_ce_delta_pct': 'ITM CE Δ%',
                'itm_pe_delta_pct': 'ITM PE Δ%',
                'itm_ce_vol_delta_pct': 'ITM CE Vol Δ%',
                'itm_pe_vol_delta_pct': 'ITM PE Vol Δ%'
            }.get(feature_key, feature_key)
            print(f"{feature_name:20s}: {corr:+.4f}")
        
        summary['correlations'] = analyses['correlations']
    
    # Combinations
    if 'combinations' in analyses:
        print("\n4. FEATURE COMBINATIONS:")
        print("-" * 60)
        if 'signal_agreement' in analyses['combinations']:
            agree = analyses['combinations']['signal_agreement']['agreement']
            disagree = analyses['combinations']['signal_agreement']['disagreement']
            print(f"Signal Agreement (OI & Volume agree):")
            print(f"  When signals agree: {agree['win_rate']:.2%} win rate ({agree['count']} trades)")
            print(f"  When signals disagree: {disagree['win_rate']:.2%} win rate ({disagree['count']} trades)")
        
        summary['combinations'] = analyses['combinations']
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Analyze ITM feature importance using reverse engineering'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days to analyze (default: 30)'
    )
    parser.add_argument(
        '--exchange',
        type=str,
        default=None,
        choices=['NSE', 'BSE'],
        help='Filter by exchange (default: all exchanges)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='reports/itm_feature_importance.json',
        help='Output file (default: reports/itm_feature_importance.json)'
    )
    
    args = parser.parse_args()
    
    print(f"Loading trade logs for last {args.days} days...")
    trades = load_trade_logs(days=args.days)
    
    if len(trades) == 0:
        print("No trade logs found. Exiting.")
        return 1
    
    print(f"Found {len(trades)} total trades")
    
    # Filter by exchange if specified
    if args.exchange:
        trades = trades[trades['exchange'] == args.exchange].copy()
        print(f"Filtered to {len(trades)} {args.exchange} trades")
        if len(trades) == 0:
            print(f"No trades found for exchange {args.exchange}. Exiting.")
            return 1
    
    # Load features from database
    print("\nEnriching trades with feature data...")
    enriched_trades = load_features_for_trades(trades)
    
    # Run all analyses
    analyses = {}
    
    analyses['distributions'] = analyze_feature_distributions(enriched_trades)
    analyses['correlations'] = analyze_feature_correlation_with_pnl(enriched_trades)
    analyses['optimal_ranges'] = find_optimal_feature_ranges(enriched_trades)
    analyses['combinations'] = analyze_feature_combinations(enriched_trades)
    
    # Generate summary
    summary = generate_feature_importance_summary(analyses)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def convert_for_json(obj):
        """Recursively convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {str(k): convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(item) for item in obj]
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    output_data = {
        'analysis_date': datetime.now().isoformat(),
        'days_analyzed': args.days,
        'exchange': args.exchange or 'ALL',
        'total_trades': len(trades),
        'trades_with_features': len(enriched_trades),
        'analyses': convert_for_json(analyses),
        'summary': convert_for_json(summary)
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n\nAnalysis saved to: {output_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
