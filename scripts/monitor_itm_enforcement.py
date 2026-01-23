#!/usr/bin/env python3
"""
Monitor ITM enforcement effectiveness in paper trading.

This script analyzes paper trading metrics to see how often ITM filters
are applied and their impact on trading performance.
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from database_new import get_db_connection
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("Warning: Database not available. Cannot analyze paper trading metrics.")


def analyze_itm_filter_effectiveness(days: int = 7):
    """
    Analyze ITM filter effectiveness from paper trading metrics.
    
    Args:
        days: Number of days to analyze (default: 7)
    """
    if not DB_AVAILABLE:
        print("Database not available. Skipping analysis.")
        return
    
    print("=" * 70)
    print("ITM Enforcement Effectiveness Analysis")
    print("=" * 70)
    print(f"\nAnalyzing last {days} days of paper trading data...")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Query paper trading metrics
        cutoff_date = datetime.now() - timedelta(days=days)
        
        query = """
        SELECT 
            executed,
            reason,
            signal,
            confidence,
            metadata,
            timestamp
        FROM paper_trading_metrics
        WHERE timestamp >= %s
        ORDER BY timestamp DESC
        """
        
        cursor.execute(query, (cutoff_date,))
        rows = cursor.fetchall()
        
        if not rows:
            print(f"\nNo paper trading data found for the last {days} days.")
            return
        
        print(f"\nFound {len(rows)} paper trading records")
        
        # Analyze data
        stats = {
            'total_signals': len(rows),
            'executed': 0,
            'skipped': 0,
            'itm_filtered': 0,
            'itm_boosted': 0,
            'itm_reasons': defaultdict(int),
            'itm_warnings': defaultdict(int),
            'confidence_adjustments': [],
            'position_adjustments': [],
        }
        
        for row in rows:
            executed, reason, signal, confidence, metadata, timestamp = row
            
            if executed:
                stats['executed'] += 1
            else:
                stats['skipped'] += 1
                
                # Check if skipped due to ITM filter
                if reason and ('ITM' in reason.upper() or 'itm' in reason.lower()):
                    stats['itm_filtered'] += 1
                    print(f"\n  ITM Filtered: {reason}")
            
            # Parse metadata for ITM information
            if metadata:
                try:
                    import json
                    if isinstance(metadata, str):
                        meta = json.loads(metadata)
                    else:
                        meta = metadata
                    
                    # Check for ITM evaluation data
                    if 'itm_score' in meta:
                        stats['itm_boosted'] += 1
                        stats['confidence_adjustments'].append(
                            meta.get('itm_confidence_multiplier', 1.0)
                        )
                        stats['position_adjustments'].append(
                            meta.get('itm_position_multiplier', 1.0)
                        )
                        
                        # Collect reasons
                        if 'itm_reasons' in meta:
                            for reason in meta['itm_reasons']:
                                stats['itm_reasons'][reason] += 1
                        
                        if 'itm_warnings' in meta:
                            for warning in meta['itm_warnings']:
                                stats['itm_warnings'][warning] += 1
                
                except Exception as e:
                    pass  # Skip if metadata parsing fails
        
        # Print statistics
        print("\n" + "=" * 70)
        print("ITM Enforcement Statistics")
        print("=" * 70)
        
        print(f"\nOverall:")
        print(f"  Total signals: {stats['total_signals']}")
        print(f"  Executed: {stats['executed']} ({stats['executed']/stats['total_signals']*100:.1f}%)")
        print(f"  Skipped: {stats['skipped']} ({stats['skipped']/stats['total_signals']*100:.1f}%)")
        
        print(f"\nITM Filter Impact:")
        print(f"  ITM filtered (skipped): {stats['itm_filtered']}")
        print(f"  ITM boosted (executed): {stats['itm_boosted']}")
        
        if stats['itm_filtered'] > 0:
            filter_rate = stats['itm_filtered'] / stats['skipped'] * 100
            print(f"  ITM filter rate: {filter_rate:.1f}% of skipped trades")
        
        if stats['confidence_adjustments']:
            avg_confidence_mult = sum(stats['confidence_adjustments']) / len(stats['confidence_adjustments'])
            min_mult = min(stats['confidence_adjustments'])
            max_mult = max(stats['confidence_adjustments'])
            print(f"\nConfidence Adjustments:")
            print(f"  Average multiplier: {avg_confidence_mult:.2f}")
            print(f"  Range: {min_mult:.2f} - {max_mult:.2f}")
            print(f"  Boosted (>1.0): {sum(1 for x in stats['confidence_adjustments'] if x > 1.0)}")
            print(f"  Reduced (<1.0): {sum(1 for x in stats['confidence_adjustments'] if x < 1.0)}")
        
        if stats['position_adjustments']:
            avg_position_mult = sum(stats['position_adjustments']) / len(stats['position_adjustments'])
            min_mult = min(stats['position_adjustments'])
            max_mult = max(stats['position_adjustments'])
            print(f"\nPosition Size Adjustments:")
            print(f"  Average multiplier: {avg_position_mult:.2f}")
            print(f"  Range: {min_mult:.2f} - {max_mult:.2f}")
            print(f"  Increased (>1.0): {sum(1 for x in stats['position_adjustments'] if x > 1.0)}")
            print(f"  Decreased (<1.0): {sum(1 for x in stats['position_adjustments'] if x < 1.0)}")
        
        if stats['itm_reasons']:
            print(f"\nTop ITM Reasons (Positive):")
            for reason, count in sorted(stats['itm_reasons'].items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {reason}: {count}")
        
        if stats['itm_warnings']:
            print(f"\nTop ITM Warnings:")
            for warning, count in sorted(stats['itm_warnings'].items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {warning}: {count}")
        
        # Calculate effectiveness
        if stats['itm_filtered'] > 0:
            print(f"\n" + "=" * 70)
            print("Effectiveness Assessment")
            print("=" * 70)
            print(f"\n✓ ITM filter is actively filtering trades")
            print(f"✓ {stats['itm_boosted']} trades were boosted by ITM conditions")
            print(f"✓ {stats['itm_filtered']} trades were filtered by ITM conditions")
            
            if stats['itm_filtered'] / stats['total_signals'] > 0.05:
                print(f"\n⚠️  High filter rate ({stats['itm_filtered']/stats['total_signals']*100:.1f}%)")
                print("   Consider reviewing ITM thresholds if this seems too high")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"\nError analyzing data: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor ITM enforcement effectiveness')
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days to analyze (default: 7)'
    )
    
    args = parser.parse_args()
    
    analyze_itm_filter_effectiveness(days=args.days)
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
