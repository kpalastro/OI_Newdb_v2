"""
Standalone Phase 2 monitoring script.

Run this script to continuously monitor Phase 2 validation metrics.
Can be run alongside the main application or as a separate service.
"""
import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from metrics.phase2_metrics import get_metrics_collector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def print_summary(summary: dict, exchange: str) -> None:
    """Print formatted summary to console."""
    print("\n" + "=" * 80)
    print(f"Phase 2 Metrics Summary - {exchange}")
    print(f"Last Updated: {summary.get('last_updated', 'N/A')}")
    print(f"Period: Last {summary.get('period_hours', 24)} hours")
    print("=" * 80)
    
    # Data Quality
    dq = summary.get('data_quality', {})
    print("\n📊 DATA QUALITY")
    print(f"  Macro Availability:        {dq.get('macro_availability_pct', 0):.1f}%")
    print(f"  Depth Capture Success:      {dq.get('depth_capture_success_rate', 0):.1f}%")
    print(f"  Feature Engineering Errors: {dq.get('feature_engineering_errors', 0)}")
    print(f"  DB Write Success Rate:      {dq.get('database_write_success_rate', 0):.1f}%")
    
    # Model Performance
    mp = summary.get('model_performance', {})
    print("\n🤖 MODEL PERFORMANCE")
    print(f"  Total Signals:              {mp.get('total_signals', 0)}")
    print(f"    BUY:  {mp.get('buy_signals', 0):3d}  |  SELL: {mp.get('sell_signals', 0):3d}  |  HOLD: {mp.get('hold_signals', 0):3d}")
    print(f"  Avg Confidence:             {mp.get('avg_confidence', 0):.2%}")
    conf_dist = mp.get('confidence_distribution', {})
    print(f"  Confidence Distribution:")
    print(f"    High (≥70%):   {conf_dist.get('high', 0):3d}")
    print(f"    Medium (50-70%): {conf_dist.get('medium', 0):3d}")
    print(f"    Low (<50%):    {conf_dist.get('low', 0):3d}")
    print(f"  Signal Frequency:           {mp.get('signal_frequency_per_hour', 0):.1f} signals/hour")
    
    # System Health
    sh = summary.get('system_health', {})
    print("\n💻 SYSTEM HEALTH")
    print(f"  Memory Usage:               {sh.get('memory_usage_mb', 0):.1f} MB")
    print(f"  CPU Usage:                  {sh.get('cpu_usage_pct', 0):.1f}%")
    print(f"  Database Size:              {sh.get('database_size_mb', 0):.1f} MB")
    print(f"  Error Rate:                 {sh.get('error_rate', 0):.2f} errors/hour")
    print(f"  Uptime:                     {sh.get('uptime_hours', 0):.1f} hours")
    
    # Paper Trading
    pt = summary.get('paper_trading', {})
    print("\n📈 PAPER TRADING")
    print(f"  Total Executions:           {pt.get('total_executions', 0)}")
    print(f"  Successful:                 {pt.get('successful_executions', 0)}")
    print(f"  Rejected:                   {pt.get('rejected_executions', 0)}")
    if pt.get('rejection_reasons'):
        print(f"  Rejection Reasons:")
        for reason, count in pt.get('rejection_reasons', {}).items():
            print(f"    - {reason}: {count}")
    print(f"  Total PnL:                  ₹{pt.get('total_pnl', 0):,.2f}")
    print(f"  Win Rate:                   {pt.get('win_rate', 0):.1f}%")
    print(f"  Avg Trade PnL:              ₹{pt.get('avg_trade_pnl', 0):,.2f}")
    print(f"  Constraint Violations:      {pt.get('portfolio_constraint_violations', 0)}")
    
    print("\n" + "=" * 80 + "\n")


def monitor_loop(exchange: str, interval: int, hours: int, output_file: Optional[str] = None):
    """Main monitoring loop."""
    collector = get_metrics_collector(exchange)
    
    LOGGER.info(f"Starting Phase 2 monitoring for {exchange}")
    LOGGER.info(f"Update interval: {interval} seconds")
    LOGGER.info(f"Metrics period: Last {hours} hours")
    
    try:
        while True:
            summary = collector.compute_summary(hours=hours)
            print_summary(summary, exchange)
            
            # Save to file if requested
            if output_file:
                output_path = Path(output_file)
                with open(output_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                LOGGER.info(f"Summary saved to {output_path}")
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        LOGGER.info("Monitoring stopped by user")
    except Exception as e:
        LOGGER.error(f"Monitoring error: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="Phase 2 validation metrics monitor")
    parser.add_argument('--exchange', choices=['NSE', 'BSE'], default='NSE',
                       help='Exchange to monitor')
    parser.add_argument('--interval', type=int, default=300,
                       help='Update interval in seconds (default: 300 = 5 minutes)')
    parser.add_argument('--hours', type=int, default=24,
                       help='Metrics period in hours (default: 24)')
    parser.add_argument('--output', type=str, default=None,
                       help='Optional output file path for JSON summary')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit (no continuous monitoring)')
    
    args = parser.parse_args()
    
    collector = get_metrics_collector(args.exchange)
    summary = collector.compute_summary(hours=args.hours)
    print_summary(summary, args.exchange)
    
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        LOGGER.info(f"Summary saved to {output_path}")
    
    if not args.once:
        monitor_loop(args.exchange, args.interval, args.hours, args.output)


if __name__ == '__main__':
    main()

