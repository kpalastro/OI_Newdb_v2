"""
NSE Multi-Expiry Data Collector

Collects and aggregates option chain data across the next 5 expiries
for a base strike calculated from market open price.

This module provides:
- Kite API integration for fetching option chain data
- Aggregation logic across multiple expiries
- Background service for minute-by-minute data collection
- Historical backfill capabilities
"""

# Try relative imports first, fallback to absolute
try:
    from .kite_fetcher import (
        get_next_5_expiries,
        fetch_kite_option_chain_for_expiry,
        aggregate_multi_expiry_data
    )
    from .collector_service import MultiExpiryCollectorService
    from .aggregator import calculate_base_strike, aggregate_option_metrics
except ImportError:
    # Fallback to absolute imports if relative fails
    from nse_multi_expiry_collector.kite_fetcher import (
        get_next_5_expiries,
        fetch_kite_option_chain_for_expiry,
        aggregate_multi_expiry_data
    )
    from nse_multi_expiry_collector.collector_service import MultiExpiryCollectorService
    from nse_multi_expiry_collector.aggregator import calculate_base_strike, aggregate_option_metrics

__all__ = [
    'get_next_5_expiries',
    'fetch_kite_option_chain_for_expiry',
    'aggregate_multi_expiry_data',
    'MultiExpiryCollectorService',
    'calculate_base_strike',
    'aggregate_option_metrics',
]