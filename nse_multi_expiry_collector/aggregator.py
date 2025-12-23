"""
Aggregation logic for multi-expiry option chain data.

Handles strike calculation and aggregation of metrics across multiple expiries.
"""
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def calculate_base_strike(open_price: float, strike_difference: int) -> float:
    """
    Calculate the nearest ATM strike price based on open price.
    
    Args:
        open_price: Market open price of the underlying
        strike_difference: Strike difference from config (e.g., 50 for NIFTY, 100 for BANKNIFTY)
    
    Returns:
        Nearest ATM strike price rounded to nearest strike_difference
    """
    nearest_strike = round(open_price / strike_difference) * strike_difference
    return nearest_strike


def aggregate_option_metrics(expiry_data_list: List[Dict]) -> Dict[str, float]:
    """
    Aggregate option metrics across multiple expiries.
    
    Args:
        expiry_data_list: List of dictionaries, each containing metrics for one expiry.
                          Each dict should have:
                          - 'oi_call': float
                          - 'oi_put': float
                          - 'oi_change_call': float
                          - 'oi_change_put': float
                          - 'volume_call': float
                          - 'volume_put': float
                          - 'iv_call': float (average IV for calls in this expiry)
                          - 'iv_put': float (average IV for puts in this expiry)
    
    Returns:
        Dictionary with aggregated metrics:
        - total_oi_call_all_expiries
        - total_oi_put_all_expiries
        - total_oi_change_call_all_expiries
        - total_oi_change_put_all_expiries
        - total_volume_call_all_expiries
        - total_volume_put_all_expiries
        - avg_iv_call_all_expiries
        - avg_iv_put_all_expiries
    """
    if not expiry_data_list:
        return {
            'total_oi_call_all_expiries': 0.0,
            'total_oi_put_all_expiries': 0.0,
            'total_oi_change_call_all_expiries': 0.0,
            'total_oi_change_put_all_expiries': 0.0,
            'total_volume_call_all_expiries': 0.0,
            'total_volume_put_all_expiries': 0.0,
            'avg_iv_call_all_expiries': 0.0,
            'avg_iv_put_all_expiries': 0.0,
        }
    
    total_oi_call = 0.0
    total_oi_put = 0.0
    total_oi_change_call = 0.0
    total_oi_change_put = 0.0
    total_volume_call = 0.0
    total_volume_put = 0.0
    
    # For IV, we'll calculate weighted average by OI or simple average
    iv_call_sum = 0.0
    iv_put_sum = 0.0
    iv_call_count = 0
    iv_put_count = 0
    
    for expiry_data in expiry_data_list:
        # Sum aggregations
        total_oi_call += float(expiry_data.get('oi_call', 0) or 0)
        total_oi_put += float(expiry_data.get('oi_put', 0) or 0)
        total_oi_change_call += float(expiry_data.get('oi_change_call', 0) or 0)
        total_oi_change_put += float(expiry_data.get('oi_change_put', 0) or 0)
        total_volume_call += float(expiry_data.get('volume_call', 0) or 0)
        total_volume_put += float(expiry_data.get('volume_put', 0) or 0)
        
        # Average IV (simple average across expiries)
        iv_call = expiry_data.get('iv_call')
        iv_put = expiry_data.get('iv_put')
        
        if iv_call is not None and iv_call > 0:
            iv_call_sum += float(iv_call)
            iv_call_count += 1
        
        if iv_put is not None and iv_put > 0:
            iv_put_sum += float(iv_put)
            iv_put_count += 1
    
    avg_iv_call = iv_call_sum / iv_call_count if iv_call_count > 0 else 0.0
    avg_iv_put = iv_put_sum / iv_put_count if iv_put_count > 0 else 0.0
    
    return {
        'total_oi_call_all_expiries': total_oi_call,
        'total_oi_put_all_expiries': total_oi_put,
        'total_oi_change_call_all_expiries': total_oi_change_call,
        'total_oi_change_put_all_expiries': total_oi_change_put,
        'total_volume_call_all_expiries': total_volume_call,
        'total_volume_put_all_expiries': total_volume_put,
        'avg_iv_call_all_expiries': avg_iv_call,
        'avg_iv_put_all_expiries': avg_iv_put,
    }