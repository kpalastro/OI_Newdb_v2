"""
Depth aggregation utilities.

Enhanced to support top-5 bid/ask levels per option for richer order book analysis.
Aggregates available bid/ask quantities to produce depth totals and imbalance ratios.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


def summarize_depth_levels(options: Iterable[Dict]) -> Tuple[float, float]:
    """
    Sum bid/ask quantities from the provided option ladder entries.
    """
    buy_total = 0.0
    sell_total = 0.0
    for opt in options:
        buy_total += float(opt.get('bid_quantity') or 0.0)
        sell_total += float(opt.get('ask_quantity') or 0.0)
    return buy_total, sell_total


def extract_top_n_depth_levels(depth_data: Dict, n_levels: int = 5) -> Dict[str, List[Dict]]:
    """
    Extract top N bid/ask levels from websocket depth data.
    
    Args:
        depth_data: Dictionary with 'buy' and 'sell' arrays from websocket tick
        n_levels: Number of levels to extract (default 5)
    
    Returns:
        Dictionary with 'buy' and 'sell' lists, each containing up to n_levels entries
        with 'price' and 'quantity' keys
    """
    result = {'buy': [], 'sell': []}
    
    buy_levels = depth_data.get('buy', [])
    sell_levels = depth_data.get('sell', [])
    
    # Extract top N buy levels
    for i, level in enumerate(buy_levels[:n_levels]):
        if isinstance(level, dict):
            price = level.get('price')
            quantity = level.get('quantity', level.get('qty', 0))
            if price is not None:
                result['buy'].append({
                    'price': float(price),
                    'quantity': float(quantity or 0.0),
                    'level': i + 1
                })
    
    # Extract top N sell levels
    for i, level in enumerate(sell_levels[:n_levels]):
        if isinstance(level, dict):
            price = level.get('price')
            quantity = level.get('quantity', level.get('qty', 0))
            if price is not None:
                result['sell'].append({
                    'price': float(price),
                    'quantity': float(quantity or 0.0),
                    'level': i + 1
                })
    
    return result


def calculate_depth_metrics_from_levels(depth_levels: Dict[str, List[Dict]]) -> Dict[str, float]:
    """
    Calculate aggregated depth metrics from top-N level data.
    
    Returns:
        Dictionary with depth_buy_total, depth_sell_total, depth_imbalance_ratio
    """
    buy_total = sum(level.get('quantity', 0.0) for level in depth_levels.get('buy', []))
    sell_total = sum(level.get('quantity', 0.0) for level in depth_levels.get('sell', []))
    total = buy_total + sell_total
    
    imbalance_ratio = ((buy_total - sell_total) / total) if total > 0 else 0.0
    
    return {
        'depth_buy_total': buy_total,
        'depth_sell_total': sell_total,
        'depth_imbalance_ratio': imbalance_ratio,
        'depth_total': total
    }

