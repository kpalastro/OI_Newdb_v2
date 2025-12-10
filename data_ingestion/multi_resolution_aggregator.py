"""
Multi-resolution bar aggregator for tick data.
Converts 5-second ticks into OHLCV bars at various timeframes.
"""

from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from time_utils import to_ist


class BarAggregator:
    """
    Aggregates tick data into OHLCV bars at specified resolution.
    
    Maintains state for partial bars (current bar being built).
    """
    
    def __init__(self, resolution: str):
        """
        Args:
            resolution: One of '1min', '5min', '15min', '1D'
        """
        self.resolution = resolution
        self._parse_resolution()
        
        # Per-token state: current bar being built
        self._current_bars: Dict[int, Dict] = {}
        
        # Per-token state: last bar timestamp (to detect new bar boundaries)
        self._last_bar_timestamp: Dict[int, datetime] = {}
        
    def _parse_resolution(self):
        """Parse resolution string into timedelta."""
        if self.resolution == '1min':
            self.delta = timedelta(minutes=1)
        elif self.resolution == '5min':
            self.delta = timedelta(minutes=5)
        elif self.resolution == '15min':
            self.delta = timedelta(minutes=15)
        elif self.resolution == '1D':
            self.delta = timedelta(days=1)
        else:
            raise ValueError(f"Unsupported resolution: {self.resolution}")
    
    def _get_bar_start(self, timestamp: datetime) -> datetime:
        """
        Calculate the start timestamp of the bar containing the given timestamp.
        
        For intraday bars, align to minute boundaries.
        For daily bars, align to market open (09:15 IST).
        """
        dt_ist = to_ist(timestamp)
        
        if self.resolution == '1D':
            # Daily bars start at market open (09:15 IST)
            bar_start = dt_ist.replace(hour=9, minute=15, second=0, microsecond=0)
            # If timestamp is before 09:15, use previous day's bar start
            if dt_ist.time() < dt_ist.replace(hour=9, minute=15).time():
                bar_start = bar_start - timedelta(days=1)
            return bar_start
        
        # Intraday bars: align to resolution boundaries
        # For 1min: 09:15, 09:16, 09:17...
        # For 5min: 09:15, 09:20, 09:25...
        # For 15min: 09:15, 09:30, 09:45...
        
        # Start from market open (09:15 IST)
        market_open = dt_ist.replace(hour=9, minute=15, second=0, microsecond=0)
        
        if dt_ist < market_open:
            # Before market open, use previous day
            market_open = market_open - timedelta(days=1)
        
        # Calculate minutes since market open
        minutes_since_open = int((dt_ist - market_open).total_seconds() / 60)
        
        # Round down to resolution boundary
        bars_since_open = minutes_since_open // (self.delta.total_seconds() / 60)
        
        bar_start = market_open + timedelta(minutes=bars_since_open * int(self.delta.total_seconds() / 60))
        
        return bar_start
    
    def add_tick(self, token: int, tick_data: Dict) -> Optional[Dict]:
        """
        Add a tick to the aggregator. Returns completed bar if bar boundary crossed.
        
        Args:
            token: Instrument token
            tick_data: Dict with keys: timestamp, ltp, volume, oi, best_bid, best_ask, spread, imbalance
        
        Returns:
            Completed bar dict if bar closed, None otherwise
        """
        timestamp = tick_data['timestamp']
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp).to_pydatetime()
        timestamp = to_ist(timestamp)
        
        bar_start = self._get_bar_start(timestamp)
        
        # Check if we need to finalize previous bar
        completed_bar = None
        if token in self._last_bar_timestamp:
            last_bar_start = self._last_bar_timestamp[token]
            if bar_start > last_bar_start:
                # Bar boundary crossed - finalize previous bar
                completed_bar = self._finalize_bar(token, last_bar_start)
        
        # Initialize or update current bar
        if token not in self._current_bars or bar_start != self._last_bar_timestamp.get(token):
            self._current_bars[token] = {
                'timestamp': bar_start,
                'token': token,
                'open_price': tick_data.get('ltp'),
                'high_price': tick_data.get('ltp'),
                'low_price': tick_data.get('ltp'),
                'close_price': tick_data.get('ltp'),
                'volume': tick_data.get('volume', 0),
                'oi_start': tick_data.get('oi'),
                'oi_end': tick_data.get('oi'),
                'vwap_numerator': tick_data.get('ltp', 0) * tick_data.get('volume', 0),
                'vwap_denominator': tick_data.get('volume', 0),
                'trade_count': 1,
                'spread_sum': tick_data.get('spread', 0.0),
                'imbalance_sum': tick_data.get('imbalance', 0.0),
                'tick_count': 1
            }
            self._last_bar_timestamp[token] = bar_start
        else:
            # Update existing bar
            bar = self._current_bars[token]
            ltp = tick_data.get('ltp')
            volume = tick_data.get('volume', 0)
            
            if ltp is not None:
                bar['high_price'] = max(bar['high_price'], ltp)
                bar['low_price'] = min(bar['low_price'], ltp)
                bar['close_price'] = ltp
            
            bar['volume'] += volume
            bar['oi_end'] = tick_data.get('oi', bar.get('oi_end'))
            bar['vwap_numerator'] += ltp * volume if ltp else 0
            bar['vwap_denominator'] += volume
            bar['trade_count'] += 1
            bar['spread_sum'] += tick_data.get('spread', 0.0)
            bar['imbalance_sum'] += tick_data.get('imbalance', 0.0)
            bar['tick_count'] += 1
        
        return completed_bar
    
    def _finalize_bar(self, token: int, bar_start: datetime) -> Dict:
        """Finalize and return completed bar."""
        if token not in self._current_bars:
            return None
        
        bar = self._current_bars[token]
        
        # Calculate VWAP
        vwap = (bar['vwap_numerator'] / bar['vwap_denominator'] 
                if bar['vwap_denominator'] > 0 else bar['close_price'])
        
        # Calculate OI change
        oi_change = None
        if bar.get('oi_start') is not None and bar.get('oi_end') is not None:
            oi_change = bar['oi_end'] - bar['oi_start']
        
        # Calculate averages
        tick_count = max(bar['tick_count'], 1)
        spread_avg = bar['spread_sum'] / tick_count
        imbalance_avg = bar['imbalance_sum'] / tick_count
        
        completed = {
            'timestamp': bar['timestamp'],
            'token': bar['token'],
            'open_price': bar['open_price'],
            'high_price': bar['high_price'],
            'low_price': bar['low_price'],
            'close_price': bar['close_price'],
            'volume': bar['volume'],
            'oi': bar['oi_end'],
            'oi_change': oi_change,
            'vwap': vwap,
            'trade_count': bar['trade_count'],
            'spread_avg': spread_avg,
            'imbalance_avg': imbalance_avg
        }
        
        # Don't remove from _current_bars - let it be overwritten by new bar
        return completed
    
    def flush_bar(self, token: int) -> Optional[Dict]:
        """Force flush current bar for a token (useful at market close)."""
        if token not in self._last_bar_timestamp:
            return None
        bar_start = self._last_bar_timestamp[token]
        return self._finalize_bar(token, bar_start)


class MultiResolutionAggregator:
    """
    Manages multiple BarAggregators for different resolutions.
    """
    
    def __init__(self, resolutions: List[str] = None):
        """
        Args:
            resolutions: List of resolutions, e.g., ['1min', '5min', '15min', '1D']
        """
        if resolutions is None:
            resolutions = ['1min', '5min', '15min', '1D']
        
        self.aggregators = {
            res: BarAggregator(res) for res in resolutions
        }
    
    def add_tick(self, token: int, tick_data: Dict) -> Dict[str, Dict]:
        """
        Add tick to all aggregators. Returns dict of completed bars per resolution.
        
        Returns:
            Dict mapping resolution -> completed bar (if any)
        """
        completed_bars = {}
        
        for resolution, aggregator in self.aggregators.items():
            completed = aggregator.add_tick(token, tick_data)
            if completed is not None:
                completed_bars[resolution] = completed
        
        return completed_bars
    
    def flush_all(self, token: int) -> Dict[str, Dict]:
        """Flush all current bars for a token."""
        completed_bars = {}
        for resolution, aggregator in self.aggregators.items():
            completed = aggregator.flush_bar(token)
            if completed is not None:
                completed_bars[resolution] = completed
        return completed_bars

