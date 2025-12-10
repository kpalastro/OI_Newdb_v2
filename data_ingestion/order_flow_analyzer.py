"""
Order Flow Analysis Module.

This module detects institutional 'smart money' activity via block trades and sweep orders.
It processes raw tick data to identify:
1. Block Trades: Large single trades exceeding a size threshold.
2. Sweep Orders: Aggressive simultaneous orders across multiple strikes.
3. Order Flow Imbalance: Net buying/selling pressure derived from trade classification.
"""
from collections import deque
from typing import Dict, List, Optional, Deque
from datetime import datetime, timedelta

class OrderFlowAnalyzer:
    """
    Analyzes tick streams for microstructure events like block trades and sweeps.
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Configuration
        self.min_block_size = self.config.get('min_block_size', 1000)
        self.sweep_window_seconds = self.config.get('sweep_window_seconds', 1.0)
        self.min_sweep_strikes = self.config.get('min_sweep_strikes', 3)
        self.block_retention_window = timedelta(minutes=5)
        
        # State
        self.recent_trades: Deque[Dict] = deque()  # For sweep detection
        self.block_trades: Deque[Dict] = deque()   # For block stats (rolling window)
        self.sweep_events: Deque[Dict] = deque()   # For sweep stats
        
    def process_tick(self, tick: Dict) -> Dict[str, float]:
        """
        Process a single tick (trade) and update flow metrics.
        
        Args:
            tick: Dict containing 'timestamp', 'volume', 'ltp', 'ask', 'bid', 'strike'
            
        Returns:
            Dict with keys: 'block_count_5m', 'sweep_score', 'net_flow_imbalance'
        """
        timestamp = tick.get('timestamp')
        if not isinstance(timestamp, datetime):
            # Attempt to parse or handle if not datetime (assuming handler provides datetime)
            # If strictly needed, we can default to now, but handlers usually provide it.
            return self._get_empty_metrics()

        # 1. Block Trade Detection
        volume = float(tick.get('volume', 0))
        ltp = float(tick.get('ltp', 0))
        ask = float(tick.get('ask_price') or tick.get('ask', 0)) # handle variations
        bid = float(tick.get('bid_price') or tick.get('bid', 0))
        
        # Classify side (Aggressor Hypothesis)
        side = 'NEUTRAL'
        if ask > 0 and ltp >= ask:
            side = 'BUY'
        elif bid > 0 and ltp <= bid:
            side = 'SELL'
        
        if volume >= self.min_block_size:
            block_trade = {
                'timestamp': timestamp,
                'volume': volume,
                'price': ltp,
                'side': side,
                'strike': tick.get('strike')
            }
            self.block_trades.append(block_trade)
            
        # 2. Sweep Detection Logic
        # Store minimal info for sweep detection
        trade_record = {
            'timestamp': timestamp,
            'strike': tick.get('strike'),
            'side': side,
            'volume': volume
        }
        self.recent_trades.append(trade_record)
        
        # Prune recent trades older than sweep window
        while self.recent_trades and (timestamp - self.recent_trades[0]['timestamp']).total_seconds() > self.sweep_window_seconds:
            self.recent_trades.popleft()
            
        # Check for sweep
        self._check_for_sweep(timestamp)
        
        # 3. Prune historical block trades for rolling stats
        while self.block_trades and (timestamp - self.block_trades[0]['timestamp']) > self.block_retention_window:
            self.block_trades.popleft()
            
        # 4. Calculate Return Metrics
        return self._calculate_metrics()

    def _check_for_sweep(self, current_time: datetime):
        """
        Check recent trades for sweep patterns (same side, multiple strikes, high volume).
        """
        if not self.recent_trades:
            return

        # Group by side
        buys = [t for t in self.recent_trades if t['side'] == 'BUY']
        sells = [t for t in self.recent_trades if t['side'] == 'SELL']
        
        for group, side in [(buys, 'BUY'), (sells, 'SELL')]:
            if not group:
                continue
                
            unique_strikes = set(t['strike'] for t in group if t['strike'] is not None)
            total_vol = sum(t['volume'] for t in group)
            
            if len(unique_strikes) >= self.min_sweep_strikes and total_vol >= (self.min_block_size * 2):
                # Heuristic: Avoid duplicates. Only register if we haven't just registered a very similar one.
                # For simplicity in this v1, we append and assume the metric calculation handles noise/decay.
                self.sweep_events.append({
                    'timestamp': current_time,
                    'side': side,
                    'strikes': len(unique_strikes),
                    'volume': total_vol
                })

        # Prune old sweep events (keep last 15 mins for scoring)
        sweep_retention = timedelta(minutes=15)
        while self.sweep_events and (current_time - self.sweep_events[0]['timestamp']) > sweep_retention:
            self.sweep_events.popleft()

    def get_current_metrics(self) -> Dict[str, float]:
        """Return current flow metrics without processing a new tick."""
        return self._calculate_metrics()

    def _calculate_metrics(self) -> Dict[str, float]:
        """Aggregate current state into feature metrics."""
        
        # Block Count (5m)
        block_count = len(self.block_trades)
        
        # Net Flow Imbalance (from blocks)
        buy_vol = sum(t['volume'] for t in self.block_trades if t['side'] == 'BUY')
        sell_vol = sum(t['volume'] for t in self.block_trades if t['side'] == 'SELL')
        imbalance = buy_vol - sell_vol
        
        # Sweep Score (Recent sweep intensity)
        # Simple count of sweeps in last few minutes (sweep_events is already pruned to 15m)
        sweep_score = len(self.sweep_events)
        
        return {
            'block_trade_count': float(block_count),
            'block_trade_imbalance': float(imbalance),
            'sweep_order_detected': 1.0 if sweep_score > 0 else 0.0, # Binary flag for now, or scaled score
            'sweep_score': float(sweep_score),
            'smart_money_flow': float(imbalance) # Alias for flow imbalance for now
        }

    def _get_empty_metrics(self) -> Dict[str, float]:
        return {
            'block_trade_count': 0.0,
            'block_trade_imbalance': 0.0,
            'sweep_order_detected': 0.0,
            'sweep_score': 0.0,
            'smart_money_flow': 0.0
        }

