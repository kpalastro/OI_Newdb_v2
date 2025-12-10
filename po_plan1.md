# P0 Critical Features Implementation Plan
## Multi-Resolution Data Pipeline, Enhanced Greeks, and OI Concentration & Skewness

**Document Version:** 2.0  
**Date:** December 2024  
**Priority Level:** P0 (Critical - Implement First)  
**Scope:** Features 1, 2, 3, 4, 5 from `new_plan.md`

---

## Executive Summary

This document provides a detailed implementation plan for five critical P0 features that form the foundation for advanced ML-based options trading capabilities:

1. **Multi-Resolution Data Pipeline** - Aggregate tick data into multiple timeframes (1m, 5m, 15m, 1d)
2. **Enhanced Greeks Features** - Calculate comprehensive options Greeks (GEX, Vanna, Charm, Speed, Zomma)
3. **OI Concentration & Skewness** - Measure OI distribution concentration, skewness, velocity, and rollover rates
4. **Advanced Risk Management** - Multi-layer risk framework with Greeks limits, correlation checks, and VaR
5. **Walk-Forward Testing** - Multi-period validation framework for strategy robustness

These features are prerequisites for multi-horizon models, advanced risk management, and sophisticated trading strategies.

---

## 1. PROJECT CONTEXT

### 1.1 Current System Architecture

**Data Flow:**
- WebSocket feeds provide 5-second tick-level data
- `handlers.py` maintains `data_reels` (deques) storing raw tick data per instrument
- `feature_engineering.py` computes 50+ features from current tick data
- `database_new.py` persists snapshots to PostgreSQL/TimescaleDB or SQLite
- Features are consumed by ML models (`ml_core.py`) and execution engine (`execution/auto_executor.py`)

**Current Limitations:**
- Single resolution (5-second ticks) stored in `data_reels`
- Basic gamma proxies using simplified formulas (no Black-Scholes Greeks)
- No OI concentration or skewness metrics
- Limited to intraday features; no multi-horizon support

**Database Schema:**
- `option_chain_snapshots` - Strike-level snapshots (timestamp, exchange, strike, option_type, oi, ltp, etc.)
- `ml_features` - Timestamp-level aggregated features
- Both tables use TimescaleDB hypertables for time-series optimization

### 1.2 Integration Points

**Files to Modify:**
- `data_ingestion/` - New aggregator modules
- `handlers.py` - Multi-resolution reel management
- `feature_engineering.py` - Enhanced Greeks and OI concentration calculations
- `database_new.py` - Schema additions and storage functions
- `config.py` - Configuration parameters
- `ml_core.py` - Feature consumption updates

**Dependencies:**
- `scipy` - For statistical calculations and Black-Scholes Greeks
- `pandas` - Time-series aggregation
- `numpy` - Numerical operations
- Existing: `time_utils.py`, `config.py`

---

## 2. FEATURE 1: MULTI-RESOLUTION DATA PIPELINE

### 2.1 Overview

Aggregate 5-second tick data into multiple timeframes to support different trading horizons:
- **1-minute bars** - For intraday models (5s → 1m)
- **5-minute bars** - For swing trading models
- **15-minute bars** - For swing trading models
- **Daily bars** - For positional models

### 2.2 Architecture Design

```
┌─────────────────┐
│  WebSocket      │
│  5-second ticks │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  handlers.py                │
│  - data_reels (5s ticks)    │
│  - Real-time aggregation    │
└────────┬────────────────────┘
         │
         ├─────────────────┬─────────────────┬─────────────────┐
         ▼                 ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  1m reels    │  │  5m reels    │  │  15m reels   │  │  1d reels    │
│  (per token) │  │  (per token) │  │  (per token) │  │  (per token) │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │                 │
       └─────────────────┴─────────────────┴─────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Database Storage    │
                    │  multi_resolution_   │
                    │  bars table          │
                    └──────────────────────┘
```

### 2.3 Database Schema Changes

**New Table: `multi_resolution_bars`**

```sql
CREATE TABLE IF NOT EXISTS multi_resolution_bars (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    exchange TEXT NOT NULL,
    resolution TEXT NOT NULL,  -- '1m', '5m', '15m', '1d'
    token INTEGER NOT NULL,     -- Instrument token
    symbol TEXT,
    open_price DOUBLE PRECISION,
    high_price DOUBLE PRECISION,
    low_price DOUBLE PRECISION,
    close_price DOUBLE PRECISION,
    volume BIGINT,
    oi BIGINT,                  -- For options/futures
    oi_change BIGINT,           -- Change in OI for the bar
    vwap DOUBLE PRECISION,      -- Volume-weighted average price
    trade_count INTEGER,        -- Number of trades in the bar
    spread_avg DOUBLE PRECISION,-- Average bid-ask spread
    imbalance_avg DOUBLE PRECISION, -- Average order book imbalance
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- Composite unique constraint: one bar per resolution per timestamp per token
    UNIQUE(timestamp, exchange, resolution, token)
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_multi_res_bars_resolution_time 
    ON multi_resolution_bars(exchange, resolution, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_multi_res_bars_token_time 
    ON multi_resolution_bars(token, timestamp DESC);

-- For TimescaleDB
SELECT create_hypertable('multi_resolution_bars', 'timestamp', if_not_exists => TRUE);
```

**SQLite Alternative:**
```sql
CREATE TABLE IF NOT EXISTS multi_resolution_bars (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP NOT NULL,
    exchange TEXT NOT NULL,
    resolution TEXT NOT NULL,
    token INTEGER NOT NULL,
    symbol TEXT,
    open_price REAL,
    high_price REAL,
    low_price REAL,
    close_price REAL,
    volume INTEGER,
    oi INTEGER,
    oi_change INTEGER,
    vwap REAL,
    trade_count INTEGER,
    spread_avg REAL,
    imbalance_avg REAL,
    created_at TIMESTAMP DEFAULT (datetime('now', 'localtime')),
    UNIQUE(timestamp, exchange, resolution, token)
);
```

### 2.4 Implementation Details

#### 2.4.1 New Module: `data_ingestion/multi_resolution_aggregator.py`

**Class: `BarAggregator`**

```python
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
```

#### 2.4.2 Modify `handlers.py`

**Add to `ExchangeDataHandler.__init__`:**

```python
from data_ingestion.multi_resolution_aggregator import MultiResolutionAggregator

class ExchangeDataHandler:
    def __init__(self, exchange_name: str, config: Dict[str, Any], data_reel_length: int):
        # ... existing initialization ...
        
        # Multi-resolution aggregation
        self.multi_res_aggregator = MultiResolutionAggregator(
            resolutions=['1min', '5min', '15min', '1D']
        )
        
        # Per-resolution reels for quick access (optional, for feature engineering)
        self.resolution_reels: DefaultDict[str, DefaultDict[int, deque]] = defaultdict(
            lambda: defaultdict(self._create_data_reel)
        )
```

**Modify tick update method** (assumes method exists in `oi_tracker_new.py` or similar):

```python
def _on_tick_update(self, token: int, tick_data: Dict):
    """
    Called when new tick arrives. Update reels and trigger aggregation.
    """
    # Existing tick handling...
    
    # Add to multi-resolution aggregator
    completed_bars = self.multi_res_aggregator.add_tick(token, {
        'timestamp': tick_data.get('timestamp', now_ist()),
        'ltp': tick_data.get('ltp'),
        'volume': tick_data.get('volume', 0),
        'oi': tick_data.get('latest_oi'),
        'best_bid': tick_data.get('best_bid'),
        'best_ask': tick_data.get('best_ask'),
        'spread': tick_data.get('spread', 0.0),
        'imbalance': tick_data.get('order_book_imbalance', 0.0)
    })
    
    # Store completed bars in resolution reels (optional)
    for resolution, bar in completed_bars.items():
        self.resolution_reels[resolution][token].append(bar)
    
    # Persist completed bars to database (async or batched)
    if completed_bars:
        self._persist_multi_resolution_bars(completed_bars, token)
```

**Add persistence method:**

```python
def _persist_multi_resolution_bars(self, completed_bars: Dict[str, Dict], token: int):
    """Persist completed bars to database (can be called async or batched)."""
    from database_new import save_multi_resolution_bars
    
    for resolution, bar in completed_bars.items():
        # Get symbol from token mapping
        symbol = self._get_symbol_for_token(token)
        
        save_multi_resolution_bars(
            exchange=self.exchange,
            resolution=resolution,
            token=token,
            symbol=symbol,
            timestamp=bar['timestamp'],
            open_price=bar['open_price'],
            high_price=bar['high_price'],
            low_price=bar['low_price'],
            close_price=bar['close_price'],
            volume=bar['volume'],
            oi=bar.get('oi'),
            oi_change=bar.get('oi_change'),
            vwap=bar.get('vwap'),
            trade_count=bar.get('trade_count'),
            spread_avg=bar.get('spread_avg'),
            imbalance_avg=bar.get('imbalance_avg')
        )
```

#### 2.4.3 Modify `database_new.py`

**Add save function:**

```python
def save_multi_resolution_bars(
    exchange: str,
    resolution: str,
    token: int,
    symbol: Optional[str],
    timestamp: datetime,
    open_price: Optional[float],
    high_price: Optional[float],
    low_price: Optional[float],
    close_price: Optional[float],
    volume: int = 0,
    oi: Optional[int] = None,
    oi_change: Optional[int] = None,
    vwap: Optional[float] = None,
    trade_count: int = 0,
    spread_avg: Optional[float] = None,
    imbalance_avg: Optional[float] = None
) -> None:
    """
    Persist a multi-resolution bar to the database.
    """
    timestamp_iso = _coerce_iso_timestamp(timestamp)
    current_time_iso = _coerce_iso_timestamp(now_ist())
    
    with db_lock:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            ph = _get_placeholder()
            
            if get_config().db_type == 'postgres':
                cursor.execute(f'''
                    INSERT INTO multi_resolution_bars (
                        timestamp, exchange, resolution, token, symbol,
                        open_price, high_price, low_price, close_price,
                        volume, oi, oi_change, vwap, trade_count,
                        spread_avg, imbalance_avg, created_at
                    ) VALUES ({', '.join([ph]*17)})
                    ON CONFLICT (timestamp, exchange, resolution, token)
                    DO UPDATE SET
                        open_price = EXCLUDED.open_price,
                        high_price = EXCLUDED.high_price,
                        low_price = EXCLUDED.low_price,
                        close_price = EXCLUDED.close_price,
                        volume = EXCLUDED.volume,
                        oi = EXCLUDED.oi,
                        oi_change = EXCLUDED.oi_change,
                        vwap = EXCLUDED.vwap,
                        trade_count = EXCLUDED.trade_count,
                        spread_avg = EXCLUDED.spread_avg,
                        imbalance_avg = EXCLUDED.imbalance_avg,
                        created_at = EXCLUDED.created_at
                ''', (
                    timestamp_iso, exchange, resolution, token, symbol,
                    open_price, high_price, low_price, close_price,
                    volume, oi, oi_change, vwap, trade_count,
                    spread_avg, imbalance_avg, current_time_iso
                ))
            else:
                cursor.execute(f'''
                    INSERT OR REPLACE INTO multi_resolution_bars (
                        timestamp, exchange, resolution, token, symbol,
                        open_price, high_price, low_price, close_price,
                        volume, oi, oi_change, vwap, trade_count,
                        spread_avg, imbalance_avg, created_at
                    ) VALUES ({', '.join([ph]*17)})
                ''', (
                    timestamp_iso, exchange, resolution, token, symbol,
                    open_price, high_price, low_price, close_price,
                    volume, oi, oi_change, vwap, trade_count,
                    spread_avg, imbalance_avg, current_time_iso
                ))
            
            conn.commit()
            release_db_connection(conn)
        except Exception as e:
            logging.error(f"Error saving multi-resolution bar: {e}", exc_info=True)
            if 'conn' in locals():
                release_db_connection(conn)
```

**Add to `initialize_database()`:**

```python
# Add to tables_ddl list in initialize_database()
'''
CREATE TABLE IF NOT EXISTS multi_resolution_bars (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    exchange TEXT NOT NULL,
    resolution TEXT NOT NULL,
    token INTEGER NOT NULL,
    symbol TEXT,
    open_price DOUBLE PRECISION,
    high_price DOUBLE PRECISION,
    low_price DOUBLE PRECISION,
    close_price DOUBLE PRECISION,
    volume BIGINT,
    oi BIGINT,
    oi_change BIGINT,
    vwap DOUBLE PRECISION,
    trade_count INTEGER,
    spread_avg DOUBLE PRECISION,
    imbalance_avg DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(timestamp, exchange, resolution, token)
)
'''
```

#### 2.4.4 Configuration Updates

**Add to `config.py`:**

```python
@dataclass
class AppConfig:
    # ... existing fields ...
    
    # Multi-resolution aggregation
    multi_resolution_enabled: bool = field(
        default_factory=lambda: _get_env_bool('OI_TRACKER_MULTI_RESOLUTION_ENABLED', True)
    )
    multi_resolution_resolutions: List[str] = field(
        default_factory=lambda: ['1min', '5min', '15min', '1D']
    )
```

### 2.5 Testing Strategy

**Unit Tests:**
1. Test `BarAggregator` with sample tick data
2. Verify bar boundary calculations (minute alignment)
3. Test VWAP, OHLC calculations
4. Test daily bar alignment to market open

**Integration Tests:**
1. End-to-end: tick → bar → database persistence
2. Verify no data loss during aggregation
3. Test concurrent access (thread safety)

**Performance Tests:**
1. Measure aggregation overhead (should be <1ms per tick)
2. Database write performance (batch writes if needed)

### 2.6 Performance Considerations

- **Memory:** Store only current bars in memory; completed bars go to DB
- **Latency:** Aggregation is O(1) per tick; database writes can be async/batched
- **Storage:** Estimate ~50KB per bar per instrument per resolution per day

---

## 3. FEATURE 2: ENHANCED GREEKS FEATURES

### 3.1 Overview

Replace basic gamma proxies with proper Black-Scholes Greeks calculations:
- **Gamma Exposure (GEX)** - Market-wide gamma exposure (OI × Gamma × Spot²)
- **Vanna** - Delta sensitivity to volatility changes
- **Charm** - Delta decay with time (theta of delta)
- **Speed** - Gamma sensitivity to underlying price moves
- **Zomma** - Gamma sensitivity to volatility changes

### 3.2 Mathematical Foundation

**Black-Scholes Model:**
- S = Spot price
- K = Strike price
- T = Time to expiry (years)
- r = Risk-free rate (default 0.10 for India)
- σ = Implied volatility
- d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T)
- d2 = d1 - σ√T

**Greeks Formulas:**
- **Delta (Call):** N(d1)
- **Delta (Put):** N(d1) - 1
- **Gamma:** N'(d1) / (S × σ × √T)
- **Vega:** S × N'(d1) × √T / 100
- **Theta (Call):** -(S × N'(d1) × σ) / (2√T) - r × K × e^(-rT) × N(d2)
- **Theta (Put):** -(S × N'(d1) × σ) / (2√T) + r × K × e^(-rT) × N(-d2)

**Higher-Order Greeks:**
- **Vanna:** -N'(d1) × d2 / σ
- **Charm:** -N'(d1) × (2rT - d2 × σ√T) / (2T × σ√T) for calls
- **Speed:** -Gamma × (d1 / (σ√T) + 1) / S
- **Zomma:** Gamma × (d1 × d2 - 1) / σ

**Gamma Exposure (GEX):**
- GEX = Σ(Contract Multiplier × OI × Gamma × Spot² × Option Type Multiplier)
- Option Type Multiplier: +1 for long calls, -1 for long puts (market maker assumption)

### 3.3 Implementation Details

#### 3.3.1 New Module: `utils/greeks_calculator.py`

```python
"""
Black-Scholes Greeks Calculator for Options Trading.

Provides accurate Greeks calculations using scipy.stats and numpy.
"""

import numpy as np
from scipy.stats import norm
from math import log, sqrt, exp
from typing import Dict, List, Optional, Tuple


class GreeksCalculator:
    """
    Calculate Black-Scholes Greeks for options.
    """
    
    def __init__(self, risk_free_rate: float = 0.10):
        """
        Args:
            risk_free_rate: Annual risk-free rate (default 0.10 = 10% for India)
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_greeks(
        self,
        spot_price: float,
        strike: float,
        time_to_expiry_years: float,
        implied_vol: float,
        option_type: str  # 'CE' or 'PE'
    ) -> Dict[str, float]:
        """
        Calculate all Greeks for an option.
        
        Returns:
            Dict with keys: delta, gamma, vega, theta, vanna, charm, speed, zomma
        """
        if time_to_expiry_years <= 0:
            # Expired option - all Greeks are 0
            return {
                'delta': 0.0,
                'gamma': 0.0,
                'vega': 0.0,
                'theta': 0.0,
                'vanna': 0.0,
                'charm': 0.0,
                'speed': 0.0,
                'zomma': 0.0
            }
        
        if implied_vol <= 0 or spot_price <= 0:
            return self._zero_greeks()
        
        is_call = option_type.upper() in ['CE', 'C', 'CALL']
        
        # Calculate d1 and d2
        d1, d2 = self._calculate_d1_d2(
            spot_price, strike, time_to_expiry_years, implied_vol
        )
        
        # PDF of standard normal at d1
        nd1_prime = norm.pdf(d1)
        
        # CDF of standard normal
        nd1 = norm.cdf(d1)
        nd2 = norm.cdf(d2) if is_call else norm.cdf(-d2)
        n_neg_d1 = norm.cdf(-d1)
        n_neg_d2 = norm.cdf(-d2)
        
        sqrt_t = sqrt(time_to_expiry_years)
        vol_sqrt_t = implied_vol * sqrt_t
        
        # Basic Greeks
        delta = nd1 if is_call else (nd1 - 1.0)
        
        gamma = nd1_prime / (spot_price * vol_sqrt_t) if vol_sqrt_t > 0 else 0.0
        
        vega = spot_price * nd1_prime * sqrt_t / 100.0  # Divided by 100 for % vol change
        
        # Theta (time decay)
        discount_factor = exp(-self.risk_free_rate * time_to_expiry_years)
        if is_call:
            theta = (
                -(spot_price * nd1_prime * implied_vol) / (2 * sqrt_t)
                - self.risk_free_rate * strike * discount_factor * nd2
            )
        else:
            theta = (
                -(spot_price * nd1_prime * implied_vol) / (2 * sqrt_t)
                + self.risk_free_rate * strike * discount_factor * n_neg_d2
            )
        
        # Higher-order Greeks
        if vol_sqrt_t > 0:
            vanna = -nd1_prime * d2 / implied_vol
            
            # Charm (delta decay)
            if is_call:
                charm = -nd1_prime * (2 * self.risk_free_rate * time_to_expiry_years - d2 * vol_sqrt_t) / (2 * time_to_expiry_years * vol_sqrt_t)
            else:
                charm = -nd1_prime * (2 * self.risk_free_rate * time_to_expiry_years + d2 * vol_sqrt_t) / (2 * time_to_expiry_years * vol_sqrt_t)
            
            # Speed (third-order price sensitivity)
            speed = -gamma * (d1 / vol_sqrt_t + 1) / spot_price if spot_price > 0 else 0.0
            
            # Zomma (gamma-vega)
            zomma = gamma * (d1 * d2 - 1) / implied_vol if implied_vol > 0 else 0.0
        else:
            vanna = 0.0
            charm = 0.0
            speed = 0.0
            zomma = 0.0
        
        return {
            'delta': float(delta),
            'gamma': float(gamma),
            'vega': float(vega),
            'theta': float(theta),
            'vanna': float(vanna),
            'charm': float(charm),
            'speed': float(speed),
            'zomma': float(zomma),
            'd1': float(d1),
            'd2': float(d2)
        }
    
    def _calculate_d1_d2(
        self,
        spot_price: float,
        strike: float,
        time_to_expiry_years: float,
        implied_vol: float
    ) -> Tuple[float, float]:
        """Calculate d1 and d2 for Black-Scholes."""
        if time_to_expiry_years <= 0 or implied_vol <= 0:
            return 0.0, 0.0
        
        sqrt_t = sqrt(time_to_expiry_years)
        vol_sqrt_t = implied_vol * sqrt_t
        
        if strike <= 0:
            return 0.0, 0.0
        
        d1 = (
            log(spot_price / strike) + 
            (self.risk_free_rate + 0.5 * implied_vol ** 2) * time_to_expiry_years
        ) / vol_sqrt_t
        
        d2 = d1 - vol_sqrt_t
        
        return d1, d2
    
    def _zero_greeks(self) -> Dict[str, float]:
        """Return zero Greeks."""
        return {
            'delta': 0.0,
            'gamma': 0.0,
            'vega': 0.0,
            'theta': 0.0,
            'vanna': 0.0,
            'charm': 0.0,
            'speed': 0.0,
            'zomma': 0.0
        }


def calculate_gamma_exposure(
    call_options: List[Dict],
    put_options: List[Dict],
    spot_price: float,
    time_to_expiry_years: float,
    risk_free_rate: float = 0.10,
    contract_multiplier: int = 50  # NIFTY lot size
) -> Dict[str, float]:
    """
    Calculate market-wide Gamma Exposure (GEX) and other aggregate Greeks.
    
    Args:
        call_options: List of call option dicts with keys: strike, latest_oi, iv
        put_options: List of put option dicts with keys: strike, latest_oi, iv
        spot_price: Current underlying spot price
        time_to_expiry_years: Time to expiry in years
        risk_free_rate: Risk-free rate (default 0.10)
        contract_multiplier: Contract multiplier (lot size)
    
    Returns:
        Dict with aggregate Greeks metrics
    """
    calculator = GreeksCalculator(risk_free_rate)
    
    total_gex = 0.0
    total_vanna_exposure = 0.0
    total_charm_exposure = 0.0
    total_speed_exposure = 0.0
    total_zomma_exposure = 0.0
    
    # Process call options (positive gamma exposure for market makers if they're short)
    for opt in call_options:
        strike = opt.get('strike')
        oi = opt.get('latest_oi', 0) or 0
        iv = opt.get('iv', 0.0) or 0.0
        
        if strike is None or oi <= 0 or iv <= 0:
            continue
        
        greeks = calculator.calculate_greeks(
            spot_price, strike, time_to_expiry_years, iv, 'CE'
        )
        
        # Gamma Exposure: OI × Gamma × Spot² × Multiplier
        # Market makers are typically short options, so we use negative sign
        # But convention is to report as if market makers are short (negative GEX = positive for price)
        gex_contribution = contract_multiplier * oi * greeks['gamma'] * (spot_price ** 2)
        total_gex += gex_contribution
        
        # Aggregate other exposures
        total_vanna_exposure += contract_multiplier * oi * greeks['vanna'] * spot_price
        total_charm_exposure += contract_multiplier * oi * greeks['charm'] * spot_price
        total_speed_exposure += contract_multiplier * oi * greeks['speed'] * (spot_price ** 3)
        total_zomma_exposure += contract_multiplier * oi * greeks['zomma'] * (spot_price ** 2)
    
    # Process put options (negative gamma exposure for market makers if they're short)
    for opt in put_options:
        strike = opt.get('strike')
        oi = opt.get('latest_oi', 0) or 0
        iv = opt.get('iv', 0.0) or 0.0
        
        if strike is None or oi <= 0 or iv <= 0:
            continue
        
        greeks = calculator.calculate_greeks(
            spot_price, strike, time_to_expiry_years, iv, 'PE'
        )
        
        # Puts contribute negatively to GEX (market makers short puts = negative gamma)
        gex_contribution = -contract_multiplier * oi * greeks['gamma'] * (spot_price ** 2)
        total_gex += gex_contribution
        
        # Aggregate other exposures (puts contribute with opposite sign)
        total_vanna_exposure -= contract_multiplier * oi * greeks['vanna'] * spot_price
        total_charm_exposure -= contract_multiplier * oi * greeks['charm'] * spot_price
        total_speed_exposure -= contract_multiplier * oi * greeks['speed'] * (spot_price ** 3)
        total_zomma_exposure -= contract_multiplier * oi * greeks['zomma'] * (spot_price ** 2)
    
    # Calculate gamma flip zones (strikes where gamma changes sign)
    # This is simplified - full implementation would find zero crossings
    gamma_flip_zones = _calculate_gamma_flip_zones(
        call_options, put_options, spot_price, time_to_expiry_years, risk_free_rate
    )
    
    return {
        'gamma_exposure': total_gex,
        'vanna_exposure': total_vanna_exposure,
        'charm_exposure': total_charm_exposure,
        'speed_exposure': total_speed_exposure,
        'zomma_exposure': total_zomma_exposure,
        'gamma_flip_zones': gamma_flip_zones
    }


def _calculate_gamma_flip_zones(
    call_options: List[Dict],
    put_options: List[Dict],
    spot_price: float,
    time_to_expiry_years: float,
    risk_free_rate: float = 0.10
) -> List[float]:
    """
    Identify strike prices where net gamma exposure changes sign (flip zones).
    
    Returns:
        List of strike prices where gamma flips
    """
    # Simplified: Find strikes where net gamma is near zero
    # Full implementation would solve for zero crossings
    calculator = GreeksCalculator(risk_free_rate)
    
    flip_zones = []
    strikes = sorted(set([
        opt.get('strike') for opt in call_options + put_options 
        if opt.get('strike') is not None
    ]))
    
    if len(strikes) < 2:
        return flip_zones
    
    # Calculate net gamma at each strike
    net_gammas = []
    for strike in strikes:
        # Find options at this strike
        call_oi = sum(
            opt.get('latest_oi', 0) or 0 
            for opt in call_options 
            if opt.get('strike') == strike
        )
        put_oi = sum(
            opt.get('latest_oi', 0) or 0 
            for opt in put_options 
            if opt.get('strike') == strike
        )
        
        if call_oi == 0 and put_oi == 0:
            net_gammas.append(0.0)
            continue
        
        # Use average IV for the strike
        call_ivs = [opt.get('iv', 0.0) for opt in call_options if opt.get('strike') == strike and opt.get('iv')]
        put_ivs = [opt.get('iv', 0.0) for opt in put_options if opt.get('strike') == strike and opt.get('iv')]
        
        avg_iv = 0.0
        if call_ivs or put_ivs:
            avg_iv = np.mean([iv for iv in call_ivs + put_ivs if iv > 0]) if (call_ivs or put_ivs) else 0.0
        
        if avg_iv <= 0:
            net_gammas.append(0.0)
            continue
        
        # Calculate gamma contributions
        call_greeks = calculator.calculate_greeks(spot_price, strike, time_to_expiry_years, avg_iv, 'CE')
        put_greeks = calculator.calculate_greeks(spot_price, strike, time_to_expiry_years, avg_iv, 'PE')
        
        net_gamma = (call_oi * call_greeks['gamma']) - (put_oi * put_greeks['gamma'])
        net_gammas.append(net_gamma)
    
    # Find sign changes (simplified - look for near-zero crossings)
    for i in range(len(net_gammas) - 1):
        if net_gammas[i] * net_gammas[i + 1] < 0:  # Sign change
            # Interpolate flip zone
            flip_zone = (strikes[i] + strikes[i + 1]) / 2.0
            flip_zones.append(flip_zone)
    
    return flip_zones
```

#### 3.3.2 Modify `feature_engineering.py`

**Update `_calculate_option_aggregates`:**

```python
from utils.greeks_calculator import calculate_gamma_exposure, GreeksCalculator

def _calculate_option_aggregates(
    call_options: List[Dict],
    put_options: List[Dict],
    atm_strike: float,
    time_to_expiry_days: float = 1.0,
    spot_price: Optional[float] = None,  # Add spot_price parameter
    risk_free_rate: float = 0.10  # Add risk_free_rate parameter
) -> OptionAggregates:
    """Calculate option aggregates with enhanced Greeks."""
    agg = OptionAggregates(gamma_flip_level=float(atm_strike or 0.0))
    
    # ... existing aggregation code ...
    
    # Enhanced Greeks calculation
    if spot_price and spot_price > 0:
        time_to_expiry_years = time_to_expiry_days / 365.25
        
        # Calculate comprehensive Greeks exposure
        greeks_exposure = calculate_gamma_exposure(
            call_options=call_options,
            put_options=put_options,
            spot_price=spot_price,
            time_to_expiry_years=time_to_expiry_years,
            risk_free_rate=risk_free_rate,
            contract_multiplier=50  # NIFTY lot size (configurable)
        )
        
        agg.net_gamma_exposure = greeks_exposure['gamma_exposure']
        agg.dealer_vanna_exposure = greeks_exposure['vanna_exposure']
        agg.dealer_charm_exposure = greeks_exposure['charm_exposure']
        
        # Store gamma flip zones (take closest to ATM)
        gamma_flip_zones = greeks_exposure.get('gamma_flip_zones', [])
        if gamma_flip_zones and atm_strike:
            # Find flip zone closest to ATM
            closest_flip = min(gamma_flip_zones, key=lambda x: abs(x - atm_strike))
            agg.gamma_flip_level = closest_flip
        
        # Calculate speed and zomma exposure (add to OptionAggregates dataclass)
        # agg.speed_exposure = greeks_exposure['speed_exposure']
        # agg.zomma_exposure = greeks_exposure['zomma_exposure']
    else:
        # Fallback to proxy calculations if spot_price unavailable
        agg.net_gamma_exposure = sum(call_stats['gamma_proxy']) - sum(put_stats['gamma_proxy'])
        agg.dealer_vanna_exposure = sum(call_stats['vanna_proxy']) - sum(put_stats['vanna_proxy'])
        agg.dealer_charm_exposure = sum(call_stats['charm_proxy']) - sum(put_stats['charm_proxy'])
    
    # ... rest of existing code ...
    
    return agg
```

**Update `engineer_live_feature_set` to pass spot_price:**

```python
def engineer_live_feature_set(
    handler,
    call_options: List[Dict],
    put_options: List[Dict],
    spot_price: float,
    atm_strike: float,
    now: datetime,
    latest_vix: Optional[float],
) -> Dict[str, float]:
    """Build the full live feature vector."""
    # ... existing code ...
    
    option_aggs = _calculate_option_aggregates(
        call_options, put_options, atm_strike,
        time_to_expiry_days=tte_hours/24.0,
        spot_price=spot_price,  # Pass spot_price
        risk_free_rate=handler.config.get('risk_free_rate', 0.10)  # From config
    )
    
    # ... rest of code ...
```

#### 3.3.3 Update `OptionAggregates` Dataclass

```python
@dataclass
class OptionAggregates:
    # ... existing fields ...
    
    # Enhanced Greeks (keep existing for backward compatibility, enhance with real calculations)
    net_gamma_exposure: float = 0.0  # Enhanced: Real GEX calculation
    dealer_vanna_exposure: float = 0.0  # Enhanced: Real Vanna calculation
    dealer_charm_exposure: float = 0.0  # Enhanced: Real Charm calculation
    
    # New higher-order Greeks (optional, can be added later)
    # speed_exposure: float = 0.0
    # zomma_exposure: float = 0.0
```

#### 3.3.4 Database Schema Updates

**Add columns to `ml_features` table (via migration):**

```sql
-- These columns already exist in ml_features, ensure they're populated correctly
-- net_gamma_exposure, dealer_vanna_exposure, dealer_charm_exposure
-- No schema change needed, but ensure values are updated with real calculations
```

### 3.4 Testing Strategy

**Unit Tests:**
1. Verify Black-Scholes Greeks against known values (test cases)
2. Test edge cases: expired options, zero IV, zero time to expiry
3. Test GEX calculation with sample option chains

**Integration Tests:**
1. End-to-end: option chain → Greeks → feature vector
2. Compare old proxy values vs new real values (should be correlated but not identical)

**Validation:**
1. Cross-reference GEX values with market data providers (if available)
2. Verify Greeks change correctly with spot/IV/time changes

### 3.5 Dependencies

**Add to `requirements.txt`:**
```
scipy>=1.9.0  # For norm.pdf, norm.cdf
```

---

## 4. FEATURE 3: OI CONCENTRATION & SKEWNESS

### 4.1 Overview

Calculate metrics describing the distribution of Open Interest across strikes:
- **OI Concentration Ratio** - Top 3 strikes OI / Total OI
- **OI Skewness** - Statistical skew of OI distribution
- **OI Velocity** - Rate of OI change per minute
- **OI Rollover Rate** - Weekly to monthly rollover rate

### 4.2 Mathematical Definitions

**OI Concentration Ratio:**
- Concentration = (Σ OI of top N strikes) / (Total OI)
- Typically N=3 (top 3 strikes by OI)

**OI Skewness:**
- Skewness = E[(X - μ)³] / σ³
- Where X is OI values across strikes, μ is mean, σ is standard deviation
- Positive skew = OI concentrated on higher strikes (calls)
- Negative skew = OI concentrated on lower strikes (puts)

**OI Velocity:**
- Velocity = (OI_current - OI_previous) / time_delta_minutes
- Measures rate of OI buildup or unwinding

**OI Rollover Rate:**
- Rollover = (Weekly OI decrease) / (Monthly OI increase)
- Measures migration from weekly to monthly expiries

### 4.3 Implementation Details

#### 4.3.1 Add to `feature_engineering.py`

```python
from scipy.stats import skew
from typing import Dict, List, Tuple

def calculate_oi_concentration(
    call_options: List[Dict],
    put_options: List[Dict],
    top_n: int = 3
) -> Dict[str, float]:
    """
    Calculate OI concentration metrics.
    
    Args:
        call_options: List of call option dicts with 'strike' and 'latest_oi'
        put_options: List of put option dicts with 'strike' and 'latest_oi'
        top_n: Number of top strikes to consider for concentration (default 3)
    
    Returns:
        Dict with concentration metrics
    """
    # Collect all OI data with strikes
    all_oi = []
    for opt in call_options + put_options:
        strike = opt.get('strike')
        oi = opt.get('latest_oi', 0) or 0
        if strike is not None and oi > 0:
            all_oi.append((strike, oi))
    
    if not all_oi:
        return {
            'oi_concentration_ratio': 0.0,
            'oi_concentration_top3_ce': 0.0,
            'oi_concentration_top3_pe': 0.0,
            'oi_skewness': 0.0,
            'oi_skewness_ce': 0.0,
            'oi_skewness_pe': 0.0
        }
    
    # Sort by OI descending
    all_oi.sort(key=lambda x: x[1], reverse=True)
    
    total_oi = sum(oi for _, oi in all_oi)
    top_n_oi = sum(oi for _, oi in all_oi[:top_n])
    
    concentration_ratio = top_n_oi / total_oi if total_oi > 0 else 0.0
    
    # Separate CE and PE concentrations
    ce_oi = [(strike, oi) for opt in call_options 
             for strike, oi in [(opt.get('strike'), opt.get('latest_oi', 0) or 0)]
             if strike is not None and oi > 0]
    pe_oi = [(strike, oi) for opt in put_options 
             for strike, oi in [(opt.get('strike'), opt.get('latest_oi', 0) or 0)]
             if strike is not None and oi > 0]
    
    ce_oi.sort(key=lambda x: x[1], reverse=True)
    pe_oi.sort(key=lambda x: x[1], reverse=True)
    
    total_ce_oi = sum(oi for _, oi in ce_oi)
    total_pe_oi = sum(oi for _, oi in pe_oi)
    
    top3_ce_oi = sum(oi for _, oi in ce_oi[:top_n])
    top3_pe_oi = sum(oi for _, oi in pe_oi[:top_n])
    
    concentration_top3_ce = top3_ce_oi / total_ce_oi if total_ce_oi > 0 else 0.0
    concentration_top3_pe = top3_pe_oi / total_pe_oi if total_pe_oi > 0 else 0.0
    
    # Calculate skewness
    oi_values = [oi for _, oi in all_oi]
    oi_skewness = float(skew(oi_values)) if len(oi_values) > 2 else 0.0
    
    ce_oi_values = [oi for _, oi in ce_oi]
    pe_oi_values = [oi for _, oi in pe_oi]
    
    ce_skewness = float(skew(ce_oi_values)) if len(ce_oi_values) > 2 else 0.0
    pe_skewness = float(skew(pe_oi_values)) if len(pe_oi_values) > 2 else 0.0
    
    return {
        'oi_concentration_ratio': concentration_ratio,
        'oi_concentration_top3_ce': concentration_top3_ce,
        'oi_concentration_top3_pe': concentration_top3_pe,
        'oi_skewness': oi_skewness,
        'oi_skewness_ce': ce_skewness,
        'oi_skewness_pe': pe_skewness
    }


def calculate_oi_velocity(
    handler,
    call_options: List[Dict],
    put_options: List[Dict],
    window_minutes: int = 5
) -> Dict[str, float]:
    """
    Calculate OI velocity (rate of change) over specified window.
    
    Uses oi_history from handler to track OI changes over time.
    
    Args:
        handler: ExchangeDataHandler instance with oi_history
        call_options: Current call options
        put_options: Current put options
        window_minutes: Time window for velocity calculation
    
    Returns:
        Dict with velocity metrics
    """
    current_time = now_ist()
    
    # Calculate current total OI
    current_ce_oi = sum(opt.get('latest_oi', 0) or 0 for opt in call_options)
    current_pe_oi = sum(opt.get('latest_oi', 0) or 0 for opt in put_options)
    current_total_oi = current_ce_oi + current_pe_oi
    
    # Find historical OI from window_minutes ago
    # This requires accessing oi_history from handler
    # For simplicity, we'll use a rolling window approach
    
    # Store current OI in handler's oi_history (if not already done)
    # This should be done in the tick update logic
    
    # For now, calculate velocity using a simplified approach
    # In practice, we'd query oi_history for values N minutes ago
    
    # Placeholder: return 0 if history unavailable
    # Full implementation would:
    # 1. Query handler.oi_history for timestamp (current_time - window_minutes)
    # 2. Calculate change = current_oi - historical_oi
    # 3. Velocity = change / window_minutes
    
    return {
        'oi_velocity_ce': 0.0,  # Placeholder
        'oi_velocity_pe': 0.0,  # Placeholder
        'oi_velocity_total': 0.0,  # Placeholder
        'oi_velocity_ce_per_minute': 0.0,
        'oi_velocity_pe_per_minute': 0.0
    }


def calculate_oi_rollover_rate(
    weekly_oi_data: Dict,
    monthly_oi_data: Dict
) -> float:
    """
    Calculate OI rollover rate from weekly to monthly expiries.
    
    Args:
        weekly_oi_data: Dict with 'total_oi' and 'timestamp' for weekly expiry
        monthly_oi_data: Dict with 'total_oi' and 'timestamp' for monthly expiry
    
    Returns:
        Rollover rate (0.0 to 1.0, or negative if monthly decreasing)
    """
    weekly_oi = weekly_oi_data.get('total_oi', 0) or 0
    monthly_oi = monthly_oi_data.get('total_oi', 0) or 0
    
    if monthly_oi == 0:
        return 0.0
    
    # Rollover rate = weekly OI decrease / monthly OI increase
    # Simplified calculation - full implementation would track changes over time
    rollover_rate = weekly_oi / monthly_oi if monthly_oi > 0 else 0.0
    
    return float(rollover_rate)
```

#### 4.3.2 Integrate into `engineer_live_feature_set`

```python
def engineer_live_feature_set(
    handler,
    call_options: List[Dict],
    put_options: List[Dict],
    spot_price: float,
    atm_strike: float,
    now: datetime,
    latest_vix: Optional[float],
) -> Dict[str, float]:
    """Build the full live feature vector."""
    # ... existing code ...
    
    # OI Concentration & Skewness
    oi_concentration = calculate_oi_concentration(call_options, put_options, top_n=3)
    features.update(oi_concentration)
    
    # OI Velocity (simplified - full implementation requires historical tracking)
    oi_velocity = calculate_oi_velocity(handler, call_options, put_options, window_minutes=5)
    features.update(oi_velocity)
    
    # OI Rollover Rate (requires weekly/monthly expiry data - can be added later)
    # oi_rollover = calculate_oi_rollover_rate(weekly_data, monthly_data)
    # features['oi_rollover_rate'] = oi_rollover
    
    # ... rest of code ...
```

#### 4.3.3 Update `REQUIRED_FEATURE_COLUMNS`

```python
REQUIRED_FEATURE_COLUMNS = [
    # ... existing columns ...
    
    # OI Concentration & Skewness
    'oi_concentration_ratio',
    'oi_concentration_top3_ce',
    'oi_concentration_top3_pe',
    'oi_skewness',
    'oi_skewness_ce',
    'oi_skewness_pe',
    'oi_velocity_ce',
    'oi_velocity_pe',
    'oi_velocity_total',
    'oi_velocity_ce_per_minute',
    'oi_velocity_pe_per_minute',
    # 'oi_rollover_rate',  # Future enhancement
]
```

#### 4.3.4 Database Schema Updates

**Add columns to `ml_features` (via migration):**

```python
# In migrate_database(), add:
new_oi_columns = [
    ('oi_concentration_ratio', 'DOUBLE PRECISION'),
    ('oi_concentration_top3_ce', 'DOUBLE PRECISION'),
    ('oi_concentration_top3_pe', 'DOUBLE PRECISION'),
    ('oi_skewness', 'DOUBLE PRECISION'),
    ('oi_skewness_ce', 'DOUBLE PRECISION'),
    ('oi_skewness_pe', 'DOUBLE PRECISION'),
    ('oi_velocity_ce', 'DOUBLE PRECISION'),
    ('oi_velocity_pe', 'DOUBLE PRECISION'),
    ('oi_velocity_total', 'DOUBLE PRECISION'),
    ('oi_velocity_ce_per_minute', 'DOUBLE PRECISION'),
    ('oi_velocity_pe_per_minute', 'DOUBLE PRECISION'),
]
```

### 4.4 Testing Strategy

**Unit Tests:**
1. Test concentration calculation with known OI distributions
2. Verify skewness matches scipy.stats.skew
3. Test edge cases: empty option lists, zero OI, single strike

**Integration Tests:**
1. End-to-end: option chain → concentration/skewness → feature vector
2. Verify values are reasonable (concentration 0-1, skewness typically -3 to +3)

### 4.5 Performance Considerations

- **Skewness calculation:** O(n) where n is number of strikes (typically <50, negligible)
- **Concentration:** O(n log n) for sorting (negligible for <100 strikes)

---

## 5. INTEGRATION & TESTING PLAN

### 5.1 Integration Order

**Phase 1: Foundation (Week 1)**
1. Implement multi-resolution aggregator (Feature 1)
2. Test aggregation logic
3. Integrate into handlers
4. Database schema updates

**Phase 2: Enhanced Greeks (Week 2)**
1. Implement Greeks calculator (Feature 2)
2. Replace proxy calculations
3. Update feature engineering
4. Validate against test cases

**Phase 3: OI Metrics (Week 3)**
1. Implement OI concentration/skewness (Feature 3)
2. Integrate into feature pipeline
3. Database migration
4. End-to-end testing

**Phase 4: Validation & Optimization (Week 4)**
1. Performance tuning
2. Cross-validation with existing features
3. Documentation
4. Production readiness review

### 5.2 Testing Checklist

**Feature 1 (Multi-Resolution):**
- [ ] Bar boundaries align correctly (1m, 5m, 15m, 1D)
- [ ] OHLCV calculations correct
- [ ] VWAP accuracy
- [ ] Database persistence works
- [ ] No data loss during aggregation
- [ ] Performance acceptable (<1ms overhead per tick)

**Feature 2 (Enhanced Greeks):**
- [ ] Black-Scholes Greeks match reference values
- [ ] Edge cases handled (expired, zero IV, zero time)
- [ ] GEX calculation reasonable
- [ ] Integration with feature pipeline works
- [ ] Feature values update correctly

**Feature 3 (OI Concentration):**
- [ ] Concentration ratio correct (0-1 range)
- [ ] Skewness matches scipy.stats
- [ ] Edge cases handled (empty lists, zero OI)
- [ ] Feature values in ml_features table
- [ ] No performance degradation

**Feature 4 (Advanced Risk Management):**
- [ ] Greeks exposure limits enforced correctly
- [ ] Correlation checks work as expected
- [ ] VaR calculation accurate
- [ ] Liquidity checks functional
- [ ] Regime-based adjustments apply correctly
- [ ] Risk monitoring alerts trigger at thresholds
- [ ] Integration with auto executor works

**Feature 5 (Walk-Forward Testing):**
- [ ] Segments generated correctly (no overlap/leakage)
- [ ] Model retraining works per segment
- [ ] Aggregated metrics calculated correctly
- [ ] Best/worst segments identified accurately
- [ ] Consistency ratio correct
- [ ] Results match individual backtests

### 5.3 Rollback Plan

**If issues arise:**
1. Feature flags in config to disable new features
2. Database columns nullable (graceful degradation)
3. Keep old proxy calculations as fallback
4. Gradual rollout (one exchange at a time)

---

## 6. DEPENDENCIES & REQUIREMENTS

### 6.1 New Dependencies

```txt
scipy>=1.9.0  # For statistical functions and Black-Scholes calculations
```

### 6.2 Configuration Changes

**config.py additions:**
```python
multi_resolution_enabled: bool = True
multi_resolution_resolutions: List[str] = ['1min', '5min', '15min', '1D']
risk_free_rate: float = 0.10  # Already exists
contract_multiplier: int = 50  # NIFTY lot size (could be per-exchange)
```

### 6.3 Database Migrations

**Migration script needed:**
1. Add `multi_resolution_bars` table
2. Add OI concentration columns to `ml_features`
3. Ensure existing Greeks columns are populated correctly

---

## 7. TIMELINE & RESOURCE ESTIMATES

### 7.1 Estimated Timeline

| Feature | Development | Testing | Integration | Total |
|---------|------------|---------|-------------|-------|
| Multi-Resolution | 3 days | 2 days | 1 day | 6 days |
| Enhanced Greeks | 3 days | 2 days | 1 day | 6 days |
| OI Concentration | 2 days | 1 day | 1 day | 4 days |
| Advanced Risk Management | 4 days | 2 days | 2 days | 8 days |
| Walk-Forward Testing | 3 days | 2 days | 1 day | 6 days |
| **Total** | **15 days** | **9 days** | **6 days** | **30 days** |

**Buffer:** +4 days for unexpected issues  
**Total Estimated Duration:** 3-4 weeks

### 7.2 Resource Requirements

- **Developer:** 1 full-time developer
- **Tester:** Part-time testing support
- **Database Admin:** Minimal (schema changes only)

---

## 8. RISK MITIGATION

### 8.1 Technical Risks

**Risk:** Aggregation overhead slows down real-time processing  
**Mitigation:** Profile and optimize; use async/batched database writes

**Risk:** Black-Scholes calculations incorrect  
**Mitigation:** Extensive unit tests against known values; cross-reference with market data

**Risk:** Database schema migration issues  
**Mitigation:** Test migrations on staging; make columns nullable initially

### 8.2 Data Quality Risks

**Risk:** Missing IV data causes incorrect Greeks  
**Mitigation:** Fallback to proxy calculations if IV unavailable

**Risk:** OI data gaps affect concentration metrics  
**Mitigation:** Handle missing data gracefully; use last known values

---

## 9. SUCCESS CRITERIA

### 9.1 Functional Requirements

- [ ] Multi-resolution bars generated correctly for all timeframes
- [ ] Enhanced Greeks calculated accurately (validated against test cases)
- [ ] OI concentration and skewness metrics available in feature vector
- [ ] All features persist to database correctly
- [ ] Feature pipeline consumes new features without errors

### 9.2 Performance Requirements

- [ ] Aggregation overhead < 1ms per tick
- [ ] Greeks calculation < 10ms for full option chain
- [ ] Database writes don't block main thread
- [ ] No increase in memory usage > 10%

### 9.3 Quality Requirements

- [ ] Unit test coverage > 80% for new modules
- [ ] Integration tests pass
- [ ] Code review completed
- [ ] Documentation updated

---

## 4. FEATURE 4: ADVANCED RISK MANAGEMENT

### 4.1 Overview

Enhance the existing risk management system (`risk_manager.py`) with comprehensive multi-layer risk controls:
- **Horizon-specific position limits** - Different limits for intraday, swing, and expiry trades
- **Greeks exposure limits** - Delta, Gamma, Vega, Theta constraints
- **Portfolio correlation checks** - Prevent over-concentration in correlated positions
- **Real-time VaR calculation** - Value at Risk monitoring
- **Liquidity risk assessment** - Check bid-ask spreads and order book depth
- **Market regime-based position sizing** - Adjust limits based on volatility regimes

### 4.2 Current State Analysis

**Existing Capabilities:**
- Basic Kelly Criterion position sizing (`get_optimal_position_size()`)
- Circuit breakers for daily loss and consecutive losses (`check_circuit_breaker()`)
- Portfolio-level constraints: net delta, max position size, total exposure (`check_portfolio_constraints()`)
- Position limit controls in `AutoExecutor` (max open positions, high confidence scaling)

**Gaps:**
- No Greeks exposure monitoring (Delta, Gamma, Vega, Theta limits)
- No correlation checks between positions
- No real-time VaR calculation
- No liquidity risk assessment
- No horizon-specific limits
- Limited regime-based adjustments

### 4.3 Architecture Design

```
┌─────────────────────┐
│  Strategy Signal    │
│  (BUY/SELL/HOLD)    │
└──────────┬──────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  AdvancedRiskManager                 │
│  ┌────────────────────────────────┐  │
│  │ Layer 1: Pre-Trade Checks      │  │
│  │ - Position size (horizon-based)│  │
│  │ - Greeks exposure limits       │  │
│  │ - Portfolio correlation        │  │
│  │ - Liquidity checks             │  │
│  │ - Market regime checks         │  │
│  └────────────────────────────────┘  │
│  ┌────────────────────────────────┐  │
│  │ Layer 2: Portfolio Monitoring  │  │
│  │ - Real-time VaR                │  │
│  │ - Greeks aggregation           │  │
│  │ - Exposure tracking            │  │
│  └────────────────────────────────┘  │
│  ┌────────────────────────────────┐  │
│  │ Layer 3: Circuit Breakers      │  │
│  │ - Daily loss limits            │  │
│  │ - Drawdown limits              │  │
│  │ - Consecutive loss limits      │  │
│  └────────────────────────────────┘  │
└──────────┬───────────────────────────┘
           │
           ▼
    ┌──────────────┐
    │ Execute /    │
    │ Reject Trade │
    └──────────────┘
```

### 4.4 Implementation Details

#### 4.4.1 Enhanced Risk Manager Class

**New Module: `risk_manager.py` (enhancement)**

```python
"""
Advanced Risk Management for OI Gemini.

Multi-layer risk framework with Greeks limits, correlation checks, VaR, and regime-based sizing.
"""

from typing import Dict, List, Optional, Tuple, Sequence
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from scipy.stats import norm

from time_utils import now_ist


@dataclass
class RiskLimitsConfig:
    """Configuration for risk limits."""
    
    # Horizon-specific position limits (as fraction of capital)
    max_position_size: Dict[str, float] = field(default_factory=lambda: {
        'intraday': 0.10,    # 10% of capital per intraday trade
        'swing': 0.20,       # 20% of capital per swing trade
        'expiry': 0.05,      # 5% of capital per expiry trade
        'default': 0.15      # 15% default
    })
    
    # Horizon-specific loss limits (as fraction of capital)
    max_loss_limits: Dict[str, float] = field(default_factory=lambda: {
        'intraday': 0.02,    # 2% of capital max loss per day (intraday)
        'swing': 0.05,       # 5% of capital max loss (swing)
        'expiry': 0.01,      # 1% of capital max loss (expiry day)
        'daily': 0.02        # 2% daily loss limit (system-wide)
    })
    
    # Greeks exposure limits
    greeks_limits: Dict[str, float] = field(default_factory=lambda: {
        'max_delta': 0.50,           # 50% of capital max delta exposure
        'max_gamma_exposure': 1000000000.0,  # Max gamma exposure (absolute)
        'max_vega_exposure': 0.30,   # 30% of capital max vega exposure
        'max_theta_exposure': -0.01, # Minimum theta (must be positive, i.e., theta > -0.01)
        'max_vanna_exposure': 0.20,  # 20% of capital max vanna exposure
    })
    
    # Correlation limits
    max_correlation_threshold: float = 0.80  # Max correlation between positions
    
    # VaR limits
    var_confidence_level: float = 0.95       # 95% VaR
    max_var_fraction: float = 0.03           # 3% of capital max VaR
    
    # Liquidity limits
    max_bid_ask_spread_pct: float = 0.01     # 1% max spread
    min_order_book_depth: float = 1000000.0  # Minimum order book depth (INR)
    
    # Market regime adjustments
    regime_adjustments: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'high_vol': {
            'position_size_multiplier': 0.7,  # Reduce size by 30%
            'max_loss_multiplier': 0.8        # Reduce loss limit by 20%
        },
        'low_vol': {
            'position_size_multiplier': 1.2,  # Increase size by 20%
            'max_loss_multiplier': 1.1        # Increase loss limit by 10%
        },
        'trending': {
            'position_size_multiplier': 1.1,
            'max_loss_multiplier': 1.0
        },
        'range_bound': {
            'position_size_multiplier': 0.9,
            'max_loss_multiplier': 1.0
        }
    })


class AdvancedRiskManager:
    """
    Multi-layer risk management framework.
    
    Provides comprehensive pre-trade checks, portfolio monitoring, and circuit breakers.
    """
    
    def __init__(self, config: Optional[RiskLimitsConfig] = None, account_size: float = 1_000_000.0):
        """
        Args:
            config: Risk limits configuration (defaults to conservative limits)
            account_size: Account size in INR
        """
        self.config = config or RiskLimitsConfig()
        self.account_size = account_size
        self.portfolio_positions: List[Dict] = []
        self.historical_returns: List[float] = []
        self.daily_pnl_history: List[float] = []
        
    def check_trade(
        self,
        proposed_trade: Dict,
        portfolio_state: Dict,
        market_regime: Optional[str] = None,
        current_prices: Optional[Dict[str, float]] = None,
        option_greeks: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Comprehensive pre-trade risk checks.
        
        Args:
            proposed_trade: Dict with keys:
                - 'signal': 'BUY' or 'SELL'
                - 'quantity_lots': int
                - 'strike': float
                - 'option_type': 'CE' or 'PE'
                - 'horizon': 'intraday', 'swing', or 'expiry'
                - 'price': float (current option price)
                - 'confidence': float
            portfolio_state: Dict with keys:
                - 'net_delta': float
                - 'total_gamma_exposure': float
                - 'total_vega_exposure': float
                - 'total_theta_exposure': float
                - 'total_mtm': float
                - 'daily_pnl': float
            market_regime: Optional market regime ('high_vol', 'low_vol', 'trending', 'range_bound')
            current_prices: Optional dict mapping symbols to current prices
            option_greeks: Optional dict mapping symbols to Greeks dicts
        
        Returns:
            Tuple of (allowed: bool, violations: List[str])
        """
        violations = []
        
        horizon = proposed_trade.get('horizon', 'default')
        signal = proposed_trade.get('signal', 'HOLD')
        quantity_lots = proposed_trade.get('quantity_lots', 0)
        option_type = proposed_trade.get('option_type', 'CE')
        strike = proposed_trade.get('strike', 0.0)
        price = proposed_trade.get('price', 0.0)
        
        # Apply regime adjustments
        regime_multipliers = self._get_regime_adjustments(market_regime)
        
        # 1. Position Size Check (Horizon-based)
        max_size_fraction = self.config.max_position_size.get(horizon, self.config.max_position_size['default'])
        max_size_fraction *= regime_multipliers['position_size_multiplier']
        
        position_value = quantity_lots * price * 50  # Assume 50 lot multiplier
        max_position_value = self.account_size * max_size_fraction
        
        if position_value > max_position_value:
            violations.append(
                f"Position size {position_value:.0f} exceeds limit {max_position_value:.0f} "
                f"for horizon {horizon} ({max_size_fraction:.1%})"
            )
        
        # 2. Greeks Exposure Check
        if option_greeks and option_type in option_greeks:
            greeks = option_greeks[option_type]
            
            # Calculate proposed Greeks contributions
            proposed_delta = self._calculate_proposed_delta(
                signal, quantity_lots, option_type, greeks.get('delta', 0.0)
            )
            proposed_gamma = self._calculate_proposed_gamma(
                signal, quantity_lots, option_type, greeks.get('gamma', 0.0), strike
            )
            proposed_vega = self._calculate_proposed_vega(
                signal, quantity_lots, option_type, greeks.get('vega', 0.0)
            )
            proposed_theta = self._calculate_proposed_theta(
                signal, quantity_lots, option_type, greeks.get('theta', 0.0)
            )
            proposed_vanna = self._calculate_proposed_vanna(
                signal, quantity_lots, option_type, greeks.get('vanna', 0.0)
            )
            
            # Check delta limit
            current_delta = portfolio_state.get('net_delta', 0.0)
            new_delta = abs(current_delta + proposed_delta)
            max_delta = self.account_size * self.config.greeks_limits['max_delta']
            if new_delta > max_delta:
                violations.append(
                    f"Net delta {new_delta:.1f} would exceed limit {max_delta:.1f} "
                    f"({self.config.greeks_limits['max_delta']:.1%} of capital)"
                )
            
            # Check gamma exposure limit
            current_gamma_exp = portfolio_state.get('total_gamma_exposure', 0.0)
            new_gamma_exp = current_gamma_exp + proposed_gamma
            max_gamma_exp = self.config.greeks_limits['max_gamma_exposure']
            if abs(new_gamma_exp) > max_gamma_exp:
                violations.append(
                    f"Gamma exposure {new_gamma_exp:.0f} would exceed limit {max_gamma_exp:.0f}"
                )
            
            # Check vega exposure limit
            current_vega_exp = portfolio_state.get('total_vega_exposure', 0.0)
            new_vega_exp = current_vega_exp + proposed_vega
            max_vega_exp = self.account_size * self.config.greeks_limits['max_vega_exposure']
            if abs(new_vega_exp) > max_vega_exp:
                violations.append(
                    f"Vega exposure {new_vega_exp:.1f} would exceed limit {max_vega_exp:.1f}"
                )
            
            # Check theta limit (must be positive, i.e., theta > threshold)
            current_theta_exp = portfolio_state.get('total_theta_exposure', 0.0)
            new_theta_exp = current_theta_exp + proposed_theta
            min_theta = self.config.greeks_limits['max_theta_exposure']
            if new_theta_exp < min_theta:
                violations.append(
                    f"Theta exposure {new_theta_exp:.4f} would be below minimum {min_theta:.4f} "
                    "(insufficient positive theta)"
                )
            
            # Check vanna exposure limit
            current_vanna_exp = portfolio_state.get('total_vanna_exposure', 0.0)
            new_vanna_exp = current_vanna_exp + proposed_vanna
            max_vanna_exp = self.account_size * self.config.greeks_limits['max_vanna_exposure']
            if abs(new_vanna_exp) > max_vanna_exp:
                violations.append(
                    f"Vanna exposure {new_vanna_exp:.1f} would exceed limit {max_vanna_exp:.1f}"
                )
        
        # 3. Portfolio Correlation Check
        correlation_violation = self._check_correlation(proposed_trade, self.portfolio_positions)
        if correlation_violation:
            violations.append(correlation_violation)
        
        # 4. Liquidity Check
        liquidity_violation = self._check_liquidity(proposed_trade, current_prices)
        if liquidity_violation:
            violations.append(liquidity_violation)
        
        # 5. VaR Check
        var_violation = self._check_var(proposed_trade, portfolio_state)
        if var_violation:
            violations.append(var_violation)
        
        # 6. Daily Loss Limit Check (Horizon-specific)
        daily_pnl = portfolio_state.get('daily_pnl', 0.0)
        max_loss_fraction = self.config.max_loss_limits.get(horizon, self.config.max_loss_limits['daily'])
        max_loss_fraction *= regime_multipliers['max_loss_multiplier']
        max_loss = self.account_size * max_loss_fraction
        
        if daily_pnl < -max_loss:
            violations.append(
                f"Daily loss {daily_pnl:.0f} exceeds limit {max_loss:.0f} "
                f"({max_loss_fraction:.1%} of capital for {horizon})"
            )
        
        return len(violations) == 0, violations
    
    def _get_regime_adjustments(self, regime: Optional[str]) -> Dict[str, float]:
        """Get position size and loss limit multipliers for market regime."""
        if not regime or regime not in self.config.regime_adjustments:
            return {'position_size_multiplier': 1.0, 'max_loss_multiplier': 1.0}
        return self.config.regime_adjustments[regime]
    
    def _calculate_proposed_delta(
        self,
        signal: str,
        quantity_lots: int,
        option_type: str,
        delta: float
    ) -> float:
        """Calculate proposed delta contribution."""
        direction = 1.0 if signal == 'BUY' else -1.0
        delta_sign = 1.0 if option_type == 'CE' else -1.0
        return direction * delta_sign * delta * quantity_lots * 50  # 50 = lot multiplier
    
    def _calculate_proposed_gamma(
        self,
        signal: str,
        quantity_lots: int,
        option_type: str,
        gamma: float,
        strike: float,
        spot_price: Optional[float] = None
    ) -> float:
        """Calculate proposed gamma exposure contribution."""
        # GEX = OI × Gamma × Spot² × Multiplier
        # Simplified: use strike as proxy for spot if not provided
        spot = spot_price or strike
        direction = 1.0 if signal == 'BUY' else -1.0
        gamma_sign = 1.0 if option_type == 'CE' else -1.0
        return direction * gamma_sign * gamma * (spot ** 2) * quantity_lots * 50
    
    def _calculate_proposed_vega(
        self,
        signal: str,
        quantity_lots: int,
        option_type: str,
        vega: float
    ) -> float:
        """Calculate proposed vega exposure contribution."""
        direction = 1.0 if signal == 'BUY' else -1.0
        return direction * vega * quantity_lots * 50
    
    def _calculate_proposed_theta(
        self,
        signal: str,
        quantity_lots: int,
        option_type: str,
        theta: float
    ) -> float:
        """Calculate proposed theta exposure contribution."""
        direction = 1.0 if signal == 'BUY' else -1.0
        return direction * theta * quantity_lots * 50
    
    def _calculate_proposed_vanna(
        self,
        signal: str,
        quantity_lots: int,
        option_type: str,
        vanna: float
    ) -> float:
        """Calculate proposed vanna exposure contribution."""
        direction = 1.0 if signal == 'BUY' else -1.0
        return direction * vanna * quantity_lots * 50
    
    def _check_correlation(
        self,
        proposed_trade: Dict,
        existing_positions: List[Dict]
    ) -> Optional[str]:
        """
        Check if proposed trade is too correlated with existing positions.
        
        Simplified correlation check: same strike/expiry = high correlation.
        Full implementation would calculate historical correlation.
        """
        proposed_strike = proposed_trade.get('strike')
        proposed_expiry = proposed_trade.get('expiry_date')
        
        if not proposed_strike or not existing_positions:
            return None
        
        # Count positions with same strike (high correlation)
        same_strike_count = sum(
            1 for pos in existing_positions
            if pos.get('strike') == proposed_strike
        )
        
        # Count positions with same expiry (moderate correlation)
        same_expiry_count = sum(
            1 for pos in existing_positions
            if pos.get('expiry_date') == proposed_expiry
        )
        
        # Warn if too many correlated positions
        if same_strike_count >= 2:
            return f"High correlation: {same_strike_count} existing positions at strike {proposed_strike}"
        
        if same_expiry_count >= 3:
            return f"Moderate correlation: {same_expiry_count} existing positions at same expiry"
        
        return None
    
    def _check_liquidity(
        self,
        proposed_trade: Dict,
        current_prices: Optional[Dict[str, float]]
    ) -> Optional[str]:
        """
        Check liquidity constraints.
        
        Args:
            proposed_trade: Trade dict with 'spread', 'bid_ask_spread_pct', 'order_book_depth'
            current_prices: Current prices dict
        """
        spread_pct = proposed_trade.get('bid_ask_spread_pct', 0.0)
        if spread_pct > self.config.max_bid_ask_spread_pct:
            return (
                f"Bid-ask spread {spread_pct:.2%} exceeds limit "
                f"{self.config.max_bid_ask_spread_pct:.2%}"
            )
        
        order_book_depth = proposed_trade.get('order_book_depth', float('inf'))
        if order_book_depth < self.config.min_order_book_depth:
            return (
                f"Order book depth {order_book_depth:.0f} below minimum "
                f"{self.config.min_order_book_depth:.0f}"
            )
        
        return None
    
    def calculate_var(
        self,
        portfolio_state: Dict,
        confidence_level: float = 0.95,
        lookback_days: int = 30
    ) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) for the portfolio.
        
        Uses historical simulation method.
        
        Args:
            portfolio_state: Portfolio state with 'total_mtm'
            confidence_level: VaR confidence level (default 0.95 = 95%)
            lookback_days: Number of days for historical returns
        
        Returns:
            Dict with 'var_absolute', 'var_percentage', 'expected_shortfall'
        """
        if not self.historical_returns or len(self.historical_returns) < lookback_days:
            # Fallback: use simple parametric VaR
            # Assume normal distribution with 20% annualized volatility
            annual_vol = 0.20
            daily_vol = annual_vol / np.sqrt(252)
            z_score = norm.ppf(1 - confidence_level)
            current_value = portfolio_state.get('total_mtm', 0.0) + self.account_size
            
            var_absolute = abs(z_score * daily_vol * current_value)
            var_percentage = abs(z_score * daily_vol)
            
            return {
                'var_absolute': var_absolute,
                'var_percentage': var_percentage,
                'expected_shortfall': var_absolute * 1.2,  # ES ~ 1.2x VaR for normal dist
                'confidence_level': confidence_level
            }
        
        # Historical simulation
        recent_returns = self.historical_returns[-lookback_days:]
        var_percentile = (1 - confidence_level) * 100
        var_return = np.percentile(recent_returns, var_percentile)
        
        current_value = portfolio_state.get('total_mtm', 0.0) + self.account_size
        var_absolute = abs(var_return * current_value)
        
        # Expected Shortfall (Conditional VaR)
        es_returns = [r for r in recent_returns if r <= var_return]
        es_return = np.mean(es_returns) if es_returns else var_return
        expected_shortfall = abs(es_return * current_value)
        
        return {
            'var_absolute': var_absolute,
            'var_percentage': abs(var_return),
            'expected_shortfall': expected_shortfall,
            'confidence_level': confidence_level
        }
    
    def _check_var(
        self,
        proposed_trade: Dict,
        portfolio_state: Dict
    ) -> Optional[str]:
        """Check if proposed trade would exceed VaR limit."""
        var_metrics = self.calculate_var(
            portfolio_state,
            confidence_level=self.config.var_confidence_level
        )
        
        max_var = self.account_size * self.config.max_var_fraction
        if var_metrics['var_absolute'] > max_var:
            return (
                f"VaR {var_metrics['var_absolute']:.0f} exceeds limit {max_var:.0f} "
                f"({self.config.max_var_fraction:.1%} of capital at "
                f"{self.config.var_confidence_level:.0%} confidence)"
            )
        
        return None
    
    def update_portfolio_state(
        self,
        positions: List[Dict],
        current_prices: Dict[str, float],
        option_greeks: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict:
        """
        Calculate and return current portfolio state.
        
        Args:
            positions: List of position dicts
            current_prices: Dict mapping symbols to current prices
            option_greeks: Optional dict mapping symbols to Greeks
        
        Returns:
            Portfolio state dict with aggregated metrics
        """
        total_mtm = 0.0
        net_delta = 0.0
        total_gamma_exposure = 0.0
        total_vega_exposure = 0.0
        total_theta_exposure = 0.0
        total_vanna_exposure = 0.0
        total_exposure = 0.0
        
        for pos in positions:
            symbol = pos.get('symbol')
            entry_price = pos.get('entry_price', 0.0)
            current_price = current_prices.get(symbol, entry_price)
            qty = pos.get('qty', 0)
            side = pos.get('side', 'B')
            option_type = pos.get('type', 'CE')
            strike = pos.get('strike', 0.0)
            
            # Calculate MTM
            if side == 'B':
                mtm = (current_price - entry_price) * qty
            else:
                mtm = (entry_price - current_price) * qty
            total_mtm += mtm
            
            # Aggregate Greeks if available
            if option_greeks and symbol in option_greeks:
                greeks = option_greeks[symbol]
                direction = 1.0 if side == 'B' else -1.0
                delta_sign = 1.0 if option_type == 'CE' else -1.0
                
                delta = greeks.get('delta', 0.0)
                gamma = greeks.get('gamma', 0.0)
                vega = greeks.get('vega', 0.0)
                theta = greeks.get('theta', 0.0)
                vanna = greeks.get('vanna', 0.0)
                
                net_delta += direction * delta_sign * delta * (qty / 50)  # Convert to lots
                
                # Gamma exposure
                spot_price = current_prices.get('underlying', strike)
                gamma_exp = direction * (1.0 if option_type == 'CE' else -1.0) * gamma * (spot_price ** 2) * (qty / 50)
                total_gamma_exposure += gamma_exp
                
                # Vega exposure
                vega_exp = direction * vega * (qty / 50)
                total_vega_exposure += vega_exp
                
                # Theta exposure
                theta_exp = direction * theta * (qty / 50)
                total_theta_exposure += theta_exp
                
                # Vanna exposure
                vanna_exp = direction * (1.0 if option_type == 'CE' else -1.0) * vanna * spot_price * (qty / 50)
                total_vanna_exposure += vanna_exp
            
            # Total exposure
            total_exposure += (qty / 50) * 175000  # Approx margin per lot
        
        self.portfolio_positions = positions
        
        return {
            'total_mtm': total_mtm,
            'net_delta': net_delta,
            'total_gamma_exposure': total_gamma_exposure,
            'total_vega_exposure': total_vega_exposure,
            'total_theta_exposure': total_theta_exposure,
            'total_vanna_exposure': total_vanna_exposure,
            'total_exposure': total_exposure,
            'exposure_pct': (total_exposure / self.account_size) * 100 if self.account_size > 0 else 0.0,
            'return_pct': (total_mtm / self.account_size) * 100 if self.account_size > 0 else 0.0
        }
```

#### 4.4.2 Integration with Auto Executor

**Modify `execution/auto_executor.py`:**

```python
from risk_manager import AdvancedRiskManager, RiskLimitsConfig

class AutoExecutor:
    def __init__(self, exchange: str, config: ExecutionConfig):
        # ... existing initialization ...
        
        # Initialize advanced risk manager
        risk_config = RiskLimitsConfig(
            max_position_size={
                'intraday': 0.10,
                'swing': 0.20,
                'expiry': 0.05,
                'default': 0.15
            },
            greeks_limits={
                'max_delta': config.max_net_delta / config.account_size if config.max_net_delta else 0.50,
                'max_gamma_exposure': 1000000000.0,
                'max_vega_exposure': 0.30,
                'max_theta_exposure': -0.01,
                'max_vanna_exposure': 0.20
            }
        )
        self.risk_manager = AdvancedRiskManager(
            config=risk_config,
            account_size=config.account_size
        )
    
    def should_execute(
        self,
        signal: StrategySignal,
        current_price: float,
        portfolio_state: Optional[Dict] = None,
        current_open_positions: Optional[Dict[str, Dict]] = None,
        market_regime: Optional[str] = None,
        option_greeks: Optional[Dict[str, Dict[str, float]]] = None
    ) -> ExecutionResult:
        """Enhanced should_execute with advanced risk checks."""
        # ... existing checks ...
        
        # Advanced risk checks
        proposed_trade = {
            'signal': signal.signal,
            'quantity_lots': signal.metadata.get('recommended_lots', 0),
            'strike': signal.metadata.get('strike', 0.0),
            'option_type': signal.metadata.get('option_type', 'CE'),
            'horizon': signal.metadata.get('horizon', 'default'),
            'price': current_price,
            'confidence': signal.confidence,
            'bid_ask_spread_pct': signal.metadata.get('spread_pct', 0.0),
            'order_book_depth': signal.metadata.get('order_book_depth', float('inf'))
        }
        
        allowed, violations = self.risk_manager.check_trade(
            proposed_trade=proposed_trade,
            portfolio_state=portfolio_state or {},
            market_regime=market_regime,
            option_greeks=option_greeks
        )
        
        if not allowed:
            result = ExecutionResult(
                executed=False,
                reason=f"Risk check failed: {'; '.join(violations)}"
            )
            self._record_paper_trade_metric(
                executed=False,
                reason=result.reason,
                signal=signal,
                quantity_lots=0,
                pnl=None,
                constraint_violation=True
            )
            return result
        
        # ... continue with existing execution logic ...
```

#### 4.4.3 Real-Time Risk Monitoring Module

**New Module: `risk_monitoring.py`:**

```python
"""
Real-time risk monitoring and alerting.

Continuously monitors portfolio risk metrics and triggers alerts when limits are approached.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from threading import Lock, Thread
import time

from risk_manager import AdvancedRiskManager
from time_utils import now_ist

LOGGER = logging.getLogger(__name__)


class RiskMonitor:
    """
    Real-time risk monitoring daemon.
    
    Monitors portfolio risk metrics and sends alerts when thresholds are breached.
    """
    
    def __init__(
        self,
        risk_manager: AdvancedRiskManager,
        check_interval_seconds: int = 60,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            risk_manager: AdvancedRiskManager instance
            check_interval_seconds: How often to check risk metrics
            alert_thresholds: Dict of threshold percentages (e.g., {'var': 0.8} = alert at 80% of limit)
        """
        self.risk_manager = risk_manager
        self.check_interval = check_interval_seconds
        self.alert_thresholds = alert_thresholds or {
            'var': 0.80,          # Alert at 80% of VaR limit
            'delta': 0.75,        # Alert at 75% of delta limit
            'gamma': 0.75,        # Alert at 75% of gamma limit
            'daily_loss': 0.75    # Alert at 75% of daily loss limit
        }
        
        self.monitoring = False
        self.monitor_thread: Optional[Thread] = None
        self.lock = Lock()
        self.last_alert_time: Dict[str, datetime] = {}
        self.alert_cooldown_seconds = 300  # 5 minutes between alerts for same metric
    
    def start_monitoring(self):
        """Start background monitoring thread."""
        if self.monitoring:
            LOGGER.warning("Risk monitoring already running")
            return
        
        self.monitoring = True
        self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        LOGGER.info("Risk monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        LOGGER.info("Risk monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                self._check_risk_metrics()
            except Exception as e:
                LOGGER.error(f"Error in risk monitoring loop: {e}", exc_info=True)
            
            time.sleep(self.check_interval)
    
    def _check_risk_metrics(self):
        """Check all risk metrics and trigger alerts if needed."""
        # This would need access to current portfolio state
        # Implementation depends on how portfolio state is accessed
        # For now, this is a placeholder structure
        
        # Example: Check VaR
        # portfolio_state = self._get_current_portfolio_state()
        # var_metrics = self.risk_manager.calculate_var(portfolio_state)
        # self._check_alert('var', var_metrics['var_absolute'], limit)
        
        pass
    
    def _check_alert(self, metric_name: str, current_value: float, limit: float):
        """Check if alert should be triggered and send if needed."""
        threshold_pct = self.alert_thresholds.get(metric_name, 0.75)
        threshold_value = limit * threshold_pct
        
        if current_value >= threshold_value:
            # Check cooldown
            last_alert = self.last_alert_time.get(metric_name)
            now = now_ist()
            
            if not last_alert or (now - last_alert).total_seconds() > self.alert_cooldown_seconds:
                self._send_alert(metric_name, current_value, limit, threshold_value)
                self.last_alert_time[metric_name] = now
    
    def _send_alert(self, metric_name: str, current_value: float, limit: float, threshold: float):
        """Send risk alert (logging, email, Slack, etc.)."""
        alert_msg = (
            f"⚠️ RISK ALERT: {metric_name.upper()} "
            f"{current_value:.0f} is {current_value/limit:.1%} of limit {limit:.0f} "
            f"(threshold: {threshold:.0f})"
        )
        LOGGER.warning(alert_msg)
        # TODO: Add email/Slack/webhook notifications
```

### 4.5 Testing Strategy

**Unit Tests:**
1. Test each risk check independently
2. Verify Greeks calculations match expected values
3. Test correlation detection
4. Test VaR calculation accuracy
5. Test regime adjustments

**Integration Tests:**
1. End-to-end: signal → risk checks → execute/reject
2. Verify alerts trigger at correct thresholds
3. Test portfolio state updates

**Performance Tests:**
1. Risk checks should complete in <10ms
2. Monitor loop should not block main thread

### 4.6 Configuration Updates

**Add to `config.py`:**

```python
@dataclass
class AppConfig:
    # ... existing fields ...
    
    # Advanced Risk Management
    risk_var_confidence_level: float = 0.95
    risk_max_var_fraction: float = 0.03
    risk_max_correlation_threshold: float = 0.80
    risk_max_bid_ask_spread_pct: float = 0.01
    risk_min_order_book_depth: float = 1000000.0
    risk_monitoring_enabled: bool = True
    risk_monitoring_interval_seconds: int = 60
```

---

## 5. FEATURE 5: WALK-FORWARD TESTING

### 5.1 Overview

Implement a comprehensive walk-forward testing framework that:
- Splits historical data into multiple train/test segments
- Trains models on training windows and tests on subsequent windows
- Evaluates performance across all segments
- Provides statistical analysis of results
- Prevents data leakage through proper temporal splits

### 5.2 Current State Analysis

**Existing Capabilities:**
- Basic backtesting engine (`backtesting/engine.py`)
- Single-period backtesting on fixed date ranges
- Walk-forward segments in training orchestrator (`train_orchestrator.py`) for model training

**Gaps:**
- No multi-period walk-forward testing for strategy validation
- No statistical aggregation across multiple test periods
- No out-of-sample robustness testing
- Limited performance analysis across different market regimes

### 5.3 Architecture Design

```
┌─────────────────────────────────┐
│  Historical Data                │
│  (Full date range)              │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│  WalkForwardTester                          │
│  ┌──────────────────────────────────────┐   │
│  │ Segment Generator                    │   │
│  │ - Train: 60 days                     │   │
│  │ - Test: 20 days                      │   │
│  │ - Step: 10 days                      │   │
│  └──────────────────────────────────────┘   │
│  ┌──────────────────────────────────────┐   │
│  │ Per-Segment Loop                     │   │
│  │ 1. Train model on train window       │   │
│  │ 2. Test on test window               │   │
│  │ 3. Calculate metrics                 │   │
│  └──────────────────────────────────────┘   │
│  ┌──────────────────────────────────────┐   │
│  │ Results Aggregation                  │   │
│  │ - Mean/std across segments           │   │
│  │ - Best/worst periods                 │   │
│  │ - Regime-specific analysis           │   │
│  └──────────────────────────────────────┘   │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  WalkForwardResult              │
│  - Segment results              │
│  - Aggregated metrics           │
│  - Statistical analysis         │
└─────────────────────────────────┘
```

### 5.4 Implementation Details

#### 5.4.1 Walk-Forward Tester Module

**New Module: `backtesting/walk_forward.py`:**

```python
"""
Walk-Forward Testing Framework for Strategy Validation.

Implements multi-period walk-forward testing to evaluate strategy robustness
across different market conditions.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

from backtesting.engine import BacktestEngine, BacktestConfig, BacktestResult
from ml_core import MLSignalGenerator
from risk_manager import calculate_trading_metrics

LOGGER = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward testing."""
    exchange: str
    start_date: date
    end_date: date
    train_days: int = 60          # Training window size (days)
    test_days: int = 20            # Test window size (days)
    step_days: int = 10            # Step size between segments (days)
    
    # Backtest configuration (passed to each segment)
    strategy: str = "ml_signal"
    holding_period_minutes: int = 15
    transaction_cost_bps: float = 2.0
    slippage_bps: float = 1.0
    min_confidence: float = 0.55
    account_size: float = 1_000_000.0
    margin_per_lot: float = 75_000.0
    max_risk_per_trade: float = 0.02
    
    # Model retraining options
    retrain_each_segment: bool = True  # Retrain model for each segment
    model_params: Optional[Dict] = None  # Fixed model params (if not retraining)
    
    def __post_init__(self):
        if self.end_date < self.start_date:
            raise ValueError("end_date must be >= start_date")
        if self.train_days <= 0 or self.test_days <= 0 or self.step_days <= 0:
            raise ValueError("train_days, test_days, and step_days must be positive")
        if self.step_days > self.test_days:
            LOGGER.warning("step_days > test_days may cause overlapping test periods")


@dataclass
class SegmentResult:
    """Results for a single walk-forward segment."""
    segment_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    
    # Test period results
    num_trades: int
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    
    # Model performance
    model_accuracy: Optional[float] = None
    model_f1_score: Optional[float] = None
    
    # Market regime indicators
    avg_volatility: Optional[float] = None
    avg_vix: Optional[float] = None
    market_trend: Optional[str] = None  # 'bull', 'bear', 'neutral'
    
    metadata: Dict = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward test results."""
    config: WalkForwardConfig
    segments: List[SegmentResult]
    
    # Aggregated metrics
    avg_sharpe: float
    std_sharpe: float
    avg_max_drawdown: float
    avg_win_rate: float
    avg_profit_factor: float
    
    # Best/worst segments
    best_segment_id: int
    worst_segment_id: int
    
    # Consistency metrics
    positive_periods: int  # Number of periods with positive returns
    consistency_ratio: float  # Ratio of positive periods to total
    
    # Statistical analysis
    sharpe_distribution: Dict[str, float]  # min, 25th, median, 75th, max
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'config': {
                'exchange': self.config.exchange,
                'start_date': self.config.start_date.isoformat(),
                'end_date': self.config.end_date.isoformat(),
                'train_days': self.config.train_days,
                'test_days': self.config.test_days,
                'step_days': self.config.step_days
            },
            'summary': {
                'num_segments': len(self.segments),
                'avg_sharpe': self.avg_sharpe,
                'std_sharpe': self.std_sharpe,
                'avg_max_drawdown': self.avg_max_drawdown,
                'avg_win_rate': self.avg_win_rate,
                'avg_profit_factor': self.avg_profit_factor,
                'consistency_ratio': self.consistency_ratio,
                'positive_periods': self.positive_periods
            },
            'segments': [
                {
                    'segment_id': seg.segment_id,
                    'train_start': seg.train_start.isoformat(),
                    'train_end': seg.train_end.isoformat(),
                    'test_start': seg.test_start.isoformat(),
                    'test_end': seg.test_end.isoformat(),
                    'num_trades': seg.num_trades,
                    'total_pnl': seg.total_pnl,
                    'sharpe_ratio': seg.sharpe_ratio,
                    'max_drawdown': seg.max_drawdown,
                    'win_rate': seg.win_rate,
                    'profit_factor': seg.profit_factor
                }
                for seg in self.segments
            ]
        }


class WalkForwardTester:
    """
    Multi-period walk-forward testing framework.
    
    Splits historical data into train/test segments and evaluates strategy
    performance across all segments to assess robustness.
    """
    
    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.segments: List[Dict] = []
    
    def generate_segments(self) -> List[Dict]:
        """
        Generate train/test segment windows.
        
        Returns:
            List of segment dicts with 'train_start', 'train_end', 'test_start', 'test_end'
        """
        segments = []
        current_start = self.config.start_date
        segment_id = 0
        
        while True:
            train_end = current_start + timedelta(days=self.config.train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.config.test_days)
            
            # Stop if test period exceeds end date
            if test_end > self.config.end_date:
                break
            
            segments.append({
                'segment_id': segment_id,
                'train_start': current_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })
            
            segment_id += 1
            current_start += timedelta(days=self.config.step_days)
        
        self.segments = segments
        LOGGER.info(f"Generated {len(segments)} walk-forward segments")
        return segments
    
    def run(self) -> WalkForwardResult:
        """
        Run walk-forward test across all segments.
        
        Returns:
            WalkForwardResult with aggregated metrics
        """
        if not self.segments:
            self.generate_segments()
        
        if not self.segments:
            LOGGER.warning("No segments generated for walk-forward test")
            return self._empty_result()
        
        segment_results: List[SegmentResult] = []
        
        for segment in self.segments:
            LOGGER.info(
                f"Processing segment {segment['segment_id']}: "
                f"Train {segment['train_start']} to {segment['train_end']}, "
                f"Test {segment['test_start']} to {segment['test_end']}"
            )
            
            # Train model on training period (if enabled)
            if self.config.retrain_each_segment:
                self._train_model_for_segment(
                    segment['train_start'],
                    segment['train_end']
                )
            
            # Run backtest on test period
            segment_result = self._run_segment_backtest(segment)
            segment_results.append(segment_result)
        
        # Aggregate results
        return self._aggregate_results(segment_results)
    
    def _train_model_for_segment(self, train_start: date, train_end: date):
        """
        Train model on training segment.
        
        This would integrate with train_orchestrator or ml_core to retrain models.
        For now, this is a placeholder - actual implementation depends on model training pipeline.
        """
        # TODO: Integrate with model training pipeline
        # Options:
        # 1. Call train_orchestrator.train_model() with train_start/train_end
        # 2. Use MLSignalGenerator's retrain capability (if available)
        # 3. Load pre-trained models for each segment
        
        LOGGER.debug(f"Training model for period {train_start} to {train_end}")
        # Placeholder - actual training logic goes here
        pass
    
    def _run_segment_backtest(self, segment: Dict) -> SegmentResult:
        """
        Run backtest for a single segment.
        
        Args:
            segment: Segment dict with train/test dates
        
        Returns:
            SegmentResult with test period metrics
        """
        # Create backtest config for test period
        backtest_config = BacktestConfig(
            exchange=self.config.exchange,
            start=segment['test_start'],
            end=segment['test_end'],
            strategy=self.config.strategy,
            holding_period_minutes=self.config.holding_period_minutes,
            transaction_cost_bps=self.config.transaction_cost_bps,
            slippage_bps=self.config.slippage_bps,
            min_confidence=self.config.min_confidence,
            account_size=self.config.account_size,
            margin_per_lot=self.config.margin_per_lot,
            max_risk_per_trade=self.config.max_risk_per_trade
        )
        
        # Run backtest
        engine = BacktestEngine(backtest_config)
        backtest_result = engine.run()
        
        # Extract metrics
        metrics = backtest_result.metrics
        
        # Calculate market regime indicators (simplified)
        avg_volatility, avg_vix, market_trend = self._analyze_market_regime(
            segment['test_start'],
            segment['test_end']
        )
        
        return SegmentResult(
            segment_id=segment['segment_id'],
            train_start=segment['train_start'],
            train_end=segment['train_end'],
            test_start=segment['test_start'],
            test_end=segment['test_end'],
            num_trades=metrics.get('num_trades', 0),
            total_pnl=metrics.get('net_total_pnl', 0.0),
            sharpe_ratio=metrics.get('sharpe_ratio', 0.0) if not np.isnan(metrics.get('sharpe_ratio', np.nan)) else 0.0,
            max_drawdown=metrics.get('net_max_drawdown', 0.0),
            win_rate=metrics.get('win_rate', 0.0),
            profit_factor=metrics.get('profit_factor', 0.0) if not np.isnan(metrics.get('profit_factor', np.nan)) else 0.0,
            avg_trade_pnl=metrics.get('avg_trade_pnl', 0.0),
            avg_volatility=avg_volatility,
            avg_vix=avg_vix,
            market_trend=market_trend,
            metadata={
                'gross_pnl': metrics.get('gross_total_pnl', 0.0),
                'cost_per_trade': metrics.get('cost_per_trade', 0.0)
            }
        )
    
    def _analyze_market_regime(
        self,
        start_date: date,
        end_date: date
    ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Analyze market regime for test period.
        
        Returns:
            Tuple of (avg_volatility, avg_vix, market_trend)
        """
        # TODO: Load historical data and calculate:
        # - Average realized volatility
        # - Average VIX
        # - Market trend (bull/bear/neutral based on returns)
        
        # Placeholder implementation
        return None, None, None
    
    def _aggregate_results(self, segment_results: List[SegmentResult]) -> WalkForwardResult:
        """
        Aggregate results across all segments.
        
        Args:
            segment_results: List of SegmentResult objects
        
        Returns:
            WalkForwardResult with aggregated metrics
        """
        if not segment_results:
            return self._empty_result()
        
        # Extract metrics arrays
        sharpe_ratios = [r.sharpe_ratio for r in segment_results]
        max_drawdowns = [r.max_drawdown for r in segment_results]
        win_rates = [r.win_rate for r in segment_results]
        profit_factors = [r.profit_factor for r in segment_results if not np.isnan(r.profit_factor)]
        total_pnls = [r.total_pnl for r in segment_results]
        
        # Calculate statistics
        avg_sharpe = float(np.mean(sharpe_ratios)) if sharpe_ratios else 0.0
        std_sharpe = float(np.std(sharpe_ratios)) if sharpe_ratios else 0.0
        avg_max_drawdown = float(np.mean(max_drawdowns)) if max_drawdowns else 0.0
        avg_win_rate = float(np.mean(win_rates)) if win_rates else 0.0
        avg_profit_factor = float(np.mean(profit_factors)) if profit_factors else 0.0
        
        # Find best/worst segments (by Sharpe ratio)
        if sharpe_ratios:
            best_idx = int(np.argmax(sharpe_ratios))
            worst_idx = int(np.argmin(sharpe_ratios))
        else:
            best_idx = 0
            worst_idx = 0
        
        # Consistency metrics
        positive_periods = sum(1 for pnl in total_pnls if pnl > 0)
        consistency_ratio = positive_periods / len(segment_results) if segment_results else 0.0
        
        # Sharpe distribution
        if sharpe_ratios:
            sharpe_sorted = sorted(sharpe_ratios)
            sharpe_distribution = {
                'min': float(sharpe_sorted[0]),
                '25th': float(np.percentile(sharpe_ratios, 25)),
                'median': float(np.median(sharpe_ratios)),
                '75th': float(np.percentile(sharpe_ratios, 75)),
                'max': float(sharpe_sorted[-1])
            }
        else:
            sharpe_distribution = {'min': 0.0, '25th': 0.0, 'median': 0.0, '75th': 0.0, 'max': 0.0}
        
        return WalkForwardResult(
            config=self.config,
            segments=segment_results,
            avg_sharpe=avg_sharpe,
            std_sharpe=std_sharpe,
            avg_max_drawdown=avg_max_drawdown,
            avg_win_rate=avg_win_rate,
            avg_profit_factor=avg_profit_factor,
            best_segment_id=segment_results[best_idx].segment_id if segment_results else 0,
            worst_segment_id=segment_results[worst_idx].segment_id if segment_results else 0,
            positive_periods=positive_periods,
            consistency_ratio=consistency_ratio,
            sharpe_distribution=sharpe_distribution
        )
    
    def _empty_result(self) -> WalkForwardResult:
        """Return empty result when no segments available."""
        return WalkForwardResult(
            config=self.config,
            segments=[],
            avg_sharpe=0.0,
            std_sharpe=0.0,
            avg_max_drawdown=0.0,
            avg_win_rate=0.0,
            avg_profit_factor=0.0,
            best_segment_id=0,
            worst_segment_id=0,
            positive_periods=0,
            consistency_ratio=0.0,
            sharpe_distribution={'min': 0.0, '25th': 0.0, 'median': 0.0, '75th': 0.0, 'max': 0.0}
        )
```

#### 5.4.2 Integration with Backtesting Engine

**Modify `backtesting/engine.py` (minimal changes needed):**

The existing `BacktestEngine` can be used as-is. The walk-forward tester wraps it.

#### 5.4.3 CLI/API Interface

**New Script: `scripts/run_walk_forward.py`:**

```python
"""
Command-line interface for walk-forward testing.
"""

import argparse
import json
from datetime import date, datetime
from pathlib import Path

from backtesting.walk_forward import WalkForwardTester, WalkForwardConfig

def main():
    parser = argparse.ArgumentParser(description='Run walk-forward backtest')
    parser.add_argument('--exchange', required=True, help='Exchange (NSE, BSE)')
    parser.add_argument('--start', required=True, type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--train-days', type=int, default=60, help='Training window (days)')
    parser.add_argument('--test-days', type=int, default=20, help='Test window (days)')
    parser.add_argument('--step-days', type=int, default=10, help='Step size (days)')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    
    args = parser.parse_args()
    
    config = WalkForwardConfig(
        exchange=args.exchange,
        start_date=date.fromisoformat(args.start),
        end_date=date.fromisoformat(args.end),
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days
    )
    
    tester = WalkForwardTester(config)
    result = tester.run()
    
    # Output results
    output_dict = result.to_dict()
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_dict, f, indent=2)
        print(f"Results saved to {output_path}")
    else:
        print(json.dumps(output_dict, indent=2))

if __name__ == '__main__':
    main()
```

### 5.5 Testing Strategy

**Unit Tests:**
1. Test segment generation logic
2. Verify no data leakage (train/test separation)
3. Test aggregation calculations
4. Test edge cases (insufficient data, overlapping segments)

**Integration Tests:**
1. End-to-end walk-forward test on sample data
2. Verify results match individual backtest results
3. Test model retraining integration

**Validation:**
1. Compare walk-forward results with single-period backtest
2. Verify consistency across segments

### 5.6 Usage Example

```python
from backtesting.walk_forward import WalkForwardTester, WalkForwardConfig
from datetime import date

config = WalkForwardConfig(
    exchange='NSE',
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 1),
    train_days=60,
    test_days=20,
    step_days=10,
    min_confidence=0.60
)

tester = WalkForwardTester(config)
result = tester.run()

print(f"Average Sharpe: {result.avg_sharpe:.2f} ± {result.std_sharpe:.2f}")
print(f"Consistency: {result.consistency_ratio:.1%} positive periods")
print(f"Best segment: {result.best_segment_id} (Sharpe: {result.segments[result.best_segment_id].sharpe_ratio:.2f})")
```

---

## 10. APPENDICES

### 10.1 File Structure

```
OI_Gemini/
├── data_ingestion/
│   └── multi_resolution_aggregator.py  # NEW
├── utils/
│   └── greeks_calculator.py             # NEW
├── feature_engineering.py               # MODIFY
├── handlers.py                          # MODIFY
├── database_new.py                      # MODIFY
├── config.py                            # MODIFY
├── risk_manager.py                      # MODIFY (enhance)
├── risk_monitoring.py                   # NEW
├── backtesting/
│   └── walk_forward.py                  # NEW
├── scripts/
│   └── run_walk_forward.py              # NEW
└── requirements.txt                     # MODIFY
```

### 10.2 Key Functions Reference

**Multi-Resolution:**
- `BarAggregator.add_tick()` - Add tick, return completed bar
- `MultiResolutionAggregator.add_tick()` - Add to all resolutions
- `save_multi_resolution_bars()` - Persist to database

**Enhanced Greeks:**
- `GreeksCalculator.calculate_greeks()` - Calculate all Greeks for one option
- `calculate_gamma_exposure()` - Aggregate GEX for option chain

**OI Concentration:**
- `calculate_oi_concentration()` - Concentration and skewness metrics
- `calculate_oi_velocity()` - OI change rate

**Advanced Risk Management:**
- `AdvancedRiskManager.check_trade()` - Comprehensive pre-trade risk checks
- `AdvancedRiskManager.calculate_var()` - Value at Risk calculation
- `AdvancedRiskManager.update_portfolio_state()` - Portfolio state aggregation
- `RiskMonitor.start_monitoring()` - Real-time risk monitoring

**Walk-Forward Testing:**
- `WalkForwardTester.generate_segments()` - Generate train/test segments
- `WalkForwardTester.run()` - Execute walk-forward test
- `WalkForwardTester._aggregate_results()` - Aggregate metrics across segments

### 10.3 References

- Black-Scholes Model: https://en.wikipedia.org/wiki/Black–Scholes_model
- Options Greeks: https://www.investopedia.com/trading/options-greeks/
- Gamma Exposure: https://spotgamma.com/
- OI Analysis: Academic papers on options market microstructure
- Value at Risk (VaR): Jorion, P. (2006). Value at Risk: The New Benchmark for Managing Financial Risk
- Walk-Forward Analysis: Prado, M. L. (2018). Advances in Financial Machine Learning

---

**Document End**

**Next Steps:**
1. Review and approve this plan
2. Set up development branch
3. Begin Phase 1 implementation (Multi-Resolution)
4. Establish testing framework
5. Schedule code review milestones
