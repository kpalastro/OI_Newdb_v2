import copy
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, date
from threading import Lock
from typing import Any, DefaultDict, Dict, List, Optional

import numpy as np

from time_utils import now_ist
from config import get_config
from data_ingestion.order_flow_analyzer import OrderFlowAnalyzer


class ExchangeDataHandler:
    """Per-exchange mutable state container."""

    def __init__(self, exchange_name: str, config: Dict[str, Any], data_reel_length: int):
        self.exchange = exchange_name
        self.config = config
        self.lock = Lock()

        # Instrument details
        self.underlying_token: Optional[int] = None
        self.option_tokens: List[int] = []
        self.expiry_date: Optional[datetime.date] = None
        self.symbol_prefix: Optional[str] = None
        self.futures_token: Optional[int] = None
        self.futures_symbol: Optional[str] = None
        self.instrument_list: List[Dict] = []

        # Real-time data
        self.latest_tick_data: Dict[int, Dict] = {}
        self.latest_tick_metadata: Dict[int, datetime] = {}
        self.latest_future_price: Optional[float] = None
        self.latest_oi_data: Dict[str, Any] = self._initial_latest_oi_state()

        # Data reels
        self._data_reel_length = data_reel_length
        self.data_reels: DefaultDict[int, deque] = defaultdict(self._create_data_reel)
        self.futures_oi_reels: deque = self._create_data_reel()
        self.data_reel_last_minute: Dict[int, datetime] = {}
        self.pending_reel_points: Dict[int, Dict] = defaultdict(dict)
        self.oi_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=data_reel_length))

        # Caches
        self.option_price_cache: Dict[int, float] = {}
        self.option_iv_cache: Dict[int, float] = {}
        self.option_pct_change_cache: DefaultDict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self.option_abs_change_cache: DefaultDict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self.microstructure_cache: Dict[int, Dict[str, float]] = defaultdict(dict)

        # State flags
        self.market_close_processed = False
        self.historical_bootstrap_completed = False
        self.last_saved_oi_state: Dict[int, float] = {}

        # Paper trading
        self.open_positions: Dict[str, Dict] = {}
        self.position_counter: int = 0
        self.total_mtm: float = 0.0
        self.closed_positions_pnl: float = 0.0
        # Timestamp of last auto-executed trade (used for cooldown)
        self.last_auto_trade_time: Optional[datetime] = None

        # ML state
        self.ml_signal = 'HOLD'
        self.ml_confidence = 0.0
        self.ml_rationale = 'Initializing...'
        self.ml_metadata = {}
        self.last_ml_signal_time: Optional[datetime] = None
        self.atm_shift_ewma: Optional[float] = None

        # Signal quality tracking
        self.recent_signal_streak: List[str] = []
        self.macro_feature_cache: Dict[str, float] = {}
        # Use aware IST timestamps so we can safely subtract from now_ist()
        self.last_macro_refresh: datetime = now_ist()
        self.last_db_save_time: datetime = now_ist()
        
        # Order Flow Analysis
        self.order_flow_analyzer = OrderFlowAnalyzer(config)
        self.flow_cache: Dict[str, float] = {}

        # Futures OI change cache (to prevent showing 0.0% during OI release gaps)
        self.cached_fut_oi_change_3m: Optional[float] = None
        self.cached_fut_oi_release_ts: Optional[datetime] = None
        
        # Multi-resolution aggregation (Feature 1: Multi-Resolution Data Pipeline)
        multi_res_enabled = get_config().multi_resolution_enabled if hasattr(get_config(), 'multi_resolution_enabled') else True
        if multi_res_enabled:
            from data_ingestion.multi_resolution_aggregator import MultiResolutionAggregator
            resolutions = get_config().multi_resolution_resolutions if hasattr(get_config(), 'multi_resolution_resolutions') else ['1min', '5min', '15min', '1D']
            self.multi_res_aggregator = MultiResolutionAggregator(resolutions=resolutions)
        else:
            self.multi_res_aggregator = None

    def _create_data_reel(self):
        return deque(maxlen=self._data_reel_length)

    def track_order_flow(self, tick: Dict):
        """Update order flow metrics with a new tick."""
        metrics = self.order_flow_analyzer.process_tick(tick)
        self.flow_cache.update(metrics)

    def _initial_latest_oi_state(self) -> dict:
        return {
            'call_options': [], 'put_options': [], 'atm_strike': None, 'underlying_price': None,
            'last_update': None, 'status': 'Initializing...', 'pcr': None,
            'exchange': self.exchange, 'underlying_name': self.config.get('underlying_name', self.config['underlying_prefix']),
            'strike_difference': self.config['strike_difference'], 'vix': None, 'diff_thresholds': {},
            'itm_oi_ce_pct_change': 0.0, 'itm_oi_pe_pct_change': 0.0, 'underlying_future_symbol': None,
            'underlying_future_price': None, 'underlying_future_oi': None, 'percent_oichange_fut_3m': None,
            'atm_shift_intensity': 0.0, 'itm_ce_breadth': 0.0, 'itm_pe_breadth': 0.0,
            'ml_signal': 'HOLD', 'ml_confidence': 0.0, 'ml_rationale': 'Initializing...',
            'ml_metadata': {}
        }

    def calculate_atm_shift_intensity_ewma(self, window: int = 10) -> float:
        try:
            if len(self.data_reels.get(self.underlying_token, [])) < window:
                return 0.0

            reel = list(self.data_reels[self.underlying_token])[-window:]
            prices = np.array([r.get('ltp', 0) for r in reel])

            if len(prices) < 2:
                return 0.0

            strike_steps = np.round(prices / self.config['strike_difference'])
            shifts = np.abs(np.diff(strike_steps))

            alpha = 2 / (window + 1)
            if self.atm_shift_ewma is None:
                self.atm_shift_ewma = shifts[0] if len(shifts) > 0 else 0

            for shift in shifts[1:]:
                self.atm_shift_ewma = alpha * shift + (1 - alpha) * self.atm_shift_ewma

            return min(1.0, self.atm_shift_ewma / window)
        except Exception:
            return 0.0

    def create_feature_snapshot(self) -> "HandlerFeatureSnapshot":
        with self.lock:
            data_reels_copy = {
                token: list(reel)
                for token, reel in self.data_reels.items()
            }
            futures_reel_copy = list(self.futures_oi_reels)
            micro_cache_copy = copy.deepcopy(self.microstructure_cache)
            macro_cache_copy = copy.deepcopy(self.macro_feature_cache)
            
            # Include cached futures OI change metadata in snapshot cache
            if self.cached_fut_oi_change_3m is not None:
                macro_cache_copy['cached_fut_oi_change_3m'] = self.cached_fut_oi_change_3m
            if self.cached_fut_oi_release_ts is not None:
                macro_cache_copy['cached_fut_oi_release_ts'] = self.cached_fut_oi_release_ts.isoformat()

            return HandlerFeatureSnapshot(
                exchange=self.exchange,
                config=copy.deepcopy(self.config),
                underlying_token=self.underlying_token,
                expiry_date=self.expiry_date,
                latest_future_price=self.latest_future_price,
                futures_symbol=self.futures_symbol,
                futures_oi_reels=futures_reel_copy,
                macro_feature_cache=macro_cache_copy,
                data_reels=data_reels_copy,
                microstructure_cache=micro_cache_copy,
                atm_shift_ewma=self.atm_shift_ewma,
                data_reel_length=self._data_reel_length
            )


@dataclass
class HandlerFeatureSnapshot:
    exchange: str
    config: Dict[str, Any]
    underlying_token: Optional[int]
    expiry_date: Optional[date]
    latest_future_price: Optional[float]
    futures_symbol: Optional[str]
    futures_oi_reels: List[Dict[str, Any]]
    macro_feature_cache: Dict[str, float]
    data_reels: Dict[int, List[Dict[str, Any]]]
    microstructure_cache: Dict[int, Dict[str, float]]
    atm_shift_ewma: Optional[float]
    data_reel_length: int

    def calculate_atm_shift_intensity_ewma(self, window: int = 10) -> float:
        try:
            reel = self.data_reels.get(self.underlying_token, [])
            if len(reel) < window or not reel:
                return 0.0

            prices = np.array([r.get('ltp', 0) for r in reel[-window:]])
            if len(prices) < 2:
                return 0.0

            strike_steps = np.round(prices / self.config['strike_difference'])
            shifts = np.abs(np.diff(strike_steps))
            alpha = 2 / (window + 1)
            atm_shift = self.atm_shift_ewma if self.atm_shift_ewma is not None else (shifts[0] if len(shifts) else 0.0)
            for shift in shifts[1:]:
                atm_shift = alpha * shift + (1 - alpha) * atm_shift
            return min(1.0, atm_shift / window)
        except Exception:
            return 0.0

    @property
    def futures_oi_reels_deque(self) -> deque:
        return deque(self.futures_oi_reels[-self.data_reel_length:], maxlen=self.data_reel_length)


class HandlerSnapshotProxy:
    """Lightweight proxy exposing the attributes Feature Engineering expects."""

    def __init__(self, snapshot: HandlerFeatureSnapshot):
        self.exchange = snapshot.exchange
        self.config = snapshot.config
        self.underlying_token = snapshot.underlying_token
        self.expiry_date = snapshot.expiry_date
        self.latest_future_price = snapshot.latest_future_price
        self.futures_symbol = snapshot.futures_symbol
        self.macro_feature_cache = snapshot.macro_feature_cache or {}
        self.microstructure_cache = snapshot.microstructure_cache or {}
        self._atm_shift_ewma = snapshot.atm_shift_ewma
        self._data_reel_length = snapshot.data_reel_length

        self.data_reels: DefaultDict[int, deque] = defaultdict(self._create_data_reel)
        for token, records in snapshot.data_reels.items():
            reel = self._create_data_reel()
            reel.extend(records[-self._data_reel_length:])
            self.data_reels[token] = reel

        self.futures_oi_reels: deque = deque(
            snapshot.futures_oi_reels[-self._data_reel_length:],
            maxlen=self._data_reel_length
        )

    def _create_data_reel(self):
        return deque(maxlen=self._data_reel_length)

    def calculate_atm_shift_intensity_ewma(self, window: int = 10) -> float:
        proxy_snapshot = HandlerFeatureSnapshot(
            exchange=self.exchange,
            config=self.config,
            underlying_token=self.underlying_token,
            expiry_date=self.expiry_date,
            latest_future_price=self.latest_future_price,
            futures_symbol=self.futures_symbol,
            futures_oi_reels=list(self.futures_oi_reels),
            macro_feature_cache=self.macro_feature_cache,
            data_reels={token: list(reel) for token, reel in self.data_reels.items()},
            microstructure_cache=self.microstructure_cache,
            atm_shift_ewma=self._atm_shift_ewma,
            data_reel_length=self._data_reel_length
        )
        return proxy_snapshot.calculate_atm_shift_intensity_ewma(window)


