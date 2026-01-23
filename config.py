"""
Centralized configuration for OI Tracker application.

All configuration values can be overridden via environment variables.
Environment variable names follow the pattern: OI_TRACKER_<CONFIG_NAME>
"""

import logging
import os
from pathlib import Path
from typing import Dict, Tuple, List
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def _get_env_int(var_name: str, default: int) -> int:
    """Safely parse environment variable as int."""
    try:
        return int(os.getenv(var_name, default))
    except (TypeError, ValueError):
        return default


def _get_env_float(var_name: str, default: float) -> float:
    """Safely parse environment variable as float."""
    try:
        return float(os.getenv(var_name, default))
    except (TypeError, ValueError):
        return default


def _get_env_bool(var_name: str, default: bool) -> bool:
    """Safely parse environment variable as bool."""
    val = os.getenv(var_name, str(default)).lower()
    return val in ('true', '1', 'yes', 'on')


@dataclass
class ExchangeConfig:
    """Configuration for a single exchange."""
    underlying_symbol: str
    underlying_prefix: str
    underlying_name: str
    strike_difference: int
    options_count: int
    options_exchange: str
    ltp_exchange: str
    display_in_ui: bool
    is_monthly: bool


@dataclass
class AppConfig:
    """Main application configuration."""
    
    # Credentials
    user_id: str = field(default_factory=lambda: os.getenv('ZERODHA_USER_ID', ''))
    password: str = field(default_factory=lambda: os.getenv('ZERODHA_PASSWORD', ''))
    
    # Exchange Configurations
    exchange_configs: Dict[str, Dict] = field(default_factory=lambda: {
        'NSE': {
            'underlying_symbol': 'NIFTY 50',
            'underlying_prefix': 'NIFTY',
            'underlying_name': 'NIFTY',
            'strike_difference': 50,
            'options_count': 5,
            'options_exchange': 'NFO',
            'ltp_exchange': 'NSE',
            'display_in_ui': True,
            'is_monthly': False,
        },
        'BSE': {
            'underlying_symbol': 'SENSEX',
            'underlying_prefix': 'SENSEX',
            'underlying_name': 'SENSEX',
            'strike_difference': 100,
            'options_count': 5,
            'options_exchange': 'BFO',
            'ltp_exchange': 'BSE',
            'display_in_ui': True,
            'is_monthly': False,
        },
        'NSE_MONTHLY': {
            'underlying_symbol': 'NIFTY 50',
            'underlying_prefix': 'NIFTY',
            'underlying_name': 'NIFTY (Monthly)',
            'strike_difference': 50,
            'options_count': 10,
            'options_exchange': 'NFO',
            'ltp_exchange': 'NSE',
            'display_in_ui': False,
            'is_monthly': True,
        },
        'BANKNIFTY_MONTHLY': {
            'underlying_symbol': 'NIFTY BANK',
            'underlying_prefix': 'BANKNIFTY',
            'underlying_name': 'BANKNIFTY (Monthly)',
            'strike_difference': 100,
            'options_count': 10,  # 10 ITM + ATM + 10 OTM = 21 strikes total
            'options_exchange': 'NFO',
            'ltp_exchange': 'NSE',
            'display_in_ui': False,  # Not shown in GUI, only saved to DB
            'is_monthly': True,
        }
    })
    
    # Data Parameters
    historical_data_minutes: int = field(default_factory=lambda: _get_env_int('OI_TRACKER_HISTORICAL_DATA_MINUTES', 40))
    data_reel_max_length: int = field(default_factory=lambda: _get_env_int('OI_TRACKER_DATA_REEL_MAX_LENGTH', None) or _get_env_int('OI_TRACKER_HISTORICAL_DATA_MINUTES', 40))
    oi_change_intervals_min: Tuple[int, ...] = field(default_factory=lambda: (3, 5, 10, 15, 30))
    historical_request_throttle_seconds: float = field(default_factory=lambda: _get_env_float('OI_TRACKER_HISTORICAL_REQUEST_THROTTLE_SECONDS', 0.4))
    
    # Financial Parameters
    risk_free_rate: float = field(default_factory=lambda: _get_env_float('OI_TRACKER_RISK_FREE_RATE', 0.10))
    tick_price_divisor: float = field(default_factory=lambda: _get_env_float('OI_TRACKER_TICK_PRICE_DIVISOR', 1.0))
    contract_multiplier: int = field(default_factory=lambda: _get_env_int('OI_TRACKER_CONTRACT_MULTIPLIER', 50))  # NIFTY lot size
    
    # Advanced Risk Management
    advanced_risk_enabled: bool = field(default_factory=lambda: _get_env_bool('OI_TRACKER_ADVANCED_RISK_ENABLED', True))
    
    # Walk-Forward Testing
    walk_forward_enabled: bool = field(default_factory=lambda: _get_env_bool('OI_TRACKER_WALK_FORWARD_ENABLED', True))
    walk_forward_train_days: int = field(default_factory=lambda: _get_env_int('OI_TRACKER_WALK_FORWARD_TRAIN_DAYS', 60))
    walk_forward_test_days: int = field(default_factory=lambda: _get_env_int('OI_TRACKER_WALK_FORWARD_TEST_DAYS', 20))
    walk_forward_step_days: int = field(default_factory=lambda: _get_env_int('OI_TRACKER_WALK_FORWARD_STEP_DAYS', 10))
    ml_signal_cooldown_seconds: int = field(default_factory=lambda: _get_env_int('OI_TRACKER_ML_SIGNAL_COOLDOWN_SECONDS', 0))
    min_confidence_for_trade: float = field(default_factory=lambda: _get_env_float('OI_TRACKER_MIN_CONFIDENCE_FOR_TRADE', 0.60))
    auto_exec_enabled: bool = field(default_factory=lambda: _get_env_bool('OI_TRACKER_AUTO_EXEC_ENABLED', True))
    auto_exec_min_kelly_fraction: float = field(default_factory=lambda: _get_env_float('OI_TRACKER_AUTO_EXEC_MIN_KELLY_FRACTION', 0.2))
    auto_exec_max_position_size_lots: int = field(default_factory=lambda: _get_env_int('OI_TRACKER_AUTO_EXEC_MAX_POSITION_SIZE_LOTS', 4))
    auto_exec_max_net_delta: float = field(default_factory=lambda: _get_env_float('OI_TRACKER_AUTO_EXEC_MAX_NET_DELTA', 3.0))
    auto_exec_session_drawdown_stop: float = field(default_factory=lambda: _get_env_float('OI_TRACKER_AUTO_EXEC_SESSION_DRAWDOWN_STOP', 0.0))
    
    # Position Limit Controls
    auto_exec_max_open_positions: int = field(default_factory=lambda: _get_env_int('OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS', 2))
    auto_exec_max_open_positions_high_confidence: int = field(default_factory=lambda: _get_env_int('OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS_HIGH_CONF', 3))
    auto_exec_max_open_positions_bullish: int = field(default_factory=lambda: _get_env_int('OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS_BULLISH', 3))
    auto_exec_max_open_positions_bearish: int = field(default_factory=lambda: _get_env_int('OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS_BEARISH', 3))
    auto_exec_high_confidence_threshold: float = field(default_factory=lambda: _get_env_float('OI_TRACKER_AUTO_EXEC_HIGH_CONFIDENCE_THRESHOLD', 0.95))
    auto_exec_cooldown_with_positions_seconds: int = field(default_factory=lambda: _get_env_int('OI_TRACKER_AUTO_EXEC_COOLDOWN_WITH_POSITIONS', 300))
    
    # RL Execution (Phase 2)
    rl_execution_enabled: bool = field(default_factory=lambda: _get_env_bool('OI_TRACKER_RL_EXECUTION_ENABLED', True))
    rl_model_path: str = field(default_factory=lambda: os.getenv('OI_TRACKER_RL_MODEL_PATH', 'models/rl_execution_model.zip'))
    rl_algorithm: str = field(default_factory=lambda: os.getenv('OI_TRACKER_RL_ALGORITHM', 'PPO'))  # PPO, DQN, or ENSEMBLE
    rl_ppo_model_path: str = field(default_factory=lambda: os.getenv('OI_TRACKER_RL_PPO_MODEL_PATH', 'models/rl_ppo_model.zip'))
    rl_dqn_model_path: str = field(default_factory=lambda: os.getenv('OI_TRACKER_RL_DQN_MODEL_PATH', 'models/rl_dqn_model.zip'))
    rl_use_ensemble: bool = field(default_factory=lambda: _get_env_bool('OI_TRACKER_RL_USE_ENSEMBLE', True))
    
    # Swing Ensemble Configuration
    use_swing_ensemble: bool = field(default_factory=lambda: _get_env_bool('OI_TRACKER_USE_SWING_ENSEMBLE', True))
    
    # End-of-Day Position Management
    auto_exec_close_all_positions_eod: bool = field(default_factory=lambda: _get_env_bool('OI_TRACKER_AUTO_EXEC_CLOSE_ALL_POSITIONS_EOD', True))
    auto_exec_eod_exit_time_hour: int = field(default_factory=lambda: _get_env_int('OI_TRACKER_AUTO_EXEC_EOD_EXIT_TIME_HOUR', 15))
    auto_exec_eod_exit_time_minute: int = field(default_factory=lambda: _get_env_int('OI_TRACKER_AUTO_EXEC_EOD_EXIT_TIME_MINUTE', 20))
    
    # UI and Logging
    ui_refresh_interval_seconds: int = field(default_factory=lambda: _get_env_int('OI_TRACKER_UI_REFRESH_INTERVAL_SECONDS', 5))
    db_save_interval_seconds: int = field(default_factory=lambda: _get_env_int('OI_TRACKER_DB_SAVE_INTERVAL_SECONDS', 30))
    pct_change_thresholds: Dict[int, float] = field(default_factory=lambda: {
        3: _get_env_float('OI_TRACKER_PCT_CHANGE_THRESHOLD_3', 8.0),
        5: _get_env_float('OI_TRACKER_PCT_CHANGE_THRESHOLD_5', 10.0),
        10: _get_env_float('OI_TRACKER_PCT_CHANGE_THRESHOLD_10', 10.0),
        15: _get_env_float('OI_TRACKER_PCT_CHANGE_THRESHOLD_15', 15.0),
        30: _get_env_float('OI_TRACKER_PCT_CHANGE_THRESHOLD_30', 25.0),
    })
    trade_log_dir: Path = field(default_factory=lambda: Path(os.getenv('OI_TRACKER_TRADE_LOG_DIR', 'trade_logs')))
    
    # Queue and Worker Settings
    feature_job_queue_size: int = field(default_factory=lambda: _get_env_int('OI_TRACKER_FEATURE_JOB_QUEUE_SIZE', 500))
    feature_processing_throttle_seconds: float = field(default_factory=lambda: _get_env_float('OI_TRACKER_FEATURE_PROCESSING_THROTTLE_SECONDS', 1.0))
    
    # WebSocket Settings
    websocket_connect_timeout_seconds: int = field(default_factory=lambda: _get_env_int('OI_TRACKER_WEBSOCKET_CONNECT_TIMEOUT_SECONDS', 30))
    websocket_max_reconnect_attempts: int = field(default_factory=lambda: _get_env_int('OI_TRACKER_WEBSOCKET_MAX_RECONNECT_ATTEMPTS', 10))
    websocket_auto_unsubscribe: bool = field(default_factory=lambda: _get_env_bool('OI_TRACKER_WEBSOCKET_AUTO_UNSUBSCRIBE', False))
    websocket_max_tokens: int = field(default_factory=lambda: _get_env_int('OI_TRACKER_WEBSOCKET_MAX_TOKENS', 2000))
    websocket_unsubscribe_threshold: int = field(default_factory=lambda: _get_env_int('OI_TRACKER_WEBSOCKET_UNSUBSCRIBE_THRESHOLD', 1800))
    
    # Flask Settings
    flask_host: str = field(default_factory=lambda: os.getenv('FLASK_HOST', '0.0.0.0'))
    flask_port: int = field(default_factory=lambda: _get_env_int('FLASK_PORT', 5004))
    flask_secret_key: str = field(default_factory=lambda: os.getenv('FLASK_SECRET_KEY', 'your-secret-key'))
    
    # Logging
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO').upper())
    log_file: str = field(default_factory=lambda: os.getenv('OI_TRACKER_LOG_FILE', 'oi_tracker.log'))
    
    # Database Configuration
    db_type: str = field(default_factory=lambda: os.getenv('OI_TRACKER_DB_TYPE', 'postgres').lower())  # Default to postgres
    db_host: str = field(default_factory=lambda: os.getenv('OI_TRACKER_DB_HOST', 'localhost'))
    db_port: int = field(default_factory=lambda: _get_env_int('OI_TRACKER_DB_PORT', 5432))
    db_name: str = field(default_factory=lambda: os.getenv('OI_TRACKER_DB_NAME', 'oi_db_new'))
    db_user: str = field(default_factory=lambda: os.getenv('OI_TRACKER_DB_USER', 'dilip'))
    db_password: str = field(default_factory=lambda: os.getenv('OI_TRACKER_DB_PASSWORD', 'ATI@dkp@1973$'))

    # Multi-resolution aggregation (Feature 1)
    multi_resolution_enabled: bool = field(default_factory=lambda: _get_env_bool('OI_TRACKER_MULTI_RESOLUTION_ENABLED', True))
    multi_resolution_resolutions: List[str] = field(default_factory=lambda: ['1min', '5min', '15min', '1D'])

    # Development / Testing
    bypass_market_hours: bool = field(default_factory=lambda: _get_env_bool('OI_TRACKER_BYPASS_MARKET_HOURS', False))
    
    # Distributed Processing (Phase 2 Feature)
    distributed_processing: bool = field(default_factory=lambda: _get_env_bool('OI_TRACKER_DISTRIBUTED_PROCESSING', False))

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.user_id or not self.password:
            logging.warning(
                "Zerodha credentials were not provided in the environment. "
                "An interactive login will be required via the web UI."
            )
        
        # Ensure trade_log_dir exists
        self.trade_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure data_reel_max_length is set
        if self.data_reel_max_length is None:
            self.data_reel_max_length = self.historical_data_minutes
    
    @property
    def all_exchanges(self) -> list:
        """Get list of all exchange keys."""
        return list(self.exchange_configs.keys())
    
    @property
    def display_exchanges(self) -> list:
        """Get list of exchanges to display in UI."""
        return [ex for ex, cfg in self.exchange_configs.items() if cfg.get('display_in_ui', False)]
    
    def get_exchange_config(self, exchange: str) -> Dict:
        """Get configuration for a specific exchange."""
        return self.exchange_configs.get(exchange, {})


# Global config instance (can be overridden for testing)
_config: AppConfig = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def set_config(config: AppConfig):
    """Set the global configuration instance (useful for testing)."""
    global _config
    _config = config


def reset_config():
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None

