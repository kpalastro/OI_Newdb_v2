"""
Feature engineering utilities shared between the real-time app and training pipeline.

This module implements the advanced feature set outlined in claud.md, including:
- Options skew, gamma exposure proxies, order-flow ratios
- Temporal/regime metadata
- Momentum and volatility statistics from rolling reels
- Microstructure metrics derived from tick depth
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import skew

from time_utils import to_ist, now_ist
from sentiment_analyzer import SentimentAnalyzer

# Global Sentiment Analyzer instance (lazy initialization or reused)
_SENTIMENT_ANALYZER = None

def get_sentiment_analyzer():
    global _SENTIMENT_ANALYZER
    if _SENTIMENT_ANALYZER is None:
        _SENTIMENT_ANALYZER = SentimentAnalyzer()
    return _SENTIMENT_ANALYZER

MARKET_CLOSE_TIME = time(15, 30)
MIN_HISTORY_MINUTES = 30

REQUIRED_FEATURE_COLUMNS = [
    'pcr_total_oi', 'pcr_itm_oi', 'pcr_total_volume',
    'futures_premium', 'time_to_expiry_hours', 'vix',
    'underlying_price', 'underlying_future_price', 'underlying_future_oi',
    'total_itm_oi_ce', 'total_itm_oi_pe',
    'atm_shift_intensity', 'itm_ce_breadth', 'itm_pe_breadth',
    'percent_oichange_fut_3m', 'itm_oi_ce_pct_change_3m_wavg',
    'itm_oi_pe_pct_change_3m_wavg', 'put_call_iv_skew', 'otm_put_premium',
    'net_gamma_exposure', 'gamma_flip_level', 'dealer_vanna_exposure', 'dealer_charm_exposure',
    'ce_volume_to_oi_ratio', 'pe_volume_to_oi_ratio',
    'ce_oi_spike', 'pe_oi_spike',
    'avg_ce_iv', 'avg_pe_iv', 'atm_iv', 'otm_pe_avg_iv',
    'bid_ask_spread', 'order_book_imbalance',
    'ce_pct_change_3m', 'pe_pct_change_3m',
    'ce_oi_momentum_5', 'ce_oi_momentum_10', 'ce_oi_momentum_20',
    'pe_oi_momentum_5', 'pe_oi_momentum_10', 'pe_oi_momentum_20',
    'price_momentum_5', 'price_momentum_10', 'price_momentum_20', 'price_momentum_30',
    'price_zscore_5', 'price_zscore_10', 'price_zscore_20', 'price_zscore_30',
    'price_roc_5m', 'price_roc_15m', 'price_roc_30m',
    'realized_vol_5m', 'vol_risk_premium', 'pcr_x_vix',
    'futures_premium_x_time', 'breadth_divergence', 'oi_pressure_ratio',
    'hour', 'minute', 'time_to_close_hours',
    'is_opening_hour', 'is_closing_hour', 'is_lunch_hour',
    'vix_contango_pct', 'term_structure_spread',
    'depth_buy_total', 'depth_sell_total', 'depth_imbalance_ratio',
    'macro_fii_dii_net', 'macro_usdinr_trend', 'macro_crude_trend',
    'macro_sentiment_score_50', 'macro_sentiment_score_100', 'macro_trin_50', 'macro_trin_100',
    'macro_banknifty_corr', 'macro_risk_on_score',
    'order_flow_toxicity', 'bid_ask_bounce_score',
    'block_trade_count', 'block_trade_imbalance', 'sweep_order_detected', 'smart_money_flow',
    'sin_time', 'cos_time',
    'total_itm_oi_ce_ratio', 'total_itm_oi_pe_ratio',
    # Feature 3: OI Concentration & Skewness
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
    # Feature 4: Expiry-Day Features
    'max_pain_distance',
    'pin_risk_probability',
    'theta_acceleration',
    # Feature 5: Sentiment Features
    'sentiment_ad_ratio_50',
    'sentiment_ad_ratio_100',
    'sentiment_trin_50',
    'sentiment_trin_100',
    'sentiment_score',
    'sentiment_fii_net_crores',
    'sentiment_dii_net_crores',
    'sentiment_inst_net_crores',
    # Volume Acceleration Features (Momentum Exhaustion Detection)
    'volume_velocity',
    'volume_acceleration',
    # ITM OI Divergence Features (Bearish/Bullish/Winding)
    'itm_oi_bearish_divergence',
    'itm_oi_bullish_divergence',
    'eod_position_winding',
    'itm_oi_flow_direction',
    'is_eod_winding_window'
]


class FeatureEngineeringError(RuntimeError):
    """Raised when live features cannot be computed due to missing data."""


@dataclass
class OptionAggregates:
    total_ce_oi: float = 0.0
    total_pe_oi: float = 0.0
    total_ce_volume: float = 0.0
    total_pe_volume: float = 0.0
    total_itm_oi_ce: float = 0.0
    total_itm_oi_pe: float = 0.0
    avg_ce_iv: float = 0.0
    avg_pe_iv: float = 0.0
    atm_iv: float = 0.0
    otm_pe_avg_iv: float = 0.0
    ce_pct_change_3m: float = 0.0
    pe_pct_change_3m: float = 0.0
    itm_ce_breadth: float = 0.0
    itm_pe_breadth: float = 0.0
    itm_oi_ce_pct_change_3m_wavg: float = 0.0
    itm_oi_pe_pct_change_3m_wavg: float = 0.0
    bid_ask_spread: float = 0.0
    order_book_imbalance: float = 0.0
    net_gamma_exposure: float = 0.0
    gamma_flip_level: float = 0.0
    dealer_vanna_exposure: float = 0.0
    dealer_charm_exposure: float = 0.0
    ce_volume_to_oi_ratio: float = 0.0
    pe_volume_to_oi_ratio: float = 0.0
    ce_oi_spike: int = 0
    pe_oi_spike: int = 0
    put_call_iv_skew: float = 0.0
    otm_put_premium: float = 0.0


def _calculate_depth_metrics(call_options: List[Dict], put_options: List[Dict]) -> Tuple[float, float, float]:
    buy_total = 0.0
    sell_total = 0.0
    for opt in call_options + put_options:
        buy_total += float(opt.get('bid_quantity') or 0.0)
        sell_total += float(opt.get('ask_quantity') or 0.0)
    total = max(buy_total + sell_total, 1e-6)
    imbalance = (buy_total - sell_total) / total
    return buy_total, sell_total, imbalance


def engineer_live_feature_set(
    handler,
    call_options: List[Dict],
    put_options: List[Dict],
    spot_price: float,
    atm_strike: float,
    now: datetime,
    latest_vix: Optional[float],
) -> Dict[str, float]:
    """
    Build the full live feature vector using tick-level reels and option ladders.
    """
    if spot_price is None:
        raise FeatureEngineeringError("Spot price unavailable.")

    price_series = _build_price_series(handler, handler.underlying_token)
    market_series = _build_market_series(handler, handler.underlying_token)
    
    if len(price_series) < MIN_HISTORY_MINUTES:
        raise FeatureEngineeringError(
            f"Require {MIN_HISTORY_MINUTES} minutes of price history, have {len(price_series)}."
        )

    tte_hours = _calc_time_to_expiry(handler.expiry_date, now)
    # Get risk_free_rate from config if available
    risk_free_rate = getattr(handler.config, 'risk_free_rate', 0.10) if hasattr(handler, 'config') else 0.10
    option_aggs = _calculate_option_aggregates(
        call_options, put_options, atm_strike, 
        time_to_expiry_days=tte_hours/24.0,
        spot_price=spot_price,
        risk_free_rate=risk_free_rate
    )
    micro_features = _calculate_microstructure_features(handler)
    temporal_features = _calculate_temporal_features(handler, now)
    momentum_features = _calculate_price_momentum(price_series)
    vpin_feature = _calculate_vpin_proxy(market_series)
    bounce_feature = _calculate_bid_ask_bounce(call_options, put_options)
    
    # Get Order Flow Metrics (cached in handler)
    flow_metrics = getattr(handler, 'flow_cache', {})
    
    # Feature 3: OI Concentration & Skewness
    oi_concentration = calculate_oi_concentration(call_options, put_options, top_n=3)
    oi_velocity = calculate_oi_velocity(handler, call_options, put_options, window_minutes=5)

    # Feature 4: Expiry Features
    max_pain = calculate_max_pain(call_options, put_options)
    max_pain_dist = spot_price - max_pain if max_pain > 0 else 0.0
    pin_risk = calculate_pin_risk(call_options, put_options, atm_strike)
    
    # Theta acceleration (Placeholder - requires history of theta, here we assume flat or calc from greeks if possible)
    # For now, we will return 0.0 or derived from option_aggs if we had it
    theta_accel = 0.0 

    # Feature 5: Sentiment Analysis
    try:
        analyzer = get_sentiment_analyzer()
        sentiment_metrics = analyzer.get_sentiment_metrics()
    except Exception as e:
        # Fallback if sentiment analysis fails
        sentiment_metrics = {
            'sentiment_ad_ratio_50': 1.0, 'sentiment_ad_ratio_100': 1.0,
            'sentiment_trin_50': 1.0, 'sentiment_trin_100': 1.0,
            'sentiment_score': 50.0,
            'sentiment_fii_net_crores': 0.0, 'sentiment_dii_net_crores': 0.0,
            'sentiment_inst_net_crores': 0.0
        }

    features: Dict[str, float] = {
        # Base PCR-style features
        'pcr_total_oi': _safe_div(option_aggs.total_pe_oi, option_aggs.total_ce_oi),
        'pcr_itm_oi': _safe_div(option_aggs.total_itm_oi_pe, option_aggs.total_itm_oi_ce, min_denominator=0.01),
        'pcr_total_volume': _safe_div(option_aggs.total_pe_volume, option_aggs.total_ce_volume),
        'total_itm_oi_ce': option_aggs.total_itm_oi_ce,
        'total_itm_oi_pe': option_aggs.total_itm_oi_pe,
        'itm_ce_breadth': option_aggs.itm_ce_breadth,
        'itm_pe_breadth': option_aggs.itm_pe_breadth,
        'itm_oi_ce_pct_change_3m_wavg': option_aggs.itm_oi_ce_pct_change_3m_wavg,
        'itm_oi_pe_pct_change_3m_wavg': option_aggs.itm_oi_pe_pct_change_3m_wavg,
        'ce_pct_change_3m': option_aggs.ce_pct_change_3m,
        'pe_pct_change_3m': option_aggs.pe_pct_change_3m,
        'bid_ask_spread': option_aggs.bid_ask_spread or micro_features.get('bid_ask_spread', 0.0),
        'order_book_imbalance': option_aggs.order_book_imbalance or micro_features.get('order_book_imbalance', 0.0),
        'put_call_iv_skew': option_aggs.put_call_iv_skew,
        'otm_put_premium': option_aggs.otm_put_premium,
        'net_gamma_exposure': option_aggs.net_gamma_exposure,
        'gamma_flip_level': option_aggs.gamma_flip_level,
        'dealer_vanna_exposure': option_aggs.dealer_vanna_exposure,
        'dealer_charm_exposure': option_aggs.dealer_charm_exposure,
        'ce_volume_to_oi_ratio': option_aggs.ce_volume_to_oi_ratio,
        'pe_volume_to_oi_ratio': option_aggs.pe_volume_to_oi_ratio,
        'ce_oi_spike': option_aggs.ce_oi_spike,
        'pe_oi_spike': option_aggs.pe_oi_spike,
        'avg_ce_iv': option_aggs.avg_ce_iv,
        'avg_pe_iv': option_aggs.avg_pe_iv,
        'atm_iv': option_aggs.atm_iv,
        'otm_pe_avg_iv': option_aggs.otm_pe_avg_iv,
    }

    # Enhanced features
    features['order_flow_toxicity'] = vpin_feature
    features['bid_ask_bounce_score'] = bounce_feature
    features['block_trade_count'] = flow_metrics.get('block_trade_count', 0.0)
    features['block_trade_imbalance'] = flow_metrics.get('block_trade_imbalance', 0.0)
    features['sweep_order_detected'] = flow_metrics.get('sweep_order_detected', 0.0)
    features['smart_money_flow'] = flow_metrics.get('smart_money_flow', 0.0)
    features['total_itm_oi_ce_ratio'] = _safe_div(features['total_itm_oi_ce'], option_aggs.total_ce_oi)
    features['total_itm_oi_pe_ratio'] = _safe_div(features['total_itm_oi_pe'], option_aggs.total_pe_oi)
    
    # Feature 3: OI Concentration & Skewness
    features['oi_concentration_ratio'] = oi_concentration.get('oi_concentration_ratio', 0.0)
    features['oi_concentration_top3_ce'] = 0.0  # Placeholder - can be enhanced
    features['oi_concentration_top3_pe'] = 0.0  # Placeholder - can be enhanced
    features['oi_skewness'] = (oi_concentration.get('ce_skewness', 0.0) + oi_concentration.get('pe_skewness', 0.0)) / 2.0
    features['oi_skewness_ce'] = oi_concentration.get('ce_skewness', 0.0)
    features['oi_skewness_pe'] = oi_concentration.get('pe_skewness', 0.0)
    features['oi_velocity_ce'] = oi_velocity.get('ce_oi_velocity', 0.0)
    features['oi_velocity_pe'] = oi_velocity.get('pe_oi_velocity', 0.0)
    features['oi_velocity_total'] = oi_velocity.get('total_oi_velocity', 0.0)
    features['oi_velocity_ce_per_minute'] = oi_velocity.get('ce_oi_velocity', 0.0)  # Same as oi_velocity_ce
    features['oi_velocity_pe_per_minute'] = oi_velocity.get('pe_oi_velocity', 0.0)  # Same as oi_velocity_pe

    # Feature 4: Expiry Features
    features['max_pain_distance'] = max_pain_dist
    features['pin_risk_probability'] = pin_risk
    features['theta_acceleration'] = theta_accel

    # Feature 5: Sentiment Features
    features.update(sentiment_metrics)

    # Market structure derived from handler
    futures_price = handler.latest_future_price or spot_price
    features['futures_premium'] = futures_price - spot_price
    features['time_to_expiry_hours'] = tte_hours
    features['vix'] = float(latest_vix) if latest_vix is not None else 0.0
    features['atm_shift_intensity'] = handler.calculate_atm_shift_intensity_ewma()

    # Derived interactions
    features['vol_risk_premium'] = features['vix'] - momentum_features['realized_vol_5m']
    features['pcr_x_vix'] = features['pcr_total_oi'] * features['vix']
    features['futures_premium_x_time'] = features['futures_premium'] * features['time_to_expiry_hours']
    features['breadth_divergence'] = features['itm_ce_breadth'] - features['itm_pe_breadth']
    features['oi_pressure_ratio'] = _safe_div(
        features['itm_oi_pe_pct_change_3m_wavg'],
        features['itm_oi_ce_pct_change_3m_wavg'],
        fallback=0.0,
        min_denominator=0.01,
    )
    features['percent_oichange_fut_3m'] = _future_oi_change(handler, 3)
    
    # Fallback to cache if 0.0 (display continuity for gaps in OI updates)
    if features['percent_oichange_fut_3m'] == 0.0:
        macro_cache = getattr(handler, 'macro_feature_cache', {}) or {}
        cached_fut = macro_cache.get('cached_fut_oi_change_3m')
        if cached_fut is not None and cached_fut != 0.0:
            features['percent_oichange_fut_3m'] = float(cached_fut)

    for window in [5, 10, 20]:
        features[f'ce_oi_momentum_{window}'] = _average_pct_change(call_options, window)
        features[f'pe_oi_momentum_{window}'] = _average_pct_change(put_options, window)

    depth_buy_total, depth_sell_total, depth_imbalance = _calculate_depth_metrics(call_options, put_options)
    features['depth_buy_total'] = depth_buy_total
    features['depth_sell_total'] = depth_sell_total
    features['depth_imbalance_ratio'] = depth_imbalance

    # Macro features with proper null handling (optional, nice-to-have)
    macro_cache = getattr(handler, 'macro_feature_cache', {}) or {}
    
    # Safely extract macro features with null handling
    # These are optional and should not break model performance if unavailable
    def safe_get_macro(key: str, default: float = 0.0) -> float:
        value = macro_cache.get(key)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    
    features['macro_fii_dii_net'] = safe_get_macro('fii_dii_net', 0.0)
    features['macro_usdinr_trend'] = safe_get_macro('usdinr_trend', 0.0)
    features['macro_crude_trend'] = safe_get_macro('crude_trend', 0.0)
    features['macro_banknifty_corr'] = safe_get_macro('banknifty_correlation', 0.0)
    features['macro_risk_on_score'] = safe_get_macro('risk_on_score', 0.0)
    # NIFTY sentiment features (market breadth indicators)
    features['macro_sentiment_score_50'] = safe_get_macro('sentiment_score_50', 50.0)  # Default to neutral (50)
    features['macro_sentiment_score_100'] = safe_get_macro('sentiment_score_100', 50.0)  # Default to neutral (50)
    features['macro_trin_50'] = safe_get_macro('trin_50', 1.0)  # Default to neutral (1.0)
    features['macro_trin_100'] = safe_get_macro('trin_100', 1.0)  # Default to neutral (1.0)
    features['vix_contango_pct'] = safe_get_macro('vix_contango_pct', 0.0)
    features['term_structure_spread'] = safe_get_macro('term_structure_spread', 0.0)

    # Temporal
    features.update(temporal_features)

    # Microstructure (averages)
    features.update(micro_features)

    # Momentum & z-scores
    features.update(momentum_features)

    # Volume Acceleration (Momentum Exhaustion Detection)
    volume_accel = _calculate_volume_acceleration(market_series, window=5)
    features['volume_velocity'] = volume_accel['volume_velocity']
    features['volume_acceleration'] = volume_accel['volume_acceleration']

    # --- ITM OI Divergence Features ---
    # 1. Bearish Divergence: ITM CALL OI increasing + ITM PUT OI decreasing
    itm_ce_change = features.get('itm_oi_ce_pct_change_3m_wavg', 0.0)
    itm_pe_change = features.get('itm_oi_pe_pct_change_3m_wavg', 0.0)

    # Bearish when CE rising (> 0) and PE falling (< 0)
    features['itm_oi_bearish_divergence'] = max(0.0, itm_ce_change) * max(0.0, -itm_pe_change)

    # 2. Bullish Divergence: ITM PUT OI increasing + ITM CALL OI decreasing
    features['itm_oi_bullish_divergence'] = max(0.0, itm_pe_change) * max(0.0, -itm_ce_change)

    # 3. EOD Winding: Both decreasing after 14:30
    is_eod_window = 1.0 if (features['hour'] == 14 and features['minute'] >= 30) or features['hour'] >= 15 else 0.0
    features['is_eod_winding_window'] = is_eod_window
    
    both_decreasing = 1.0 if itm_ce_change < 0 and itm_pe_change < 0 else 0.0
    # Strength is sum of abs changes
    features['eod_position_winding'] = is_eod_window * both_decreasing * (abs(itm_ce_change) + abs(itm_pe_change))

    # 4. Net Flow Direction: +ve = bear pressure (CE adds / PE leaves), -ve = bull pressure
    features['itm_oi_flow_direction'] = itm_ce_change - itm_pe_change

    # Final sanitation + ensure no NaNs
    sanitized = {k: _sanitize_float(v) for k, v in features.items()}
    return sanitized


def prepare_training_features(raw_features: pd.DataFrame, required_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Normalize historical ML feature frames loaded from the database.
    Ensures all advanced feature columns exist and fills missing values.
    Also computes stationary transformations (e.g., Z-scores) for non-stationary features.
    """
    if raw_features is None or raw_features.empty:
        return pd.DataFrame()

    df = raw_features.copy()
    if 'timestamp' not in df.columns:
        raise FeatureEngineeringError("Expected 'timestamp' column in historical feature set.")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)
    
    # Remove duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]

    required = required_columns or REQUIRED_FEATURE_COLUMNS
    # Ensure unique
    required = list(dict.fromkeys(required))
    
    for column in required:
        if column not in df.columns:
            df[column] = 0.0

    numeric_cols = [col for col in required if col in df.columns]
    # Ensure numeric_cols is unique
    numeric_cols = list(dict.fromkeys(numeric_cols))
    
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)

    # Convert non-stationary features to stationary (Z-scores)
    # PCR can drift over time, so use Z-score over rolling window (5 days = ~750 minutes in trading hours)
    if 'pcr_total_oi' in df.columns:
        window_minutes = 750  # Approximately 5 trading days
        rolling_mean = df['pcr_total_oi'].rolling(window=window_minutes, min_periods=50).mean()
        rolling_std = df['pcr_total_oi'].rolling(window=window_minutes, min_periods=50).std()
        
        # Calculate Z-score: (value - rolling_mean) / rolling_std
        # Use fillna to handle initial periods with insufficient data
        df['pcr_total_oi_zscore'] = ((df['pcr_total_oi'] - rolling_mean) / rolling_std).fillna(0.0)
        
        # Replace infinite values with 0 (can occur if std is 0)
        df['pcr_total_oi_zscore'] = df['pcr_total_oi_zscore'].replace([np.inf, -np.inf], 0.0)
        
        # Add to required columns if not already present
        if 'pcr_total_oi_zscore' not in required:
            required = list(required) + ['pcr_total_oi_zscore']

    # Rebuild temporal metadata to avoid stale payload values
    index = df.index
    df['hour'] = index.hour
    df['minute'] = index.minute
    df['time_to_close_hours'] = index.map(_time_to_close_from_timestamp)
    df['is_opening_hour'] = (df['hour'] == 9).astype(int)
    df['is_closing_hour'] = (df['hour'] >= 15).astype(int)
    df['is_lunch_hour'] = ((df['hour'] >= 12) & (df['hour'] < 14)).astype(int)
    
    # Cyclical time encoding
    minutes_since_midnight = df['hour'] * 60 + df['minute']
    df['sin_time'] = np.sin(2 * np.pi * minutes_since_midnight / 1440)
    df['cos_time'] = np.cos(2 * np.pi * minutes_since_midnight / 1440)

    return df


def _build_price_series(handler, token: Optional[int]) -> pd.Series:
    """Convert handler reels into a minute-level pandas Series."""
    reel = []
    if token and token in handler.data_reels:
        reel = list(handler.data_reels[token])

    if not reel:
        return pd.Series(dtype=float)

    data: Dict[datetime, float] = {}
    for entry in reel:
        ts = entry.get('timestamp')
        ltp = entry.get('ltp')
        if ts is None or ltp is None:
            continue
        if isinstance(ts, datetime):
            ts = to_ist(ts)
        data[ts] = float(ltp)

    if not data:
        return pd.Series(dtype=float)

    series = pd.Series(data).sort_index()
    series = series.resample('1min').last().ffill()
    return series


def _build_market_series(handler, token: Optional[int]) -> pd.DataFrame:
    """Convert handler reels into a minute-level pandas DataFrame with price and volume."""
    reel = []
    if token and token in handler.data_reels:
        reel = list(handler.data_reels[token])

    if not reel:
        return pd.DataFrame()

    data: List[Dict[str, object]] = []
    for entry in reel:
        ts = entry.get('timestamp')
        ltp = entry.get('ltp')
        if ts is None or ltp is None:
            continue
        if isinstance(ts, datetime):
            ts = to_ist(ts)
        row = {
            'timestamp': ts,
            'ltp': float(ltp),
            'volume': float(entry.get('volume') or 0.0),
        }
        data.append(row)
    
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    # Resample to ensure 1-minute bars
    df_resampled = df.resample('1min').agg({
        'ltp': 'last',
        'volume': 'sum'
    }).ffill()
    
    return df_resampled


def _calculate_vpin_proxy(market_data: pd.DataFrame, window: int = 30) -> float:
    """
    Calculate VPIN-style order flow toxicity proxy.
    Approximates buy/sell volume from 1-minute price changes.
    """
    if market_data.empty or len(market_data) < window:
        return 0.0
    
    df = market_data.iloc[-window:].copy()
    df['price_change'] = df['ltp'].diff().fillna(0.0)
    
    # Classify volume
    # If price rose, assume buy volume. If fell, sell volume. If unchanged, split 50/50
    conditions = [
        df['price_change'] > 0,
        df['price_change'] < 0
    ]
    choices_buy = [df['volume'], 0.0]
    choices_sell = [0.0, df['volume']]
    
    df['buy_vol'] = np.select(conditions, choices_buy, default=df['volume'] * 0.5)
    df['sell_vol'] = np.select(conditions, choices_sell, default=df['volume'] * 0.5)
    
    # Calculate VPIN-like metric: sum(|buy - sell|) / sum(total_volume)
    total_volume = df['volume'].sum()
    if total_volume == 0:
        return 0.0
        
    vol_imbalance = np.abs(df['buy_vol'] - df['sell_vol']).sum()
    return vol_imbalance / total_volume


def _calculate_volume_acceleration(market_data: pd.DataFrame, window: int = 5) -> Dict[str, float]:
    """
    Calculate volume velocity and acceleration for momentum exhaustion detection.
    
    Volume Velocity: First derivative - rate of volume change
    Volume Acceleration: Second derivative - rate of velocity change
    
    Trading Signal Interpretation:
        - Rising volume + Positive acceleration = Strong momentum building (trend continuation)
        - Rising volume + Negative acceleration = Momentum slowing (potential reversal)
        - Falling volume + Positive acceleration = Selling exhaustion (potential bottom)
        - Falling volume + Negative acceleration = Continued weakness (avoid entry)
    
    Args:
        market_data: DataFrame with 'volume' column (minute-level)
        window: Rolling window for averaging derivatives
        
    Returns:
        Dict with 'volume_velocity' and 'volume_acceleration'
    """
    result = {
        'volume_velocity': 0.0,
        'volume_acceleration': 0.0
    }
    
    if market_data.empty or 'volume' not in market_data.columns:
        return result
        
    if len(market_data) < window * 2:
        return result
    
    try:
        vol = market_data['volume'].values
        
        # First derivative (velocity) - rate of volume change per minute
        # Positive = volume increasing, Negative = volume decreasing
        velocity = np.diff(vol)
        
        if len(velocity) >= window:
            # Use recent window for average velocity
            avg_velocity = np.mean(velocity[-window:])
            result['volume_velocity'] = float(avg_velocity)
            
            # Second derivative (acceleration) - rate of velocity change
            # Positive = velocity increasing (momentum building)
            # Negative = velocity decreasing (momentum fading)
            acceleration = np.diff(velocity)
            
            if len(acceleration) >= window:
                avg_acceleration = np.mean(acceleration[-window:])
                result['volume_acceleration'] = float(avg_acceleration)
                
    except Exception:
        # Return zeros on any error
        pass
        
    return result


def _calculate_bid_ask_bounce(call_options: List[Dict], put_options: List[Dict]) -> float:
    """
    Estimate bid-ask bounce intensity.
    Positive score = hitting ask (aggressive buying).
    Negative score = hitting bid (aggressive selling).
    """
    bounce_score_sum = 0.0
    count = 0
    
    for opt in call_options + put_options:
        ltp = opt.get('ltp')
        best_bid = opt.get('best_bid')
        best_ask = opt.get('best_ask')
        
        if ltp is None or best_bid is None or best_ask is None:
            continue
            
        # Check if LTP is at Bid or Ask
        # Use a small epsilon for float comparison
        epsilon = 0.05
        
        if abs(ltp - best_ask) < epsilon:
            bounce_score_sum += 1.0
            count += 1
        elif abs(ltp - best_bid) < epsilon:
            bounce_score_sum -= 1.0
            count += 1
            
    if count == 0:
        return 0.0
        
    # Normalize to [-1, 1]
    return bounce_score_sum / count


def _calculate_option_aggregates(
    call_options: List[Dict],
    put_options: List[Dict],
    atm_strike: float,
    time_to_expiry_days: float = 1.0,
    spot_price: Optional[float] = None,
    risk_free_rate: float = 0.10
) -> OptionAggregates:
    agg = OptionAggregates(gamma_flip_level=float(atm_strike or 0.0))

    def _collect_stats(options: List[Dict], opt_type: str) -> Dict[str, List[float]]:
        raw = {
            'oi': [],
            'volume': [],
            'iv': [],
            'pct_change_3m': [],
            'itm_flags': [],
            'itm_pct_change_3m': [],
            'oi_for_weight': [],
            'bid_ask_spread': [],
            'imbalance': [],
            'gamma_proxy': [],
            'vanna_proxy': [],
            'charm_proxy': [],
        }
        for opt in options:
            latest_oi = float(opt.get('latest_oi') or 0.0)
            raw['oi'].append(latest_oi)
            raw['volume'].append(float(opt.get('volume') or 0.0))
            if opt.get('iv') is not None:
                raw['iv'].append(float(opt['iv']))
            pct = opt.get('pct_changes', {}).get('3m')
            if pct is not None:
                raw['pct_change_3m'].append(float(pct))
            is_itm = _is_itm(opt, opt_type)
            raw['itm_flags'].append(1 if is_itm else 0)
            if is_itm and pct is not None:
                raw['itm_pct_change_3m'].append(float(pct))
                raw['oi_for_weight'].append(latest_oi)
            spread = opt.get('spread')
            if spread is not None:
                raw['bid_ask_spread'].append(float(spread))
            imbalance = opt.get('order_book_imbalance')
            if imbalance is not None:
                raw['imbalance'].append(float(imbalance))
            
            # Advanced Greeks Proxies
            # Position is distance from ATM (0 = ATM, 1 = 1 strike away, etc.)
            pos = abs(opt.get('position') or 0)
            
            # Gamma: Highest ATM, decays rapidly. 
            # 1.0 / (pos^2 + 1) gives 1.0 at ATM, 0.5 at 1st strike, 0.2 at 2nd.
            gamma_weight = 1.0 / (float(pos)**2 + 1.0)
            gamma_proxy = latest_oi * gamma_weight
            
            # Vanna: Sensitivity to Vol. Higher OTM.
            # Grows with distance from ATM until wings.
            vanna_weight = float(pos) / (float(pos) + 5.0)
            vanna_proxy = latest_oi * vanna_weight
            
            # Charm: Delta decay. Highest ATM, magnified by nearness to expiry.
            # 1 / sqrt(T) logic usually.
            tte_factor = 1.0 / max(time_to_expiry_days, 0.1)
            charm_weight = gamma_weight * tte_factor
            charm_proxy = latest_oi * charm_weight
            
            raw['gamma_proxy'].append(gamma_proxy)
            raw['vanna_proxy'].append(vanna_proxy)
            raw['charm_proxy'].append(charm_proxy)
            
        return raw

    call_stats = _collect_stats(call_options, 'ce')
    put_stats = _collect_stats(put_options, 'pe')

    agg.total_ce_oi = sum(call_stats['oi'])
    agg.total_pe_oi = sum(put_stats['oi'])
    agg.total_ce_volume = sum(call_stats['volume'])
    agg.total_pe_volume = sum(put_stats['volume'])
    agg.total_itm_oi_ce = sum(
        oi for oi, flag in zip(call_stats['oi'], call_stats['itm_flags']) if flag
    )
    agg.total_itm_oi_pe = sum(
        oi for oi, flag in zip(put_stats['oi'], put_stats['itm_flags']) if flag
    )
    agg.avg_ce_iv = _safe_mean(call_stats['iv'])
    agg.avg_pe_iv = _safe_mean(put_stats['iv'])
    agg.atm_iv = _safe_mean(
        [opt.get('iv') for opt in call_options + put_options if opt.get('position') == 0]
    )
    agg.otm_pe_avg_iv = _safe_mean(
        [opt.get('iv') for opt in put_options if (opt.get('position') or 0) < 0]
    )
    agg.put_call_iv_skew = (agg.avg_pe_iv or 0.0) - (agg.avg_ce_iv or 0.0)
    agg.otm_put_premium = (agg.otm_pe_avg_iv or 0.0) - (agg.atm_iv or 0.0)
    agg.ce_pct_change_3m = _safe_mean(call_stats['pct_change_3m'])
    agg.pe_pct_change_3m = _safe_mean(put_stats['pct_change_3m'])
    agg.itm_ce_breadth = _safe_ratio(sum(call_stats['itm_flags']), len(call_options))
    agg.itm_pe_breadth = _safe_ratio(sum(put_stats['itm_flags']), len(put_options))
    agg.itm_oi_ce_pct_change_3m_wavg = _weighted_change(call_stats)
    agg.itm_oi_pe_pct_change_3m_wavg = _weighted_change(put_stats)
    agg.bid_ask_spread = _safe_mean(call_stats['bid_ask_spread'] + put_stats['bid_ask_spread'])
    agg.order_book_imbalance = _safe_mean(call_stats['imbalance'] + put_stats['imbalance'])
    
    # Feature 2: Enhanced Greeks - Use real Black-Scholes calculations instead of proxies
    if spot_price and spot_price > 0:
        time_to_expiry_years = time_to_expiry_days / 365.25
        
        # Calculate comprehensive Greeks exposure using real Black-Scholes
        from utils.greeks_calculator import calculate_gamma_exposure
        
        try:
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
        except Exception as e:
            # Fallback to proxy calculations if real Greeks calculation fails
            import logging
            logging.debug(f"Enhanced Greeks calculation failed, using proxies: {e}")
            agg.net_gamma_exposure = sum(call_stats['gamma_proxy']) - sum(put_stats['gamma_proxy'])
            agg.dealer_vanna_exposure = sum(call_stats['vanna_proxy']) - sum(put_stats['vanna_proxy'])
            agg.dealer_charm_exposure = sum(call_stats['charm_proxy']) - sum(put_stats['charm_proxy'])
    else:
        # Fallback to proxy calculations if spot_price unavailable
        agg.net_gamma_exposure = sum(call_stats['gamma_proxy']) - sum(put_stats['gamma_proxy'])
        agg.dealer_vanna_exposure = sum(call_stats['vanna_proxy']) - sum(put_stats['vanna_proxy'])
        agg.dealer_charm_exposure = sum(call_stats['charm_proxy']) - sum(put_stats['charm_proxy'])
    
    agg.ce_volume_to_oi_ratio = _safe_div(agg.total_ce_volume, agg.total_ce_oi)
    agg.pe_volume_to_oi_ratio = _safe_div(agg.total_pe_volume, agg.total_pe_oi)
    agg.ce_oi_spike = _detect_spike(call_stats['pct_change_3m'])
    agg.pe_oi_spike = _detect_spike(put_stats['pct_change_3m'])
    return agg


def _calculate_temporal_features(handler, now: datetime) -> Dict[str, float]:
    """Temporal features based on IST wall-clock time."""
    now_ist = to_ist(now)

    market_close_dt = to_ist(datetime.combine(now_ist.date(), MARKET_CLOSE_TIME))
    time_to_close = max((market_close_dt - now_ist).total_seconds() / 3600.0, 0.0)
    
    # Cyclical time encoding (minutes since midnight)
    minutes_since_midnight = now_ist.hour * 60 + now_ist.minute
    sin_time = np.sin(2 * np.pi * minutes_since_midnight / 1440)
    cos_time = np.cos(2 * np.pi * minutes_since_midnight / 1440)
    
    return {
        'hour': now_ist.hour,
        'minute': now_ist.minute,
        'time_to_close_hours': time_to_close,
        'is_opening_hour': 1 if now_ist.hour == 9 else 0,
        'is_closing_hour': 1 if now_ist.hour >= 15 else 0,
        'is_lunch_hour': 1 if 12 <= now_ist.hour < 14 else 0,
        'sin_time': sin_time,
        'cos_time': cos_time,
    }


def _time_to_close_from_timestamp(ts: pd.Timestamp) -> float:
    if pd.isna(ts):
        return 0.0
    dt_value = ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts
    dt_ist = to_ist(dt_value)
    market_close_dt = to_ist(datetime.combine(dt_ist.date(), MARKET_CLOSE_TIME))
    return max((market_close_dt - dt_ist).total_seconds() / 3600.0, 0.0)


def _calculate_microstructure_features(handler) -> Dict[str, float]:
    spreads = []
    imbalances = []
    for stats in handler.microstructure_cache.values():
        spread = stats.get('spread')
        imb = stats.get('imbalance')
        if spread is not None:
            spreads.append(float(spread))
        if imb is not None:
            imbalances.append(float(imb))

    return {
        'bid_ask_spread': _safe_mean(spreads),
        'order_book_imbalance': _safe_mean(imbalances),
    }


def _calculate_price_momentum(price_series: pd.Series) -> Dict[str, float]:
    result: Dict[str, float] = {}
    closes = price_series.dropna()
    if len(closes) < MIN_HISTORY_MINUTES:
        raise FeatureEngineeringError("Insufficient price history for momentum features.")

    windows = [5, 10, 20, 30]
    for window in windows:
        if len(closes) > window:
            earlier = closes.shift(window).iloc[-1]
            result[f'price_momentum_{window}'] = closes.iloc[-1] / earlier - 1 if earlier else 0.0
            rolling_mean = closes.rolling(window).mean().iloc[-1]
            rolling_std = closes.rolling(window).std().iloc[-1]
            if rolling_std and not np.isnan(rolling_std):
                result[f'price_zscore_{window}'] = (closes.iloc[-1] - rolling_mean) / rolling_std
            else:
                result[f'price_zscore_{window}'] = 0.0

    returns = closes.pct_change()
    result['realized_vol_5m'] = returns.rolling(5).std().iloc[-1] * np.sqrt(252 * 375) if len(returns) >= 5 else 0.0
    result['price_roc_5m'] = _ratio(closes, 5)
    result['price_roc_15m'] = _ratio(closes, 15)
    result['price_roc_30m'] = _ratio(closes, 30)

    for window in windows:
        result.setdefault(f'price_momentum_{window}', 0.0)
        result.setdefault(f'price_zscore_{window}', 0.0)

    return result


def _ratio(series: pd.Series, window: int) -> float:
    if len(series) <= window:
        return 0.0
    earlier = series.shift(window).iloc[-1]
    if earlier is None or (isinstance(earlier, float) and np.isnan(earlier)) or earlier == 0:
        return 0.0
    return (series.iloc[-1] / earlier - 1) * 100


def _calc_time_to_expiry(expiry_date: Optional[date], now: datetime) -> float:
    if not expiry_date:
        return 0.0
    # Treat expiry at 15:30 IST on the expiry date
    expiry_dt_naive = datetime.combine(expiry_date, MARKET_CLOSE_TIME)
    expiry_dt = to_ist(expiry_dt_naive)
    now_ist = to_ist(now)
    return max((expiry_dt - now_ist).total_seconds() / 3600.0, 0.0)


def _is_itm(option: Dict, opt_type: str) -> bool:
    pos = option.get('position') or 0
    if opt_type == 'ce':
        return pos < 0  # strikes below ATM
    return pos > 0  # puts ITM when strike above ATM


def _safe_mean(values: Iterable[float]) -> float:
    filtered = [float(v) for v in values if v is not None]
    return float(np.mean(filtered)) if filtered else 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _safe_div(numerator: float, denominator: float, fallback: float = 0.0, min_denominator: float = 1.0) -> float:
    if denominator is None:
        return fallback
    denom = denominator if abs(denominator) >= min_denominator else min_denominator
    return numerator / denom if denom else fallback


def _weighted_change(stats: Dict[str, List[float]]) -> float:
    weights = stats.get('oi_for_weight', [])
    changes = stats.get('itm_pct_change_3m', [])
    if not weights or not changes:
        return 0.0
    weights = np.array(weights)
    changes = np.array(changes)
    total = weights.sum()
    if total == 0:
        return 0.0
    return float(np.dot(weights, changes) / total)


def _detect_spike(changes: List[float]) -> int:
    filtered = [float(x) for x in changes if x is not None]
    if len(filtered) < 5:
        return 0
    mean = np.mean(filtered)
    std = np.std(filtered)
    latest = filtered[-1]
    return 1 if latest > mean + 2 * std else 0


def _average_pct_change(options: List[Dict], window: int) -> float:
    key = f'{window}m'
    values = [
        float(opt.get('pct_changes', {}).get(key))
        for opt in options
        if opt.get('pct_changes', {}).get(key) is not None
    ]
    return _safe_mean(values)


def _future_oi_change(handler, minutes: int) -> float:
    """
    Calculate futures OI percent change over specified minutes.
    Uses futures_oi_reels which is specifically for tracking futures OI.
    """
    # CRITICAL FIX: Use futures_oi_reels instead of data_reels
    # futures_oi_reels is specifically maintained for futures OI tracking
    futures_reel = getattr(handler, 'futures_oi_reels', None)
    if not futures_reel or len(futures_reel) == 0:
        return 0.0
    
    # Convert deque to list for easier indexing
    reel_list = list(futures_reel)
    
    # Need at least (minutes + 1) entries to calculate change
    if len(reel_list) <= minutes:
        return 0.0
    
    # Get current and past OI values
    latest_entry = reel_list[-1]
    past_entry = reel_list[-minutes - 1]  # minutes entries before last
    
    latest_oi = latest_entry.get('oi')
    past_oi = past_entry.get('oi')
    
    # Validate data
    if latest_oi is None or past_oi is None:
        return 0.0
    if past_oi == 0 or (isinstance(past_oi, float) and np.isnan(past_oi)):
        return 0.0
    if isinstance(latest_oi, float) and np.isnan(latest_oi):
        return 0.0
    
    # Calculate percent change
    return ((latest_oi - past_oi) / past_oi) * 100


def _sanitize_float(value: Optional[float]) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return float(value)


def calculate_oi_concentration(
    call_options: List[Dict],
    put_options: List[Dict],
    top_n: int = 3
) -> Dict[str, float]:
    """
    Calculate OI concentration metrics.
    
    Returns:
        Dict with keys:
        - oi_concentration_ratio: Ratio of top N strikes OI to total OI
        - ce_skewness: Skewness of call OI distribution
        - pe_skewness: Skewness of put OI distribution
    """
    result = {
        'oi_concentration_ratio': 0.0,
        'ce_skewness': 0.0,
        'pe_skewness': 0.0
    }
    
    # Calculate total OI
    total_ce_oi = sum(opt.get('latest_oi', 0) or 0 for opt in call_options)
    total_pe_oi = sum(opt.get('latest_oi', 0) or 0 for opt in put_options)
    total_oi = total_ce_oi + total_pe_oi
    
    if total_oi == 0:
        return result
    
    # Get top N strikes by OI (combining CE and PE)
    all_strikes = {}
    for opt in call_options + put_options:
        strike = opt.get('strike')
        oi = opt.get('latest_oi', 0) or 0
        if strike is not None:
            all_strikes[strike] = all_strikes.get(strike, 0) + oi
    
    if not all_strikes:
        return result
    
    # Sort by OI and get top N
    sorted_strikes = sorted(all_strikes.items(), key=lambda x: x[1], reverse=True)
    top_n_oi = sum(oi for _, oi in sorted_strikes[:top_n])
    result['oi_concentration_ratio'] = top_n_oi / total_oi if total_oi > 0 else 0.0
    
    # Calculate skewness for CE and PE OI distributions
    ce_oi_values = [opt.get('latest_oi', 0) or 0 for opt in call_options]
    pe_oi_values = [opt.get('latest_oi', 0) or 0 for opt in put_options]
    
    if len(ce_oi_values) > 2:
        try:
            result['ce_skewness'] = float(skew(ce_oi_values))
        except:
            result['ce_skewness'] = 0.0
    
    if len(pe_oi_values) > 2:
        try:
            result['pe_skewness'] = float(skew(pe_oi_values))
        except:
            result['pe_skewness'] = 0.0
    
    return result


def calculate_max_pain(call_options: List[Dict], put_options: List[Dict]) -> float:
    """
    Calculate Max Pain strike price.
    Max Pain is the strike with the lowest total intrinsic value for all option holders.
    """
    strikes = set()
    for opt in call_options + put_options:
        if opt.get('strike'):
            strikes.add(float(opt['strike']))
            
    if not strikes:
        return 0.0
        
    sorted_strikes = sorted(list(strikes))
    min_pain = float('inf')
    max_pain_strike = 0.0
    
    for strike in sorted_strikes:
        total_pain = 0.0
        # Call Pain: Max(0, spot - strike) * OI
        # Put Pain: Max(0, strike - spot) * OI
        # Here 'spot' is simulated as if the market expires at 'strike'
        
        for opt in call_options:
            s = float(opt.get('strike', 0))
            oi = float(opt.get('latest_oi', 0) or 0)
            if s > 0:
                total_pain += max(0, strike - s) * oi
                
        for opt in put_options:
            s = float(opt.get('strike', 0))
            oi = float(opt.get('latest_oi', 0) or 0)
            if s > 0:
                total_pain += max(0, s - strike) * oi
                
        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = strike
            
    return max_pain_strike


def calculate_pin_risk(call_options: List[Dict], put_options: List[Dict], atm_strike: float) -> float:
    """
    Calculate Pin Risk Probability.
    Heuristic based on OI concentration at ATM relative to total OI.
    """
    if not atm_strike:
        return 0.0
        
    total_oi = sum(float(opt.get('latest_oi', 0) or 0) for opt in call_options + put_options)
    if total_oi == 0:
        return 0.0
        
    atm_oi = 0.0
    # Sum OI for strikes very close to ATM (e.g. within 0.1%)
    threshold = atm_strike * 0.001 
    for opt in call_options + put_options:
        strike = float(opt.get('strike', 0))
        if abs(strike - atm_strike) <= threshold:
            atm_oi += float(opt.get('latest_oi', 0) or 0)
            
    return atm_oi / total_oi


def calculate_oi_velocity(
    handler,
    call_options: List[Dict],
    put_options: List[Dict],
    window_minutes: int = 5
) -> Dict[str, float]:
    """
    Calculate OI velocity (rate of change) for calls, puts, and total.
    
    Args:
        handler: ExchangeDataHandler instance with oi_history
        call_options: List of call option dicts
        put_options: List of put option dicts
        window_minutes: Time window for velocity calculation
    
    Returns:
        Dict with keys: ce_oi_velocity, pe_oi_velocity, total_oi_velocity
    """
    result = {
        'ce_oi_velocity': 0.0,
        'pe_oi_velocity': 0.0,
        'total_oi_velocity': 0.0
    }
    
    # Get OI history from handler
    oi_history = getattr(handler, 'oi_history', None)
    if not oi_history or len(oi_history) < 2:
        return result
    
    # Calculate current total OI
    current_ce_oi = sum(opt.get('latest_oi', 0) or 0 for opt in call_options)
    current_pe_oi = sum(opt.get('latest_oi', 0) or 0 for opt in put_options)
    current_total_oi = current_ce_oi + current_pe_oi
    
    # Get historical OI values (need at least window_minutes entries)
    # Assuming oi_history stores dicts with 'ce_oi', 'pe_oi', 'total_oi', 'timestamp'
    history_list = list(oi_history)
    if len(history_list) < window_minutes:
        return result
    
    # Find entry from window_minutes ago
    # oi_history is a deque, so we can index from the end
    past_entry = history_list[-window_minutes] if len(history_list) >= window_minutes else history_list[0]
    
    past_ce_oi = past_entry.get('ce_oi', 0) or 0
    past_pe_oi = past_entry.get('pe_oi', 0) or 0
    past_total_oi = past_entry.get('total_oi', 0) or (past_ce_oi + past_pe_oi)
    
    # Calculate velocity (change per minute)
    if window_minutes > 0:
        result['ce_oi_velocity'] = (current_ce_oi - past_ce_oi) / window_minutes
        result['pe_oi_velocity'] = (current_pe_oi - past_pe_oi) / window_minutes
        result['total_oi_velocity'] = (current_total_oi - past_total_oi) / window_minutes
    
    return result


def calculate_oi_rollover_rate(
    handler,
    expiry_date: Optional[date],
    now: datetime
) -> float:
    """
    Calculate OI rollover rate (placeholder for future enhancement).
    
    This would track how much OI is rolling from near expiry to next expiry.
    """
    # Placeholder implementation
    return 0.0
