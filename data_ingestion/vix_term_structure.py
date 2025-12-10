"""
VIX term structure recorder for India VIX.

Since India VIX is an index (not futures), we track:
- Current VIX value
- VIX vs Realized Volatility (volatility term structure)
- VIX vs Historical VIX (rolling averages)
- VIX trend metrics

This provides similar insights to futures-based term structure.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import database_new as db

LOGGER = logging.getLogger(__name__)


def calculate_vix_historical_metrics(exchange: str, current_vix: Optional[float] = None) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Calculate VIX historical metrics from database.
    
    Returns:
        Tuple of (vix_ma_5d, vix_ma_20d, vix_trend_1d, vix_trend_5d)
    """
    conn = None
    try:
        # Get historical VIX data from database
        # Note: VIX is stored in ml_features table as 'vix' column
        conn = db.get_db_connection()
        
        from config import get_config
        config = get_config()
        ph = '%s' if config.db_type == 'postgres' else '?'
        
        query = f"""
            SELECT timestamp, vix FROM ml_features
            WHERE exchange = {ph} AND vix IS NOT NULL AND vix > 0
            ORDER BY timestamp DESC
            LIMIT 30
        """
        df = pd.read_sql_query(query, conn, params=(exchange,))
        
        # If we have current VIX, append it to the dataframe for calculation
        if current_vix is not None:
            new_row = pd.DataFrame({'timestamp': [datetime.now()], 'vix': [current_vix]})
            if df.empty:
                df = new_row
            else:
                df = pd.concat([df, new_row], ignore_index=True)

        if df.empty:
            return None, None, None, None
        
        # Ensure proper sorting
        if 'timestamp' in df.columns:
             df['timestamp'] = pd.to_datetime(df['timestamp'])
             df = df.sort_values('timestamp')
        
        vix_series = df['vix']
        
        # Calculate moving averages (relaxed to available data)
        vix_ma_5d = float(vix_series.tail(5).mean()) if not vix_series.empty else None
        vix_ma_20d = float(vix_series.tail(20).mean()) if not vix_series.empty else None
        
        # Calculate trends
        current = float(vix_series.iloc[-1])
        # Use whatever history is available for trend, or None
        vix_1d_ago = float(vix_series.iloc[-2]) if len(vix_series) >= 2 else None
        vix_5d_ago = float(vix_series.iloc[-6]) if len(vix_series) >= 6 else None
        
        vix_trend_1d = ((current - vix_1d_ago) / vix_1d_ago * 100) if vix_1d_ago else 0.0
        vix_trend_5d = ((current - vix_5d_ago) / vix_5d_ago * 100) if vix_5d_ago else 0.0
        
        return vix_ma_5d, vix_ma_20d, vix_trend_1d, vix_trend_5d
        
    except Exception as e:
        LOGGER.error(f"Error calculating VIX metrics: {e}")
        return None, None, None, None
    finally:
        if conn:
            db.release_db_connection(conn)


def get_realized_volatility(exchange: str) -> Optional[float]:
    """
    Get realized volatility from recent price data.
    This can be calculated from the main app's feature engineering.
    
    Returns:
        Realized volatility value or None
    """
    conn = None
    try:
        # Get recent underlying price data to calculate realized vol
        conn = db.get_db_connection()
        
        from config import get_config
        config = get_config()
        ph = '%s' if config.db_type == 'postgres' else '?'
        
        query = f"""
            SELECT timestamp, underlying_price FROM ml_features
            WHERE exchange = {ph} AND underlying_price IS NOT NULL AND underlying_price > 0
            ORDER BY timestamp DESC
            LIMIT 20
        """
        df = pd.read_sql_query(query, conn, params=(exchange,))
        
        if df.empty or len(df) < 5:
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        prices = df['underlying_price']
        
        # Calculate realized volatility (5-minute returns, annualized)
        # Assuming 375 minutes trading day
        returns = prices.pct_change().dropna()
        if len(returns) >= 5:
            realized_vol = float(returns.tail(5).std() * (252 * 375) ** 0.5 * 100)  # Annualized %
            return realized_vol
        
        return None
        
    except Exception as e:
        LOGGER.error(f"Error calculating realized volatility: {e}")
        return None
    finally:
        if conn:
            db.release_db_connection(conn)


def record_vix_term_structure(
    exchange: str,
    current_vix: float,
    realized_vol: Optional[float] = None,
    vix_ma_5d: Optional[float] = None,
    vix_ma_20d: Optional[float] = None,
    vix_trend_1d: Optional[float] = None,
    vix_trend_5d: Optional[float] = None,
    source: str | None = None,
    timestamp: datetime | None = None,
) -> None:
    """
    Record VIX term structure metrics for India VIX.
    
    Since India VIX is an index (not futures), we use:
    - Current VIX vs Realized Volatility (volatility premium/discount)
    - Current VIX vs Historical VIX (rolling averages)
    - VIX trend metrics
    
    Args:
        exchange: "NSE" or "BSE"
        current_vix: Current India VIX index value
        realized_vol: Realized volatility (for comparison)
        vix_ma_5d: 5-day moving average of VIX
        vix_ma_20d: 20-day moving average of VIX
        vix_trend_1d: VIX % change over 1 day
        vix_trend_5d: VIX % change over 5 days
        source: Data source identifier
        timestamp: When data was captured
    """
    # Calculate volatility premium (VIX - Realized Vol)
    vol_premium = None
    if realized_vol is not None:
        vol_premium = current_vix - realized_vol
    
    # Calculate VIX vs historical (similar to contango concept)
    vix_vs_ma5 = None
    if vix_ma_5d is not None:
        vix_vs_ma5 = ((current_vix - vix_ma_5d) / max(vix_ma_5d, 1.0)) * 100
    
    vix_vs_ma20 = None
    if vix_ma_20d is not None:
        vix_vs_ma20 = ((current_vix - vix_ma_20d) / max(vix_ma_20d, 1.0)) * 100
    
    # For backward compatibility, map to old structure
    # "front_month" = current VIX, "next_month" = historical average
    front_month_price = current_vix
    next_month_price = vix_ma_5d or vix_ma_20d or current_vix
    
    # Calculate "contango" as VIX vs historical (positive = VIX above average)
    contango_pct = vix_vs_ma5 or vix_vs_ma20 or 0.0
    backwardation_pct = -contango_pct if contango_pct else None
    
    db.save_vix_term_structure(
        exchange=exchange,
        front_month_price=front_month_price,
        next_month_price=next_month_price,
        contango_pct=contango_pct,
        backwardation_pct=backwardation_pct,
        timestamp=timestamp,
        source=source,
        current_vix=current_vix,
        realized_vol=realized_vol,
        vix_ma_5d=vix_ma_5d,
        vix_ma_20d=vix_ma_20d,
        vix_trend_1d=vix_trend_1d,
        vix_trend_5d=vix_trend_5d,
    )


def record_vix_simple(
    exchange: str,
    current_vix: float,
    source: str | None = None,
    timestamp: datetime | None = None,
) -> None:
    """
    Simple VIX recorder - just stores current VIX value.
    Use this if you only have the current VIX index value.
    
    Args:
        exchange: "NSE" or "BSE"
        current_vix: Current India VIX index value
        source: Data source identifier
        timestamp: When data was captured
    """
    record_vix_term_structure(
        exchange=exchange,
        current_vix=current_vix,
        source=source,
        timestamp=timestamp,
    )
