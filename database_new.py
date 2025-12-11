# Database Module for OI Tracker
# Handles PostgreSQL/TimescaleDB persistence with separate tables for raw data and ML features

import json
import logging
import math
import os
import warnings
try:
    import psycopg2
    from psycopg2.extras import DictCursor
    from psycopg2 import pool
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    raise ImportError("psycopg2 is required. Please install it: pip install psycopg2-binary")

# Suppress pandas UserWarning about non-SQLAlchemy connections
warnings.filterwarnings('ignore', message='.*pandas only supports SQLAlchemy connectable.*')

from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional

from time_utils import now_ist, to_ist
from threading import Lock
from config import get_config

import numpy as np
import pandas as pd

DB_FILE = "db_new.db"
TRAINING_EXPORT_DIR = Path("exports") / "training_batches"
REPORTS_DIR = Path("reports")
db_lock = Lock()

# Global pool for Postgres
pg_pool = None

def get_db_connection():
    """Create and return a thread-safe database connection (PostgreSQL only)."""
    config = get_config()
    
    if config.db_type != 'postgres':
        raise ValueError(f"Only PostgreSQL is supported. Current db_type: {config.db_type}. Please set OI_TRACKER_DB_TYPE=postgres")
    
    if not POSTGRES_AVAILABLE:
        raise ImportError("psycopg2 is not installed. Please run 'pip install psycopg2-binary'")
        
    global pg_pool
    if pg_pool is None:
        try:
            pg_pool = psycopg2.pool.SimpleConnectionPool(
                1, 50,
                user=config.db_user,
                password=config.db_password,
                host=config.db_host,
                port=config.db_port,
                database=config.db_name
            )
            logging.info(f"✓ Connected to PostgreSQL: {config.db_name}@{config.db_host}")
        except Exception as e:
            logging.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    conn = pg_pool.getconn()
    conn.autocommit = False  # manage transactions manually
    return conn

def release_db_connection(conn):
    """Release connection back to pool."""
    if pg_pool:
        pg_pool.putconn(conn)
    else:
        conn.close()

def _sanitize_feature_value(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (np.floating, np.integer)):
        val = float(value)
        return 0.0 if math.isnan(val) else val
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _serialize_feature_dict(feature_dict: dict | None) -> str | None:
    if not feature_dict:
        return None
    sanitized = {k: _sanitize_feature_value(v) for k, v in feature_dict.items()}
    return json.dumps(sanitized)


def _deserialize_feature_series(series: pd.Series) -> pd.DataFrame:
    if series is None or series.empty:
        return pd.DataFrame()
    payloads = []
    for item in series:
        if not item:
            payloads.append({})
            continue
        try:
            payloads.append(json.loads(item))
        except json.JSONDecodeError:
            payloads.append({})
    if not payloads:
        return pd.DataFrame()
    return pd.json_normalize(payloads)


def _ensure_directory(path: Path | str):
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _coerce_iso_timestamp(value: datetime | date | str) -> str:
    if isinstance(value, datetime):
        # Ensure all timestamps are normalised to IST and serialised without microseconds.
        dt_ist = to_ist(value)
        return dt_ist.replace(microsecond=0).strftime('%Y-%m-%d %H:%M:%S')
    if isinstance(value, date):
        dt_ist = datetime.combine(value, datetime.min.time())
        dt_ist = to_ist(dt_ist)
        return dt_ist.strftime('%Y-%m-%d %H:%M:%S')
    # assume already str
    return str(value)


def _safe_json_dumps(payload: dict | None) -> str | None:
    if payload is None:
        return None
    try:
        return json.dumps(payload)
    except (TypeError, ValueError):
        logging.warning("Failed to serialize training batch metadata", exc_info=True)
        return None


def _get_placeholder():
    return '%s' if get_config().db_type == 'postgres' else '?'

def initialize_database():
    """Initialize complete database schema with separate ML features table (PostgreSQL only)."""
    config = get_config()
    
    if config.db_type != 'postgres':
        raise ValueError(f"Only PostgreSQL is supported. Current db_type: {config.db_type}")
    
    # Use a separate connection for initialization to avoid pool state issues
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # 1. Enable Extensions
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            conn.commit()
        except Exception:
            logging.warning("TimescaleDB extension not available or error enabling it")
            conn.rollback()

        # 2. Create Tables (One by one with commit)
        tables_ddl = [
                '''
                CREATE TABLE IF NOT EXISTS option_chain_snapshots (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    exchange TEXT NOT NULL,
                    strike DOUBLE PRECISION NOT NULL,
                    option_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    oi BIGINT,
                    ltp DOUBLE PRECISION,
                    token BIGINT NOT NULL,
                    underlying_price DOUBLE PRECISION,
                    moneyness TEXT,
                    time_to_expiry_seconds INTEGER,
                    pct_change_3m DOUBLE PRECISION,
                    pct_change_5m DOUBLE PRECISION,
                    pct_change_10m DOUBLE PRECISION,
                    pct_change_15m DOUBLE PRECISION,
                    pct_change_30m DOUBLE PRECISION,
                    iv DOUBLE PRECISION,
                    volume BIGINT,
                    best_bid DOUBLE PRECISION,
                    best_ask DOUBLE PRECISION,
                    bid_quantity DOUBLE PRECISION,
                    ask_quantity DOUBLE PRECISION,
                    spread DOUBLE PRECISION,
                    order_book_imbalance DOUBLE PRECISION,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(timestamp, exchange, strike, option_type)
                )
                ''',
                '''
                CREATE TABLE IF NOT EXISTS ml_features (
                    timestamp TIMESTAMP NOT NULL,
                    exchange TEXT NOT NULL,
                    pcr_total_oi DOUBLE PRECISION,
                    pcr_itm_oi DOUBLE PRECISION,
                    pcr_total_volume DOUBLE PRECISION,
                    futures_premium DOUBLE PRECISION,
                    time_to_expiry_hours DOUBLE PRECISION,
                    vix DOUBLE PRECISION,
                    underlying_price DOUBLE PRECISION,
                    underlying_future_price DOUBLE PRECISION,
                    underlying_future_oi DOUBLE PRECISION,
                    total_itm_oi_ce DOUBLE PRECISION,
                    total_itm_oi_pe DOUBLE PRECISION,
                    atm_shift_intensity DOUBLE PRECISION,
                    itm_ce_breadth DOUBLE PRECISION,
                    itm_pe_breadth DOUBLE PRECISION,
                    percent_oichange_fut_3m DOUBLE PRECISION,
                    itm_oi_ce_pct_change_3m_wavg DOUBLE PRECISION,
                    itm_oi_pe_pct_change_3m_wavg DOUBLE PRECISION,
                    created_at TIMESTAMP DEFAULT NOW(),
                    feature_payload TEXT,
                    PRIMARY KEY (timestamp, exchange)
                )
                ''',
                '''
                CREATE TABLE IF NOT EXISTS exchange_metadata (
                    exchange TEXT PRIMARY KEY,
                    last_update_time TIMESTAMP NOT NULL,
                    last_atm_strike DOUBLE PRECISION,
                    last_underlying_price DOUBLE PRECISION,
                    last_future_price DOUBLE PRECISION,
                    last_future_oi BIGINT,
                    updated_at TIMESTAMP DEFAULT NOW()
                )
                ''',
                '''
                CREATE TABLE IF NOT EXISTS training_batches (
                    id SERIAL PRIMARY KEY,
                    exchange TEXT NOT NULL,
                    start_timestamp TIMESTAMP NOT NULL,
                    end_timestamp TIMESTAMP NOT NULL,
                    model_hash TEXT,
                    artifact_path TEXT,
                    csv_path TEXT,
                    parquet_path TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    dataset_version TEXT
                )
                ''',
                '''
                CREATE TABLE IF NOT EXISTS vix_term_structure (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    exchange TEXT NOT NULL,
                    front_month_price DOUBLE PRECISION,
                    next_month_price DOUBLE PRECISION,
                    contango_pct DOUBLE PRECISION,
                    backwardation_pct DOUBLE PRECISION,
                    current_vix DOUBLE PRECISION,
                    realized_vol DOUBLE PRECISION,
                    vix_ma_5d DOUBLE PRECISION,
                    vix_ma_20d DOUBLE PRECISION,
                    vix_trend_1d DOUBLE PRECISION,
                    vix_trend_5d DOUBLE PRECISION,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
                ''',
                '''
                CREATE TABLE IF NOT EXISTS macro_signals (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    exchange TEXT NOT NULL,
                    fii_flow DOUBLE PRECISION,
                    dii_flow DOUBLE PRECISION,
                    fii_dii_net DOUBLE PRECISION,
                    usdinr DOUBLE PRECISION,
                    usdinr_trend DOUBLE PRECISION,
                    crude_price DOUBLE PRECISION,
                    crude_trend DOUBLE PRECISION,
                    banknifty_correlation DOUBLE PRECISION,
                    macro_spread DOUBLE PRECISION,
                    risk_on_score DOUBLE PRECISION,
                    metadata TEXT,
                    sentiment_score_50 DOUBLE PRECISION,
                    sentiment_confidence_50 DOUBLE PRECISION,
                    trin_50 DOUBLE PRECISION,
                    sentiment_score_100 DOUBLE PRECISION,
                    sentiment_confidence_100 DOUBLE PRECISION,
                    trin_100 DOUBLE PRECISION,
                    sentiment_score DOUBLE PRECISION,
                    sentiment_confidence DOUBLE PRECISION,
                    created_at TIMESTAMP DEFAULT NOW()
                )
                ''',
                '''
                CREATE TABLE IF NOT EXISTS order_book_depth_snapshots (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    exchange TEXT NOT NULL,
                    depth_buy_total DOUBLE PRECISION,
                    depth_sell_total DOUBLE PRECISION,
                    depth_imbalance_ratio DOUBLE PRECISION,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
                ''',
                '''
                CREATE TABLE IF NOT EXISTS paper_trading_metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    exchange TEXT NOT NULL,
                    executed BOOLEAN NOT NULL,
                    reason TEXT,
                    signal TEXT,
                    confidence DOUBLE PRECISION,
                    quantity_lots INTEGER,
                    pnl DOUBLE PRECISION,
                    constraint_violation BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT NOW()
                )
                ''',
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
            ]

        for ddl in tables_ddl:
            try:
                cursor.execute(ddl)
                conn.commit()
            except Exception as e:
                logging.error(f"Table creation failed: {e}")
                conn.rollback()

        # 3. Hypertables (TimescaleDB)
        hypertables = [
            "SELECT create_hypertable('option_chain_snapshots', 'timestamp', if_not_exists => TRUE);",
            "SELECT create_hypertable('ml_features', 'timestamp', if_not_exists => TRUE);",
            "SELECT create_hypertable('vix_term_structure', 'timestamp', if_not_exists => TRUE);",
            "SELECT create_hypertable('macro_signals', 'timestamp', if_not_exists => TRUE);",
            "SELECT create_hypertable('order_book_depth_snapshots', 'timestamp', if_not_exists => TRUE);",
            "SELECT create_hypertable('paper_trading_metrics', 'timestamp', if_not_exists => TRUE);",
            "SELECT create_hypertable('multi_resolution_bars', 'timestamp', if_not_exists => TRUE);"
        ]
        for ht in hypertables:
            try:
                cursor.execute(ht)
                conn.commit()
            except Exception:
                conn.rollback() # Timescale might not be installed

        # 4. Indexes
        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_snapshots_ts_exchange ON option_chain_snapshots(timestamp, exchange)',
            'CREATE INDEX IF NOT EXISTS idx_ml_features_ts_exchange ON ml_features(timestamp, exchange)',
            'CREATE INDEX IF NOT EXISTS idx_training_batches_exchange ON training_batches(exchange, start_timestamp)',
            'CREATE INDEX IF NOT EXISTS idx_vix_term_structure_ts ON vix_term_structure(timestamp DESC)',
            'CREATE INDEX IF NOT EXISTS idx_macro_signals_exchange ON macro_signals(exchange, timestamp DESC)',
            'CREATE INDEX IF NOT EXISTS idx_depth_snapshots_exchange ON order_book_depth_snapshots(exchange, timestamp DESC)',
            'CREATE INDEX IF NOT EXISTS idx_paper_trading_metrics_exchange_ts ON paper_trading_metrics(exchange, timestamp DESC)',
            'CREATE INDEX IF NOT EXISTS idx_multi_res_bars_resolution_time ON multi_resolution_bars(exchange, resolution, timestamp DESC)',
            'CREATE INDEX IF NOT EXISTS idx_multi_res_bars_token_time ON multi_resolution_bars(token, timestamp DESC)'
        ]
        for idx in indexes:
            try:
                cursor.execute(idx)
                conn.commit()
            except Exception as e:
                logging.warning(f"Index creation skipped/failed: {e}")
                conn.rollback()

        # PostgreSQL only - no SQLite support

    except Exception as e:
        logging.error(f"Database initialization failed: {e}")
        conn.rollback()
        raise
    finally:
        release_db_connection(conn)
        logging.info(f"✓ Database initialized: PostgreSQL")

def migrate_database():
    """Add missing columns to existing database without rebuilding (PostgreSQL only)."""
    config = get_config()
    
    if config.db_type != 'postgres':
        raise ValueError(f"Only PostgreSQL is supported. Current db_type: {config.db_type}")
    
    with db_lock:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get existing columns (PostgreSQL only)
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'option_chain_snapshots'
            """)
            snap_cols = {row[0] for row in cursor.fetchall()}
            
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'ml_features'
            """)
            ml_cols = {row[0] for row in cursor.fetchall()}

            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'training_batches'
            """)
            training_cols = {row[0] for row in cursor.fetchall()}
            
            # Add missing columns to snapshots
            new_snap_cols = [
                    ('time_to_expiry_seconds', 'INTEGER'),
                    ('time_to_expiry', 'DROP'),  # Remove old column if exists
                    ('volume', 'BIGINT'),
                    ('iv', 'DOUBLE PRECISION'),
                    ('best_bid', 'DOUBLE PRECISION'),
                    ('best_ask', 'DOUBLE PRECISION'),
                    ('bid_quantity', 'DOUBLE PRECISION'),
                    ('ask_quantity', 'DOUBLE PRECISION'),
                    ('spread', 'DOUBLE PRECISION'),
                    ('order_book_imbalance', 'DOUBLE PRECISION')
            ]
            for col_name, col_type in new_snap_cols:
                if col_type == 'DROP':
                    # Only drop if it exists
                    if col_name in snap_cols:
                        cursor.execute(f'ALTER TABLE option_chain_snapshots DROP COLUMN {col_name}')
                        logging.info(f"Dropped column {col_name} from snapshots")
                elif col_name not in snap_cols:
                    cursor.execute(f'ALTER TABLE option_chain_snapshots ADD COLUMN {col_name} {col_type}')
                    logging.info(f"Added column {col_name} to snapshots")

            # Add missing columns to macro_signals
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'macro_signals'
            """)
            macro_cols = {row[0] for row in cursor.fetchall()}
            
            # Remove old sentiment columns (if they exist) - but keep sentiment_score and sentiment_confidence
            old_sentiment_cols = [
                'sentiment_summary', 
                'sentiment_drivers', 'news_sentiment_score', 'news_sentiment_summary'
            ]
            for col in old_sentiment_cols:
                if col in macro_cols:
                    try:
                        cursor.execute(f'ALTER TABLE macro_signals DROP COLUMN {col}')
                        logging.info(f"Removed column {col} from macro_signals")
                    except Exception as e:
                        logging.warning(f"Could not remove column {col}: {e}")
            
            # Add new NIFTY sentiment columns
            new_sentiment_cols = [
                ('sentiment_score_50', 'DOUBLE PRECISION'),
                ('sentiment_confidence_50', 'DOUBLE PRECISION'),
                ('trin_50', 'DOUBLE PRECISION'),
                ('sentiment_score_100', 'DOUBLE PRECISION'),
                ('sentiment_confidence_100', 'DOUBLE PRECISION'),
                ('trin_100', 'DOUBLE PRECISION'),
            ]
            for col_name, col_type in new_sentiment_cols:
                if col_name not in macro_cols:
                    cursor.execute(f'ALTER TABLE macro_signals ADD COLUMN {col_name} {col_type}')
                    logging.info(f"Added column {col_name} to macro_signals")
            
            # Add aggregated sentiment_score and sentiment_confidence columns (computed from _50 and _100)
            # These are convenience columns that aggregate the NIFTY50 and NIFTY100 sentiment
            aggregated_sentiment_cols = [
                ('sentiment_score', 'DOUBLE PRECISION'),
                ('sentiment_confidence', 'DOUBLE PRECISION'),
            ]
            for col_name, col_type in aggregated_sentiment_cols:
                if col_name not in macro_cols:
                    cursor.execute(f'ALTER TABLE macro_signals ADD COLUMN {col_name} {col_type}')
                    logging.info(f"Added column {col_name} to macro_signals")

            # Add missing columns to ml_features
            ml_feature_cols = [
                ('atm_shift_intensity', 'DOUBLE PRECISION'),
                ('itm_ce_breadth', 'DOUBLE PRECISION'),
                ('itm_pe_breadth', 'DOUBLE PRECISION'),
                ('percent_oichange_fut_3m', 'DOUBLE PRECISION'),
                ('itm_oi_ce_pct_change_3m_wavg', 'DOUBLE PRECISION'),
                ('itm_oi_pe_pct_change_3m_wavg', 'DOUBLE PRECISION'),
                ('futures_premium', 'DOUBLE PRECISION'),
                ('feature_payload', 'TEXT'),
                ('underlying_price', 'DOUBLE PRECISION'),
                ('underlying_future_price', 'DOUBLE PRECISION'),
                ('underlying_future_oi', 'DOUBLE PRECISION'),
                ('dealer_vanna_exposure', 'DOUBLE PRECISION'),
                ('dealer_charm_exposure', 'DOUBLE PRECISION'),
                ('net_gamma_exposure', 'DOUBLE PRECISION'),
                ('gamma_flip_level', 'DOUBLE PRECISION'),
                ('ce_volume_to_oi_ratio', 'DOUBLE PRECISION'),
                ('pe_volume_to_oi_ratio', 'DOUBLE PRECISION'),
                ('news_sentiment_score', 'DOUBLE PRECISION'),
                ('sentiment_score_50', 'DOUBLE PRECISION'),
                ('sentiment_score_100', 'DOUBLE PRECISION'),
                ('trin_50', 'DOUBLE PRECISION'),
                ('trin_100', 'DOUBLE PRECISION')
            ]
            for col, col_type in ml_feature_cols:
                if col not in ml_cols:
                    cursor.execute(f'ALTER TABLE ml_features ADD COLUMN {col} {col_type}')
                    logging.info(f"Added column {col} to ml_features")

            # Add missing columns to training_batches
            if 'dataset_version' not in training_cols:
                cursor.execute("ALTER TABLE training_batches ADD COLUMN dataset_version TEXT")
                logging.info("Added column dataset_version to training_batches")

            # Add missing columns to vix_term_structure
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'vix_term_structure'
            """)
            vix_cols = {row[0] for row in cursor.fetchall()}
            
            new_vix_cols = [
                ('current_vix', 'DOUBLE PRECISION'),
                ('realized_vol', 'DOUBLE PRECISION'),
                ('vix_ma_5d', 'DOUBLE PRECISION'),
                ('vix_ma_20d', 'DOUBLE PRECISION'),
                ('vix_trend_1d', 'DOUBLE PRECISION'),
                ('vix_trend_5d', 'DOUBLE PRECISION')
            ]
            for col_name, col_type in new_vix_cols:
                if col_name not in vix_cols:
                    cursor.execute(f'ALTER TABLE vix_term_structure ADD COLUMN {col_name} {col_type}')
                    logging.info(f"Added column {col_name} to vix_term_structure")

            # PostgreSQL only - no SQLite support
            
            conn.commit()
            release_db_connection(conn)
        except Exception as e:
            logging.error(f"Migration error: {e}")
            if 'conn' in locals():
                release_db_connection(conn)
            # Fallback to full initialization if migration fails
            # initialize_database()

# Initialize on import (create then migrate to latest schema)
initialize_database()
migrate_database()

def save_option_chain_snapshot(exchange, call_options, put_options, underlying_price=None, 
                               atm_strike=None, expiry_date=None, timestamp=None, vix_value=None,
                               underlying_future_price=None, underlying_future_oi=None,
                               ml_features_dict=None):
    """
    Save option chain data and ML features atomically.
    """
    if timestamp is None:
        timestamp = now_ist()
    
    # Ensure proper datetime format (IST)
    timestamp_iso = _coerce_iso_timestamp(timestamp)
    current_time_iso = _coerce_iso_timestamp(now_ist())
    
    with db_lock:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Calculate time to expiry
            time_to_expiry_seconds = None
            if expiry_date:
                # Treat expiry at 15:30 IST on the expiry date
                expiry_dt = datetime.combine(expiry_date, datetime.strptime('15:30:00', '%H:%M:%S').time())
                expiry_dt = to_ist(expiry_dt)
                ts_ist = to_ist(timestamp)
                time_to_expiry_seconds = int((expiry_dt - ts_ist).total_seconds())
            
            # Check for OI changes
            p = _get_placeholder()
            cursor.execute(f'''
                SELECT token, oi FROM option_chain_snapshots 
                WHERE exchange={p} AND timestamp=(
                    SELECT MAX(timestamp) FROM option_chain_snapshots WHERE exchange={p}
                )
            ''', (exchange, exchange))
            last_oi = {row[0]: row[1] for row in cursor.fetchall()}
            
            oi_changed = False
            for opt in call_options + put_options:
                token = opt.get('token')
                current_oi = opt.get('latest_oi')
                if last_oi.get(token) != current_oi:
                    oi_changed = True
                    break
            
            if not oi_changed and last_oi:
                logging.info(f"{exchange}: ⊘ OI unchanged - skipping save")
                release_db_connection(conn)
                return
            
            # Save strike-level data
            records = []
            for opt in call_options + put_options:
                if not all(k in opt for k in ['token', 'symbol', 'strike']):
                    continue
                
                record = (
                    timestamp_iso, exchange, opt['strike'], opt.get('type', 'CE' if 'CE' in opt['symbol'] else 'PE'),
                    opt['symbol'], opt.get('latest_oi'), opt.get('ltp'), opt['token'],
                    underlying_price, opt.get('moneyness', 'OTM'), time_to_expiry_seconds,
                    opt.get('pct_changes', {}).get('3m'), opt.get('pct_changes', {}).get('5m'),
                    opt.get('pct_changes', {}).get('10m'), opt.get('pct_changes', {}).get('15m'),
                    opt.get('pct_changes', {}).get('30m'), opt.get('iv'), opt.get('volume'),
                    opt.get('best_bid'), opt.get('best_ask'),
                    opt.get('bid_quantity'), opt.get('ask_quantity'),
                    opt.get('spread'), opt.get('order_book_imbalance'),
                    current_time_iso, current_time_iso
                )
                records.append(record)
            
            # Note: The INSERT statement for snapshots does not need to change
            # because we provide all the values explicitly. The DEFAULT only applies
            # if the column is omitted, which we are not doing.
            if records:
                config = get_config()
                placeholders = ', '.join([_get_placeholder()] * len(records[0]))
                
                if config.db_type == 'postgres':
                    # Postgres "INSERT ... ON CONFLICT"
                    # Assuming UNIQUE constraint on (timestamp, exchange, strike, option_type)
                    # We construct DO UPDATE SET ... to behave like REPLACE
                    
                    # Construct column list manually to ensure correct mapping
                    cols = [
                        "timestamp", "exchange", "strike", "option_type", "symbol", "oi", "ltp", "token", 
                        "underlying_price", "moneyness", "time_to_expiry_seconds", "pct_change_3m", 
                        "pct_change_5m", "pct_change_10m", "pct_change_15m", "pct_change_30m", "iv", "volume",
                        "best_bid", "best_ask", "bid_quantity", "ask_quantity", "spread", "order_book_imbalance",
                        "created_at", "updated_at"
                    ]
                    
                    # Build update clause: "oi = EXCLUDED.oi, ltp = EXCLUDED.ltp, ..."
                    update_clause = ", ".join([f"{col} = EXCLUDED.{col}" for col in cols if col != "id"])
                    
                    query = f'''
                        INSERT INTO option_chain_snapshots 
                        ({', '.join(cols)})
                        VALUES ({placeholders})
                        ON CONFLICT (timestamp, exchange, strike, option_type)
                        DO UPDATE SET {update_clause}
                    '''
                    # Need to strip 'id' from records if it was auto-generated? 
                    # records tuple doesn't have 'id'. Correct.
                    # But records tuple has 25 items. My cols list has 24.
                    # Let's count.
                    # record = (timestamp_iso, exchange, strike, type, symbol, oi, ltp, token, underlying, moneyness, tte,
                    #           3m, 5m, 10m, 15m, 30m, iv, vol, bid, ask, bidq, askq, spread, imbalance) -> 24 items.
                    # Wait, the original INSERT has 25 columns in VALUES (NULL, ...). 
                    # SQLite uses NULL for auto-increment ID.
                    # My `records` list does NOT include NULL for ID.
                    # The original code: 
                    # VALUES (NULL, ?, ?, ...)
                    # records elements correspond to the `?` placeholders.
                    # So `records` has 24 items. 
                    # The VALUES clause had 25 slots because of NULL for ID.
                    
                    # For Postgres, we omit ID column in INSERT to auto-increment.
                    cursor.executemany(query, records)
                    
                else:
                    # SQLite
                    cursor.executemany(f'''
                        INSERT OR REPLACE INTO option_chain_snapshots 
                        (id, timestamp, exchange, strike, option_type, symbol, oi, ltp, token, 
                         underlying_price, moneyness, time_to_expiry_seconds, pct_change_3m, 
                         pct_change_5m, pct_change_10m, pct_change_15m, pct_change_30m, iv, volume,
                         best_bid, best_ask, bid_quantity, ask_quantity, spread, order_book_imbalance,
                         created_at, updated_at)
                        VALUES (NULL, {placeholders})
                    ''', records)

            # Save ML features
            if ml_features_dict:
                config = get_config()
                ph = _get_placeholder()
                
                feature_payload = _serialize_feature_dict(ml_features_dict)
                
                # Sanitize all ML feature values to standard Python types to avoid "np.float64" db errors
                # Extract sentiment values from macro features (they're prefixed with 'macro_' in the dict)
                raw_vals = [
                    ml_features_dict.get('pcr_total_oi'),
                    ml_features_dict.get('pcr_itm_oi'),
                    ml_features_dict.get('pcr_total_volume'),
                    ml_features_dict.get('futures_premium'),
                    ml_features_dict.get('time_to_expiry_hours'),
                    vix_value,
                    underlying_price,
                    underlying_future_price,
                    underlying_future_oi,
                    ml_features_dict.get('total_itm_oi_ce'),
                    ml_features_dict.get('total_itm_oi_pe'),
                    ml_features_dict.get('atm_shift_intensity'),
                    ml_features_dict.get('itm_ce_breadth'),
                    ml_features_dict.get('itm_pe_breadth'),
                    ml_features_dict.get('percent_oichange_fut_3m'),
                    ml_features_dict.get('itm_oi_ce_pct_change_3m_wavg'),
                    ml_features_dict.get('itm_oi_pe_pct_change_3m_wavg'),
                    ml_features_dict.get('dealer_vanna_exposure'),
                    ml_features_dict.get('dealer_charm_exposure'),
                    ml_features_dict.get('net_gamma_exposure'),
                    ml_features_dict.get('gamma_flip_level'),
                    ml_features_dict.get('ce_volume_to_oi_ratio'),
                    ml_features_dict.get('pe_volume_to_oi_ratio'),
                    ml_features_dict.get('news_sentiment_score'),
                    # NIFTY sentiment features (extract from macro_ prefixed keys)
                    ml_features_dict.get('macro_sentiment_score_50'),
                    ml_features_dict.get('macro_sentiment_score_100'),
                    ml_features_dict.get('macro_trin_50'),
                    ml_features_dict.get('macro_trin_100'),
                ]
                # Apply sanitization (converts np.float/int to python float/int)
                sanitized_vals = [_sanitize_feature_value(v) for v in raw_vals]
                
                ml_record = (
                    timestamp_iso, exchange,
                    *sanitized_vals,
                    current_time_iso,
                    feature_payload
                )
                
                if config.db_type == 'postgres':
                    cols = [
                        "timestamp", "exchange", "pcr_total_oi", "pcr_itm_oi", "pcr_total_volume", 
                        "futures_premium", "time_to_expiry_hours", "vix", "underlying_price",
                        "underlying_future_price", "underlying_future_oi", "total_itm_oi_ce", 
                        "total_itm_oi_pe", "atm_shift_intensity", "itm_ce_breadth", "itm_pe_breadth", 
                        "percent_oichange_fut_3m", "itm_oi_ce_pct_change_3m_wavg", 
                        "itm_oi_pe_pct_change_3m_wavg",
                        "dealer_vanna_exposure", "dealer_charm_exposure", "net_gamma_exposure",
                        "gamma_flip_level", "ce_volume_to_oi_ratio", "pe_volume_to_oi_ratio",
                        "news_sentiment_score",
                        "sentiment_score_50", "sentiment_score_100", "trin_50", "trin_100",
                        "created_at", "feature_payload"
                    ]
                    placeholders_str = ', '.join([ph] * len(cols))
                    update_clause = ", ".join([f"{col} = EXCLUDED.{col}" for col in cols])
                    
                    query = f'''
                        INSERT INTO ml_features ({', '.join(cols)})
                        VALUES ({placeholders_str})
                        ON CONFLICT (timestamp, exchange)
                        DO UPDATE SET {update_clause}
                    '''
                    cursor.execute(query, ml_record)
                else:
                    cursor.execute(f'''
                        INSERT OR REPLACE INTO ml_features 
                        (timestamp, exchange, pcr_total_oi, pcr_itm_oi, pcr_total_volume, 
                         futures_premium, time_to_expiry_hours, vix, underlying_price,
                         underlying_future_price, underlying_future_oi, total_itm_oi_ce, 
                         total_itm_oi_pe, atm_shift_intensity, itm_ce_breadth, itm_pe_breadth, 
                         percent_oichange_fut_3m, itm_oi_ce_pct_change_3m_wavg, 
                         itm_oi_pe_pct_change_3m_wavg, dealer_vanna_exposure, dealer_charm_exposure,
                         net_gamma_exposure, gamma_flip_level, ce_volume_to_oi_ratio, pe_volume_to_oi_ratio,
                         news_sentiment_score,
                         sentiment_score_50, sentiment_score_100, trin_50, trin_100,
                         created_at, feature_payload)
                        VALUES ({', '.join([ph]*32)})
                    ''', ml_record)
            
            # Update metadata
            ph = _get_placeholder()
            if get_config().db_type == 'postgres':
                cursor.execute(f'''
                    INSERT INTO exchange_metadata 
                    (exchange, last_update_time, last_atm_strike, last_underlying_price, 
                     last_future_price, last_future_oi, updated_at)
                    VALUES ({', '.join([ph]*7)})
                    ON CONFLICT (exchange) DO UPDATE SET
                    last_update_time = EXCLUDED.last_update_time,
                    last_atm_strike = EXCLUDED.last_atm_strike,
                    last_underlying_price = EXCLUDED.last_underlying_price,
                    last_future_price = EXCLUDED.last_future_price,
                    last_future_oi = EXCLUDED.last_future_oi,
                    updated_at = EXCLUDED.updated_at
                ''', (exchange, timestamp_iso, atm_strike, underlying_price, underlying_future_price, underlying_future_oi, current_time_iso))
            else:
                cursor.execute(f'''
                    INSERT OR REPLACE INTO exchange_metadata 
                    (exchange, last_update_time, last_atm_strike, last_underlying_price, 
                     last_future_price, last_future_oi, updated_at)
                    VALUES ({', '.join([ph]*7)})
                ''', (exchange, timestamp_iso, atm_strike, underlying_price, underlying_future_price, underlying_future_oi, current_time_iso))
            
            conn.commit()
            logging.info(f"✓ Saved {len(records)} records + ML features for {exchange}")
            release_db_connection(conn)
            
        except Exception as e:
            logging.error(f"Error saving snapshot: {e}", exc_info=True)
            if 'conn' in locals():
                release_db_connection(conn) # Ensure release on error


def load_historical_data_for_ml(exchange: str, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Load timestamp-level features for ML training.
    """
    logging.info(f"Loading ML features for {exchange} from {start_date} to {end_date}")
    
    with db_lock:
        try:
            conn = get_db_connection()
            ph = _get_placeholder()
            query = f"""
                SELECT * FROM ml_features
                WHERE exchange = {ph} AND timestamp >= {ph} AND timestamp <= {ph}
                ORDER BY timestamp ASC
            """
            # psycopg2 prefers standard SQL params, pandas read_sql handles execution
            # But read_sql with psycopg2 connection might need params as list/tuple
            df = pd.read_sql_query(query, conn, 
                                 params=(exchange, start_date.isoformat(), end_date.isoformat()))
            release_db_connection(conn)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                if 'feature_payload' in df.columns:
                    extra = _deserialize_feature_series(df['feature_payload'])
                    df = pd.concat([df.drop(columns=['feature_payload']), extra], axis=1)
                logging.info(f"✓ Loaded {len(df)} feature records")
            return df
            
        except Exception as e:
            logging.error(f"Error loading ML data: {e}")
            if 'conn' in locals():
                release_db_connection(conn)
            return pd.DataFrame()

def cleanup_old_data(days_to_keep=30):
    """Delete data older than N days."""
    with db_lock:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cutoff = _coerce_iso_timestamp(now_ist() - timedelta(days=days_to_keep))
            ph = _get_placeholder()
            
            cursor.execute(f"DELETE FROM option_chain_snapshots WHERE timestamp < {ph}", (cutoff,))
            cursor.execute(f"DELETE FROM ml_features WHERE timestamp < {ph}", (cutoff,))
            
            deleted = cursor.rowcount
            conn.commit()
            release_db_connection(conn)
            logging.info(f"✓ Cleaned up {deleted} old records")
        except Exception as e:
            logging.error(f"Cleanup error: {e}")
            if 'conn' in locals():
                release_db_connection(conn)


def record_paper_trading_metric(
    exchange: str,
    timestamp,
    executed: bool,
    reason: str,
    signal: str,
    confidence: float,
    quantity_lots: int,
    pnl: float | None,
    constraint_violation: bool,
) -> None:
    """
    Persist a single paper trading metric event to the database.
    """
    with db_lock:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            ph = _get_placeholder()
            cursor.execute(f'''
                INSERT INTO paper_trading_metrics (
                    timestamp, exchange, executed, reason, signal,
                    confidence, quantity_lots, pnl, constraint_violation,
                    created_at
                ) VALUES ({', '.join([ph]*10)})
            ''', (
                _coerce_iso_timestamp(timestamp),
                exchange,
                bool(executed),
                reason,
                signal,
                float(confidence) if confidence is not None else None,
                int(quantity_lots) if quantity_lots is not None else 0,
                float(pnl) if pnl is not None else None,
                bool(constraint_violation),
                _coerce_iso_timestamp(now_ist())
            ))
            conn.commit()
            release_db_connection(conn)
        except Exception as exc:
            logging.error("Failed to record paper trading metric: %s", exc, exc_info=True)
            if 'conn' in locals():
                release_db_connection(conn)

def record_training_batch(
    exchange: str,
    start_timestamp,
    end_timestamp,
    model_hash: str | None = None,
    artifact_path: str | None = None,
    csv_path: str | None = None,
    parquet_path: str | None = None,
    metadata: dict | None = None,
    dataset_version: str | None = None,
) -> int | None:
    """
    Persist a training batch entry for traceability between exports, models, and backtests.
    """
    with db_lock:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            ph = _get_placeholder()
            cursor.execute(f'''
                INSERT INTO training_batches (
                    exchange, start_timestamp, end_timestamp, model_hash,
                    artifact_path, csv_path, parquet_path, metadata,
                    created_at, dataset_version
                ) VALUES ({', '.join([ph]*10)})
            ''', (
                exchange,
                _coerce_iso_timestamp(start_timestamp),
                _coerce_iso_timestamp(end_timestamp),
                model_hash,
                artifact_path,
                csv_path,
                parquet_path,
                _safe_json_dumps(metadata),
                _coerce_iso_timestamp(now_ist()),
                dataset_version,
            ))
            conn.commit()
            row_id = cursor.lastrowid
            release_db_connection(conn)
            logging.info("✓ Recorded training batch %s for %s", row_id, exchange)
            return row_id
        except Exception as exc:
            logging.error("Failed to record training batch: %s", exc, exc_info=True)
            if 'conn' in locals():
                release_db_connection(conn)
            return None


def list_training_batches(exchange: str | None = None, limit: int = 50) -> list[dict]:
    """
    Retrieve recent training batch metadata for dashboards or orchestration.
    """
    with db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        ph = _get_placeholder()
        if exchange:
            cursor.execute(f'''
                SELECT * FROM training_batches
                WHERE exchange = {ph}
                ORDER BY created_at DESC
                LIMIT {ph}
            ''', (exchange, limit))
        else:
            cursor.execute(f'''
                SELECT * FROM training_batches
                ORDER BY created_at DESC
                LIMIT {ph}
            ''', (limit,))
        rows = cursor.fetchall()
        release_db_connection(conn)
        return rows


def export_training_window(
    exchange: str,
    start_timestamp,
    end_timestamp,
    output_dir: str | Path | None = None,
    file_prefix: str | None = None,
    include_payload: bool = True,
    dataset_version: str | None = None,
) -> dict:
    """
    Export ML feature history for the provided window to CSV & Parquet for offline experiments.
    """
    start_iso = _coerce_iso_timestamp(start_timestamp)
    end_iso = _coerce_iso_timestamp(end_timestamp)
    logging.info("Exporting training window for %s between %s and %s", exchange, start_iso, end_iso)

    with db_lock:
        try:
            conn = get_db_connection()
            ph = _get_placeholder()
            query = f"""
                SELECT * FROM ml_features
                WHERE exchange = {ph} AND timestamp >= {ph} AND timestamp <= {ph}
                ORDER BY timestamp ASC
            """
            df = pd.read_sql_query(query, conn, params=(exchange, start_iso, end_iso))
            release_db_connection(conn)
        except Exception as exc:
            logging.error("Failed reading ML features for export: %s", exc, exc_info=True)
            if 'conn' in locals():
                release_db_connection(conn)
            return {}

    if df.empty:
        logging.warning("No ML feature rows found for %s between %s and %s", exchange, start_iso, end_iso)
        return {}

    if include_payload and 'feature_payload' in df.columns:
        payload_df = _deserialize_feature_series(df['feature_payload'])
        df = pd.concat([df.drop(columns=['feature_payload']), payload_df], axis=1)

    export_dir = _ensure_directory(output_dir or TRAINING_EXPORT_DIR)
    timestamp_slug = now_ist().strftime("%Y%m%d_%H%M%S")
    start_slug = start_iso.replace(':', '').replace(' ', '_').replace('-', '')
    end_slug = end_iso.replace(':', '').replace(' ', '_').replace('-', '')
    stem = file_prefix or f"{exchange}_{start_slug}_{end_slug}_{timestamp_slug}"
    stem_path = Path(export_dir) / stem

    csv_path = stem_path.with_suffix(".csv")
    parquet_path = stem_path.with_suffix(".parquet")

    df.to_csv(csv_path, index=False)

    try:
        df.to_parquet(parquet_path, index=False)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        logging.warning("Parquet export unavailable: %s", exc)
        parquet_path = None

    record_training_batch(
        exchange=exchange,
        start_timestamp=start_iso,
        end_timestamp=end_iso,
        csv_path=str(csv_path),
        parquet_path=str(parquet_path) if parquet_path else None,
        metadata={"row_count": len(df)},
        dataset_version=dataset_version,
    )

    return {
        "csv_path": str(csv_path),
        "parquet_path": str(parquet_path) if parquet_path else None,
        "row_count": len(df)
    }


def save_vix_term_structure(exchange: str, front_month_price: float, next_month_price: float,
                            timestamp: datetime | None = None, source: str | None = None,
                            contango_pct: float | None = None, backwardation_pct: float | None = None,
                            current_vix: float | None = None, realized_vol: float | None = None,
                            vix_ma_5d: float | None = None, vix_ma_20d: float | None = None,
                            vix_trend_1d: float | None = None, vix_trend_5d: float | None = None) -> None:
    """
    Persist VIX term structure snapshot.
    """
    if timestamp is None:
        timestamp = now_ist()
    if contango_pct is None and front_month_price and next_month_price:
        contango_pct = ((next_month_price - front_month_price) / max(front_month_price, 1e-6)) * 100
    if backwardation_pct is None:
        backwardation_pct = -contango_pct if contango_pct else None
    
    if current_vix is None:
        current_vix = front_month_price

    with db_lock:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            ph = _get_placeholder()
            cursor.execute(f'''
                INSERT INTO vix_term_structure (
                    timestamp, exchange, front_month_price, next_month_price,
                    contango_pct, backwardation_pct, source,
                    current_vix, realized_vol, vix_ma_5d, vix_ma_20d,
                    vix_trend_1d, vix_trend_5d, created_at
                ) VALUES ({', '.join([ph]*14)})
            ''', (
                _coerce_iso_timestamp(timestamp),
                exchange,
                front_month_price,
                next_month_price,
                contango_pct,
                backwardation_pct,
                source,
                current_vix,
                realized_vol,
                vix_ma_5d,
                vix_ma_20d,
                vix_trend_1d,
                vix_trend_5d,
                _coerce_iso_timestamp(now_ist())
            ))
            conn.commit()
            release_db_connection(conn)
        except Exception:
            if 'conn' in locals():
                release_db_connection(conn)
            raise


def save_macro_signals(exchange: str, fii_flow: float | None = None, dii_flow: float | None = None,
                       usdinr: float | None = None, usdinr_trend: float | None = None,
                       crude_price: float | None = None, crude_trend: float | None = None,
                       banknifty_correlation: float | None = None, macro_spread: float | None = None,
                       risk_on_score: float | None = None, metadata: dict | None = None,
                       timestamp: datetime | None = None,
                       sentiment_score_50: float | None = None, sentiment_confidence_50: float | None = None,
                       trin_50: float | None = None, sentiment_score_100: float | None = None,
                       sentiment_confidence_100: float | None = None, trin_100: float | None = None) -> None:
    """
    Insert macro or fund-flow snapshot with NIFTY sentiment data.
    
    Args:
        exchange: Exchange name (e.g., 'NSE')
        fii_flow: FII net flow
        dii_flow: DII net flow
        usdinr: USD/INR price
        usdinr_trend: USD/INR trend
        crude_price: Crude oil price
        crude_trend: Crude oil trend
        banknifty_correlation: BankNifty correlation
        macro_spread: Macro spread
        risk_on_score: Risk-on score
        metadata: Additional metadata dict
        timestamp: Timestamp (defaults to now)
        sentiment_score_50: NIFTY50 sentiment score (0-100)
        sentiment_confidence_50: NIFTY50 confidence (0-100)
        trin_50: NIFTY50 TRIN value
        sentiment_score_100: NIFTY100 sentiment score (0-100)
        sentiment_confidence_100: NIFTY100 confidence (0-100)
        trin_100: NIFTY100 TRIN value
    
    Note:
        sentiment_score and sentiment_confidence are computed as:
        - Average of _50 and _100 values if both are available
        - Otherwise, use whichever value is available
    """
    if timestamp is None:
        timestamp = now_ist()
    fii_dii_net = None
    if fii_flow is not None or dii_flow is not None:
        # fii_dii_net is the sum of both flows (not subtraction)
        fii_dii_net = (fii_flow or 0.0) + (dii_flow or 0.0)

    with db_lock:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            ph = _get_placeholder()
            # Compute aggregated sentiment_score and sentiment_confidence
            # Use average of _50 and _100 if both are available, otherwise use whichever is available
            sentiment_score = None
            sentiment_confidence = None
            if sentiment_score_50 is not None and sentiment_score_100 is not None:
                sentiment_score = (sentiment_score_50 + sentiment_score_100) / 2.0
            elif sentiment_score_50 is not None:
                sentiment_score = sentiment_score_50
            elif sentiment_score_100 is not None:
                sentiment_score = sentiment_score_100
            
            if sentiment_confidence_50 is not None and sentiment_confidence_100 is not None:
                sentiment_confidence = (sentiment_confidence_50 + sentiment_confidence_100) / 2.0
            elif sentiment_confidence_50 is not None:
                sentiment_confidence = sentiment_confidence_50
            elif sentiment_confidence_100 is not None:
                sentiment_confidence = sentiment_confidence_100
            
            cursor.execute(f'''
                INSERT INTO macro_signals (
                    timestamp, exchange, fii_flow, dii_flow, fii_dii_net,
                    usdinr, usdinr_trend, crude_price, crude_trend,
                    banknifty_correlation, macro_spread, risk_on_score, metadata,
                    sentiment_score_50, sentiment_confidence_50, trin_50,
                    sentiment_score_100, sentiment_confidence_100, trin_100,
                    sentiment_score, sentiment_confidence,
                    created_at
                ) VALUES ({', '.join([ph]*22)})
            ''', (
                _coerce_iso_timestamp(timestamp),
                exchange,
                fii_flow,
                dii_flow,
                fii_dii_net,
                usdinr,
                usdinr_trend,
                crude_price,
                crude_trend,
                banknifty_correlation,
                macro_spread,
                risk_on_score,
                _safe_json_dumps(metadata),
                sentiment_score_50,
                sentiment_confidence_50,
                trin_50,
                sentiment_score_100,
                sentiment_confidence_100,
                trin_100,
                sentiment_score,
                sentiment_confidence,
                _coerce_iso_timestamp(now_ist())
            ))
            conn.commit()
            release_db_connection(conn)
        except Exception:
            if 'conn' in locals():
                release_db_connection(conn)
            raise


def save_order_book_depth_snapshot(exchange: str, depth_buy_total: float,
                                   depth_sell_total: float, depth_imbalance_ratio: float,
                                   timestamp: datetime | None = None, source: str | None = None) -> None:
    """Store aggregated order-book depth metrics."""
    if timestamp is None:
        timestamp = now_ist()

    with db_lock:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            ph = _get_placeholder()
            cursor.execute(f'''
                INSERT INTO order_book_depth_snapshots (
                    timestamp, exchange, depth_buy_total, depth_sell_total,
                    depth_imbalance_ratio, source, created_at
                ) VALUES ({', '.join([ph]*7)})
            ''', (
                _coerce_iso_timestamp(timestamp),
                exchange,
                depth_buy_total,
                depth_sell_total,
                depth_imbalance_ratio,
                source,
                _coerce_iso_timestamp(now_ist())
            ))
            conn.commit()
            release_db_connection(conn)
        except Exception:
            if 'conn' in locals():
                release_db_connection(conn)
            raise


def get_latest_vix_term_structure(exchange: str) -> dict:
    with db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        ph = _get_placeholder()
        cursor.execute(f'''
            SELECT * FROM vix_term_structure
            WHERE exchange = {ph}
            ORDER BY timestamp DESC
            LIMIT 1
        ''', (exchange,))
        row = cursor.fetchone()
        
        # Safe conversion before release
        payload = {}
        if row:
            if hasattr(cursor, 'description'):
                 cols = [d[0] for d in cursor.description]
                 payload = dict(zip(cols, row))
            else:
                 # Try generic conversion
                 try:
                    payload = dict(row)
                 except Exception:
                    payload = {}

        release_db_connection(conn)
        return payload


def get_latest_macro_signals(exchange: str) -> dict:
    with db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        ph = _get_placeholder()
        cursor.execute(f'''
            SELECT * FROM macro_signals
            WHERE exchange = {ph}
            ORDER BY timestamp DESC
            LIMIT 1
        ''', (exchange,))
        row = cursor.fetchone()
        
        # Convert row to dict BEFORE closing connection
        payload = {}
        if row:
            if hasattr(cursor, 'description'):
                # Postgres standard cursor returns tuple, need description for keys
                cols = [d[0] for d in cursor.description]
                payload = dict(zip(cols, row))
            elif hasattr(row, 'keys'):
                 # Psycopg2 DictRow
                payload = dict(row)
            else:
                # Fallback for tuple without description (shouldn't happen if we check description above)
                payload = {'raw_row': row}

        release_db_connection(conn)

        if not payload:
            return {}
            
        if payload.get('metadata'):
            try:
                payload['metadata'] = json.loads(payload['metadata'])
            except (json.JSONDecodeError, TypeError):
                payload['metadata'] = {}
        return payload


def get_historical_macro_signals(exchange: str, limit: int = 30) -> list:
    """Fetch historical macro signals for correlation calculation."""
    with db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        ph = _get_placeholder()
        cursor.execute(f'''
            SELECT timestamp, usdinr_trend, crude_trend, fii_flow, dii_flow, risk_on_score
            FROM macro_signals
            WHERE exchange = {ph} AND usdinr_trend IS NOT NULL AND crude_trend IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT {ph}
        ''', (exchange, limit))
        rows = cursor.fetchall()
        
        results = []
        if rows:
            if hasattr(cursor, 'description'):
                cols = [d[0] for d in cursor.description]
                results = [dict(zip(cols, r)) for r in rows]
            else:
                 # Try generic conversion
                 try:
                    results = [dict(r) for r in rows]
                 except Exception:
                    pass

        release_db_connection(conn)
        return results


def get_latest_depth_snapshot(exchange: str) -> dict:
    with db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        ph = _get_placeholder()
        cursor.execute(f'''
            SELECT * FROM order_book_depth_snapshots
            WHERE exchange = {ph}
            ORDER BY timestamp DESC
            LIMIT 1
        ''', (exchange,))
        row = cursor.fetchone()
        
        # Safe conversion before release
        payload = {}
        if row:
            if hasattr(cursor, 'description'):
                 cols = [d[0] for d in cursor.description]
                 payload = dict(zip(cols, row))
            else:
                 # Try generic conversion
                 try:
                    payload = dict(row)
                 except Exception:
                    payload = {}

        release_db_connection(conn)
        return payload


def get_latest_macro_price_row(exchange: str) -> dict:
    with db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        ph = _get_placeholder()
        cursor.execute(f'''
            SELECT * FROM macro_signals
            WHERE exchange = {ph} AND usdinr IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 1
        ''', (exchange,))
        row = cursor.fetchone()
        
        # Safe conversion before release
        payload = {}
        if row:
            if hasattr(cursor, 'description'):
                 cols = [d[0] for d in cursor.description]
                 payload = dict(zip(cols, row))
            else:
                 # Try generic conversion
                 try:
                    payload = dict(row)
                 except Exception:
                    payload = {}

        release_db_connection(conn)
        return payload


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