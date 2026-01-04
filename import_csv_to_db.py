#!/usr/bin/env python3
"""
Import CSV files from tmp folder into database.
For ml_feature table, parse feature_payload JSON to populate individual columns.
"""

import pandas as pd
import json
import logging
from datetime import datetime
from database_new import get_db_connection, release_db_connection, _get_placeholder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def import_csv_to_table(csv_path: str, table_name: str):
    """Import a generic CSV file into its corresponding table."""
    logging.info(f"Importing {csv_path} into table '{table_name}'...")
    
    try:
        # Define column names for tables based on table name
        if table_name == 'option_chain_snapshots':
            column_names = [
                'id', 'timestamp', 'exchange', 'strike', 'option_type', 'symbol',
                'oi', 'ltp', 'token', 'underlying_price', 'moneyness',
                'time_to_expiry_seconds', 'pct_change_3m', 'pct_change_5m',
                'pct_change_10m', 'pct_change_15m', 'pct_change_30m', 'iv',
                'volume', 'best_bid', 'best_ask', 'bid_quantity', 'ask_quantity',
                'spread', 'order_book_imbalance', 'created_at', 'updated_at'
            ]
            df = pd.read_csv(csv_path, header=None, names=column_names)
        elif table_name == 'multi_resolution_bars':
            column_names = [
                'id', 'timestamp', 'exchange', 'resolution', 'token', 'symbol',
                'open_price', 'high_price', 'low_price', 'close_price',
                'volume', 'oi', 'oi_change', 'vwap', 'trade_count',
                'spread_avg', 'imbalance_avg', 'created_at'
            ]
            df = pd.read_csv(csv_path, header=None, names=column_names)
        else:
            # Read CSV with headers
            df = pd.read_csv(csv_path)
        
        logging.info(f"Loaded {len(df)} rows from {csv_path}")
        
        if len(df) == 0:
            logging.warning(f"No data to import from {csv_path}")
            return
        
        # Remove 'id' column if it exists (auto-increment, shouldn't be imported)
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
            logging.info("Dropped 'id' column (auto-increment)")
        
        # Handle NULL/NaN values for numeric columns (convert to None for PostgreSQL NULL)
        # Replace all NaN/NA values with None
        import numpy as np
        df = df.replace([np.nan, pd.NA, pd.NaT, ''], None)
        df = df.where(pd.notnull(df), None)
        
        # Get database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        ph = _get_placeholder()
        
        # Convert DataFrame to list of tuples, handling NaN values
        records = []
        for _, row in df.iterrows():
            # Convert numpy types to python types and NaN to None
            record = tuple(None if pd.isna(val) else (int(val) if isinstance(val, np.integer) else float(val) if isinstance(val, np.floating) else val) 
                          for val in row)
            records.append(record)
        columns = df.columns.tolist()
        
        # Build INSERT query with ON CONFLICT handling
        placeholders = ', '.join([ph] * len(columns))
        cols_str = ', '.join(columns)
        
        # Determine conflict columns based on table
        if table_name == 'ml_features':
            conflict_cols = 'timestamp, exchange'
        elif table_name == 'option_chain_snapshots':
            conflict_cols = 'timestamp, exchange, strike, option_type'
        elif table_name == 'multi_resolution_bars':
            conflict_cols = 'timestamp, exchange, resolution, token'
        else:
            conflict_cols = 'id'  # fallback
        
        # Build update clause
        update_clause = ', '.join([f"{col} = EXCLUDED.{col}" for col in columns if col != 'id'])
        
        query = f'''
            INSERT INTO {table_name} ({cols_str})
            VALUES ({placeholders})
            ON CONFLICT ({conflict_cols}) DO UPDATE SET {update_clause}
        '''
        
        # Execute inserts one by one (executemany has issues with None values)
        imported_count = 0
        failed_count = 0
        for record in records:
            try:
                cursor.execute(query, record)
                imported_count += 1
                if imported_count % 100 == 0:
                    conn.commit()
                    logging.info(f"Progress: {imported_count}/{len(records)} rows imported into {table_name}")
            except Exception as e:
                # Rollback the transaction and continue
                conn.rollback()
                logging.error(f"Error importing row: {e}")
                failed_count += 1
                continue
        
        conn.commit()
        logging.info(f"✓ Imported {imported_count} rows into {table_name}")
        
        release_db_connection(conn)
        return True
        
    except Exception as e:
        logging.error(f"Error importing {csv_path}: {e}", exc_info=True)
        if 'conn' in locals():
            release_db_connection(conn)
        return False


def import_ml_features_with_payload(csv_path: str):
    """
    Import ml_feature.csv into ml_features table.
    Parse feature_payload JSON column to populate individual feature columns.
    """
    logging.info(f"Importing {csv_path} into ml_features table with feature_payload parsing...")
    
    try:
        # Read CSV - ml_feature.csv has no header, so we need to define columns
        # Structure: 35 columns total
        column_names = [
            'timestamp', 'exchange', 'pcr_total_oi', 'pcr_itm_oi', 'pcr_total_volume',
            'futures_premium', 'time_to_expiry_hours', 'vix', 'underlying_price',
            'underlying_future_price', 'underlying_future_oi', 'total_itm_oi_ce',
            'total_itm_oi_pe', 'atm_shift_intensity', 'itm_ce_breadth', 'itm_pe_breadth',
            'percent_oichange_fut_3m', 'itm_oi_ce_pct_change_3m_wavg',
            'itm_oi_pe_pct_change_3m_wavg', 'created_at', 'feature_payload',
            'dealer_vanna_exposure', 'dealer_charm_exposure', 'net_gamma_exposure',
            'gamma_flip_level', 'ce_volume_to_oi_ratio', 'pe_volume_to_oi_ratio',
            'news_sentiment_score', 'sentiment_score_50', 'sentiment_score_100',
            'trin_50', 'trin_100', 'sentiment_confidence_50', 'sentiment_confidence_100',
            'market_breadth'
        ]
        
        df = pd.read_csv(csv_path, header=None, names=column_names)
        logging.info(f"Loaded {len(df)} rows from {csv_path}")
        
        if len(df) == 0:
            logging.warning(f"No data to import from {csv_path}")
            return
        
        # Get database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        ph = _get_placeholder()
        
        imported_count = 0
        failed_count = 0
        
        for idx, row in df.iterrows():
            try:
                # Parse feature_payload JSON
                feature_payload_json = row['feature_payload']
                if isinstance(feature_payload_json, str):
                    feature_dict = json.loads(feature_payload_json)
                else:
                    feature_dict = {}
                
                # Extract nse_next_* fields from feature_payload
                nse_next_oi_call_total = feature_dict.get('nse_next_oi_call_total')
                nse_next_oi_put_total = feature_dict.get('nse_next_oi_put_total')
                nse_next_oi_change_call_total = feature_dict.get('nse_next_oi_change_call_total')
                nse_next_oi_change_put_total = feature_dict.get('nse_next_oi_change_put_total')
                nse_next_volume_call_total = feature_dict.get('nse_next_volume_call_total')
                nse_next_volume_put_total = feature_dict.get('nse_next_volume_put_total')
                nse_next_oi_change_diff_put_call = feature_dict.get('nse_next_oi_change_diff_put_call')
                oi_next_sentiment = feature_dict.get('oi_next_sentiment')
                
                # Build INSERT query
                query = f'''
                    INSERT INTO ml_features (
                        timestamp, exchange, pcr_total_oi, pcr_itm_oi, pcr_total_volume,
                        futures_premium, time_to_expiry_hours, vix, underlying_price,
                        underlying_future_price, underlying_future_oi, total_itm_oi_ce,
                        total_itm_oi_pe, atm_shift_intensity, itm_ce_breadth, itm_pe_breadth,
                        percent_oichange_fut_3m, itm_oi_ce_pct_change_3m_wavg,
                        itm_oi_pe_pct_change_3m_wavg,
                        dealer_vanna_exposure, dealer_charm_exposure, net_gamma_exposure,
                        gamma_flip_level, ce_volume_to_oi_ratio, pe_volume_to_oi_ratio,
                        news_sentiment_score, sentiment_score_50, sentiment_score_100,
                        trin_50, trin_100,
                        oi_next_sentiment,
                        nse_next_oi_call_total, nse_next_oi_put_total,
                        nse_next_oi_change_call_total, nse_next_oi_change_put_total,
                        nse_next_volume_call_total, nse_next_volume_put_total,
                        nse_next_oi_change_diff_put_call,
                        created_at, feature_payload
                    ) VALUES ({', '.join([ph] * 40)})
                    ON CONFLICT (timestamp, exchange) DO UPDATE SET
                        pcr_total_oi = EXCLUDED.pcr_total_oi,
                        pcr_itm_oi = EXCLUDED.pcr_itm_oi,
                        pcr_total_volume = EXCLUDED.pcr_total_volume,
                        futures_premium = EXCLUDED.futures_premium,
                        time_to_expiry_hours = EXCLUDED.time_to_expiry_hours,
                        vix = EXCLUDED.vix,
                        underlying_price = EXCLUDED.underlying_price,
                        underlying_future_price = EXCLUDED.underlying_future_price,
                        underlying_future_oi = EXCLUDED.underlying_future_oi,
                        total_itm_oi_ce = EXCLUDED.total_itm_oi_ce,
                        total_itm_oi_pe = EXCLUDED.total_itm_oi_pe,
                        atm_shift_intensity = EXCLUDED.atm_shift_intensity,
                        itm_ce_breadth = EXCLUDED.itm_ce_breadth,
                        itm_pe_breadth = EXCLUDED.itm_pe_breadth,
                        percent_oichange_fut_3m = EXCLUDED.percent_oichange_fut_3m,
                        itm_oi_ce_pct_change_3m_wavg = EXCLUDED.itm_oi_ce_pct_change_3m_wavg,
                        itm_oi_pe_pct_change_3m_wavg = EXCLUDED.itm_oi_pe_pct_change_3m_wavg,
                        dealer_vanna_exposure = EXCLUDED.dealer_vanna_exposure,
                        dealer_charm_exposure = EXCLUDED.dealer_charm_exposure,
                        net_gamma_exposure = EXCLUDED.net_gamma_exposure,
                        gamma_flip_level = EXCLUDED.gamma_flip_level,
                        ce_volume_to_oi_ratio = EXCLUDED.ce_volume_to_oi_ratio,
                        pe_volume_to_oi_ratio = EXCLUDED.pe_volume_to_oi_ratio,
                        news_sentiment_score = EXCLUDED.news_sentiment_score,
                        sentiment_score_50 = EXCLUDED.sentiment_score_50,
                        sentiment_score_100 = EXCLUDED.sentiment_score_100,
                        trin_50 = EXCLUDED.trin_50,
                        trin_100 = EXCLUDED.trin_100,
                        oi_next_sentiment = EXCLUDED.oi_next_sentiment,
                        nse_next_oi_call_total = EXCLUDED.nse_next_oi_call_total,
                        nse_next_oi_put_total = EXCLUDED.nse_next_oi_put_total,
                        nse_next_oi_change_call_total = EXCLUDED.nse_next_oi_change_call_total,
                        nse_next_oi_change_put_total = EXCLUDED.nse_next_oi_change_put_total,
                        nse_next_volume_call_total = EXCLUDED.nse_next_volume_call_total,
                        nse_next_volume_put_total = EXCLUDED.nse_next_volume_put_total,
                        nse_next_oi_change_diff_put_call = EXCLUDED.nse_next_oi_change_diff_put_call,
                        feature_payload = EXCLUDED.feature_payload
                '''
                
                record = (
                    row['timestamp'], row['exchange'], row['pcr_total_oi'], row['pcr_itm_oi'],
                    row['pcr_total_volume'], row['futures_premium'], row['time_to_expiry_hours'],
                    row['vix'], row['underlying_price'], row['underlying_future_price'],
                    row['underlying_future_oi'], row['total_itm_oi_ce'], row['total_itm_oi_pe'],
                    row['atm_shift_intensity'], row['itm_ce_breadth'], row['itm_pe_breadth'],
                    row['percent_oichange_fut_3m'], row['itm_oi_ce_pct_change_3m_wavg'],
                    row['itm_oi_pe_pct_change_3m_wavg'],
                    row['dealer_vanna_exposure'], row['dealer_charm_exposure'],
                    row['net_gamma_exposure'], row['gamma_flip_level'],
                    row['ce_volume_to_oi_ratio'], row['pe_volume_to_oi_ratio'],
                    row['news_sentiment_score'], row['sentiment_score_50'],
                    row['sentiment_score_100'], row['trin_50'], row['trin_100'],
                    oi_next_sentiment,
                    nse_next_oi_call_total, nse_next_oi_put_total,
                    nse_next_oi_change_call_total, nse_next_oi_change_put_total,
                    nse_next_volume_call_total, nse_next_volume_put_total,
                    nse_next_oi_change_diff_put_call,
                    row['created_at'], feature_payload_json
                )
                
                cursor.execute(query, record)
                imported_count += 1
                
                if imported_count % 100 == 0:
                    conn.commit()
                    logging.info(f"Progress: {imported_count}/{len(df)} rows imported")
                
            except Exception as e:
                logging.error(f"Error importing row {idx}: {e}")
                failed_count += 1
                continue
        
        conn.commit()
        logging.info(f"✓ Imported {imported_count} rows into ml_features (failed: {failed_count})")
        
        release_db_connection(conn)
        return True
        
    except Exception as e:
        logging.error(f"Error importing {csv_path}: {e}", exc_info=True)
        if 'conn' in locals():
            release_db_connection(conn)
        return False


def main():
    """Main import function."""
    tmp_dir = '/Users/kpal/projects/dilip/OI_Newdb_v2/tmp'
    
    # Import order: start with simpler tables first
    imports = [
        (f'{tmp_dir}/option_chain_snapshots.csv', 'option_chain_snapshots', False),
        (f'{tmp_dir}/multi_resolution_bars.csv', 'multi_resolution_bars', False),
        (f'{tmp_dir}/ml_feature.csv', 'ml_features', True),  # Special handling for ml_features
    ]
    
    for csv_path, table_name, use_special_handler in imports:
        logging.info(f"\n{'='*60}")
        if use_special_handler:
            success = import_ml_features_with_payload(csv_path)
        else:
            success = import_csv_to_table(csv_path, table_name)
        
        if success:
            logging.info(f"✓ Successfully imported {table_name}")
        else:
            logging.error(f"✗ Failed to import {table_name}")
    
    logging.info(f"\n{'='*60}")
    logging.info("Import completed!")


if __name__ == '__main__':
    main()
