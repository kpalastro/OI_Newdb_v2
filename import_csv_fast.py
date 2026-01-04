#!/usr/bin/env python3
"""
Fast CSV import using PostgreSQL COPY command.
Handles CSV files with no headers, skipping id column for multi_resolution_bars.
"""

import pandas as pd
import json
import logging
import sys
import io
from database_new import get_db_connection, release_db_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def import_with_copy(csv_path: str, table_name: str, column_definitions: list):
    """Import CSV using PostgreSQL COPY command for speed."""
    logging.info(f"Importing {csv_path} into {table_name} using COPY...")
    
    try:
        # Read CSV
        df = pd.read_csv(csv_path, header=None, names=column_definitions)
        logging.info(f"Loaded {len(df)} rows from {csv_path}")
        
        # Remove id column if present
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
            logging.info("Dropped 'id' column")
        
        # Convert columns to proper types for bigint/integer columns FIRST
        # Use Int64 (nullable integer) for bigint columns to handle NaN properly
        import numpy as np
        
        bigint_columns = []
        integer_columns = []
        
        if table_name == 'multi_resolution_bars':
            bigint_columns = ['volume', 'oi', 'oi_change']
            integer_columns = ['trade_count']
        elif table_name == 'option_chain_snapshots':
            bigint_columns = ['volume', 'oi']
            integer_columns = ['bid_quantity', 'ask_quantity', 'time_to_expiry_seconds']
        
        # Convert bigint columns to Int64 (nullable integer)
        for col in bigint_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        
        # Convert integer columns to Int64
        for col in integer_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        
        # Now handle remaining NaN/None values
        df = df.replace([np.nan, pd.NA, pd.NaT, ''], None)
        
        # Get connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create in-memory CSV for COPY
        output = io.StringIO()
        df.to_csv(output, sep='\t', header=False, index=False, na_rep='\\N')
        output.seek(0)
        
        columns = df.columns.tolist()
        cols_str = ', '.join(columns)
        
        # Use COPY command for fast bulk insert
        try:
            cursor.copy_from(
                output,
                table_name,
                sep='\t',
                null='\\N',
                columns=columns
            )
            conn.commit()
            logging.info(f"✓ Imported {len(df)} rows into {table_name}")
            release_db_connection(conn)
            return True
        except Exception as copy_error:
            conn.rollback()
            logging.warning(f"COPY failed ({copy_error}), falling back to row-by-row INSERT...")
            
            # Fallback to row-by-row INSERT with conflict handling
            from database_new import _get_placeholder
            ph = _get_placeholder()
            placeholders = ', '.join([ph] * len(columns))
            
            # Determine conflict columns
            if table_name == 'ml_features':
                conflict_cols = 'timestamp, exchange'
            elif table_name == 'option_chain_snapshots':
                conflict_cols = 'timestamp, exchange, strike, option_type'
            elif table_name == 'multi_resolution_bars':
                conflict_cols = 'timestamp, exchange, resolution, token'
            else:
                conflict_cols = ''
            
            update_clause = ', '.join([f"{col} = EXCLUDED.{col}" for col in columns if col != 'id'])
            
            if conflict_cols:
                query = f'''
                    INSERT INTO {table_name} ({cols_str})
                    VALUES ({placeholders})
                    ON CONFLICT ({conflict_cols}) DO UPDATE SET {update_clause}
                '''
            else:
                query = f'''
                    INSERT INTO {table_name} ({cols_str})
                    VALUES ({placeholders})
                '''
            
            imported_count = 0
            failed_count = 0
            for idx, row in df.iterrows():
                try:
                    record = tuple(row)
                    cursor.execute(query, record)
                    imported_count += 1
                    
                    if imported_count % 500 == 0:
                        conn.commit()
                        logging.info(f"Progress: {imported_count}/{len(df)} rows")
                except Exception as e:
                    conn.rollback()
                    failed_count += 1
                    if failed_count < 5:  # Only log first few errors
                        logging.error(f"Row {idx} error: {e}")
                    continue
            
            conn.commit()
            logging.info(f"✓ Imported {imported_count} rows (failed: {failed_count})")
            release_db_connection(conn)
            return imported_count > 0
            
    except Exception as e:
        logging.error(f"Error importing {csv_path}: {e}", exc_info=True)
        if 'conn' in locals():
            conn.rollback()
            release_db_connection(conn)
        return False


def import_ml_features(csv_path: str):
    """Import ml_features with feature_payload parsing."""
    logging.info(f"Importing {csv_path} into ml_features...")
    
    try:
        # Define columns for ml_feature.csv (35 columns, no header)
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
        logging.info(f"Loaded {len(df)} rows")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        from database_new import _get_placeholder
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
                
                # Extract nse_next_* fields
                nse_next_oi_call = feature_dict.get('nse_next_oi_call_total')
                nse_next_oi_put = feature_dict.get('nse_next_oi_put_total')
                nse_next_oi_change_call = feature_dict.get('nse_next_oi_change_call_total')
                nse_next_oi_change_put = feature_dict.get('nse_next_oi_change_put_total')
                nse_next_volume_call = feature_dict.get('nse_next_volume_call_total')
                nse_next_volume_put = feature_dict.get('nse_next_volume_put_total')
                nse_next_oi_change_diff = feature_dict.get('nse_next_oi_change_diff_put_call')
                oi_next_sentiment = feature_dict.get('oi_next_sentiment')
                
                # Build INSERT query (40 columns total)
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
                    nse_next_oi_call, nse_next_oi_put,
                    nse_next_oi_change_call, nse_next_oi_change_put,
                    nse_next_volume_call, nse_next_volume_put,
                    nse_next_oi_change_diff,
                    row['created_at'], feature_payload_json
                )
                
                cursor.execute(query, record)
                imported_count += 1
                
                if imported_count % 100 == 0:
                    conn.commit()
                    sys.stdout.write(f"\rProgress: {imported_count}/{len(df)} rows")
                    sys.stdout.flush()
                
            except Exception as e:
                conn.rollback()
                failed_count += 1
                if failed_count <= 5:
                    logging.error(f"Row {idx} error: {e}")
                continue
        
        conn.commit()
        print(f"\n✓ Imported {imported_count} rows (failed: {failed_count})")
        release_db_connection(conn)
        return True
        
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        if 'conn' in locals():
            conn.rollback()
            release_db_connection(conn)
        return False


def main():
    """Import all CSV files."""
    tmp_dir = '/Users/kpal/projects/dilip/OI_Newdb_v2/tmp'
    
    # 1. option_chain_snapshots
    print("\n" + "="*60)
    option_cols = [
        'id', 'timestamp', 'exchange', 'strike', 'option_type', 'symbol',
        'oi', 'ltp', 'token', 'underlying_price', 'moneyness',
        'time_to_expiry_seconds', 'pct_change_3m', 'pct_change_5m',
        'pct_change_10m', 'pct_change_15m', 'pct_change_30m', 'iv',
        'volume', 'best_bid', 'best_ask', 'bid_quantity', 'ask_quantity',
        'spread', 'order_book_imbalance', 'created_at', 'updated_at'
    ]
    import_with_copy(f'{tmp_dir}/option_chain_snapshots.csv', 'option_chain_snapshots', option_cols)
    
    # 2. multi_resolution_bars
    print("\n" + "="*60)
    bars_cols = [
        'id', 'timestamp', 'exchange', 'resolution', 'token', 'symbol',
        'open_price', 'high_price', 'low_price', 'close_price',
        'volume', 'oi', 'oi_change', 'vwap', 'trade_count',
        'spread_avg', 'imbalance_avg', 'created_at'
    ]
    import_with_copy(f'{tmp_dir}/multi_resolution_bars.csv', 'multi_resolution_bars', bars_cols)
    
    # 3. ml_features
    print("\n" + "="*60)
    import_ml_features(f'{tmp_dir}/ml_feature.csv')
    
    print("\n" + "="*60)
    print("Import completed!")


if __name__ == '__main__':
    main()
