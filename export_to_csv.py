# Export Database to CSV Files
# Creates separate CSV files for each table for easy viewing and analysis
# Updated to match latest database_new.py schema (8 tables)

import pandas as pd
from datetime import datetime, date
import logging

from time_utils import now_ist, today_ist
from database_new import get_db_connection, release_db_connection, _get_placeholder, get_config

# Configure logging to show info during import/init
logging.basicConfig(level=logging.INFO, format='%(message)s')

def export_all_data():
    """Export all database tables to separate CSV files."""
    
    config = get_config()
    db_name = config.db_name if config.db_type == 'postgres' else "db_new.db"
    
    print("=" * 70)
    print(f"DATABASE EXPORT FROM '{db_name}' TO CSV")
    print("=" * 70)
    
    conn = None
    try:
        conn = get_db_connection()
        
        # 1. Export option_chain_snapshots (today's data)
        print("\n1. Exporting today's option chain snapshots...")
        today_str = today_ist().strftime('%Y-%m-%d')
        
        # Use simple string formatting for the date to avoid placeholder issues across DBs 
        # since pandas read_sql params handling can be tricky with raw psycopg2 connections
        query = f"""
            SELECT *
            FROM option_chain_snapshots
            WHERE date(timestamp) = '{today_str}'
            ORDER BY timestamp DESC, exchange, strike, option_type
        """
        
        df_snapshots = pd.read_sql_query(query, conn)
        
        filename_snapshots = f"option_chain_snapshots_today_{today_ist().strftime('%Y%m%d')}.csv"
        df_snapshots.to_csv(filename_snapshots, index=False)
        print(f"   ✓ Saved: {filename_snapshots} ({len(df_snapshots)} records)")

        # 2. Export ml_features
        print("\n2. Exporting ML features...")
        df_ml = pd.read_sql_query(
            """
            SELECT * FROM ml_features
            ORDER BY timestamp DESC
            """,
            conn
        )
        filename_ml = f"ml_features_{now_ist().strftime('%Y%m%d_%H%M%S')}.csv"
        df_ml.to_csv(filename_ml, index=False)
        print(f"   ✓ Saved: {filename_ml} ({len(df_ml)} records)")
        
        # 3. Export exchange metadata
        print("\n3. Exporting exchange metadata...")
        df_meta = pd.read_sql_query(
            """
            SELECT * FROM exchange_metadata
            ORDER BY exchange
            """, 
            conn
        )
        filename_meta = f"exchange_metadata_{now_ist().strftime('%Y%m%d_%H%M%S')}.csv"
        df_meta.to_csv(filename_meta, index=False)
        print(f"   ✓ Saved: {filename_meta} ({len(df_meta)} records)")
        
        # 4. Export training batches
        print("\n4. Exporting training batches...")
        df_training = pd.read_sql_query(
            """
            SELECT * FROM training_batches
            ORDER BY created_at DESC
            """,
            conn
        )
        filename_training = f"training_batches_{now_ist().strftime('%Y%m%d_%H%M%S')}.csv"
        df_training.to_csv(filename_training, index=False)
        print(f"   ✓ Saved: {filename_training} ({len(df_training)} records)")
        
        # 5. Export VIX term structure
        print("\n5. Exporting VIX term structure...")
        df_vix = pd.read_sql_query(
            """
            SELECT * FROM vix_term_structure
            ORDER BY timestamp DESC
            """,
            conn
        )
        filename_vix = f"vix_term_structure_{now_ist().strftime('%Y%m%d_%H%M%S')}.csv"
        df_vix.to_csv(filename_vix, index=False)
        print(f"   ✓ Saved: {filename_vix} ({len(df_vix)} records)")
        
        # 6. Export macro signals
        print("\n6. Exporting macro signals...")
        df_macro = pd.read_sql_query(
            """
            SELECT * FROM macro_signals
            ORDER BY timestamp DESC
            """,
            conn
        )
        filename_macro = f"macro_signals_{now_ist().strftime('%Y%m%d_%H%M%S')}.csv"
        df_macro.to_csv(filename_macro, index=False)
        print(f"   ✓ Saved: {filename_macro} ({len(df_macro)} records)")
        
        # 7. Export order book depth snapshots
        print("\n7. Exporting order book depth snapshots...")
        df_depth = pd.read_sql_query(
            """
            SELECT * FROM order_book_depth_snapshots
            ORDER BY timestamp DESC
            """,
            conn
        )
        filename_depth = f"order_book_depth_snapshots_{now_ist().strftime('%Y%m%d_%H%M%S')}.csv"
        df_depth.to_csv(filename_depth, index=False)
        print(f"   ✓ Saved: {filename_depth} ({len(df_depth)} records)")
        
        # 8. Export paper trading metrics
        print("\n8. Exporting paper trading metrics...")
        df_paper = pd.read_sql_query(
            """
            SELECT * FROM paper_trading_metrics
            ORDER BY timestamp DESC
            """,
            conn
        )
        filename_paper = f"paper_trading_metrics_{now_ist().strftime('%Y%m%d_%H%M%S')}.csv"
        df_paper.to_csv(filename_paper, index=False)
        print(f"   ✓ Saved: {filename_paper} ({len(df_paper)} records)")
        
        # 9. Export summary statistics
        print("\n9. Creating summary report...")
        summary_data = []
        
        cursor = conn.cursor()
        ph = _get_placeholder()
        
        for exchange in ['NSE', 'BSE', 'NSE_MONTHLY']:
            try:
                # Use param binding for exchange name
                query = f"""
                    SELECT 
                        {ph} as exchange,
                        COUNT(*) as total_records,
                        MIN(timestamp) as first_record,
                        MAX(timestamp) as last_record,
                        COUNT(DISTINCT date(timestamp)) as days_of_data
                    FROM option_chain_snapshots
                    WHERE exchange = {ph}
                """
                cursor.execute(query, (exchange, exchange))
                
                row = cursor.fetchone()
                if row:
                    summary_data.append(row)
            except Exception as e:
                print(f"   Could not generate summary for {exchange}: {e}")

        if summary_data:
            df_summary = pd.DataFrame(summary_data, columns=[
                'exchange', 'total_records', 'first_record', 'last_record', 
                'days_of_data'
            ])
            filename_summary = f"database_summary_{now_ist().strftime('%Y%m%d_%H%M%S')}.csv"
            df_summary.to_csv(filename_summary, index=False)
            print(f"   ✓ Saved: {filename_summary}")
        
    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            release_db_connection(conn)
    
    print("\n" + "=" * 70)
    print("✓ EXPORT COMPLETE!")
    print("=" * 70)

if __name__ == '__main__':
    export_all_data()
