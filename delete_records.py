# Script to delete records from database
# Deletes today's records from specified time onwards (e.g. to clean up after bad runs)

from datetime import datetime
import sqlite3

from time_utils import today_ist
from database_new import get_db_connection, release_db_connection, _get_placeholder, get_config

def delete_records_from_time(time_str="15:25:00"):
    """Delete today's records from specified time onwards."""
    
    # Get today's date (IST)
    today = today_ist()
    try:
        cutoff_time = datetime.strptime(time_str, "%H:%M:%S").time()
    except ValueError:
        print(f"✗ Invalid time format: {time_str}. Use HH:MM:SS")
        return

    cutoff_datetime = datetime.combine(today, cutoff_time)
    
    print("=" * 70)
    print("DATABASE CLEANUP - DELETE RECORDS FROM TIME")
    print("=" * 70)
    print(f"\nCutoff timestamp: {cutoff_datetime}")
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        ph = _get_placeholder()
        
        # Tables that contain time-series data (timestamp column)
        tables = [
            'option_chain_snapshots',
            'ml_features',
            'vix_term_structure',
            'macro_signals',
            'order_book_depth_snapshots',
            'paper_trading_metrics'
        ]

        # Count records to be deleted
        total_to_delete = 0
        table_counts = {}
        
        print("\nChecking records to delete...")
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE timestamp >= {ph}", (cutoff_datetime,))
                count = cursor.fetchone()[0]
                if count > 0:
                    print(f"  - {table}: {count:,} records")
                    table_counts[table] = count
                    total_to_delete += count
                else:
                    pass
            except Exception as e:
                # print(f"  ⚠ {table}: Error checking table ({e})")
                pass

        if total_to_delete == 0:
            print("\n✓ No records found to delete.")
            return
        
        # Confirm deletion
        print(f"\n⚠️  WARNING: This will delete {total_to_delete:,} records permanently!")
        confirm = input(f"Type 'YES' to confirm deleting records from {time_str} onwards: ")
        
        if confirm != 'YES':
            print("\n✗ Deletion cancelled.")
            return
        
        # Delete records
        print("\nDeleting records...")
        for table in tables:
            if table in table_counts:
                try:
                    cursor.execute(f"DELETE FROM {table} WHERE timestamp >= {ph}", (cutoff_datetime,))
                    deleted = cursor.rowcount
                    print(f"  ✓ Deleted {deleted:,} records from {table}")
                except Exception as e:
                    print(f"  ✗ Failed to delete from {table}: {e}")
        
        # Update exchange_metadata to reflect new latest time
        print("\nUpdating exchange metadata...")
        exchanges = ['NSE', 'BSE', 'NSE_MONTHLY']
        
        for exchange in exchanges:
            try:
                cursor.execute(f"SELECT MAX(timestamp) FROM option_chain_snapshots WHERE exchange = {ph}", (exchange,))
                result = cursor.fetchone()
                last_time = result[0] if result else None
                
                if last_time:
                    cursor.execute(f'''
                        UPDATE exchange_metadata
                        SET last_update_time = {ph}, updated_at = {ph}
                        WHERE exchange = {ph}
                    ''', (last_time, datetime.now(), exchange))
                    print(f"  ✓ Updated {exchange} metadata to {last_time}")
                else:
                    print(f"  - No records left for {exchange}")
                    
            except Exception:
                pass
        
        conn.commit()
        
        # Vacuum if SQLite
        try:
            config = get_config()
            if config.db_type != 'postgres':
                print("\nOptimizing database...")
                conn.execute('VACUUM')
                print("✓ Database optimized")
        except Exception:
            pass
        
        print("\n" + "=" * 70)
        print("✓ CLEANUP COMPLETE")
        print("=" * 70)

    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            release_db_connection(conn)

if __name__ == '__main__':
    import sys
    # Default time or argument
    time_arg = "15:25:00"
    if len(sys.argv) > 1:
        time_arg = sys.argv[1]
        
    print("USAGE: python delete_records.py [HH:MM:SS]")
    print(f"Default time: {time_arg}\n")
    
    delete_records_from_time(time_arg)
