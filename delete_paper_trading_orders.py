# Delete All Paper Trading Orders from Database
# Deletes all system-generated orders stored in paper_trading_metrics table

from database_new import get_db_connection, release_db_connection, get_config, db_lock, _get_placeholder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def delete_all_paper_trading_orders():
    """Delete all records from paper_trading_metrics table (system-generated orders)."""
    
    config = get_config()
    db_name = config.db_name if config.db_type == 'postgres' else "db_new.db"
    
    print("=" * 70)
    print("DELETE ALL PAPER TRADING ORDERS")
    print("=" * 70)
    print(f"Target Database: {db_name}")
    print(f"Database Type: {config.db_type.upper()}")
    print("=" * 70)
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        ph = _get_placeholder()
        
        # Count existing records
        print("\nChecking paper_trading_metrics table...")
        try:
            cursor.execute("SELECT COUNT(*) FROM paper_trading_metrics")
            total_records = cursor.fetchone()[0]
            print(f"  Total records found: {total_records:,}")
        except Exception as e:
            print(f"  ⚠ Error counting records: {e}")
            print("  This table may not exist or may be empty.")
            return
        
        if total_records == 0:
            print("\n✓ No paper trading orders found. Database is already clean.")
            return
        
        # Get breakdown by exchange
        print("\nBreakdown by exchange:")
        try:
            cursor.execute(f"SELECT exchange, COUNT(*) as count FROM paper_trading_metrics GROUP BY exchange ORDER BY count DESC")
            exchange_counts = cursor.fetchall()
            for exchange, count in exchange_counts:
                print(f"  - {exchange}: {count:,} orders")
        except Exception as e:
            print(f"  ⚠ Could not get exchange breakdown: {e}")
        
        # Get breakdown by executed status
        print("\nBreakdown by execution status:")
        try:
            cursor.execute(f"SELECT executed, COUNT(*) as count FROM paper_trading_metrics GROUP BY executed ORDER BY count DESC")
            execution_counts = cursor.fetchall()
            for executed, count in execution_counts:
                status = "EXECUTED" if executed else "REJECTED"
                print(f"  - {status}: {count:,} orders")
        except Exception as e:
            print(f"  ⚠ Could not get execution breakdown: {e}")
        
        # Confirm deletion
        print("\n" + "=" * 70)
        print("⚠️  WARNING: This will DELETE ALL paper trading orders permanently!")
        print("⚠️  This includes all executed and rejected trade records.")
        print("⚠️  The table schema will be kept, only data will be deleted.")
        print("=" * 70)
        confirm = input("\nType 'DELETE ORDERS' to confirm: ")
        
        if confirm != 'DELETE ORDERS':
            print("\n✗ Deletion cancelled. No changes made.")
            return
        
        # Delete all records with proper locking
        print("\nDeleting all paper trading orders...")
        deleted_count = 0
        with db_lock:
            try:
                cursor.execute("DELETE FROM paper_trading_metrics")
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    print(f"  ✓ Successfully deleted {deleted_count:,} records")
                else:
                    print("  ⚠ No records were deleted (table may have been empty)")
                    
            except Exception as e:
                conn.rollback()
                print(f"  ✗ Error during deletion: {e}")
                logging.error(f"Failed to delete paper trading orders: {e}", exc_info=True)
                return
        
        # Verify deletion
        print("\nVerifying deletion...")
        try:
            cursor.execute("SELECT COUNT(*) FROM paper_trading_metrics")
            remaining = cursor.fetchone()[0]
            
            if remaining == 0:
                print("  ✓ Verification successful: All records deleted")
            else:
                print(f"  ⚠ Warning: {remaining} records still remain in the table")
        except Exception as e:
            print(f"  ⚠ Could not verify deletion: {e}")
        
        # Vacuum to reclaim disk space (SQLite only)
        try:
            if config.db_type != 'postgres':
                print("\nOptimizing database and reclaiming disk space...")
                cursor.execute('VACUUM')
                print("  ✓ Database optimized")
        except Exception as e:
            logging.debug(f"Vacuum skipped: {e}")
        
        print("\n" + "=" * 70)
        print("✓ DELETION COMPLETE!")
        print("=" * 70)
        print(f"\nRecords deleted: {deleted_count:,}")
        print("\nAll system-generated paper trading orders have been removed.")
        print("The table is now empty and ready for new orders.")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ AN ERROR OCCURRED: {e}")
        logging.error(f"Error in delete_all_paper_trading_orders: {e}", exc_info=True)
        if conn:
            conn.rollback()
    finally:
        if conn:
            release_db_connection(conn)

if __name__ == '__main__':
    delete_all_paper_trading_orders()

