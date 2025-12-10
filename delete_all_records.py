# Delete All Records from Database
# Complete database cleanup - keeps schema, deletes all data
# Updated to match latest database_new.py schema (8 tables)

from database_new import get_db_connection, release_db_connection, get_config

def delete_all_records():
    """Delete all records from database tables."""
    
    config = get_config()
    db_name = config.db_name if config.db_type == 'postgres' else "db_new.db"
    
    print("=" * 70)
    print("DELETE ALL DATABASE RECORDS")
    print(f"Target Database: {db_name}")
    print("=" * 70)
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Count existing records - All 8 tables in the database
        tables = [
            'option_chain_snapshots',
            'ml_features',
            'exchange_metadata',
            'training_batches',
            'vix_term_structure',
            'macro_signals',
            'order_book_depth_snapshots',
            'paper_trading_metrics'
        ]
        
        print(f"\nCurrent records:")
        total_records = 0
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"  - {table}: {count:,}")
                total_records += count
            except Exception as e:
                # print(f"  - {table}: {e} (skipping)")
                pass
        
        if total_records == 0:
            print("\n✓ Database is already empty.")
            return
        
        # Confirm deletion
        print(f"\n⚠️  WARNING: This will DELETE ALL {total_records:,} RECORDS permanently!")
        print("⚠️  The database schema will be kept, only data will be deleted.")
        confirm = input("\nType 'DELETE ALL' to confirm: ")
        
        if confirm != 'DELETE ALL':
            print("\n✗ Deletion cancelled. No changes made.")
            return
        
        print("\nDeleting all records...")
        
        deleted_counts = {}
        for table in tables:
            try:
                cursor.execute(f'DELETE FROM {table}')
                deleted_count = cursor.rowcount
                deleted_counts[table] = deleted_count
                if deleted_count > 0:
                    print(f"  ✓ Deleted {deleted_count:,} records from {table}")
            except Exception as e:
                print(f"  ⚠ Skipped {table}: {e}")
        
        # Commit changes
        conn.commit()
        
        # Vacuum to reclaim disk space (SQLite only)
        try:
            if config.db_type != 'postgres':
                print("\nOptimizing database and reclaiming disk space...")
                conn.execute('VACUUM')
                print("  ✓ Database optimized")
        except:
            pass
        
        # Verify deletion
        print("\nVerifying deletion...")
        remaining_total = 0
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                verify_count = cursor.fetchone()[0]
                if verify_count > 0:
                    print(f"  ⚠ {table}: {verify_count} records still remain")
                    remaining_total += verify_count
            except:
                pass
        
        total_deleted = sum(deleted_counts.values())
        print("\n" + "=" * 70)
        print("✓ DELETION COMPLETE!")
        print("=" * 70)
        print(f"\nRecords deleted: {total_deleted:,}")
        print(f"Remaining records: {remaining_total}")
        print("\nDatabase schema is intact and ready for fresh data.")
        print("=" * 70)

    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            release_db_connection(conn)

if __name__ == '__main__':
    delete_all_records()
