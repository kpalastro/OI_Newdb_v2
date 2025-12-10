# Delete Records from Previous Days After Market Hours
# Deletes all records from specified number of previous days starting from 15:30 (after market close)
# to next day 09:14 (just before market open at 09:15)

from datetime import datetime, timedelta
import logging

from time_utils import today_ist
from database_new import get_db_connection, release_db_connection, _get_placeholder, get_config

def delete_from_time(days_back=1, start_hour=15, start_minute=30, end_hour=9, end_minute=14):
    """
    Delete records from previous days from market close (15:30) to next day market open (09:14).
    """
    
    if days_back < 1:
        print("✗ Number of days must be at least 1")
        return
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        ph = _get_placeholder()
        
        # Tables that have timestamp columns
        timestamp_tables = [
            ('option_chain_snapshots', 'timestamp'),
            ('ml_features', 'timestamp'),
            ('vix_term_structure', 'timestamp'),
            ('macro_signals', 'timestamp'),
            ('order_book_depth_snapshots', 'timestamp'),
            ('paper_trading_metrics', 'timestamp'),
            ('training_batches', 'created_at')
        ]
        
        # Calculate date ranges for each day in IST
        today = today_ist()
        date_ranges = []
        
        for i in range(1, days_back + 1):
            target_date = today - timedelta(days=i)
            next_date = target_date + timedelta(days=1)
            
            # Start: Day N at 15:30:00 (market close)
            start_datetime = datetime(target_date.year, target_date.month, target_date.day, start_hour, start_minute, 0)
            # End: Day N+1 at 09:14:59 (just before market open at 09:15)
            end_datetime = datetime(next_date.year, next_date.month, next_date.day, end_hour, end_minute, 59)
            
            date_ranges.append((target_date, next_date, start_datetime, end_datetime))
        
        print("=" * 70)
        print("DELETE DATABASE RECORDS - AFTER MARKET HOURS CLEANUP")
        print("=" * 70)
        print(f"\nWill delete records from {days_back} day(s) back:")
        print("(From market close to next day market open)")
        for target_date, next_date, start_dt, end_dt in date_ranges:
            print(f"  - {target_date.strftime('%Y-%m-%d')} {start_dt.strftime('%H:%M:%S')} to {next_date.strftime('%Y-%m-%d')} {end_dt.strftime('%H:%M:%S')}")
        
        # Count records to be deleted
        total_to_delete = 0
        table_counts = {}
        day_counts = {}
        
        print("\nCounting records to delete...")
        for table_name, timestamp_col in timestamp_tables:
            table_total = 0
            for target_date, next_date, start_dt, end_dt in date_ranges:
                try:
                    start_str = start_dt.strftime('%Y-%m-%d %H:%M:%S')
                    end_str = end_dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    cursor.execute(f'''
                        SELECT COUNT(*) FROM {table_name}
                        WHERE {timestamp_col} >= {ph} AND {timestamp_col} <= {ph}
                    ''', (start_str, end_str))
                    
                    count = cursor.fetchone()[0]
                    table_total += count
                    
                    # Track per day range
                    day_key = f"{target_date.strftime('%Y-%m-%d')} to {next_date.strftime('%Y-%m-%d')}"
                    if day_key not in day_counts:
                        day_counts[day_key] = {}
                    if table_name not in day_counts[day_key]:
                        day_counts[day_key][table_name] = 0
                    day_counts[day_key][table_name] += count
                    
                except Exception as e:
                    pass
            
            table_counts[table_name] = table_total
            total_to_delete += table_total
            if table_total > 0:
                print(f"  - {table_name}: {table_total:,} records")
        
        if total_to_delete == 0:
            print("\n✓ No records found to delete.")
            return
        
        print(f"\nTotal records to delete: {total_to_delete:,}")
        
        # Show breakdown by day
        if day_counts:
            print("\nBreakdown by day:")
            print("-" * 70)
            for day_key in sorted(day_counts.keys()):
                day_total = sum(day_counts[day_key].values())
                print(f"  {day_key}: {day_total:,} records")
                for table_name, count in day_counts[day_key].items():
                    if count > 0:
                        print(f"    - {table_name}: {count:,}")
            print("-" * 70)
        
        # Confirm deletion
        print(f"\n⚠️  WARNING: This will delete {total_to_delete:,} records permanently!")
        confirm = input("\nType 'DELETE' to confirm: ")
        
        if confirm != 'DELETE':
            print("\n✗ Deletion cancelled. No changes made.")
            return
        
        print("\nDeleting records...")
        
        deleted_counts = {}
        for table_name, timestamp_col in timestamp_tables:
            table_deleted = 0
            for target_date, next_date, start_dt, end_dt in date_ranges:
                try:
                    start_str = start_dt.strftime('%Y-%m-%d %H:%M:%S')
                    end_str = end_dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    cursor.execute(f'''
                        DELETE FROM {table_name}
                        WHERE {timestamp_col} >= {ph} AND {timestamp_col} <= {ph}
                    ''', (start_str, end_str))
                    
                    deleted_count = cursor.rowcount
                    table_deleted += deleted_count
                    
                except Exception as e:
                    print(f"  ⚠ Skipped {table_name}: {e}")
            
            deleted_counts[table_name] = table_deleted
            if table_deleted > 0:
                print(f"  ✓ Deleted {table_deleted:,} records from {table_name}")
        
        conn.commit()
        
        # Vacuum if SQLite
        try:
            config = get_config()
            if config.db_type != 'postgres':
                print("\nOptimizing database...")
                conn.execute('VACUUM')
                print("✓ Database optimized")
        except:
            pass
        
        total_deleted = sum(deleted_counts.values())
        print(f"\n✓ Successfully deleted {total_deleted:,} records")
        
        print("\n" + "=" * 70)
        print("✓ CLEANUP COMPLETE!")
        print("=" * 70)

    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            release_db_connection(conn)

if __name__ == '__main__':
    print("=" * 70)
    print("DELETE RECORDS AFTER MARKET HOURS")
    print("=" * 70)
    print("\nThis script deletes records from previous days:")
    print("  - From: 15:30 (market close) of Day N")
    print("  - To:   09:14 (just before market open) of Day N+1")
    print("\nThis helps remove duplicate/garbage data saved during program")
    print("modifications after market hours (15:30) until next market open (09:15).")
    print()
    
    try:
        days_input = input("Enter number of days to go back (default: 1): ").strip()
        days_back = int(days_input) if days_input else 1
        
        if days_back < 1:
            print("✗ Number of days must be at least 1. Using default: 1")
            days_back = 1
    except ValueError:
        print("✗ Invalid input. Using default: 1 day")
        days_back = 1
    
    delete_from_time(days_back=days_back, start_hour=15, start_minute=30, end_hour=9, end_minute=14)
