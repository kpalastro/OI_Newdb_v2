# Quick Database Viewer
# View database contents in terminal

import sqlite3
import pandas as pd
from datetime import date

from export_to_csv import export_all_data
from time_utils import today_ist

DB_FILE = "oi_tracker.db"

BASE_COLUMN_SQL = """
    timestamp,
    exchange,
    strike,
    option_type,
    symbol,
    token,
    oi,
    ltp,
    underlying_price,
    moneyness,
    time_to_expiry,
    pct_change_3m,
    pct_change_5m,
    pct_change_10m,
    pct_change_15m,
    pct_change_30m,
    vix,
    iv,
    volume,
    strike_minus_100_oi_change,
    strike_plus_100_oi_change,
    created_at,
    updated_at
"""

DISPLAY_COLUMNS = [
    "timestamp", "exchange", "strike", "option_type", "moneyness",
    "oi", "ltp", "underlying_price", "pct_change_3m", "pct_change_5m",
    "pct_change_10m", "pct_change_15m", "pct_change_30m",
    "iv", "volume", "vix"
]


def _print_df(df: pd.DataFrame, title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    if df.empty:
        print("\n(No data available)")
    else:
        print("\n" + df.to_string(index=False))

def view_database():
    """Display database contents in terminal."""
    
    conn = sqlite3.connect(DB_FILE)
    
    print("=" * 70)
    print("DATABASE VIEWER")
    print("=" * 70)
    
    # 1. Database statistics
    print("\n" + "=" * 70)
    print("DATABASE STATISTICS")
    print("=" * 70)
    
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM option_chain_snapshots")
    total = cursor.fetchone()[0]
    print(f"\nTotal Records: {total:,}")
    
    df_exchange = pd.read_sql_query(
        """
        SELECT exchange, COUNT(*) AS records
        FROM option_chain_snapshots
        GROUP BY exchange
        """,
        conn,
    )
    _print_df(df_exchange, "Records by Exchange")
    
    df_dates = pd.read_sql_query(
        """
        SELECT MIN(timestamp) AS first_record,
               MAX(timestamp) AS last_record
        FROM option_chain_snapshots
        """,
        conn,
    )
    _print_df(df_dates, "Date Range")
    
    # 2. Exchange metadata
    df_meta = pd.read_sql_query("SELECT * FROM exchange_metadata", conn)
    _print_df(df_meta, "EXCHANGE METADATA (Current Values)")
    
    # 3. Latest snapshots
    df_latest = pd.read_sql_query(
        f"""
        SELECT {BASE_COLUMN_SQL}
        FROM option_chain_snapshots
        WHERE timestamp = (SELECT MAX(timestamp) FROM option_chain_snapshots)
        ORDER BY exchange, strike, option_type
        """,
        conn,
    )
    _print_df(df_latest[DISPLAY_COLUMNS], "LATEST SNAPSHOTS (Most Recent Data)")
    
    # 4. Today's summary
    df_today_summary = pd.read_sql_query(
        """
        SELECT 
            exchange,
            COUNT(*) AS records,
            MIN(timestamp) AS first_update,
            MAX(timestamp) AS last_update,
            AVG(pct_change_10m) AS avg_10m_change,
            MAX(pct_change_10m) AS max_10m_increase,
            MIN(pct_change_10m) AS max_10m_decrease,
            AVG(pct_change_3m) AS avg_3m_change,
            MAX(pct_change_3m) AS max_3m_increase,
            MIN(pct_change_3m) AS max_3m_decrease,
            AVG(iv) AS avg_iv,
            AVG(volume) AS avg_volume,
            AVG(vix) AS avg_vix
        FROM option_chain_snapshots
        WHERE DATE(timestamp) = DATE('now')
        GROUP BY exchange
        """,
        conn,
    )
    _print_df(df_today_summary, "TODAY'S SUMMARY")
    
    conn.close()
    
    print("\n" + "=" * 70)
    print("✓ VIEW COMPLETE")
    print("=" * 70)

def export_today_to_csv():
    """Export only today's data to CSV."""
    
    conn = sqlite3.connect(DB_FILE)
    today = today_ist()
    
    print("\n" + "=" * 70)
    print("EXPORTING TODAY'S DATA TO CSV")
    print("=" * 70)
    
    df = pd.read_sql_query(
        f"""
        SELECT {BASE_COLUMN_SQL}
        FROM option_chain_snapshots
        WHERE DATE(timestamp) = DATE('now')
        ORDER BY timestamp DESC, exchange, strike, option_type
        """,
        conn,
    )
    
    filename = f"today_data_{today.strftime('%Y%m%d')}.csv"
    df.to_csv(filename, index=False)
    
    print(f"\n✓ Exported {len(df)} records to: {filename}")
    print("\nYou can open this file in:")
    print("  - Microsoft Excel")
    print("  - Google Sheets")
    print("  - Any spreadsheet application")
    
    conn.close()
    
    return filename

def export_custom_query(query, filename):
    """Export custom SQL query to CSV."""
    
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(query, conn)
    df.to_csv(filename, index=False)
    conn.close()
    
    print(f"✓ Exported {len(df)} records to: {filename}")
    return filename

if __name__ == '__main__':
    # Show menu
    print("\n" + "=" * 70)
    print("DATABASE EXPORT MENU")
    print("=" * 70)
    print("\n1. View database in terminal")
    print("2. Export today's data to CSV")
    print("3. Export all data to multiple CSV files")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        view_database()
    elif choice == '2':
        filename = export_today_to_csv()
        print(f"\n✓ File created: {filename}")
    elif choice == '3':
        export_all_data()
    elif choice == '4':
        print("\nExiting...")
    else:
        print("\nInvalid choice. Please run again.")

