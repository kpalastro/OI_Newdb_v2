#!/usr/bin/env python3
"""
Backfill BSE sentiment scores from NSE sentiment data (proxy) and/or real BSE API data.

This script has two phases:
1. NSE Proxy Phase: Copy NSE sentiment_score_50 → bse_sentiment_score_100
                    Copy NSE sentiment_score_100 → bse_sentiment_score_200
2. Real BSE Phase: Fetch real BSE sentiment from API and update

Usage:
    python backfill_bse_sentiment.py [--phase PHASE] [--start-date DATE] [--end-date DATE] [--batch-size N]
    
Examples:
    # Backfill using NSE proxy data
    python backfill_bse_sentiment.py --phase nse_proxy
    
    # Backfill using real BSE API data (for recent dates)
    python backfill_bse_sentiment.py --phase real_bse --start-date 2025-01-01
    
    # Backfill both phases
    python backfill_bse_sentiment.py --phase both
"""

import argparse
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Tuple

from database_new import get_db_connection, release_db_connection, get_config, save_macro_signals
from time_utils import now_ist

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backfill_nse_proxy(start_date: Optional[str] = None, end_date: Optional[str] = None, 
                       batch_size: int = 1000) -> Tuple[int, int]:
    """
    Backfill BSE sentiment scores using NSE sentiment data as proxy.
    
    For each BSE ml_features record:
    - Find matching NSE record (same timestamp)
    - Copy NSE sentiment_score_50 → BSE bse_sentiment_score_100
    - Copy NSE sentiment_score_100 → BSE bse_sentiment_score_200
    - Update macro_signals and sync to ml_features
    
    Args:
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        batch_size: Number of records to process per batch
    
    Returns:
        Tuple of (records_updated, records_skipped)
    """
    config = get_config()
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build date filter
        date_filter = ""
        date_params = []
        if start_date:
            date_filter += " AND mf.timestamp >= %s"
            date_params.append(start_date)
        if end_date:
            date_filter += " AND mf.timestamp <= %s"
            date_params.append(end_date + " 23:59:59")
        
        logger.info("="*60)
        logger.info("BSE Sentiment Backfill - NSE Proxy Phase")
        logger.info("="*60)
        logger.info(f"Date range: {start_date or 'all'} to {end_date or 'all'}")
        logger.info(f"Batch size: {batch_size}")
        
        updated_count = 0
        skipped_count = 0
        total_processed = 0
        
        # Loop until all records are processed
        while True:
            # Find BSE records missing sentiment data
            query = f"""
                SELECT 
                    mf.timestamp,
                    mf.exchange,
                    mse.sentiment_score_50 as nse_score_50,
                    mse.sentiment_score_100 as nse_score_100
                FROM ml_features mf
                LEFT JOIN ml_features mse 
                    ON DATE_TRUNC('minute', mf.timestamp) = DATE_TRUNC('minute', mse.timestamp)
                    AND mse.exchange = 'NSE'
                    AND mse.sentiment_score_50 IS NOT NULL
                    AND mse.sentiment_score_100 IS NOT NULL
                WHERE mf.exchange = 'BSE'
                  AND (mf.bse_sentiment_score_100 IS NULL OR mf.bse_sentiment_score_200 IS NULL)
                  {date_filter}
                ORDER BY mf.timestamp
                LIMIT %s
            """
            
            cursor.execute(query, date_params + [batch_size])
            records = cursor.fetchall()
            
            if not records:
                logger.info("No more BSE records found missing sentiment data")
                break
            
            logger.info(f"Processing batch: {len(records)} records (Total processed so far: {total_processed})")
            batch_updated = 0
            batch_skipped = 0
            
            for record in records:
                timestamp, exchange, nse_score_50, nse_score_100 = record
                
                # Skip if no NSE data available
                if nse_score_50 is None or nse_score_100 is None:
                    batch_skipped += 1
                    skipped_count += 1
                    logger.debug(f"Skipping {timestamp}: No matching NSE data")
                    continue
                
                try:
                    # Update macro_signals first
                    # Check if macro_signals record exists for this timestamp
                    check_query = """
                        SELECT id FROM macro_signals
                        WHERE exchange = 'BSE'
                          AND DATE_TRUNC('minute', timestamp) = DATE_TRUNC('minute', %s)
                        LIMIT 1
                    """
                    cursor.execute(check_query, (timestamp,))
                    macro_record = cursor.fetchone()
                    
                    if macro_record:
                        # Update existing record
                        update_macro_query = """
                            UPDATE macro_signals
                            SET 
                                bse_sentiment_score_100 = %s,
                                bse_sentiment_score_200 = %s,
                                bse_sentiment_confidence_100 = 75.0,
                                bse_sentiment_confidence_200 = 75.0,
                                bse_trin_100 = 1.0,
                                bse_trin_200 = 1.0
                            WHERE id = %s
                        """
                        cursor.execute(update_macro_query, (nse_score_50, nse_score_100, macro_record[0]))
                    else:
                        # Insert new record
                        save_macro_signals(
                            exchange='BSE',
                            bse_sentiment_score_100=nse_score_50,
                            bse_sentiment_score_200=nse_score_100,
                            bse_sentiment_confidence_100=75.0,
                            bse_sentiment_confidence_200=75.0,
                            bse_trin_100=1.0,
                            bse_trin_200=1.0,
                            timestamp=timestamp,
                            metadata={'source': 'nse_proxy_backfill', 'backfilled_at': now_ist().isoformat()}
                        )
                    
                    # Update ml_features
                    # Use DATE_TRUNC for timestamp matching to handle any microsecond differences
                    update_ml_query = """
                        UPDATE ml_features
                        SET 
                            bse_sentiment_score_100 = %s,
                            bse_sentiment_score_200 = %s
                        WHERE DATE_TRUNC('minute', timestamp) = DATE_TRUNC('minute', %s)
                          AND exchange = %s
                    """
                    cursor.execute(update_ml_query, (nse_score_50, nse_score_100, timestamp, exchange))
                
                    rows_updated = cursor.rowcount
                    if rows_updated == 0:
                        logger.warning(f"No rows updated for {timestamp} ({exchange}) - timestamp might not match")
                        batch_skipped += 1
                        skipped_count += 1
                        continue
                    elif rows_updated > 1:
                        logger.warning(f"Multiple rows ({rows_updated}) updated for {timestamp} ({exchange})")
                    
                    batch_updated += 1
                    updated_count += 1
                    logger.debug(f"Updated {timestamp}: bse_sentiment_score_100={nse_score_50}, bse_sentiment_score_200={nse_score_100}")
                    
                except Exception as e:
                    logger.error(f"Error processing record at {timestamp} ({exchange}): {e}", exc_info=True)
                    batch_skipped += 1
                    skipped_count += 1
                    continue
            
            # Commit after each batch
            conn.commit()
            total_processed += len(records)
            logger.info(f"Batch complete: Updated {batch_updated}, Skipped {batch_skipped} (Total: {updated_count} updated, {skipped_count} skipped)")
            
            # If we processed fewer records than batch_size, we're done
            if len(records) < batch_size:
                break
        
        release_db_connection(conn)
        
        logger.info("="*60)
        logger.info(f"Backfill complete: Updated {updated_count} records, Skipped {skipped_count} records")
        logger.info(f"Total records processed: {total_processed}")
        logger.info("="*60)
        
        return updated_count, skipped_count
        
    except Exception as e:
        logger.error(f"Error in backfill: {e}", exc_info=True)
        if 'conn' in locals():
            release_db_connection(conn)
        raise


def backfill_real_bse(start_date: Optional[str] = None, end_date: Optional[str] = None,
                     batch_size: int = 100) -> Tuple[int, int]:
    """
    Backfill BSE sentiment scores using real BSE API data.
    
    This fetches real sentiment data from BSE API for each timestamp.
    Note: This is slower and may hit rate limits, so use for recent dates only.
    
    Args:
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        batch_size: Number of records to process per batch
    
    Returns:
        Tuple of (records_updated, records_skipped)
    """
    try:
        from data_ingestion.macro_loader import fetch_bse_sentiment
        import time
    except ImportError:
        logger.error("Failed to import fetch_bse_sentiment. Make sure macro_loader is available.")
        return 0, 0
    
    config = get_config()
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build date filter
        date_filter = ""
        date_params = []
        if start_date:
            date_filter += " AND mf.timestamp >= %s"
            date_params.append(start_date)
        if end_date:
            date_filter += " AND mf.timestamp <= %s"
            date_params.append(end_date + " 23:59:59")
        
        logger.info("="*60)
        logger.info("BSE Sentiment Backfill - Real BSE API Phase")
        logger.info("="*60)
        logger.info(f"Date range: {start_date or 'all'} to {end_date or 'all'}")
        logger.info(f"Batch size: {batch_size}")
        logger.warning("This phase fetches real data from BSE API - may be slow due to rate limits")
        
        # Find BSE records missing sentiment data (prioritize recent dates)
        query = f"""
            SELECT DISTINCT DATE_TRUNC('minute', mf.timestamp) as minute_timestamp
            FROM ml_features mf
            WHERE mf.exchange = 'BSE'
              AND (mf.bse_sentiment_score_100 IS NULL OR mf.bse_sentiment_score_200 IS NULL)
              {date_filter}
            ORDER BY minute_timestamp DESC
            LIMIT %s
        """
        
        cursor.execute(query, date_params + [batch_size])
        timestamps = cursor.fetchall()
        
        if not timestamps:
            logger.info("No BSE records found missing sentiment data")
            return 0, 0
        
        logger.info(f"Found {len(timestamps)} unique timestamps to process")
        
        updated_count = 0
        skipped_count = 0
        
        for (timestamp,) in timestamps:
            try:
                # Fetch real BSE sentiment
                logger.info(f"Fetching BSE sentiment for {timestamp}...")
                bse_sentiment = fetch_bse_sentiment()
                
                if not bse_sentiment or not any(v is not None for v in bse_sentiment.values()):
                    logger.warning(f"No valid BSE sentiment data for {timestamp}")
                    skipped_count += 1
                    time.sleep(1)  # Rate limiting
                    continue
                
                bse_score_100 = bse_sentiment.get('bse_sentiment_score_100')
                bse_score_200 = bse_sentiment.get('bse_sentiment_score_200')
                bse_conf_100 = bse_sentiment.get('bse_sentiment_confidence_100')
                bse_conf_200 = bse_sentiment.get('bse_sentiment_confidence_200')
                bse_trin_100 = bse_sentiment.get('bse_trin_100')
                bse_trin_200 = bse_sentiment.get('bse_trin_200')
                
                # Update all ml_features records for this timestamp
                update_query = """
                    UPDATE ml_features
                    SET 
                        bse_sentiment_score_100 = %s,
                        bse_sentiment_score_200 = %s
                    WHERE exchange = 'BSE'
                      AND DATE_TRUNC('minute', timestamp) = DATE_TRUNC('minute', %s)
                """
                cursor.execute(update_query, (bse_score_100, bse_score_200, timestamp))
                rows_updated = cursor.rowcount
                
                # Update or insert macro_signals
                check_query = """
                    SELECT id FROM macro_signals
                    WHERE exchange = 'BSE'
                      AND DATE_TRUNC('minute', timestamp) = DATE_TRUNC('minute', %s)
                    LIMIT 1
                """
                cursor.execute(check_query, (timestamp,))
                macro_record = cursor.fetchone()
                
                if macro_record:
                    update_macro_query = """
                        UPDATE macro_signals
                        SET 
                            bse_sentiment_score_100 = %s,
                            bse_sentiment_score_200 = %s,
                            bse_sentiment_confidence_100 = %s,
                            bse_sentiment_confidence_200 = %s,
                            bse_trin_100 = %s,
                            bse_trin_200 = %s
                        WHERE id = %s
                    """
                    cursor.execute(update_macro_query, (
                        bse_score_100, bse_score_200, bse_conf_100, bse_conf_200,
                        bse_trin_100, bse_trin_200, macro_record[0]
                    ))
                else:
                    save_macro_signals(
                        exchange='BSE',
                        bse_sentiment_score_100=bse_score_100,
                        bse_sentiment_score_200=bse_score_200,
                        bse_sentiment_confidence_100=bse_conf_100,
                        bse_sentiment_confidence_200=bse_conf_200,
                        bse_trin_100=bse_trin_100,
                        bse_trin_200=bse_trin_200,
                        timestamp=timestamp,
                        metadata={'source': 'bse_api_backfill', 'backfilled_at': now_ist().isoformat()}
                    )
                
                updated_count += rows_updated
                conn.commit()
                
                logger.info(f"Updated {rows_updated} records for {timestamp}")
                
                # Rate limiting - wait 2 seconds between API calls
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error processing timestamp {timestamp}: {e}")
                skipped_count += 1
                time.sleep(1)
                continue
        
        release_db_connection(conn)
        
        logger.info("="*60)
        logger.info(f"Backfill complete: Updated {updated_count} records, Skipped {skipped_count} records")
        logger.info("="*60)
        
        return updated_count, skipped_count
        
    except Exception as e:
        logger.error(f"Error in backfill: {e}", exc_info=True)
        if 'conn' in locals():
            release_db_connection(conn)
        raise


def preview_backfill(phase: str = 'nse_proxy', start_date: Optional[str] = None, 
                     end_date: Optional[str] = None) -> None:
    """
    Preview how many records would be backfilled without actually updating.
    """
    config = get_config()
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        date_filter = ""
        date_params = []
        if start_date:
            date_filter += " AND mf.timestamp >= %s"
            date_params.append(start_date)
        if end_date:
            date_filter += " AND mf.timestamp <= %s"
            date_params.append(end_date + " 23:59:59")
        
        if phase == 'nse_proxy':
            query = f"""
                SELECT 
                    COUNT(*) as total_bse_records,
                    COUNT(mse.sentiment_score_50) as records_with_nse_data,
                    COUNT(mf.bse_sentiment_score_100) as records_already_filled
                FROM ml_features mf
                LEFT JOIN ml_features mse 
                    ON DATE_TRUNC('minute', mf.timestamp) = DATE_TRUNC('minute', mse.timestamp)
                    AND mse.exchange = 'NSE'
                WHERE mf.exchange = 'BSE'
                  AND (mf.bse_sentiment_score_100 IS NULL OR mf.bse_sentiment_score_200 IS NULL)
                  {date_filter}
            """
        else:
            query = f"""
                SELECT 
                    COUNT(DISTINCT DATE_TRUNC('minute', timestamp)) as unique_timestamps,
                    COUNT(*) as total_records
                FROM ml_features
                WHERE exchange = 'BSE'
                  AND (bse_sentiment_score_100 IS NULL OR bse_sentiment_score_200 IS NULL)
                  {date_filter}
            """
        
        cursor.execute(query, date_params)
        results = cursor.fetchall()
        release_db_connection(conn)
        
        logger.info("\n" + "="*60)
        logger.info(f"Preview: Records that would be backfilled ({phase})")
        logger.info("="*60)
        
        if phase == 'nse_proxy':
            total, with_nse, already_filled = results[0]
            logger.info(f"Total BSE records missing sentiment: {total}")
            logger.info(f"Records with matching NSE data: {with_nse}")
            logger.info(f"Records already filled: {already_filled}")
            logger.info(f"Records that would be updated: {with_nse - already_filled}")
        else:
            unique_ts, total = results[0]
            logger.info(f"Unique timestamps to process: {unique_ts}")
            logger.info(f"Total records to update: {total}")
        
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error previewing backfill: {e}", exc_info=True)
        if 'conn' in locals():
            release_db_connection(conn)
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Backfill BSE sentiment scores from NSE proxy or real BSE API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview NSE proxy backfill
  python backfill_bse_sentiment.py --phase nse_proxy --preview
  
  # Backfill using NSE proxy (all dates)
  python backfill_bse_sentiment.py --phase nse_proxy
  
  # Backfill using NSE proxy (date range)
  python backfill_bse_sentiment.py --phase nse_proxy --start-date 2024-01-01 --end-date 2024-12-31
  
  # Backfill using real BSE API (recent dates only)
  python backfill_bse_sentiment.py --phase real_bse --start-date 2025-01-01
  
  # Backfill both phases
  python backfill_bse_sentiment.py --phase both
        """
    )
    parser.add_argument('--phase', type=str, default='nse_proxy',
                       choices=['nse_proxy', 'real_bse', 'both'],
                       help='Backfill phase: nse_proxy (copy from NSE), real_bse (fetch from API), or both')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date in YYYY-MM-DD format')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Number of records to process per batch (default: 1000)')
    parser.add_argument('--preview', action='store_true',
                       help='Preview how many records would be updated without actually updating')
    
    args = parser.parse_args()
    
    if args.preview:
        if args.phase in ('nse_proxy', 'both'):
            preview_backfill('nse_proxy', args.start_date, args.end_date)
        if args.phase in ('real_bse', 'both'):
            preview_backfill('real_bse', args.start_date, args.end_date)
    else:
        if args.phase in ('nse_proxy', 'both'):
            logger.info("Starting NSE proxy backfill phase...")
            updated, skipped = backfill_nse_proxy(args.start_date, args.end_date, args.batch_size)
            logger.info(f"NSE proxy phase complete: {updated} updated, {skipped} skipped")
        
        if args.phase in ('real_bse', 'both'):
            logger.info("Starting real BSE API backfill phase...")
            updated, skipped = backfill_real_bse(args.start_date, args.end_date, min(args.batch_size, 100))
            logger.info(f"Real BSE phase complete: {updated} updated, {skipped} skipped")


if __name__ == '__main__':
    main()
