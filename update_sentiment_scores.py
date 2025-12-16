#!/usr/bin/env python3
"""
Update sentiment_score_50 and sentiment_score_100 in ml_features table
from macro_signals table for each exchange.

This script updates ml_features by matching records from macro_signals
based on exchange and timestamp.

Usage:
    python update_sentiment_scores.py [--exchange EXCHANGE] [--exact-match]
"""

import argparse
import logging
from datetime import datetime, timedelta

from database_new import get_db_connection, release_db_connection, get_config
from time_utils import now_ist

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def update_sentiment_scores(exchange: str = None, exact_match: bool = True):
    """
    Update sentiment_score_50 and sentiment_score_100 in ml_features from macro_signals.
    
    Args:
        exchange: Exchange name to update (if None, updates all exchanges)
        exact_match: If True, match timestamps up to minutes (ignoring seconds). 
                     If False, match nearest within 5 minutes (not currently implemented)
    """
    config = get_config()
    db_type = config.db_type
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        ph = '?' if db_type == 'sqlite' else '%s'
        
        # Build exchange filter
        exchange_filter = ""
        exchange_params = []
        if exchange:
            exchange_filter = f" AND mf.exchange = {ph} AND ms.exchange = {ph}"
            exchange_params = [exchange, exchange]
        
        logger.info(f"Updating sentiment scores{' for exchange ' + exchange if exchange else ' for all exchanges'}")
        logger.info(f"Using timestamp matching up to minutes (ignoring seconds)")
        
        if exact_match:
            # Match timestamps up to minutes (ignore seconds)
            if db_type == 'postgres':
                query = f"""
                    UPDATE ml_features mf
                    SET 
                        sentiment_score_50 = ms.sentiment_score_50,
                        sentiment_score_100 = ms.sentiment_score_100
                    FROM macro_signals ms
                    WHERE mf.exchange = ms.exchange
                      AND DATE_TRUNC('minute', mf.timestamp) = DATE_TRUNC('minute', ms.timestamp)
                      AND ms.sentiment_score_50 IS NOT NULL
                      AND ms.sentiment_score_100 IS NOT NULL
                      {exchange_filter}
                """
                cursor.execute(query, exchange_params)
            else:
                # SQLite syntax - match timestamps up to minutes (ignore seconds)
                if exchange:
                    # SQLite doesn't support FROM clause in UPDATE
                    query = f"""
                        UPDATE ml_features
                        SET 
                            sentiment_score_50 = (
                                SELECT sentiment_score_50 
                                FROM macro_signals ms
                                WHERE ms.exchange = ml_features.exchange
                                  AND strftime('%Y-%m-%d %H:%M', ms.timestamp) = strftime('%Y-%m-%d %H:%M', ml_features.timestamp)
                                  AND ms.sentiment_score_50 IS NOT NULL
                                  AND ms.sentiment_score_100 IS NOT NULL
                                LIMIT 1
                            ),
                            sentiment_score_100 = (
                                SELECT sentiment_score_100 
                                FROM macro_signals ms
                                WHERE ms.exchange = ml_features.exchange
                                  AND strftime('%Y-%m-%d %H:%M', ms.timestamp) = strftime('%Y-%m-%d %H:%M', ml_features.timestamp)
                                  AND ms.sentiment_score_50 IS NOT NULL
                                  AND ms.sentiment_score_100 IS NOT NULL
                                LIMIT 1
                            )
                        WHERE EXISTS (
                            SELECT 1 FROM macro_signals ms
                            WHERE ms.exchange = ml_features.exchange
                              AND strftime('%Y-%m-%d %H:%M', ms.timestamp) = strftime('%Y-%m-%d %H:%M', ml_features.timestamp)
                              AND ms.sentiment_score_50 IS NOT NULL
                              AND ms.sentiment_score_100 IS NOT NULL
                        )
                        AND ml_features.exchange = {ph}
                    """
                    params = [exchange]
                else:
                    query = f"""
                        UPDATE ml_features
                        SET 
                            sentiment_score_50 = (
                                SELECT sentiment_score_50 
                                FROM macro_signals ms
                                WHERE ms.exchange = ml_features.exchange
                                  AND strftime('%Y-%m-%d %H:%M', ms.timestamp) = strftime('%Y-%m-%d %H:%M', ml_features.timestamp)
                                  AND ms.sentiment_score_50 IS NOT NULL
                                  AND ms.sentiment_score_100 IS NOT NULL
                                LIMIT 1
                            ),
                            sentiment_score_100 = (
                                SELECT sentiment_score_100 
                                FROM macro_signals ms
                                WHERE ms.exchange = ml_features.exchange
                                  AND strftime('%Y-%m-%d %H:%M', ms.timestamp) = strftime('%Y-%m-%d %H:%M', ml_features.timestamp)
                                  AND ms.sentiment_score_50 IS NOT NULL
                                  AND ms.sentiment_score_100 IS NOT NULL
                                LIMIT 1
                            )
                        WHERE EXISTS (
                            SELECT 1 FROM macro_signals ms
                            WHERE ms.exchange = ml_features.exchange
                              AND strftime('%Y-%m-%d %H:%M', ms.timestamp) = strftime('%Y-%m-%d %H:%M', ml_features.timestamp)
                              AND ms.sentiment_score_50 IS NOT NULL
                              AND ms.sentiment_score_100 IS NOT NULL
                        )
                    """
                    params = []
                cursor.execute(query, params)
            
            rows_updated = cursor.rowcount
            conn.commit()
            
            logger.info(f"✓ Updated {rows_updated} records (matched timestamps up to minutes, ignoring seconds)")
            
        else:
            # Nearest timestamp match (within 5 minutes)
            # This is more complex and only works well with PostgreSQL
            if db_type != 'postgres':
                logger.warning("Nearest timestamp matching requires PostgreSQL. Falling back to exact match.")
                return update_sentiment_scores(exchange, exact_match=True)
            
            logger.warning("Nearest timestamp matching is complex. Using exact match instead.")
            return update_sentiment_scores(exchange, exact_match=True)
        
        # Verify the update
        if exchange:
            verify_query = f"""
                SELECT 
                    COUNT(*) as total,
                    COUNT(sentiment_score_50) as with_score_50,
                    COUNT(sentiment_score_100) as with_score_100
                FROM ml_features
                WHERE exchange = {ph}
            """
            cursor.execute(verify_query, (exchange,))
        else:
            verify_query = """
                SELECT 
                    exchange,
                    COUNT(*) as total,
                    COUNT(sentiment_score_50) as with_score_50,
                    COUNT(sentiment_score_100) as with_score_100
                FROM ml_features
                GROUP BY exchange
            """
            cursor.execute(verify_query)
        
        results = cursor.fetchall()
        release_db_connection(conn)
        
        logger.info("\n" + "="*60)
        logger.info("Update Summary:")
        logger.info("="*60)
        if exchange:
            total, score_50, score_100 = results[0]
            logger.info(f"Exchange: {exchange}")
            logger.info(f"  Total records: {total}")
            logger.info(f"  Records with sentiment_score_50: {score_50}")
            logger.info(f"  Records with sentiment_score_100: {score_100}")
        else:
            for row in results:
                exch, total, score_50, score_100 = row
                logger.info(f"Exchange: {exch}")
                logger.info(f"  Total records: {total}")
                logger.info(f"  Records with sentiment_score_50: {score_50}")
                logger.info(f"  Records with sentiment_score_100: {score_100}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error updating sentiment scores: {e}", exc_info=True)
        if 'conn' in locals():
            release_db_connection(conn)
        raise


def preview_update(exchange: str = None):
    """
    Preview how many records would be updated without actually updating.
    """
    config = get_config()
    db_type = config.db_type
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        ph = '?' if db_type == 'sqlite' else '%s'
        
        exchange_filter = ""
        exchange_params = []
        if exchange:
            exchange_filter = f" AND mf.exchange = {ph}"
            exchange_params = [exchange]
        
        # Count how many records would be updated (matching timestamps up to minutes)
        if db_type == 'postgres':
            query = f"""
                SELECT 
                    mf.exchange,
                    COUNT(*) as total_ml_features,
                    COUNT(ms.sentiment_score_50) as records_with_macro_data
                FROM ml_features mf
                LEFT JOIN macro_signals ms 
                    ON mf.exchange = ms.exchange 
                    AND DATE_TRUNC('minute', mf.timestamp) = DATE_TRUNC('minute', ms.timestamp)
                    AND ms.sentiment_score_50 IS NOT NULL
                    AND ms.sentiment_score_100 IS NOT NULL
                WHERE 1=1 {exchange_filter}
                GROUP BY mf.exchange
                ORDER BY mf.exchange
            """
            cursor.execute(query, exchange_params)
        else:
            query = f"""
                SELECT 
                    mf.exchange,
                    COUNT(*) as total_ml_features,
                    SUM(CASE WHEN ms.sentiment_score_50 IS NOT NULL THEN 1 ELSE 0 END) as records_with_macro_data
                FROM ml_features mf
                LEFT JOIN macro_signals ms 
                    ON mf.exchange = ms.exchange 
                    AND strftime('%Y-%m-%d %H:%M', mf.timestamp) = strftime('%Y-%m-%d %H:%M', ms.timestamp)
                    AND ms.sentiment_score_50 IS NOT NULL
                    AND ms.sentiment_score_100 IS NOT NULL
                WHERE 1=1 {exchange_filter}
                GROUP BY mf.exchange
                ORDER BY mf.exchange
            """
            cursor.execute(query, exchange_params)
        
        results = cursor.fetchall()
        release_db_connection(conn)
        
        logger.info("\n" + "="*60)
        logger.info("Preview: Records that would be updated")
        logger.info("="*60)
        for row in results:
            exch, total, with_macro = row
            logger.info(f"Exchange: {exch}")
            logger.info(f"  Total ml_features records: {total}")
            logger.info(f"  Records with matching macro_signals: {with_macro}")
            logger.info(f"  Coverage: {with_macro/total*100:.1f}%")
        logger.info("="*60)
        
        return results
        
    except Exception as e:
        logger.error(f"Error previewing update: {e}", exc_info=True)
        if 'conn' in locals():
            release_db_connection(conn)
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Update sentiment_score_50 and sentiment_score_100 in ml_features from macro_signals',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview update for all exchanges
  python update_sentiment_scores.py --preview
  
  # Update all exchanges
  python update_sentiment_scores.py
  
  # Update specific exchange
  python update_sentiment_scores.py --exchange NSE
  
  # Preview for specific exchange
  python update_sentiment_scores.py --exchange NSE --preview
        """
    )
    parser.add_argument('--exchange', type=str, default=None,
                       help='Exchange name to update (default: all exchanges)')
    parser.add_argument('--preview', action='store_true',
                       help='Preview how many records would be updated without actually updating')
    parser.add_argument('--exact-match', action='store_true', default=True,
                       help='Use exact timestamp matching (default: True)')
    
    args = parser.parse_args()
    
    if args.preview:
        preview_update(args.exchange)
    else:
        update_sentiment_scores(args.exchange, args.exact_match)


if __name__ == '__main__':
    main()
