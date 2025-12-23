#!/usr/bin/env python3
"""
Standalone script to run the multi-expiry data collector service.

This script starts a background service that collects aggregated option chain data
across the next 5 expiries every minute during market hours.
"""
import argparse
import logging
import signal
import sys
from kite_trade import KiteApp, get_enctoken

from config import get_config
from nse_multi_expiry_collector.collector_service import MultiExpiryCollectorService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Run multi-expiry option chain data collector service'
    )
    parser.add_argument(
        '--exchange',
        type=str,
        default='NSE',
        help='Exchange name (default: NSE)'
    )
    parser.add_argument(
        '--user-id',
        type=str,
        help='Zerodha user ID (default: from config/env)'
    )
    parser.add_argument(
        '--password',
        type=str,
        help='Zerodha password (default: from config/env)'
    )
    
    args = parser.parse_args()
    
    config = get_config()
    user_id = args.user_id or config.user_id
    password = args.password or config.password
    
    if not user_id or not password:
        logger.error("User ID and password required (via args or config/env)")
        sys.exit(1)
    
    # Initialize Kite
    logger.info("Initializing Kite connection...")
    try:
        twofa_code = input("Enter 2FA code: ").strip()
        enctoken = get_enctoken(user_id, password, twofa_code)
        
        if not enctoken:
            logger.error("Failed to obtain enctoken")
            sys.exit(1)
        
        kite_obj = KiteApp(enctoken=enctoken)
        profile = kite_obj.profile()
        logger.info(f"Connected as: {profile.get('user_id')}")
    except Exception as e:
        logger.error(f"Failed to initialize Kite: {e}")
        sys.exit(1)
    
    # Create collector service
    service = MultiExpiryCollectorService(
        kite_obj=kite_obj,
        exchange=args.exchange
    )
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal. Stopping service...")
        service.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start service
    try:
        logger.info(f"Starting multi-expiry collector service for {args.exchange}...")
        service.start()
        
        # Keep running
        logger.info("Service is running. Press Ctrl+C to stop.")
        while service.running:
            import time
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Stopping service...")
    finally:
        service.stop()
        logger.info("Service stopped.")


if __name__ == '__main__':
    main()
