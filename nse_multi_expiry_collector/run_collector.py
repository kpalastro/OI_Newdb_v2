"""
Standalone script to run the multi-expiry collector service.

This script runs as a background service that collects minute-by-minute
option chain data across the next 5 expiries during market hours.

Usage:
    python nse_multi_expiry_collector/run_collector.py --exchange NSE
"""
import argparse
import logging
import sys
import signal
from pathlib import Path

# Add parent directory to path
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from kite_trade import KiteApp, get_enctoken
from config import get_config
from nse_multi_expiry_collector.collector_service import MultiExpiryCollectorService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global service instance for graceful shutdown
service: MultiExpiryCollectorService = None


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    logger.info("Received shutdown signal. Stopping service...")
    if service:
        service.stop()
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='Run multi-expiry option chain collector service'
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
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    config = get_config()
    user_id = args.user_id or config.user_id
    password = args.password or config.password
    
    if not user_id or not password:
        logger.error("User ID and password required (via args or config/env)")
        return
    
    # Initialize Kite connection
    logger.info("Initializing Kite connection...")
    try:
        twofa_code = input("Enter 2FA code: ").strip()
        enctoken = get_enctoken(user_id, password, twofa_code)
        
        if not enctoken:
            logger.error("Failed to obtain enctoken")
            return
        
        kite_obj = KiteApp(enctoken=enctoken)
        profile = kite_obj.profile()
        logger.info(f"Connected as: {profile.get('user_id')} ({profile.get('user_name')})")
    except Exception as e:
        logger.error(f"Failed to initialize Kite: {e}")
        return
    
    # Create and start service
    global service
    service = MultiExpiryCollectorService(
        kite_obj=kite_obj,
        exchange=args.exchange
    )
    
    try:
        logger.info(f"Starting multi-expiry collector service for {args.exchange}...")
        service.start()
        
        # Keep the main thread alive
        logger.info("Service running. Press Ctrl+C to stop.")
        while service.running:
            import time
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if service:
            service.stop()
        logger.info("Service stopped")


if __name__ == '__main__':
    main()