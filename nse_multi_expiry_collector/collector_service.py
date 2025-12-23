"""
Background service for collecting multi-expiry option chain data every minute.

Runs during market hours (9:15 AM - 3:30 PM IST) and collects aggregated data
across the next 5 expiries for the base strike calculated from market open price.
"""
import logging
import time
import threading
from datetime import datetime, date, time as dt_time
from typing import Optional, Dict, List
from kite_trade import KiteApp

from time_utils import now_ist
from config import get_config
from database_new import save_multi_expiry_minute_data
from nse_multi_expiry_collector.kite_fetcher import get_next_5_expiries, aggregate_multi_expiry_data
from nse_multi_expiry_collector.aggregator import calculate_base_strike

logger = logging.getLogger(__name__)


class MultiExpiryCollectorService:
    """
    Background service that collects multi-expiry option chain data every minute.
    
    On market open (9:15 AM):
    - Gets market open price
    - Calculates base strike
    - Identifies next 5 expiries
    
    Then every minute during market hours:
    - Fetches option chain data for base strike across all 5 expiries
    - Aggregates metrics
    - Saves to database (if not already exists)
    """
    
    def __init__(self, kite_obj: Optional[KiteApp] = None, exchange: str = "NSE"):
        """
        Initialize the collector service.
        
        Args:
            kite_obj: KiteApp instance (will create new if None)
            exchange: Exchange name (default: "NSE")
        """
        self.kite_obj = kite_obj
        self.exchange = exchange
        self.config = get_config()
        self.exchange_config = self.config.exchange_configs.get(exchange, {})
        
        # Daily initialization values (set once at market open)
        self.open_price: Optional[float] = None
        self.base_strike: Optional[float] = None
        self.expiry_dates: List[date] = []
        self.underlying_prefix: str = self.exchange_config.get('underlying_prefix', 'NIFTY')
        self.options_exchange: str = self.exchange_config.get('options_exchange', 'NFO')
        self.strike_difference: int = self.exchange_config.get('strike_difference', 50)
        
        # Service state
        self.running = False
        self.collector_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Websocket data cache (optional, for faster data access)
        self.websocket_data: Dict[int, Dict] = {}
        
        # Previous OI tracking (for change in OI calculation)
        self.previous_oi: Dict[int, float] = {}
        
        # Current spot price (for IV calculation)
        self.current_spot_price: Optional[float] = None
        
    def _get_or_create_kite(self) -> KiteApp:
        """Get existing Kite instance or create new one."""
        if self.kite_obj is not None:
            return self.kite_obj
        
        # Try to get from connector if available
        try:
            from connector import Connector
            # This would need access to the connector instance
            # For now, return None and let caller handle
            logger.warning("Kite instance not provided and connector not accessible. Service will not start.")
            return None
        except:
            logger.warning("Cannot create Kite instance. Service requires Kite connection.")
            return None
    
    def _get_market_open_price(self) -> Optional[float]:
        """
        Get market open price from handler or database.
        
        Returns:
            Open price or None if not available
        """
        # Try to get from handler if available in main app context
        # For standalone service, we'll need to fetch from database or Kite
        try:
            # Option 1: Get from latest underlying price at 9:15 AM
            # This is a simplified approach - in production, you'd get actual open price
            from database_new import get_db_connection, release_db_connection, _get_placeholder
            
            conn = get_db_connection()
            cursor = conn.cursor()
            ph = _get_placeholder()
            
            # Get today's first price from ml_features or option_chain_snapshots
            today = now_ist().date()
            query = f"""
                SELECT underlying_price 
                FROM ml_features 
                WHERE exchange = {ph} 
                  AND DATE(timestamp) = {ph}
                  AND underlying_price IS NOT NULL
                ORDER BY timestamp ASC
                LIMIT 1
            """
            cursor.execute(query, (self.exchange, today))
            row = cursor.fetchone()
            release_db_connection(conn)
            
            if row and row[0]:
                return float(row[0])
            
            return None
        except Exception as e:
            logger.error(f"Error getting market open price: {e}")
            return None
    
    def _initialize_daily_parameters(self):
        """Initialize daily parameters once at market open."""
        try:
            # Get market open price
            self.open_price = self._get_market_open_price()
            
            if self.open_price is None:
                logger.warning(f"Could not get market open price for {self.exchange}. Skipping initialization.")
                return False
            
            # Calculate base strike
            self.base_strike = calculate_base_strike(self.open_price, self.strike_difference)
            
            # Get next 5 expiries
            kite = self._get_or_create_kite()
            if kite is None:
                logger.error("Kite instance not available. Cannot get expiry dates.")
                return False
            
            self.expiry_dates = get_next_5_expiries(
                kite,
                self.exchange,
                self.underlying_prefix,
                self.options_exchange
            )
            
            if len(self.expiry_dates) < 1:
                logger.warning(f"Could not find expiry dates for {self.exchange}")
                return False
            
            logger.info(f"Daily initialization complete for {self.exchange}:")
            logger.info(f"  Open Price: {self.open_price}")
            logger.info(f"  Base Strike: {self.base_strike}")
            logger.info(f"  Expiry Dates: {self.expiry_dates}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing daily parameters: {e}", exc_info=True)
            return False
    
    def _is_market_hours(self, now: datetime) -> bool:
        """Check if current time is during market hours (9:15 AM - 3:30 PM IST)."""
        market_open = dt_time(9, 15)
        market_close = dt_time(15, 30)
        current_time = now.time()
        return market_open <= current_time <= market_close
    
    def _get_current_spot_price(self) -> Optional[float]:
        """Get current spot price for IV calculation."""
        # Try to get from handler if available
        if self.current_spot_price is not None:
            return self.current_spot_price
        
        # Fallback: get from database
        try:
            from database_new import get_db_connection, release_db_connection, _get_placeholder
            
            conn = get_db_connection()
            cursor = conn.cursor()
            ph = _get_placeholder()
            
            query = f"""
                SELECT underlying_price 
                FROM ml_features 
                WHERE exchange = {ph}
                  AND timestamp >= NOW() - INTERVAL '5 minutes'
                  AND underlying_price IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 1
            """
            cursor.execute(query, (self.exchange,))
            row = cursor.fetchone()
            release_db_connection(conn)
            
            if row and row[0]:
                return float(row[0])
            
            return None
        except Exception as e:
            logger.debug(f"Error getting spot price: {e}")
            return None
    
    def _update_previous_oi(self, expiry_data_list: List[Dict]):
        """Update previous OI tracking from current expiry data."""
        # This will be populated from the instruments found in fetch
        # For now, we'll track it in the next iteration
        pass
    
    def _collect_minute_data(self, timestamp: datetime) -> bool:
        """
        Collect and save data for a specific minute.
        
        Args:
            timestamp: Minute-level timestamp
            
        Returns:
            True if collected successfully, False otherwise
        """
        if self.base_strike is None or not self.expiry_dates:
            logger.warning("Daily parameters not initialized. Skipping collection.")
            return False
        
        try:
            kite = self._get_or_create_kite()
            if kite is None:
                return False
            
            # Get current spot price for IV calculation
            spot_price = self._get_current_spot_price()
            if spot_price:
                self.current_spot_price = spot_price
            
            # Fetch and aggregate data across all expiries
            # Round timestamp to minute boundary
            target_timestamp = timestamp.replace(second=0, microsecond=0)
            
            expiry_data_list, aggregated = aggregate_multi_expiry_data(
                kite,
                self.underlying_prefix,
                self.base_strike,
                self.expiry_dates,
                self.options_exchange,
                use_websocket_data=self.websocket_data if self.websocket_data else None,
                previous_oi=self.previous_oi if self.previous_oi else None,
                spot_price=spot_price,
                target_timestamp=target_timestamp
            )
            
            # Check if we got valid data (at least one non-zero value in key metrics)
            has_valid_data = (
                aggregated.get('total_oi_call_all_expiries', 0) > 0 or
                aggregated.get('total_oi_put_all_expiries', 0) > 0 or
                aggregated.get('total_volume_call_all_expiries', 0) > 0 or
                aggregated.get('total_volume_put_all_expiries', 0) > 0
            )
            
            # Also check that change in OI is not zero for both (indicates meaningful data)
            oi_change_call = aggregated.get('total_oi_change_call_all_expiries', 0) or 0
            oi_change_put = aggregated.get('total_oi_change_put_all_expiries', 0) or 0
            has_oi_change = (oi_change_call != 0 or oi_change_put != 0)
            
            if not has_valid_data:
                logger.warning(
                    f"All aggregated metrics are zero for {self.exchange} at {timestamp}. "
                    f"This might indicate data is not available yet. Skipping save."
                )
                # Don't save zeros - wait for valid data
                return False
            
            if not has_oi_change:
                logger.debug(
                    f"Change in OI is zero for both calls and puts at {timestamp}. "
                    f"Skipping save (no meaningful change detected)."
                )
                return False
            
            # Update previous OI for next iteration
            # Extract OI from expiry data and store by instrument token
            # This requires finding tokens for each expiry - for now, we'll rely on websocket data
            # or historical comparison within fetch function
            
            # Save to database
            success = save_multi_expiry_minute_data(
                exchange=self.exchange,
                timestamp=timestamp,
                open_price=self.open_price,
                base_strike=self.base_strike,
                expiry_dates=self.expiry_dates,
                aggregated_metrics=aggregated
            )
            
            if success:
                logger.info(
                    f"✓ Collected data for {self.exchange} at {timestamp} (strike: {self.base_strike}): "
                    f"CE_OI={aggregated.get('total_oi_call_all_expiries', 0):,.0f}, "
                    f"PE_OI={aggregated.get('total_oi_put_all_expiries', 0):,.0f}"
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Error collecting minute data: {e}", exc_info=True)
            return False
    
    def _collection_loop(self):
        """Main collection loop that runs every minute during market hours."""
        logger.info(f"Multi-expiry collector service started for {self.exchange}")
        
        while not self._stop_event.is_set():
            try:
                now = now_ist()
                
                # Check if market hours
                if not self._is_market_hours(now):
                    # If before market open, wait until 9:15 AM
                    if now.time() < dt_time(9, 15):
                        # Wait until market open
                        next_open = datetime.combine(now.date(), dt_time(9, 15))
                        wait_seconds = (next_open - now).total_seconds()
                        if wait_seconds > 0:
                            logger.info(f"Waiting {wait_seconds/60:.1f} minutes until market open...")
                            time.sleep(min(wait_seconds, 300))  # Sleep max 5 minutes at a time
                        continue
                    else:
                        # After market close, stop for today
                        logger.info("Market closed. Collector stopping for today.")
                        break
                
                # Initialize daily parameters if not done (at market open)
                if self.base_strike is None:
                    if now.time().hour == 9 and now.time().minute == 15:
                        logger.info("Market open detected. Initializing daily parameters...")
                        if not self._initialize_daily_parameters():
                            logger.error("Daily initialization failed. Retrying in 1 minute...")
                            time.sleep(60)
                            continue
                    else:
                        # Not yet market open, wait
                        time.sleep(10)
                        continue
                
                # Collect data for current minute (rounded down)
                current_minute = now.replace(second=0, microsecond=0)
                
                if not self._collect_minute_data(current_minute):
                    logger.warning(f"Failed to collect data for {current_minute}")
                
                # Wait until next minute
                next_minute = current_minute.replace(minute=current_minute.minute + 1)
                wait_seconds = (next_minute - now).total_seconds()
                
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
                else:
                    # Already past the minute mark, wait a bit
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in collection loop: {e}", exc_info=True)
                time.sleep(60)  # Wait 1 minute before retrying
        
        logger.info(f"Multi-expiry collector service stopped for {self.exchange}")
    
    def start(self):
        """Start the background collection service."""
        if self.running:
            logger.warning("Collector service is already running")
            return
        
        self.running = True
        self._stop_event.clear()
        self.collector_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collector_thread.start()
        logger.info(f"Started multi-expiry collector service for {self.exchange}")
    
    def stop(self):
        """Stop the background collection service."""
        if not self.running:
            return
        
        self.running = False
        self._stop_event.set()
        
        if self.collector_thread:
            self.collector_thread.join(timeout=5.0)
        
        logger.info(f"Stopped multi-expiry collector service for {self.exchange}")
    
    def update_websocket_data(self, tick_data: Dict[int, Dict]):
        """
        Update websocket tick data cache for faster access.
        
        Args:
            tick_data: Dictionary mapping instrument_token to tick data
        """
        self.websocket_data.update(tick_data)