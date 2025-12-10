import logging
import time
from typing import Dict, List, Tuple, Optional, Callable, Set
from datetime import datetime
from threading import Lock

from kite_trade import get_enctoken, KiteApp
from kiteconnect import KiteTicker

from config import get_config

# ... [Keep existing helper functions: initialize_kite_session, fetch_all_instruments, etc.] ...
# ... [No changes needed until the Connector class] ...

def initialize_kite_session(user_id: str, password: str) -> KiteApp:
    """Authenticate with Zerodha Kite and return a KiteApp session."""
    logging.info("=" * 70)
    logging.info("🔒 Authenticating with Zerodha Kite")
    logging.info("=" * 70)

    enctoken = get_enctoken(user_id, password,
                            input(f"Login ID: {user_id}\nEnter 2FA code: ").strip())

    if not enctoken:
        raise ConnectionError("Failed to obtain enctoken")

    kite_session = KiteApp(enctoken=enctoken)
    profile = kite_session.profile()
    kite_session.user_id = profile.get('user_id')

    logging.info(f"✓ Connected as: {kite_session.user_id} ({profile.get('user_name')})")
    return kite_session


def fetch_all_instruments(kite_obj: KiteApp, exchange_segments: List[str]) -> Dict[str, List[Dict]]:
    """Fetch instrument lists with error handling."""
    logging.info("=" * 70)
    logging.info("FETCHING INSTRUMENT LISTS")
    logging.info("=" * 70)

    all_instruments = {}

    for segment in exchange_segments:
        try:
            instruments = kite_obj.instruments(segment)
            all_instruments[segment] = instruments
            logging.info(f"✓ Fetched {len(instruments)} {segment} instruments")
        except Exception as e:
            logging.error(f"✗ Failed to fetch {segment} instruments: {e}")
            raise

    return all_instruments


def configure_exchange_handlers(all_instruments: Dict[str, List[Dict]],
                                exchange_handlers: Dict[str, object],
                                exchange_configs: Dict[str, dict],
                                find_instrument_token_for_symbol,
                                get_nearest_expiry,
                                get_nearest_futures_contract) -> Tuple[Dict[str, object], int]:
    """Configure handler metadata and resolve VIX token. Returns handlers and VIX token."""
    logging.info("=" * 70)
    logging.info("CONFIGURING EXCHANGES")
    logging.info("=" * 70)

    vix_token = None

    try:
        if 'NSE' in all_instruments:
            vix_token = find_instrument_token_for_symbol(
                all_instruments['NSE'], 'INDIAVIX', 'NSE'
            )
            logging.info(f"✓ VIX token resolved to: {vix_token}")

        for ex_key, handler in exchange_handlers.items():
            cfg = exchange_configs[ex_key]

            spot_list = all_instruments.get(cfg['ltp_exchange'], [])
            option_list = all_instruments.get(cfg['options_exchange'], [])

            if not spot_list or not option_list:
                raise ValueError(f"Missing instruments for {ex_key}")

            handler.instrument_list = option_list

            handler.underlying_token = find_instrument_token_for_symbol(
                spot_list, cfg['underlying_symbol'], cfg['ltp_exchange']
            )
            if not handler.underlying_token:
                raise ValueError(f"Could not find token for {cfg['underlying_symbol']}")

            expiry_info = get_nearest_expiry(
                option_list, cfg['underlying_prefix'],
                cfg['options_exchange'], cfg.get('is_monthly', False)
            )
            if not expiry_info:
                raise ValueError(f"Could not determine expiry for {cfg['underlying_prefix']}")

            handler.expiry_date, handler.symbol_prefix = expiry_info['expiry'], expiry_info['symbol_prefix']

            future = get_nearest_futures_contract(option_list, cfg['underlying_prefix'])
            if future:
                handler.futures_token = future.get('instrument_token')
                handler.futures_symbol = future.get('tradingsymbol')

            logging.info(
                f"✓ {ex_key}: Underlying={handler.underlying_token}, "
                f"Expiry={handler.expiry_date:%d-%b-%Y}, Future={handler.futures_symbol or 'N/A'}"
            )
    except Exception as e:
        logging.error(f"Exchange configuration error: {e}")
        raise

    return exchange_handlers, vix_token


def bootstrap_initial_prices(kite_obj: KiteApp,
                             exchange_handlers: Dict[str, object],
                             normalize_price,
                             strip_timezone,
                             get_session_range):
    """Bootstrap initial prices from historical data."""
    logging.info("=" * 70)
    logging.info("BOOTSTRAPPING INITIAL PRICES")
    logging.info("=" * 70)

    from_dt, to_dt = get_session_range()

    for handler in exchange_handlers.values():
        if not handler.underlying_token:
            continue

        try:
            candles = kite_obj.historical_data(
                handler.underlying_token, from_dt, to_dt, 'minute',
                continuous=False, oi=True
            )

            if not candles:
                logging.warning(f"[{handler.exchange}] No historical data available for bootstrap")
                continue

            latest_candle = next((c for c in reversed(candles) if c.get('close') is not None), None)
            if latest_candle:
                normalized_price = normalize_price(latest_candle.get('close'))
                candle_time = strip_timezone(latest_candle.get('date'))

                logging.info(
                    f"✓ Initial {handler.config['underlying_symbol']} LTP: {normalized_price} "
                    f"(as of {candle_time:%H:%M:%S})"
                )

                with handler.lock:
                    handler.latest_tick_data[handler.underlying_token] = {
                        'last_price': latest_candle.get('close')
                    }
                    handler.latest_oi_data['underlying_price'] = normalized_price
        except Exception as e:
            logging.warning(f"[{handler.exchange}] Initial price bootstrap failed: {e}")
            # Continue anyway - system will use live data when available


class Connector:
    """Manages Kite session and WebSocket lifecycle with improved retry logic."""
    
    def __init__(self, 
                 user_id: str,
                 password: str,
                 twofa_callback: Optional[Callable[[], str]] = None,
                 on_tick_callback: Optional[Callable] = None,
                 vix_token: Optional[int] = None,
                 exchange_handlers: Optional[Dict] = None,
                 all_subscribed_tokens: Optional[Set[int]] = None):
        """
        Initialize connector.
        """
        self.user_id = user_id
        self.password = password
        self.twofa_callback = twofa_callback
        self.on_tick_callback = on_tick_callback
        self.vix_token = vix_token
        self.exchange_handlers = exchange_handlers or {}
        
        # Use the same set reference if provided to keep state synced with app_manager
        if all_subscribed_tokens is not None:
            self.all_subscribed_tokens = all_subscribed_tokens
        else:
            self.all_subscribed_tokens = set()
            
        # Set to hold manually added protected tokens (like Macro tokens)
        self.additional_protected_tokens = set()
        
        self.kite: Optional[KiteApp] = None
        self.kws: Optional[KiteTicker] = None
        self._subscription_lock = Lock()
        self._reconnect_attempts = 0
        self._shutting_down = False  # Flag to prevent reconnection during shutdown
        self._connection_established = False  # Flag set by _on_connect callback
        
    def initialize_kite(self, twofa_code: Optional[str] = None) -> KiteApp:
        """Initialize and authenticate Kite session."""
        # Reset shutdown flag when initializing new session
        self._shutting_down = False
        
        if self.kite:
            return self.kite
            
        logging.info("=" * 70)
        logging.info("🔒 Authenticating with Zerodha Kite")
        logging.info("=" * 70)

        code = twofa_code
        if not code and self.twofa_callback:
            try:
                code = self.twofa_callback()
            except Exception as callback_error:
                logging.error(f"Two-factor callback failed: {callback_error}")
                code = None
        if not code:
            code = input(f"Login ID: {self.user_id}\nEnter 2FA code: ").strip()

        enctoken = get_enctoken(self.user_id, self.password, code)

        if not enctoken:
            raise ConnectionError("Failed to obtain enctoken")

        self.kite = KiteApp(enctoken=enctoken)
        profile = self.kite.profile()
        self.kite.user_id = profile.get('user_id')

        logging.info(f"✓ Connected as: {self.kite.user_id} ({profile.get('user_name')})")
        return self.kite
    
    def connect_websocket(self, timeout_seconds: int = 30, max_attempts: int = 2) -> bool:
        """Connect WebSocket with progressive backoff retry logic.
        
        - Tries up to `max_attempts` connection attempts.
        - Between attempts, closes any stale WebSocket instance and waits briefly.
        - Returns False only if all attempts fail.
        """
        if not self.kite:
            raise RuntimeError("Kite session not initialized. Call initialize_kite() first.")
        
        # Reset shutdown flag and connection flag for new connection
        self._shutting_down = False
        self._connection_established = False
        
        if self.kws and (self._connection_established or self.kws.is_connected()):
            logging.info("WebSocket already connected")
            return True
        
        self._reconnect_attempts = 0
        
        for attempt_index in range(1, max_attempts + 1):
            logging.info("=" * 70)
            logging.info(f"🔌 Initializing WebSocket Connection (attempt {attempt_index}/{max_attempts})")
            logging.info("=" * 70)
            
            try:
                # Clean up any existing, possibly stale WebSocket instance
                if self.kws:
                    try:
                        if self.kws.is_connected():
                            self.kws.close()
                            time.sleep(0.5)
                    except Exception:
                        # Ignore cleanup errors; we'll replace kws below
                        pass
                    finally:
                        self.kws = None
                
                # Validate we have valid credentials before creating KiteTicker
                if not self.kite or not self.kite.enctoken:
                    raise RuntimeError("Kite session not properly initialized - missing enctoken")
                
                logging.info("Creating KiteTicker instance...")
                self.kws = KiteTicker(
                    api_key="TradeViaPython",
                    access_token=self.kite.enctoken + "&user_id=" + self.kite.user_id
                )
                
                logging.info("Setting up KiteTicker callbacks...")
                self.kws.on_ticks = self._on_ticks
                self.kws.on_connect = self._on_connect
                self.kws.on_error = self._on_error
                self.kws.on_close = self._on_close
                
                # Reset connection flag before attempting connection
                self._connection_established = False
                
                logging.info("Calling kws.connect(threaded=True)...")
                try:
                    self.kws.connect(threaded=True)
                    logging.info("kws.connect() call completed (connection is asynchronous)")
                except Exception as connect_ex:
                    logging.error(f"Exception during kws.connect(): {connect_ex}", exc_info=True)
                    raise
                
                start_time = time.time()
                wait_step = 0
                while time.time() - start_time < timeout_seconds:
                    # Check both the callback flag and is_connected() for reliability
                    if self._connection_established or (self.kws and self.kws.is_connected()):
                        logging.info("✓ WebSocket connected successfully")
                        self._reconnect_attempts = 0
                        return True
                    
                    # Log progress every 5 seconds
                    elapsed = time.time() - start_time
                    if wait_step % 10 == 0 and elapsed > 5:
                        logging.debug(
                            f"Waiting for WebSocket connection... "
                            f"({elapsed:.1f}s elapsed, "
                            f"callback_flag={self._connection_established}, "
                            f"is_connected={self.kws.is_connected() if self.kws else 'N/A'})"
                        )
                    
                    wait_time = min(0.5 + (wait_step * 0.5), 2.0)
                    time.sleep(wait_time)
                    wait_step += 1
                
                logging.warning(
                    f"✗ WebSocket attempt {attempt_index}/{max_attempts} "
                    f"failed to connect within {timeout_seconds} seconds"
                )
            except Exception as e:
                logging.error(
                    f"WebSocket connection error on attempt {attempt_index}/{max_attempts}: {e}",
                    exc_info=True
                )
            
            # If not last attempt, wait a bit before retrying
            if attempt_index < max_attempts:
                time.sleep(3.0)
        
        logging.error(
            f"✗ WebSocket failed to connect after {max_attempts} attempts "
            f"(each {timeout_seconds} seconds)"
        )
        return False
    
    def _on_ticks(self, ws, ticks):
        if self.on_tick_callback:
            try:
                self.on_tick_callback(ws, ticks)
            except Exception as e:
                logging.error(f"Tick callback error: {e}", exc_info=True)
    
    def _on_connect(self, ws, response):
        """Handle WebSocket connection (initial and reconnection)."""
        if self._shutting_down:
            logging.debug("WebSocket connection attempt during shutdown (ignoring)")
            return
        
        # Set connection flag immediately when callback fires
        self._connection_established = True
        
        logging.info("=" * 70)
        logging.info("✓ WebSocket connected (on_connect callback fired)")
        logging.info("=" * 70)
        
        self._reconnect_attempts = 0
        
        try:
            # Build subscription list
            tokens = set()
            
            # 1. Add VIX
            if self.vix_token:
                tokens.add(self.vix_token)
            
            # 2. Add Underlying and Futures from all handlers
            for handler in self.exchange_handlers.values():
                if handler.underlying_token:
                    tokens.add(handler.underlying_token)
                if handler.futures_token:
                    tokens.add(handler.futures_token)
            
            # 3. Add all previously subscribed option/macro tokens
            # This ensures macro tokens (added via update() in main app) are included
            tokens.update(self.all_subscribed_tokens)
            
            # 4. Add additional protected tokens (e.g. manually added macro tokens)
            tokens.update(self.additional_protected_tokens)
            
            token_list = list(filter(None, tokens))
            
            if token_list:
                with self._subscription_lock:
                    logging.info(f"✓ Subscribing to {len(token_list)} total tokens")
                    ws.subscribe(token_list)
                    ws.set_mode(ws.MODE_FULL, token_list)
                    
                    # Update the tracking set
                    self.all_subscribed_tokens.update(token_list)
                    logging.info(f"✓ Subscription request sent for {len(token_list)} instruments")
            else:
                logging.warning("No instruments to subscribe to")
            
            time.sleep(0.5)
            
        except Exception as e:
            logging.error(f"WebSocket connection handler error: {e}", exc_info=True)
            self._schedule_subscription_restore()
    
    def _on_error(self, ws, code, reason):
        if self._shutting_down:
            logging.debug(f"WebSocket error {code} during shutdown (ignoring): {reason}")
            return
        # Log errors more prominently during connection attempts
        if code == 1006:
            logging.warning(f"WebSocket connection error {code}: {reason} (will retry automatically)")
        else:
            logging.error(f"WebSocket error {code}: {reason}")
        # Reset connection flag on error
        self._connection_established = False
    
    def _on_close(self, ws, code, reason):
        if self._shutting_down:
            logging.debug(f"WebSocket closed during shutdown (code {code}): {reason}")
            return
        # Reset connection flag on close
        self._connection_established = False
        if code == 1006:
            logging.warning(f"WebSocket closed uncleanly (code {code}): {reason}. Retrying...")
        else:
            logging.warning(f"WebSocket closed {code}: {reason}")
    
    def _schedule_subscription_restore(self):
        if self._shutting_down:
            logging.debug("Skipping scheduled subscription restore during shutdown")
            return
        from threading import Thread
        def delayed_restore():
            time.sleep(2)
            if not self._shutting_down:
                self.restore_subscriptions()
        Thread(target=delayed_restore, daemon=True).start()
    
    def restore_subscriptions(self, max_wait_seconds: int = 10) -> bool:
        """Restore all subscriptions after WebSocket reconnection."""
        if self._shutting_down:
            logging.debug("Skipping subscription restoration during shutdown")
            return False
        if not self.kws:
            return False
        
        for _ in range(max_wait_seconds * 2):
            if self.kws.is_connected():
                break
            time.sleep(0.5)
        
        if not self.kws.is_connected():
            return False
        
        try:
            # Re-trigger logic in _on_connect which rebuilds the full list
            self._on_connect(self.kws, None)
            return True
        except Exception as e:
            logging.error(f"Error restoring subscriptions: {e}")
            return False
    
    def update_subscriptions(self, new_tokens: List[int], max_retries: int = 5) -> bool:
        """
        Update subscriptions: Add new tokens, optionally remove unused (excluding protected).
        
        By default, tokens are NOT unsubscribed to avoid frequent add/remove cycles.
        Unsubscription only occurs if:
        1. websocket_auto_unsubscribe is enabled, OR
        2. Total subscribed tokens exceed unsubscribe_threshold (default: 1800)
        """
        if self._shutting_down:
            logging.debug("Skipping subscription update during shutdown")
            return False
        if not self.kws:
            logging.debug("WebSocket not available for subscription update")
            return False
        try:
            config = get_config()
            with self._subscription_lock:
                # 1. Identify Protected Tokens (Critical Core + Macro)
                protected_tokens = set()
                
                if self.vix_token:
                    protected_tokens.add(self.vix_token)
                
                for handler in self.exchange_handlers.values():
                    if handler.underlying_token:
                        protected_tokens.add(handler.underlying_token)
                    if handler.futures_token:
                        protected_tokens.add(handler.futures_token)
                
                # Add manually registered protected tokens (Macro tokens)
                protected_tokens.update(self.additional_protected_tokens)

                # 2. Add new tokens
                to_add = set(new_tokens) - self.all_subscribed_tokens
                if to_add:
                    self.kws.subscribe(list(to_add))
                    self.kws.set_mode(self.kws.MODE_FULL, list(to_add))
                    self.all_subscribed_tokens.update(to_add)
                    logging.info(f"✓ Added {len(to_add)} new subscriptions (Total: {len(self.all_subscribed_tokens)})")
                
                # 3. Calculate removals (Unused - Protected)
                # Only remove if:
                # - auto_unsubscribe is enabled, OR
                # - We're approaching the max token limit (above threshold)
                candidates_for_removal = self.all_subscribed_tokens - set(new_tokens)
                to_remove = candidates_for_removal - protected_tokens
                
                total_subscribed = len(self.all_subscribed_tokens)
                should_unsubscribe = (
                    config.websocket_auto_unsubscribe or 
                    (total_subscribed > config.websocket_unsubscribe_threshold and to_remove)
                )
                
                if to_remove and should_unsubscribe:
                    # If we're above threshold, only remove enough to get back below threshold
                    if total_subscribed > config.websocket_unsubscribe_threshold:
                        tokens_to_remove_count = total_subscribed - config.websocket_unsubscribe_threshold
                        to_remove_list = list(to_remove)[:tokens_to_remove_count]
                        logging.info(
                            f"⚠ Approaching token limit ({total_subscribed}/{config.websocket_max_tokens}), "
                            f"removing {len(to_remove_list)} unused tokens"
                        )
                    else:
                        to_remove_list = list(to_remove)
                        logging.info(f"✓ Removed {len(to_remove_list)} subscriptions (Protected {len(protected_tokens)} core tokens)")
                    
                    self.kws.unsubscribe(to_remove_list)
                    self.all_subscribed_tokens -= set(to_remove_list)
                elif to_remove and not should_unsubscribe:
                    # Log that we're keeping tokens (only log occasionally to avoid spam)
                    if len(to_remove) > 10:  # Only log if significant number of tokens
                        logging.debug(
                            f"Keeping {len(to_remove)} unused tokens subscribed "
                            f"(Total: {total_subscribed}, Auto-unsubscribe: {config.websocket_auto_unsubscribe})"
                        )
                
                return True
        except Exception as e:
            logging.error(f"Error updating subscriptions: {e}")
            # Fallback: Ensure we at least track the new ones
            self.all_subscribed_tokens.update(new_tokens)
            return False

    def close(self):
        """Close WebSocket connection and prevent reconnection attempts."""
        self._shutting_down = True
        
        if self.kws:
            try:
                # Disconnect if connected
                if self.kws.is_connected():
                    self.kws.close()
                    # Wait a moment for clean close
                    import time
                    time.sleep(0.5)
                # Set kws to None to prevent further operations
                self.kws = None
                logging.debug("WebSocket closed and shutdown flag set")
            except Exception as e:
                logging.debug(f"Error closing WebSocket: {e}")
                # Still set to None even if close failed
                self.kws = None
    
    def is_connected(self) -> bool:
        return self.kws is not None and self.kws.is_connected()