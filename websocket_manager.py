import logging
import threading
import time
from typing import Callable, Optional, Set

from kiteconnect import KiteTicker


class PersistentWebSocketManager:
    """Manages a persistent WebSocket connection that survives logout/login cycles."""

    def __init__(self):
        self.kws: Optional[KiteTicker] = None
        self.access_token: Optional[str] = None
        self.api_key: Optional[str] = None
        self.on_tick_callback: Optional[Callable] = None
        self.subscribed_tokens: Set[int] = set()
        self.is_active = False
        self.lock = threading.Lock()
        self._connection_event = threading.Event()

    def initialize(
        self,
        access_token: str,
        api_key: str,
        on_tick_callback: Callable,
        timeout_seconds: int = 30,
    ) -> bool:
        """
        Initialize or reinitialize the WebSocket connection.

        Args:
            access_token: Zerodha access token (e.g. enctoken&user_id=<id>)
            api_key: Zerodha API key (string used when constructing KiteTicker)
            on_tick_callback: Callback function for tick data
            timeout_seconds: Time to wait for connection before failing

        Returns:
            True if connection successful, False otherwise.
        """
        with self.lock:
            # If we already have a connection with same credentials, just update callback
            if (
                self.kws
                and self.is_active
                and self.access_token == access_token
                and self.api_key == api_key
            ):
                logging.info("Reusing existing WebSocket connection")
                self.on_tick_callback = on_tick_callback
                return True

            # Store credentials
            self.access_token = access_token
            self.api_key = api_key
            self.on_tick_callback = on_tick_callback

            # If we have an old connection, close it first (without touching reactor)
            if self.kws:
                try:
                    self.kws.close()
                    time.sleep(0.5)
                except Exception as e:
                    logging.debug(f"Error closing old WebSocket: {e}")

            # Create new KiteTicker instance
            self.kws = KiteTicker(api_key=api_key, access_token=access_token)
            self._connection_event.clear()

            # Setup callbacks
            def on_connect(ws, response):
                logging.info("WebSocket connected successfully")
                self.is_active = True
                self._connection_event.set()

                # Resubscribe to tokens if any
                if self.subscribed_tokens:
                    try:
                        token_list = list(self.subscribed_tokens)
                        ws.subscribe(token_list)
                        ws.set_mode(ws.MODE_FULL, token_list)
                        logging.info(
                            "Restored %s subscriptions", len(token_list)
                        )
                    except Exception as exc:
                        logging.error("Error restoring subscriptions: %s", exc)

            def on_close(ws, code, reason):
                logging.warning("WebSocket closed: %s - %s", code, reason)
                self.is_active = False
                # Attempt reconnection after 5 seconds
                threading.Timer(5.0, self._attempt_reconnect).start()

            def on_error(ws, code, reason):
                logging.error("WebSocket error: %s - %s", code, reason)

            def on_reconnect(ws, attempts_count):
                logging.info(
                    "WebSocket reconnecting (attempt %s)...", attempts_count
                )

            def on_noreconnect(ws):
                logging.error("WebSocket failed to reconnect")
                self.is_active = False

            def on_ticks(ws, ticks):
                if self.on_tick_callback:
                    try:
                        self.on_tick_callback(ws, ticks)
                    except Exception as exc:
                        logging.error("Error in tick callback: %s", exc)

            # Assign callbacks
            self.kws.on_connect = on_connect
            self.kws.on_close = on_close
            self.kws.on_error = on_error
            self.kws.on_reconnect = on_reconnect
            self.kws.on_noreconnect = on_noreconnect
            self.kws.on_ticks = on_ticks

            # Start connection in background thread
            try:
                self.kws.connect(threaded=True)
                logging.info("WebSocket connection initiated...")
            except Exception as exc:
                logging.error("Failed to start WebSocket: %s", exc)
                return False

        # Wait for connection with timeout (outside lock to avoid blocking others)
        if self._connection_event.wait(timeout=timeout_seconds):
            logging.info("WebSocket connection established")
            return True

        logging.error("WebSocket connection timeout after %s seconds", timeout_seconds)
        return False

    def _attempt_reconnect(self):
        """Attempt to reconnect the WebSocket using existing credentials."""
        if not self.is_active and self.access_token and self.api_key:
            logging.info("Attempting automatic WebSocket reconnection...")
            # Reuse last known callback
            self.initialize(
                self.access_token,
                self.api_key,
                self.on_tick_callback,
            )

    def update_credentials(
        self,
        access_token: str,
        api_key: str,
        on_tick_callback: Callable,
        timeout_seconds: int = 30,
    ) -> bool:
        """
        Update credentials and reconnect if needed.

        This is called during (re)login with new credentials.
        """
        with self.lock:
            credentials_changed = (
                self.access_token != access_token or self.api_key != api_key
            )

        if credentials_changed:
            logging.info("WebSocket credentials changed - reconnecting...")
            # Clear old subscriptions since we're changing user
            self.clear_subscriptions()
            return self.initialize(
                access_token, api_key, on_tick_callback, timeout_seconds=timeout_seconds
            )

        # Credentials unchanged: just update callback
        self.on_tick_callback = on_tick_callback
        return self.is_active

    def subscribe(self, tokens: list):
        """Subscribe to tokens. If not active yet, queue them for later."""
        if not tokens:
            return

        token_set = set(tokens)

        if not self.kws or not self.is_active:
            logging.debug(
                "WebSocket not active - queuing %s tokens for subscription",
                len(token_set),
            )
            self.subscribed_tokens.update(token_set)
            return

        try:
            self.kws.subscribe(tokens)
            self.kws.set_mode(self.kws.MODE_FULL, tokens)
            self.subscribed_tokens.update(token_set)
            logging.debug(
                "Subscribed to %s tokens (total tracked: %s)",
                len(token_set),
                len(self.subscribed_tokens),
            )
        except Exception as exc:
            logging.error("Subscription failed: %s", exc)

    def unsubscribe(self, tokens: list):
        """Unsubscribe from tokens."""
        if not tokens:
            return

        if not self.kws or not self.is_active:
            # Still update local tracking so future restores are correct
            self.subscribed_tokens.difference_update(tokens)
            return

        try:
            self.kws.unsubscribe(tokens)
            self.subscribed_tokens.difference_update(tokens)
            logging.debug(
                "Unsubscribed from %s tokens (total tracked: %s)",
                len(tokens),
                len(self.subscribed_tokens),
            )
        except Exception as exc:
            logging.error("Unsubscription failed: %s", exc)

    def clear_subscriptions(self):
        """Clear all subscriptions (for logout)."""
        if self.subscribed_tokens and self.kws and self.is_active:
            try:
                self.kws.unsubscribe(list(self.subscribed_tokens))
            except Exception as exc:
                logging.debug("Error clearing subscriptions: %s", exc)
        self.subscribed_tokens.clear()

    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self.is_active and self.kws is not None

    def soft_reset(self):
        """
        Soft reset for logout - clear subscriptions but keep connection alive.

        This allows fast re-login without WebSocket reconnection and avoids
        Twisted reactor restart issues.
        """
        logging.info("Performing soft WebSocket reset...")
        self.clear_subscriptions()
        logging.info("WebSocket reset complete (connection maintained)")

    def shutdown(self):
        """Complete shutdown - only call when application is closing."""
        with self.lock:
            self.is_active = False
            if self.kws:
                try:
                    self.kws.close()
                    logging.info("WebSocket connection closed")
                except Exception as exc:
                    logging.debug("Error closing WebSocket: %s", exc)
                self.kws = None
            self.subscribed_tokens.clear()


_ws_manager: Optional[PersistentWebSocketManager] = None


def get_websocket_manager() -> PersistentWebSocketManager:
    """Get the global WebSocket manager instance."""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = PersistentWebSocketManager()
    return _ws_manager


