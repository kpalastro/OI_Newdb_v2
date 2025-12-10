"""
Rate limiter for Zerodha API calls.
Enforces 3 requests per second limit for historical_data() calls.
"""
import time
import threading
from collections import deque
from typing import Optional


class ZerodhaRateLimiter:
    """
    Thread-safe rate limiter for Zerodha historical_data API.
    Enforces maximum 3 requests per second.
    """
    
    def __init__(self, max_requests: int = 3, time_window: float = 1.0):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed
            time_window: Time window in seconds (default: 1.0 for per-second limit)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_times = deque()
        self.lock = threading.Lock()
        # Use 350ms minimum spacing to ensure we stay under 3 req/sec
        self.min_interval = time_window / max_requests  # ~0.333s, but use 0.35s for safety
        
    def wait_if_needed(self) -> None:
        """
        Wait if necessary to respect rate limit.
        This should be called before making each API request.
        """
        with self.lock:
            now = time.perf_counter()
            
            # Remove requests older than time window
            while self.request_times and (now - self.request_times[0]) >= self.time_window:
                self.request_times.popleft()
            
            # Check if we're at the limit
            if len(self.request_times) >= self.max_requests:
                # Calculate how long to wait
                oldest_request_time = self.request_times[0]
                wait_time = self.time_window - (now - oldest_request_time)
                if wait_time > 0:
                    time.sleep(wait_time)
                    now = time.perf_counter()
                    # Clean up old requests again after waiting
                    while self.request_times and (now - self.request_times[0]) >= self.time_window:
                        self.request_times.popleft()
            
            # Ensure minimum interval between requests
            if self.request_times:
                last_request_time = self.request_times[-1]
                elapsed = now - last_request_time
                if elapsed < self.min_interval:
                    sleep_time = self.min_interval - elapsed
                    time.sleep(sleep_time)
                    now = time.perf_counter()
            
            # Record this request
            self.request_times.append(now)
    
    def get_wait_time(self) -> float:
        """
        Get the time we need to wait before next request (without actually waiting).
        Useful for logging or planning.
        
        Returns:
            Seconds to wait (0 if no wait needed)
        """
        with self.lock:
            now = time.perf_counter()
            
            # Remove old requests
            while self.request_times and (now - self.request_times[0]) >= self.time_window:
                self.request_times.popleft()
            
            if len(self.request_times) >= self.max_requests:
                oldest_request_time = self.request_times[0]
                wait_time = self.time_window - (now - oldest_request_time)
                return max(0.0, wait_time)
            
            if self.request_times:
                last_request_time = self.request_times[-1]
                elapsed = now - last_request_time
                if elapsed < self.min_interval:
                    return self.min_interval - elapsed
            
            return 0.0


# Global rate limiter instance
_global_rate_limiter: Optional[ZerodhaRateLimiter] = None
_rate_limiter_lock = threading.Lock()


def get_rate_limiter() -> ZerodhaRateLimiter:
    """Get or create global rate limiter instance."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        with _rate_limiter_lock:
            if _global_rate_limiter is None:
                _global_rate_limiter = ZerodhaRateLimiter(max_requests=3, time_window=1.0)
    return _global_rate_limiter


def reset_rate_limiter() -> None:
    """Reset the global rate limiter (useful for testing)."""
    global _global_rate_limiter
    with _rate_limiter_lock:
        _global_rate_limiter = None

