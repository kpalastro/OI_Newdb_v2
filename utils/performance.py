"""
Performance instrumentation and timing utilities.
"""
import time
import functools
import logging
from typing import Callable, Any, Dict
from collections import defaultdict
from threading import Lock


# Performance metrics storage
_performance_metrics: Dict[str, list] = defaultdict(list)
_metrics_lock = Lock()


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure and log function execution time.
    
    Usage:
        @timing_decorator
        def my_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start_time
            func_name = f"{func.__module__}.{func.__name__}"
            logging.debug(f"[PERF] {func_name} took {elapsed:.3f}s")
            
            # Store metric
            with _metrics_lock:
                _performance_metrics[func_name].append(elapsed)
                # Keep only last 100 measurements
                if len(_performance_metrics[func_name]) > 100:
                    _performance_metrics[func_name] = _performance_metrics[func_name][-100:]
    
    return wrapper


def get_performance_stats() -> Dict[str, Dict[str, float]]:
    """
    Get performance statistics for all timed functions.
    
    Returns:
        Dictionary mapping function names to stats (min, max, avg, count)
    """
    with _metrics_lock:
        stats = {}
        for func_name, times in _performance_metrics.items():
            if times:
                stats[func_name] = {
                    'min': min(times),
                    'max': max(times),
                    'avg': sum(times) / len(times),
                    'count': len(times),
                    'total': sum(times)
                }
        return stats


def clear_performance_stats() -> None:
    """Clear all performance statistics."""
    with _metrics_lock:
        _performance_metrics.clear()


class PerformanceTimer:
    """
    Context manager for timing code blocks.
    
    Usage:
        with PerformanceTimer("operation_name"):
            # code to time
            ...
    """
    
    def __init__(self, name: str, log_level: int = logging.DEBUG):
        self.name = name
        self.log_level = log_level
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        logging.log(self.log_level, f"[PERF] {self.name} took {elapsed:.3f}s")
        return False


def log_slow_operation(threshold: float = 1.0):
    """
    Decorator that logs warnings for operations taking longer than threshold.
    
    Args:
        threshold: Time in seconds to consider as "slow"
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start_time
                if elapsed > threshold:
                    func_name = f"{func.__module__}.{func.__name__}"
                    logging.warning(
                        f"[PERF] SLOW OPERATION: {func_name} took {elapsed:.3f}s "
                        f"(threshold: {threshold:.3f}s)"
                    )
        
        return wrapper
    return decorator

