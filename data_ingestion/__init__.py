"""
Data ingestion helpers for Phase 2 expansion.

These modules centralise reusable routines for recording VIX term structure,
macro/fund-flow signals, and order-book depth aggregates. Actual vendor/API
integration can plug into these helpers without touching the runtime.
"""

from .vix_term_structure import record_vix_term_structure  # noqa: F401
from .macro_feeds import record_macro_snapshot  # noqa: F401
from .depth_capture import summarize_depth_levels  # noqa: F401

