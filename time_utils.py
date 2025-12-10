"""
time_utils.py

Centralised helpers for working with Indian Standard Time (IST).

All application timestamps and market‑time calculations should use these
helpers so that behaviour is consistent regardless of the server's local
time zone (e.g., when deploying to cloud regions outside India).
"""
from __future__ import annotations

from datetime import datetime, date, timedelta, timezone

try:
    # Python 3.9+ standard library time zone support
    from zoneinfo import ZoneInfo  # type: ignore[attr-defined]

    try:
        IST = ZoneInfo("Asia/Kolkata")
    except Exception:  # pragma: no cover - environment may lack tzdata
        IST = None
except Exception:  # pragma: no cover - very old Python fallback
    ZoneInfo = None  # type: ignore
    IST = None


def now_ist() -> datetime:
    """
    Return the current time in Indian Standard Time as an aware datetime.

    If the system does not have the Asia/Kolkata time zone definition
    available, this falls back to UTC+05:30 using a fixed offset.
    """
    if IST is not None:
        return datetime.now(IST)
    # Fallback: fixed offset of +5:30 hours from UTC
    return datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)


def today_ist() -> date:
    """Return today's date in IST."""
    return now_ist().date()


def to_ist(dt: datetime) -> datetime:
    """
    Convert a datetime to an aware IST datetime.

    If `dt` is naive, it is assumed to already represent an IST wall‑clock
    time and is tagged with the IST time zone without shifting the clock.
    """
    if IST is None:
        # Without proper tz info, just return the original datetime
        return dt
    if dt.tzinfo is None:
        return dt.replace(tzinfo=IST)
    return dt.astimezone(IST)


