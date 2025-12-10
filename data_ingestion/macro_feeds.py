"""
Macro / fund-flow ingestion helpers.

Provides a tiny abstraction so scheduled jobs or external scripts can push
new macro datapoints into SQLite without duplicating SQL logic.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import database_new as db


def record_macro_snapshot(
    exchange: str,
    fii_flow: float | None = None,
    dii_flow: float | None = None,
    usdinr: float | None = None,
    usdinr_trend: float | None = None,
    crude_price: float | None = None,
    crude_trend: float | None = None,
    banknifty_correlation: float | None = None,
    macro_spread: float | None = None,
    risk_on_score: float | None = None,
    metadata: Optional[Dict[str, Any]] = None,
    timestamp: datetime | None = None,
) -> None:
    """
    Persist a macro snapshot. Designed to be called by scheduled ingestion jobs.
    """
    db.save_macro_signals(
        exchange=exchange,
        fii_flow=fii_flow,
        dii_flow=dii_flow,
        usdinr=usdinr,
        usdinr_trend=usdinr_trend,
        crude_price=crude_price,
        crude_trend=crude_trend,
        banknifty_correlation=banknifty_correlation,
        macro_spread=macro_spread,
        risk_on_score=risk_on_score,
        metadata=metadata,
        timestamp=timestamp,
    )

