from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from handlers import HandlerFeatureSnapshot


@dataclass
class FeatureJob:
    exchange: str
    timestamp: datetime
    spot_ltp: float
    atm: float
    time_yrs: float
    fut_oi: Optional[float]
    futures_price: Optional[float]
    vix_value: Optional[float]
    calls: List[Dict[str, Any]]
    puts: List[Dict[str, Any]]
    handler_snapshot: HandlerFeatureSnapshot


@dataclass
class ResultJob:
    exchange: str
    timestamp: datetime
    calls: List[Dict[str, Any]]
    puts: List[Dict[str, Any]]
    latest_oi_data: Dict[str, Any]
    ml_signal: str
    ml_confidence: float
    ml_rationale: str
    ml_metadata: Dict[str, Any]
    ml_features: Dict[str, Any]
    spot_ltp: float
    atm: float
    fut_oi: Optional[float]
    futures_price: Optional[float]
    iv_updates: Dict[int, float]

