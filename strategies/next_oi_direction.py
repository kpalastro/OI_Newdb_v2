"""
Next-Expiry OI Direction Strategy.

Based on validated outcome (NSE, last ~60 days, 3-minute horizon, F=3.5):
- Rule: When CE-dominant → expect down (bearish) → SELL.
        When PE-dominant → expect up (bullish) → BUY.
- Regimes: CE-dominant when call OI change dominates put by factor F;
           PE-dominant when put OI change dominates call by factor F.

Uses features: nse_next_oi_change_call_total, nse_next_oi_change_put_total,
oi_next_sentiment (put_change - call_change).
"""
from typing import Dict, Any
from .base_strategy import BaseStrategy, TradeRecommendation

# Dominance factor from validation (best directional accuracy ~56% at F=3.0–3.5).
DOMINANCE_FACTOR = 3.5
_EPS = 1e-9

# Base confidence from per-day accuracy (e.g. Tuesday/Thursday ~61–62% on NSE).
BASE_CONFIDENCE = 0.60
MAX_CONFIDENCE_AGREEMENT = 0.85


def _ce_dominant(call_change: float, put_change: float, f: float) -> bool:
    """True when CE OI change dominates PE by factor F (expect bearish)."""
    if call_change <= 0:
        return False
    return put_change <= 0 or call_change >= f * (abs(put_change) + _EPS)


def _pe_dominant(call_change: float, put_change: float, f: float) -> bool:
    """True when PE OI change dominates CE by factor F (expect bullish)."""
    if put_change <= 0:
        return False
    return call_change <= 0 or put_change >= f * (abs(call_change) + _EPS)


class NextOIDirectionStrategy(BaseStrategy):
    """
    Directional strategy from next-expiry CE/PE dominance (F=3.5):
    - CE-dominant → bearish → SELL (suggested_contract PE).
    - PE-dominant → bullish → BUY (suggested_contract CE).
    """

    def analyze(
        self,
        signal: Dict[str, Any],
        features: Dict[str, Any],
        market_state: Dict[str, Any],
    ) -> TradeRecommendation:
        oi_call_raw = features.get("nse_next_oi_change_call_total")
        oi_put_raw = features.get("nse_next_oi_change_put_total")
        try:
            oi_call = float(oi_call_raw) if oi_call_raw is not None else 0.0
            oi_put = float(oi_put_raw) if oi_put_raw is not None else 0.0
        except (TypeError, ValueError):
            oi_call, oi_put = 0.0, 0.0

        sentiment_raw = features.get("oi_next_sentiment")
        try:
            sentiment = float(sentiment_raw) if sentiment_raw is not None else 0.0
        except (TypeError, ValueError):
            sentiment = 0.0

        ml_signal = signal.get("signal", "HOLD")
        ml_conf = float(signal.get("confidence", 0.0) or 0.0)
        rationale = []
        strategy_signal = "HOLD"
        confidence = 0.0
        suggested_contract = "ATM"

        ce_dom = _ce_dominant(oi_call, oi_put, DOMINANCE_FACTOR)
        pe_dom = _pe_dominant(oi_call, oi_put, DOMINANCE_FACTOR)

        if ce_dom:
            strategy_signal = "SELL"
            confidence = BASE_CONFIDENCE
            suggested_contract = "PE"
            rationale.append(f"CE-dominant (F={DOMINANCE_FACTOR}): expect down (bearish)")
            if ml_signal == "SELL":
                confidence = min(ml_conf * 1.1, MAX_CONFIDENCE_AGREEMENT)
                rationale.append("ML agrees SELL")
        elif pe_dom:
            strategy_signal = "BUY"
            confidence = BASE_CONFIDENCE
            suggested_contract = "CE"
            rationale.append(f"PE-dominant (F={DOMINANCE_FACTOR}): expect up (bullish)")
            if ml_signal == "BUY":
                confidence = min(ml_conf * 1.1, MAX_CONFIDENCE_AGREEMENT)
                rationale.append("ML agrees BUY")
        else:
            rationale.append("No CE/PE dominance (F=3.5); neutral")
            if ml_signal in ("BUY", "SELL"):
                strategy_signal = ml_signal
                confidence = ml_conf
                suggested_contract = "CE" if ml_signal == "BUY" else "PE"

        return TradeRecommendation(
            signal=strategy_signal,
            confidence=min(max(confidence, 0.0), 1.0),
            strategy_name="NextOIDirection",
            rationale="; ".join(rationale),
            suggested_contract=suggested_contract,
            metadata={
                "oi_next_sentiment": sentiment,
                "nse_next_oi_change_call_total": oi_call,
                "nse_next_oi_change_put_total": oi_put,
                "ce_dominant": ce_dom,
                "pe_dominant": pe_dom,
                "dominance_factor": DOMINANCE_FACTOR,
            },
        )
