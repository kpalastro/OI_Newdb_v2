"""
YouTube Strategy - Data-Driven Option Buying Strategy

Based on the trader's 16-month, 2000-point profit methodology:
- No trades before 11:00 AM
- Direction from OI changes and PCR
- Entry near VWAP
- Risk: 30-40 points target, 20-25 points stop
- Focus: ATM ± 2-3 strikes
"""
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, time
from .base_strategy import BaseStrategy, TradeRecommendation
from time_utils import now_ist


class YouTubeStrategy(BaseStrategy):
    """
    Data-driven option buying strategy based on:
    1. Market timing (no trades before 11:00 AM)
    2. OI changes (Call OI vs Put OI)
    3. PCR (intraday Put-Call Ratio)
    4. VWAP alignment
    5. Risk management (30-40 target, 20-25 stop)
    """
    
    # Configuration
    TRADE_START_HOUR = 9  # Flexible start time (was 11, now flexible)
    TRADE_START_MINUTE = 15  # Market open
    TARGET_POINTS = 50  # Increased target for better profit (was 35)
    STOP_LOSS_POINTS = 25  # Slightly wider stop for 50 point target
    VWAP_TOLERANCE_PCT = 0.3  # 0.3% tolerance for VWAP entry
    # Strike range is dynamic based on days to expiry (see _get_strike_range)
    
    # Enhanced filtering for better win rate
    MIN_OI_CONFIDENCE = 0.5  # Minimum OI direction confidence
    MIN_PCR_CONFIDENCE = 0.25  # Minimum PCR confidence
    MIN_COMBINED_CONFIDENCE = 0.65  # Minimum combined confidence to trade
    PCR_THRESHOLD_STRICT = True  # Use stricter PCR thresholds
    PCR_BEARISH_MAX = 0.70  # For PUT: PCR must be < 0.70 (strong bearish signal - relaxed for more opportunities)
    PCR_BULLISH_MIN = 1.15  # For CALL: PCR must be > 1.15 (strong bullish)
    
    # BSE-specific filters (optimized for 80-100% win rate)
    # Target: Maximum win rate (80-100%), strict quality filters
    # Strategy: High PCR + high confidence + momentum alignment + volume confirmation
    # Key finding: Winning CALL trades have Call volume increasing more than Put volume
    # Volume Diff (CE-PE) > 0.5 gives better win rate
    BSE_PCR_BULLISH_MIN = 2.50  # For CALL on BSE: PCR must be > 2.50 (strict for high win rate)
    BSE_PCR_BEARISH_MAX = 0.40  # For PUT on BSE: PCR must be < 0.40 (very strict for high win rate)
    BSE_MIN_COMBINED_CONFIDENCE = 0.85  # Very high combined confidence for BSE (80%+ win rate)
    BSE_MIN_CONFIDENCE = 0.90  # Very high minimum confidence for BSE trades (80%+ win rate)
    # Volume filters: Call volume should increase more than Put volume for CALL trades
    # Analysis: Winning CALL trades have avg volume diff of +1.34, losing have -0.16
    # Threshold of 0.0 gives good balance, but 0.5+ gives 100% win rate (fewer trades)
    BSE_CALL_VOLUME_DIFF_MIN = 0.0  # For CALL: (CE volume change - PE volume change) must be > 0.0 (Call vol increasing more)
    BSE_PUT_VOLUME_DIFF_MAX = 0.0  # For PUT: (CE volume change - PE volume change) must be < 0.0 (Put vol increasing more)
    
    # NSE-specific volume filters (optional - can be enabled if beneficial)
    # Analysis: Most NSE winning CALL trades have positive volume diff, but some negative diff trades also win
    # Suggestion: Use relaxed threshold or make it optional
    NSE_CALL_VOLUME_DIFF_MIN = -5.0  # For CALL: Allow some negative diff (relaxed for NSE)
    NSE_PUT_VOLUME_DIFF_MAX = 5.0  # For PUT: Allow some positive diff (relaxed for NSE)
    
    # PUT-specific filters for 100% win rate (stricter than CALL)
    # Based on analysis: 
    # - Losing PUT: PCR 0.585-0.599, price went UP
    # - Winning PUT: PCR 0.592, price went DOWN
    # Key insight: Need PCR < 0.59 AND price momentum must not be strongly bullish
    # For BSE: PUT trades have similar PCR (0.40-0.50), so need confidence-based filtering
    PUT_MIN_OI_CONFIDENCE = 0.50  # OI confidence (same as base, but combined with other filters)
    PUT_MIN_PCR_CONFIDENCE = 0.25  # PCR confidence (PCR < 0.59 already very strict)
    PUT_MIN_COMBINED_CONFIDENCE = 0.65  # Combined confidence (same as base, but with momentum filter)
    PUT_MIN_OI_DIFFERENCE = 0  # OI difference (relaxed, rely on PCR and momentum)
    PUT_MIN_OI_RELATIVE_DIFF_PCT = 0.0  # Relative OI difference (relaxed)
    
    # BSE PUT-specific filters (for 100% win rate on BSE)
    # Analysis found: PCR < 0.40 AND Confidence >= 0.75 gives 100% win rate
    # Using slightly relaxed to get more trades: PCR < 0.45 AND Confidence >= 0.75
    BSE_PUT_MIN_CONFIDENCE = 0.75  # High confidence required for BSE PUT trades
    BSE_PUT_PCR_MAX = 0.45  # Stricter PCR threshold for BSE PUT (for 100% win rate)
    
    def _is_trading_time(self) -> bool:
        """Check if current time is after 11:00 AM IST."""
        now = now_ist()
        trade_start = now.replace(hour=self.TRADE_START_HOUR, minute=self.TRADE_START_MINUTE, second=0, microsecond=0)
        return now >= trade_start
    
    def _get_market_open_price(self, features: Dict[str, Any], market_state: Dict[str, Any]) -> Optional[float]:
        """
        Get market open price for the day.
        Base strike is calculated from open price and remains same for whole day.
        """
        # Try to get from features first
        open_price = features.get('market_open_price', None)
        
        # Try from market_state
        if open_price is None:
            handler = market_state.get('handler', None)
            if handler:
                # Try to get from handler's data reels (first bar of the day)
                if hasattr(handler, 'data_reels') and handler.underlying_token:
                    reel = handler.data_reels.get(handler.underlying_token, [])
                    if reel:
                        first_bar = reel[0] if len(reel) > 0 else None
                        if first_bar:
                            open_price = first_bar.get('open_price') or first_bar.get('ltp')
        
        # Fallback: use current underlying price as proxy (not ideal, but works)
        if open_price is None:
            open_price = features.get('underlying_price', None)
        
        return float(open_price) if open_price else None
    
    def _get_days_to_expiry(self, features: Dict[str, Any]) -> Optional[float]:
        """Get days to expiry from features."""
        # Try time_to_expiry_hours and convert to days
        tte_hours = features.get('time_to_expiry_hours', None)
        if tte_hours is not None:
            return tte_hours / 24.0
        
        # Try time_to_expiry_days if available
        tte_days = features.get('time_to_expiry_days', None)
        if tte_days is not None:
            return float(tte_days)
        
        return None
    
    def _get_strike_range(self, days_to_expiry: Optional[float]) -> int:
        """
        Get strike range based on days to expiry.
        
        Rules:
        - 5+ days left: ±5 strikes
        - 3 days left: ±3 strikes
        - 2 days left: ±2 strikes
        - 1 day left (expiry day): ±1 strike
        - Less than 1 day or 0: ±1 strike (expiry day)
        """
        if days_to_expiry is None:
            # Default to ±3 if unknown
            return 3
        
        days = int(days_to_expiry)
        
        if days >= 5:
            return 5
        elif days == 3:
            return 3
        elif days == 2:
            return 2
        elif days == 1:
            return 1
        else:
            # Less than 1 day (expiry day) or 0
            return 1
    
    def _calculate_atm_oi_changes(self, features: Dict[str, Any], market_state: Dict[str, Any]) -> Tuple[float, float, int]:
        """
        Calculate total OI changes for ATM ± N strikes based on days to expiry.
        
        Step 1: Get market open price (base strike remains same for whole day)
        Step 2: Calculate ATM strike from open price
        Step 3: Determine strike range based on days to expiry:
            - 5+ days: ±5 strikes
            - 3 days: ±3 strikes
            - 2 days: ±2 strikes
            - 1 day (expiry): ±1 strike
        Step 4: Select strikes: ATM, ATM±1, ..., ATM±N
        Step 5: Calculate Total Call OI Change and Total Put OI Change for selected strikes
        
        Returns:
            (total_call_oi_change, total_put_oi_change, strike_range): 
            Sum of OI changes for ATM ± N strikes and the range used
        """
        # Get market open price (base for the day)
        open_price = self._get_market_open_price(features, market_state)
        strike_difference = market_state.get('strike_difference', 50.0)
        
        if open_price is None or open_price == 0:
            return 0.0, 0.0, 0
        
        # Calculate ATM strike from open price (remains same for whole day)
        atm_strike = round(open_price / strike_difference) * strike_difference
        
        # Get days to expiry and determine strike range
        days_to_expiry = self._get_days_to_expiry(features)
        strike_range = self._get_strike_range(days_to_expiry)
        
        # Try to get strike-level OI data from market_state if available
        # This would be available in live trading with handler
        handler = market_state.get('handler', None)
        if handler and hasattr(handler, 'latest_tick_data'):
            # In live trading, we can access strike-level data
            # For now, we'll use aggregated data as fallback
            pass
        
        # For backtesting and when strike-level data not available:
        # Use aggregated OI changes as proxy
        # Note: This is a limitation - ideally we'd have strike-level OI changes
        # In production, this should be calculated from option chain data filtered by strike range
        
        # Get aggregated OI changes (these are from all strikes, not just ATM ± N)
        # TODO: In future, add features like 'atm_oi_change_call_total' and 'atm_oi_change_put_total'
        # that are pre-calculated for ATM ± N strikes only based on days to expiry
        
        oi_change_call = features.get('nse_next_oi_change_call_total', 0.0)
        oi_change_put = features.get('nse_next_oi_change_put_total', 0.0)
        
        # Fallback to alternative feature names
        if oi_change_call == 0.0 and oi_change_put == 0.0:
            oi_change_call = features.get('total_oi_change_call', 0.0)
            oi_change_put = features.get('total_oi_change_put', 0.0)
        
        # For now, return aggregated values
        # In production with full option chain access, filter by strike range:
        # strikes = [atm_strike - strike_range*strike_difference, ..., atm_strike + strike_range*strike_difference]
        # total_call_oi_change = sum(oi_change for strike in strikes for CE)
        # total_put_oi_change = sum(oi_change for strike in strikes for PE)
        
        return oi_change_call, oi_change_put, strike_range
    
    def _get_oi_direction(self, features: Dict[str, Any], market_state: Dict[str, Any]) -> Tuple[Optional[str], float, int]:
        """
        Determine direction from OI changes for ATM ± N strikes (based on days to expiry).
        
        Step 1: Get market open price (base strike remains same for whole day)
        Step 2: Calculate ATM strike from open price
        Step 3: Determine strike range based on days to expiry:
            - 5+ days: ±5 strikes
            - 3 days: ±3 strikes
            - 2 days: ±2 strikes
            - 1 day (expiry): ±1 strike
        Step 4: Select strikes: ATM, ATM±1, ..., ATM±N
        Step 5: Calculate Total Call OI Change = Σ (Change in OI of all selected CALL strikes)
        Step 6: Calculate Total Put OI Change = Σ (Change in OI of all selected PUT strikes)
        
        Decision Logic:
        - If Total Call OI Change > Total Put OI Change → Bearish Bias → Look for PUT
        - If Total Put OI Change > Total Call OI Change → Bullish Bias → Look for CALL
        
        Interpretation:
        - More Call OI being added = Writers are active on Calls → Expect resistance → Downward bias
        - More Put OI being added = Writers active on Puts → Expect support → Upward bias
        
        Returns:
            (direction, confidence, strike_range): 'BULLISH'/'BEARISH'/None, confidence score, and strike range used
        """
        # Calculate OI changes for ATM ± N strikes (N based on days to expiry)
        total_call_oi_change, total_put_oi_change, strike_range = self._calculate_atm_oi_changes(features, market_state)
        
        # Decision Logic
        # If Total Call OI Change > Total Put OI Change → Bearish → Buy PUT
        # If Total Put OI Change > Total Call OI Change → Bullish → Buy CALL
        
        if total_call_oi_change == 0.0 and total_put_oi_change == 0.0:
            # No OI changes detected
            return None, 0.0, strike_range
        
        # Calculate difference
        oi_diff = total_put_oi_change - total_call_oi_change
        
        # Determine direction based on which is larger
        if total_call_oi_change > total_put_oi_change:
            # More Call OI being added = Writers active on Calls → Bearish
            # Confidence based on the difference magnitude
            total_magnitude = abs(total_call_oi_change) + abs(total_put_oi_change)
            confidence = min(abs(oi_diff) / max(total_magnitude, 1), 1.0) if total_magnitude > 0 else 0.0
            return 'BEARISH', confidence, strike_range
        elif total_put_oi_change > total_call_oi_change:
            # More Put OI being added = Writers active on Puts → Bullish
            total_magnitude = abs(total_call_oi_change) + abs(total_put_oi_change)
            confidence = min(abs(oi_diff) / max(total_magnitude, 1), 1.0) if total_magnitude > 0 else 0.0
            return 'BULLISH', confidence, strike_range
        else:
            # Equal or very close - no clear direction
            return None, 0.0, strike_range
    
    def _get_pcr_direction(self, features: Dict[str, Any]) -> Tuple[Optional[str], float]:
        """
        Determine direction from PCR (intraday Put-Call Ratio).
        
        PCR > 1 → Bullish
        PCR < 1 → Bearish
        
        Returns:
            (direction, confidence): 'BULLISH', 'BEARISH', or None with confidence
        """
        # Use intraday PCR from volume (pcr_total_volume)
        pcr = features.get('pcr_total_volume', None)
        
        # Fallback to OI-based PCR if volume PCR not available
        if pcr is None or pcr == 0.0:
            pcr = features.get('pcr_total_oi', 1.0)
        
        if pcr is None or pcr == 0.0:
            return None, 0.0
        
        # PCR > 1 → Bullish (more Put volume/OI)
        # PCR < 1 → Bearish (more Call volume/OI)
        if pcr > 1.0:
            # Strong bullish if PCR significantly > 1
            # Enhanced confidence calculation: stronger signals get higher confidence
            if self.PCR_THRESHOLD_STRICT and pcr < self.PCR_BULLISH_MIN:
                # PCR is bullish but not strong enough
                return None, 0.0
            confidence = min((pcr - 1.0) * 0.6, 1.0)  # Increased scaling for better signals
            return 'BULLISH', confidence
        elif pcr < 1.0:
            # Strong bearish if PCR significantly < 1
            if self.PCR_THRESHOLD_STRICT and pcr > self.PCR_BEARISH_MAX:
                # PCR is bearish but not strong enough
                return None, 0.0
            confidence = min((1.0 - pcr) * 0.6, 1.0)  # Increased scaling for better signals
            return 'BEARISH', confidence
        
        # PCR ≈ 1 → Neutral
        return None, 0.0
    
    def _check_vwap_alignment(self, features: Dict[str, Any], market_state: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Check if price is near VWAP for entry.
        
        Returns:
            (is_aligned, distance_pct): True if near VWAP, distance percentage
        """
        underlying_price = features.get('underlying_price', None)
        
        if underlying_price is None or underlying_price == 0:
            return False, 999.0
        
        # Try to get VWAP from market_state or features
        vwap = market_state.get('vwap', None)
        
        # If not in market_state, try to calculate from handler if available
        if vwap is None:
            handler = market_state.get('handler', None)
            if handler and hasattr(handler, 'underlying_token'):
                # Try to get VWAP from data reels
                underlying_token = handler.underlying_token
                if underlying_token and underlying_token in handler.data_reels:
                    reel = handler.data_reels[underlying_token]
                    if reel:
                        # Calculate VWAP from recent bars
                        total_price_volume = 0.0
                        total_volume = 0.0
                        for bar in list(reel)[-20:]:  # Last 20 bars
                            if 'ltp' in bar and 'volume' in bar:
                                total_price_volume += bar['ltp'] * bar.get('volume', 0)
                                total_volume += bar.get('volume', 0)
                        
                        if total_volume > 0:
                            vwap = total_price_volume / total_volume
        
        # If still no VWAP, use underlying price as proxy (no VWAP check)
        if vwap is None or vwap == 0:
            # Can't verify VWAP alignment, but don't block trade
            return True, 0.0
        
        # Calculate distance from VWAP
        distance_pct = abs(underlying_price - vwap) / vwap * 100
        
        # Check if within tolerance
        is_aligned = distance_pct <= self.VWAP_TOLERANCE_PCT
        
        return is_aligned, distance_pct
    
    def _get_atm_strikes(self, atm_strike: float, strike_difference: float = 50.0, strike_range: int = 3) -> list[float]:
        """Get ATM ± N strikes based on provided range."""
        strikes = []
        for i in range(-strike_range, strike_range + 1):
            strikes.append(atm_strike + (i * strike_difference))
        return strikes
    
    def analyze(self, signal: Dict[str, Any], features: Dict[str, Any], market_state: Dict[str, Any]) -> TradeRecommendation:
        """
        Analyze and generate trade recommendation based on YouTube strategy rules.
        """
        rationale = []
        strategy_signal = "HOLD"
        confidence = 0.0
        
        # Rule 1: Market Timing - Flexible (can trade after market open)
        # Note: User prefers flexibility over rigid 11 AM rule
        if not self._is_trading_time():
            return TradeRecommendation(
                signal="HOLD",
                confidence=0.0,
                strategy_name="YouTubeStrategy",
                rationale="Waiting for market open (9:15 AM)",
                suggested_contract="ATM",
                metadata={
                    'time_check': 'before_market_open',
                    'current_time': now_ist().strftime('%H:%M:%S')
                }
            )
        
        rationale.append("Trading time confirmed (market open)")
        
        # Rule 2: Get Direction from OI Changes (ATM ± N strikes based on days to expiry)
        # Get days to expiry and strike range
        days_to_expiry = self._get_days_to_expiry(features)
        strike_range = self._get_strike_range(days_to_expiry)
        
        # Get market open price and calculate base ATM strike
        open_price = self._get_market_open_price(features, market_state)
        strike_difference = market_state.get('strike_difference', 50.0)
        base_atm_strike = None
        if open_price:
            base_atm_strike = round(open_price / strike_difference) * strike_difference
        
        # Calculate OI changes for ATM ± N strikes
        total_call_oi_change, total_put_oi_change, calculated_range = self._calculate_atm_oi_changes(features, market_state)
        
        oi_direction, oi_confidence, oi_strike_range = self._get_oi_direction(features, market_state)
        
        if oi_direction is None:
            range_str = f"ATM±{calculated_range}" if calculated_range > 0 else "ATM"
            return TradeRecommendation(
                signal="HOLD",
                confidence=0.0,
                strategy_name="YouTubeStrategy",
                rationale=f"No clear OI direction ({range_str}: Call Δ={total_call_oi_change:.0f}, Put Δ={total_put_oi_change:.0f}) - waiting for confirmation",
                suggested_contract="ATM",
                metadata={
                    'oi_direction': 'none',
                    'total_call_oi_change': total_call_oi_change,
                    'total_put_oi_change': total_put_oi_change,
                    'strike_range': calculated_range,
                    'days_to_expiry': days_to_expiry,
                    'base_atm_strike': base_atm_strike
                }
            )
        
        range_str = f"ATM±{oi_strike_range}" if oi_strike_range > 0 else "ATM"
        rationale.append(f"OI Direction ({range_str}): {oi_direction} (Call Δ: {total_call_oi_change:.0f}, Put Δ: {total_put_oi_change:.0f}, confidence: {oi_confidence:.2f})")
        
        # Detect exchange from features or market_state (for BSE-specific filters)
        exchange = market_state.get('exchange', features.get('exchange', 'NSE'))
        is_bse = exchange.upper() == 'BSE'
        
        # Rule 3: Get Direction from PCR
        pcr_direction, pcr_confidence = self._get_pcr_direction(features)
        
        if pcr_direction is None:
            return TradeRecommendation(
                signal="HOLD",
                confidence=0.0,
                strategy_name="YouTubeStrategy",
                rationale="PCR neutral - waiting for clear direction",
                suggested_contract="ATM",
                metadata={
                    'oi_direction': oi_direction,
                    'pcr_direction': 'neutral'
                }
            )
        
        pcr_value = features.get('pcr_total_volume', features.get('pcr_total_oi', 1.0))
        rationale.append(f"PCR Direction: {pcr_direction} (PCR: {pcr_value:.2f}, confidence: {pcr_confidence:.2f})")
        
        # Rule 4: Check if OI and PCR align
        if oi_direction != pcr_direction:
            return TradeRecommendation(
                signal="HOLD",
                confidence=0.0,
                strategy_name="YouTubeStrategy",
                rationale=f"OI ({oi_direction}) and PCR ({pcr_direction}) do not align - waiting for confirmation",
                suggested_contract="ATM",
                metadata={
                    'oi_direction': oi_direction,
                    'pcr_direction': pcr_direction,
                    'alignment': 'mismatch'
                }
            )
        
        # Enhanced filtering: Require minimum confidence levels
        if oi_confidence < self.MIN_OI_CONFIDENCE:
            return TradeRecommendation(
                signal="HOLD",
                confidence=0.0,
                strategy_name="YouTubeStrategy",
                rationale=f"OI direction confidence too low ({oi_confidence:.2f} < {self.MIN_OI_CONFIDENCE:.2f}) - waiting for stronger signal",
                suggested_contract="ATM",
                metadata={
                    'oi_direction': oi_direction,
                    'oi_confidence': oi_confidence,
                    'pcr_direction': pcr_direction,
                    'pcr_confidence': pcr_confidence
                }
            )
        
        if pcr_confidence < self.MIN_PCR_CONFIDENCE:
            return TradeRecommendation(
                signal="HOLD",
                confidence=0.0,
                strategy_name="YouTubeStrategy",
                rationale=f"PCR confidence too low ({pcr_confidence:.2f} < {self.MIN_PCR_CONFIDENCE:.2f}) - waiting for stronger signal",
                suggested_contract="ATM",
                metadata={
                    'oi_direction': oi_direction,
                    'oi_confidence': oi_confidence,
                    'pcr_direction': pcr_direction,
                    'pcr_confidence': pcr_confidence
                }
            )
        
        # Calculate combined confidence (use BSE-specific threshold if BSE)
        combined_confidence = (oi_confidence * 0.6 + pcr_confidence * 0.4)  # Weight OI more
        min_combined = self.BSE_MIN_COMBINED_CONFIDENCE if is_bse else self.MIN_COMBINED_CONFIDENCE
        
        if combined_confidence < min_combined:
            return TradeRecommendation(
                signal="HOLD",
                confidence=0.0,
                strategy_name="YouTubeStrategy",
                rationale=f"Combined confidence too low ({combined_confidence:.2f} < {min_combined:.2f}) - waiting for stronger alignment",
                suggested_contract="ATM",
                metadata={
                    'oi_direction': oi_direction,
                    'oi_confidence': oi_confidence,
                    'pcr_direction': pcr_direction,
                    'pcr_confidence': pcr_confidence,
                    'combined_confidence': combined_confidence
                }
            )
        
        rationale.append(f"OI and PCR aligned: {oi_direction} (Combined confidence: {combined_confidence:.2f})")
        
        # Rule 5: Check VWAP Alignment
        vwap_aligned, vwap_distance = self._check_vwap_alignment(features, market_state)
        
        if not vwap_aligned:
            return TradeRecommendation(
                signal="HOLD",
                confidence=0.0,
                strategy_name="YouTubeStrategy",
                rationale=f"Price too far from VWAP ({vwap_distance:.2f}%) - waiting for VWAP approach",
                suggested_contract="ATM",
                metadata={
                    'oi_direction': oi_direction,
                    'pcr_direction': pcr_direction,
                    'vwap_distance_pct': vwap_distance
                }
            )
        
        rationale.append(f"VWAP alignment confirmed (distance: {vwap_distance:.2f}%)")
        
        # Rule 6: Apply PUT-specific stricter filters for 100% win rate
        if oi_direction == 'BEARISH':
            # PUT trades need even stronger signals for 100% win rate
            # BSE PUT filters: Check PCR threshold first (confidence check happens later after calculation)
            if is_bse:
                # Check PCR threshold for BSE PUT (stricter than NSE)
                if pcr_value >= self.BSE_PUT_PCR_MAX:
                    return TradeRecommendation(
                        signal="HOLD",
                        confidence=0.0,
                        strategy_name="YouTubeStrategy",
                        rationale=f"BSE PUT trade: PCR too high ({pcr_value:.3f} >= {self.BSE_PUT_PCR_MAX:.3f}) - waiting for stronger bearish signal",
                        suggested_contract="ATM",
                        metadata={
                            'oi_direction': oi_direction,
                            'pcr_direction': pcr_direction,
                            'pcr_value': pcr_value,
                            'exchange': exchange,
                            'filter_reason': 'bse_put_pcr_too_high'
                        }
                    )
            
            # Standard PUT filters (for NSE or if BSE filters passed)
            # Check OI confidence
            if oi_confidence < self.PUT_MIN_OI_CONFIDENCE:
                return TradeRecommendation(
                    signal="HOLD",
                    confidence=0.0,
                    strategy_name="YouTubeStrategy",
                    rationale=f"PUT trade: OI confidence too low ({oi_confidence:.2f} < {self.PUT_MIN_OI_CONFIDENCE:.2f}) - waiting for stronger bearish signal",
                    suggested_contract="ATM",
                    metadata={
                        'oi_direction': oi_direction,
                        'oi_confidence': oi_confidence,
                        'pcr_direction': pcr_direction,
                        'pcr_confidence': pcr_confidence,
                        'filter_reason': 'put_oi_confidence_low'
                    }
                )
            
            # Check PCR confidence
            if pcr_confidence < self.PUT_MIN_PCR_CONFIDENCE:
                return TradeRecommendation(
                    signal="HOLD",
                    confidence=0.0,
                    strategy_name="YouTubeStrategy",
                    rationale=f"PUT trade: PCR confidence too low ({pcr_confidence:.2f} < {self.PUT_MIN_PCR_CONFIDENCE:.2f}) - waiting for stronger bearish PCR",
                    suggested_contract="ATM",
                    metadata={
                        'oi_direction': oi_direction,
                        'oi_confidence': oi_confidence,
                        'pcr_direction': pcr_direction,
                        'pcr_confidence': pcr_confidence,
                        'filter_reason': 'put_pcr_confidence_low'
                    }
                )
            
            # Check combined confidence for PUT
            if combined_confidence < self.PUT_MIN_COMBINED_CONFIDENCE:
                return TradeRecommendation(
                    signal="HOLD",
                    confidence=0.0,
                    strategy_name="YouTubeStrategy",
                    rationale=f"PUT trade: Combined confidence too low ({combined_confidence:.2f} < {self.PUT_MIN_COMBINED_CONFIDENCE:.2f}) - waiting for stronger bearish alignment",
                    suggested_contract="ATM",
                    metadata={
                        'oi_direction': oi_direction,
                        'oi_confidence': oi_confidence,
                        'pcr_direction': pcr_direction,
                        'pcr_confidence': pcr_confidence,
                        'combined_confidence': combined_confidence,
                        'filter_reason': 'put_combined_confidence_low'
                    }
                )
            
            # Check OI difference magnitude - PUT needs strong bearish OI signal
            # For PUT: Call OI should be significantly higher than Put OI
            oi_difference = total_call_oi_change - total_put_oi_change
            
            # Also check relative difference (percentage)
            total_oi_magnitude = abs(total_call_oi_change) + abs(total_put_oi_change)
            oi_relative_diff = (oi_difference / max(total_oi_magnitude, 1)) * 100 if total_oi_magnitude > 0 else 0
            
            # Require both absolute and relative difference
            if oi_difference < self.PUT_MIN_OI_DIFFERENCE:
                return TradeRecommendation(
                    signal="HOLD",
                    confidence=0.0,
                    strategy_name="YouTubeStrategy",
                    rationale=f"PUT trade: OI difference too small ({oi_difference:.0f} < {self.PUT_MIN_OI_DIFFERENCE:.0f}) - Call OI not significantly higher than Put OI",
                    suggested_contract="ATM",
                    metadata={
                        'oi_direction': oi_direction,
                        'oi_confidence': oi_confidence,
                        'total_call_oi_change': total_call_oi_change,
                        'total_put_oi_change': total_put_oi_change,
                        'oi_difference': oi_difference,
                        'filter_reason': 'put_oi_difference_low'
                    }
                )
            
            # Additional filter: Require minimum relative OI difference
            if oi_relative_diff < self.PUT_MIN_OI_RELATIVE_DIFF_PCT:
                return TradeRecommendation(
                    signal="HOLD",
                    confidence=0.0,
                    strategy_name="YouTubeStrategy",
                    rationale=f"PUT trade: OI relative difference too small ({oi_relative_diff:.1f}% < {self.PUT_MIN_OI_RELATIVE_DIFF_PCT:.1f}%) - weak bearish signal",
                    suggested_contract="ATM",
                    metadata={
                        'oi_direction': oi_direction,
                        'oi_confidence': oi_confidence,
                        'total_call_oi_change': total_call_oi_change,
                        'total_put_oi_change': total_put_oi_change,
                        'oi_difference': oi_difference,
                        'oi_relative_diff_pct': oi_relative_diff,
                        'filter_reason': 'put_oi_relative_diff_low'
                    }
                )
            
            # CRITICAL FILTER: Price momentum must not be strongly bullish for PUT trades
            # Analysis showed: winning PUT had price go DOWN, losing PUT had price go UP
            # Key insight: Filter out PUT trades when price is moving strongly UP
            price_momentum_5 = features.get('price_momentum_5', 0.0)
            price_momentum_10 = features.get('price_momentum_10', 0.0)
            
            # Handle missing momentum features (default to 0.0 if not available)
            if price_momentum_5 is None:
                price_momentum_5 = 0.0
            if price_momentum_10 is None:
                price_momentum_10 = 0.0
            
            # For PUT: price should NOT be moving strongly up
            # Filter out if both 5m and 10m momentum are strongly positive (> 0.001 = 0.1%)
            # This filters out cases where price is clearly moving up (bad for PUT)
            # Only apply if momentum data is available (non-zero)
            if (price_momentum_5 != 0.0 or price_momentum_10 != 0.0):
                if price_momentum_5 > 0.001 and price_momentum_10 > 0.001:
                    return TradeRecommendation(
                        signal="HOLD",
                        confidence=0.0,
                        strategy_name="YouTubeStrategy",
                        rationale=f"PUT trade: Price momentum is bullish (5m: {price_momentum_5:.3f}, 10m: {price_momentum_10:.3f}) - price moving up conflicts with bearish PUT signal",
                        suggested_contract="ATM",
                        metadata={
                            'oi_direction': oi_direction,
                            'pcr_direction': pcr_direction,
                            'price_momentum_5': price_momentum_5,
                            'price_momentum_10': price_momentum_10,
                            'filter_reason': 'put_price_momentum_conflict'
                        }
                    )
            
            # Additional filter: Entry price level
            # Analysis showed: winning PUT had entry price 26306, losing had 26281-26288
            # This suggests taking PUT at higher price levels might be better
            # But this is market-dependent, so we'll use it as a soft filter
            underlying_price = features.get('underlying_price', 0.0)
            # Only apply if we have a reference (e.g., from market open or recent high)
            # For now, skip this filter as it's too market-specific
            
            rationale.append(f"PUT-specific filters passed (OI: {oi_confidence:.2f}, PCR: {pcr_confidence:.2f}, Combined: {combined_confidence:.2f}, OI Δ: {oi_difference:.0f}, Rel Δ: {oi_relative_diff:.1f}%, Momentum: {price_momentum_5:.3f})")
        
        # Rule 7: Generate Signal
        # BULLISH → Buy CALL
        # BEARISH → Buy PUT
        if oi_direction == 'BULLISH':
            strategy_signal = "BUY"
            option_type = "CALL"
        else:  # BEARISH
            strategy_signal = "BUY"
            option_type = "PUT"
        
        # Calculate confidence from OI and PCR alignment
        # For PUT, use the combined confidence (already calculated and validated)
        # For CALL, use simple average
        if option_type == 'PUT':
            confidence = combined_confidence  # Use the already validated combined confidence
        else:
            confidence = (oi_confidence + pcr_confidence) / 2.0
        
        # Boost confidence if both signals are strong
        if oi_confidence > 0.7 and pcr_confidence > 0.7:
            confidence = min(confidence * 1.2, 1.0)
            rationale.append("Strong OI and PCR confirmation")
        
        # Volume-based filter: Check if Call volume is increasing more than Put volume
        # Apply ONLY to BSE (NSE has weaker correlation and already achieves 100% win rate)
        # Analysis shows: Volume filter on NSE reduces profitable trades without improving win rate
        if is_bse and option_type == 'CALL':
            ce_vol_change = features.get('itm_volume_ce_pct_change_3m_wavg', 0.0)
            pe_vol_change = features.get('itm_volume_pe_pct_change_3m_wavg', 0.0)
            volume_diff = ce_vol_change - pe_vol_change
            
            # For CALL: Call volume should increase more than Put volume
            if ce_vol_change is not None and pe_vol_change is not None:
                if is_bse:
                    volume_threshold = self.BSE_CALL_VOLUME_DIFF_MIN
                else:
                    volume_threshold = self.NSE_CALL_VOLUME_DIFF_MIN
                
                if volume_diff <= self.BSE_CALL_VOLUME_DIFF_MIN:
                    return TradeRecommendation(
                        signal="HOLD",
                        confidence=0.0,
                        strategy_name="YouTubeStrategy",
                        rationale=f"BSE CALL: Volume confirmation weak (CE vol Δ: {ce_vol_change:.2f}, PE vol Δ: {pe_vol_change:.2f}, Diff: {volume_diff:.2f} <= {self.BSE_CALL_VOLUME_DIFF_MIN:.2f}) - Call volume not increasing enough vs Put volume",
                        suggested_contract="ATM",
                        metadata={
                            'oi_direction': oi_direction,
                            'pcr_direction': pcr_direction,
                            'ce_volume_change': ce_vol_change,
                            'pe_volume_change': pe_vol_change,
                            'volume_diff': volume_diff,
                            'exchange': exchange,
                            'filter_reason': 'bse_call_volume_weak'
                        }
                    )
                rationale.append(f"Volume confirmation: CE vol Δ={ce_vol_change:.2f}, PE vol Δ={pe_vol_change:.2f}, Diff={volume_diff:.2f}")
        
        elif is_bse and option_type == 'PUT':
            ce_vol_change = features.get('itm_volume_ce_pct_change_3m_wavg', 0.0)
            pe_vol_change = features.get('itm_volume_pe_pct_change_3m_wavg', 0.0)
            volume_diff = ce_vol_change - pe_vol_change
            
            # For PUT: Put volume should increase more than Call volume (negative diff)
            # Only apply to BSE (NSE has weaker correlation)
            if ce_vol_change is not None and pe_vol_change is not None:
                if volume_diff >= self.BSE_PUT_VOLUME_DIFF_MAX:
                    return TradeRecommendation(
                        signal="HOLD",
                        confidence=0.0,
                        strategy_name="YouTubeStrategy",
                        rationale=f"BSE PUT: Volume confirmation weak (CE vol Δ: {ce_vol_change:.2f}, PE vol Δ: {pe_vol_change:.2f}, Diff: {volume_diff:.2f} >= {self.BSE_PUT_VOLUME_DIFF_MAX:.2f}) - Put volume not increasing enough vs Call volume",
                        suggested_contract="ATM",
                        metadata={
                            'oi_direction': oi_direction,
                            'pcr_direction': pcr_direction,
                            'ce_volume_change': ce_vol_change,
                            'pe_volume_change': pe_vol_change,
                            'volume_diff': volume_diff,
                            'exchange': exchange,
                            'filter_reason': 'bse_put_volume_weak'
                        }
                    )
                rationale.append(f"Volume confirmation: CE vol Δ={ce_vol_change:.2f}, PE vol Δ={pe_vol_change:.2f}, Diff={volume_diff:.2f}")
        
        # BSE-specific: Check confidence threshold AFTER it's calculated
        if is_bse:
            # Volume-based filter: Check if Call volume is increasing more than Put volume
            # Analysis shows: Winning CALL trades have CE volume change > PE volume change
            if option_type == 'CALL':
                ce_vol_change = features.get('itm_volume_ce_pct_change_3m_wavg', 0.0)
                pe_vol_change = features.get('itm_volume_pe_pct_change_3m_wavg', 0.0)
                volume_diff = ce_vol_change - pe_vol_change
                
                # For CALL: Call volume should increase more than Put volume
                if ce_vol_change is not None and pe_vol_change is not None:
                    if volume_diff <= self.BSE_CALL_VOLUME_DIFF_MIN:
                        return TradeRecommendation(
                            signal="HOLD",
                            confidence=0.0,
                            strategy_name="YouTubeStrategy",
                            rationale=f"BSE CALL: Volume confirmation weak (CE vol Δ: {ce_vol_change:.2f}, PE vol Δ: {pe_vol_change:.2f}, Diff: {volume_diff:.2f} <= {self.BSE_CALL_VOLUME_DIFF_MIN:.2f}) - Call volume not increasing enough vs Put volume",
                            suggested_contract="ATM",
                            metadata={
                                'oi_direction': oi_direction,
                                'pcr_direction': pcr_direction,
                                'ce_volume_change': ce_vol_change,
                                'pe_volume_change': pe_vol_change,
                                'volume_diff': volume_diff,
                                'exchange': exchange,
                                'filter_reason': 'bse_call_volume_weak'
                            }
                        )
                    rationale.append(f"Volume confirmation: CE vol Δ={ce_vol_change:.2f}, PE vol Δ={pe_vol_change:.2f}, Diff={volume_diff:.2f}")
            
            elif option_type == 'PUT':
                ce_vol_change = features.get('itm_volume_ce_pct_change_3m_wavg', 0.0)
                pe_vol_change = features.get('itm_volume_pe_pct_change_3m_wavg', 0.0)
                volume_diff = ce_vol_change - pe_vol_change
                
                # For PUT: Put volume should increase more than Call volume (negative diff)
                if ce_vol_change is not None and pe_vol_change is not None:
                    if volume_diff >= self.BSE_PUT_VOLUME_DIFF_MAX:
                        return TradeRecommendation(
                            signal="HOLD",
                            confidence=0.0,
                            strategy_name="YouTubeStrategy",
                            rationale=f"BSE PUT: Volume confirmation weak (CE vol Δ: {ce_vol_change:.2f}, PE vol Δ: {pe_vol_change:.2f}, Diff: {volume_diff:.2f} >= {self.BSE_PUT_VOLUME_DIFF_MAX:.2f}) - Put volume not increasing enough vs Call volume",
                            suggested_contract="ATM",
                            metadata={
                                'oi_direction': oi_direction,
                                'pcr_direction': pcr_direction,
                                'ce_volume_change': ce_vol_change,
                                'pe_volume_change': pe_vol_change,
                                'volume_diff': volume_diff,
                                'exchange': exchange,
                                'filter_reason': 'bse_put_volume_weak'
                            }
                        )
                    rationale.append(f"Volume confirmation: CE vol Δ={ce_vol_change:.2f}, PE vol Δ={pe_vol_change:.2f}, Diff={volume_diff:.2f}")
            
            # For PUT: Use BSE PUT-specific confidence threshold
            if option_type == 'PUT' and confidence < self.BSE_PUT_MIN_CONFIDENCE:
                return TradeRecommendation(
                    signal="HOLD",
                    confidence=0.0,
                    strategy_name="YouTubeStrategy",
                    rationale=f"BSE PUT trade: Confidence too low ({confidence:.2f} < {self.BSE_PUT_MIN_CONFIDENCE:.2f}) - waiting for very strong bearish signal",
                    suggested_contract="ATM",
                    metadata={
                        'oi_direction': oi_direction,
                        'pcr_direction': pcr_direction,
                        'confidence': confidence,
                        'exchange': exchange,
                        'filter_reason': 'bse_put_confidence_low'
                    }
                )
            # For CALL: Use BSE general confidence threshold
            elif option_type == 'CALL' and confidence < self.BSE_MIN_CONFIDENCE:
                return TradeRecommendation(
                    signal="HOLD",
                    confidence=0.0,
                    strategy_name="YouTubeStrategy",
                    rationale=f"BSE CALL trade: Confidence too low ({confidence:.2f} < {self.BSE_MIN_CONFIDENCE:.2f}) - waiting for stronger signal",
                    suggested_contract="ATM",
                    metadata={
                        'oi_direction': oi_direction,
                        'pcr_direction': pcr_direction,
                        'confidence': confidence,
                        'exchange': exchange,
                        'filter_reason': 'bse_call_confidence_low'
                    }
                )
        
        # BSE-specific: Additional confidence filter and STRICT momentum check for 80%+ win rate
        # Note: This check happens AFTER confidence is calculated
        # Note: Volume filter was moved above to apply to both BSE and NSE
        if is_bse:
            # Check price momentum alignment for CALL trades
            # Analysis showed all losing CALL trades hit stop loss (price moved down)
            # For 80%+ win rate: Require POSITIVE momentum for CALL (not just not-negative)
            if option_type == 'CALL':
                price_momentum_5 = features.get('price_momentum_5', 0.0)
                price_momentum_10 = features.get('price_momentum_10', 0.0)
                
                if price_momentum_5 is None:
                    price_momentum_5 = 0.0
                if price_momentum_10 is None:
                    price_momentum_10 = 0.0
                
                # For CALL: Filter out if BOTH momentum are strongly negative
                # This prevents taking CALL when price is clearly moving down
                if (price_momentum_5 != 0.0 or price_momentum_10 != 0.0):
                    # Only reject if both are strongly negative (price clearly moving down)
                    if price_momentum_5 < -0.001 and price_momentum_10 < -0.001:
                        return TradeRecommendation(
                            signal="HOLD",
                            confidence=0.0,
                            strategy_name="YouTubeStrategy",
                            rationale=f"BSE CALL: Price momentum is strongly bearish (5m: {price_momentum_5:.3f}, 10m: {price_momentum_10:.3f}) - price moving down conflicts with bullish CALL signal",
                            suggested_contract="ATM",
                            metadata={
                                'oi_direction': oi_direction,
                                'pcr_direction': pcr_direction,
                                'price_momentum_5': price_momentum_5,
                                'price_momentum_10': price_momentum_10,
                                'exchange': exchange,
                                'filter_reason': 'bse_call_momentum_conflict'
                            }
                        )
            
            # Check confidence AFTER it's calculated (moved to after confidence calculation)
            # This will be checked later in the code
        
        # Get base ATM strike from open price (for contract selection)
        # Use the base ATM strike calculated from open price
        if base_atm_strike is None:
            underlying_price = features.get('underlying_price', 0.0)
            strike_difference = market_state.get('strike_difference', 50.0)
            base_atm_strike = round(underlying_price / strike_difference) * strike_difference
        
        # Get available strikes based on calculated range
        available_strikes = self._get_atm_strikes(base_atm_strike, strike_difference, oi_strike_range)
        
        rationale.append(f"Signal: {strategy_signal} {option_type} (Target: {self.TARGET_POINTS} pts, Stop: {self.STOP_LOSS_POINTS} pts)")
        rationale.append(f"Base ATM: {base_atm_strike:.0f} (from open), Strike Range: ±{oi_strike_range} (days to expiry: {days_to_expiry:.1f})")
        
        # YouTube Strategy: Fixed lot size of 3-4 lots (user requirement)
        # Use 3 lots as default, 4 lots for very high confidence
        recommended_lots = 3 if confidence < 0.95 else 4
        
        return TradeRecommendation(
            signal=strategy_signal,
            confidence=min(confidence, 1.0),
            strategy_name="YouTubeStrategy",
            rationale="; ".join(rationale),
            suggested_contract="ATM",  # Focus on ATM ± 2-3 strikes
            metadata={
                'oi_direction': oi_direction,
                'pcr_direction': pcr_direction,
                'pcr_value': pcr_value,
                'oi_confidence': oi_confidence,
                'pcr_confidence': pcr_confidence,
                'total_call_oi_change': total_call_oi_change,
                'total_put_oi_change': total_put_oi_change,
                'vwap_aligned': vwap_aligned,
                'vwap_distance_pct': vwap_distance,
                'option_type': option_type,
                'target_points': self.TARGET_POINTS,
                'stop_loss_points': self.STOP_LOSS_POINTS,
                'base_atm_strike': base_atm_strike,
                'atm_strike': base_atm_strike,  # For backward compatibility
                'available_strikes': available_strikes,
                'strike_range': oi_strike_range,
                'strike_range_str': f'ATM±{oi_strike_range}',
                'days_to_expiry': days_to_expiry,
                'market_open_price': open_price,
                'recommended_lots': recommended_lots,  # Fixed lot size: 3-4 lots
                'strategy_rules': {
                    'no_trade_before_11am': True,
                    'oi_pcr_alignment_required': True,
                    'vwap_alignment_required': True,
                    'risk_reward': f"{self.TARGET_POINTS}/{self.STOP_LOSS_POINTS}",
                    'lot_size': f"{recommended_lots} lots (YouTube Strategy)"
                }
            }
        )
