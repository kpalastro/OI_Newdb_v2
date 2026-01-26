"""
ITM Feature Evaluator - Enforces reverse-engineered insights from trade log analysis.

This module evaluates ITM features and provides trading decision modifiers based on
statistical analysis of winning vs losing trades.

Key Findings (Updated Jan 2026):

NSE:
- ITM PE Vol Δ% is the most important feature (correlation: -0.30)
- ITM CE Vol Δ% is highly important (correlation: -0.17)
- Lower volume changes = Better for trades (contrarian indicator)

BSE (Jan 2026 Analysis):
- ITM PE Vol Δ%: correlation +0.55 (HIGHER is better - OPPOSITE of NSE!)
- ITM CE Vol Δ%: correlation -0.32 (lower is better, same as NSE)
- ITM PE Δ%: correlation +0.21 (higher is better)
- ITM CE Δ%: correlation +0.04 (neutral)

NEW Bottom Detection Findings (Jan 2026 Analysis):
- Bottom signals have STRONG correlation with profitability
- Bottom signals vs PnL: +0.399 correlation
- Bottom signals vs Win Rate: +0.575 correlation
- High bottom detection days: Avg PnL +₹27,659 (vs -₹20,394 for low bottom days)
- Type 1 Bottom: CE+/PE- with divergence > 10% (smart money accumulation)
- Type 2 Bottom: PE+/CE- with divergence < -8% (traditional bullish)
"""

from typing import Dict, Any, List, Tuple, Optional

# ============================================================================
# EXCHANGE-SPECIFIC THRESHOLDS
# ============================================================================

# NSE Thresholds (Default - from original analysis)
NSE_THRESHOLDS = {
    'itm_pe_vol_optimal': -3.0,       # Below this is optimal (volume decreasing)
    'itm_ce_vol_optimal': 10.0,       # Below this is optimal
    'itm_ce_vol_warning': 16.0,       # Above this is warning (volume spike)
    'itm_pe_delta_optimal_min': -0.65,
    'itm_pe_delta_optimal_max': 0.0,
    'itm_ce_delta_optimal_min': 0.3,
    'itm_ce_delta_optimal_max': 2.5,
    # For NSE: Lower PE Vol is better (negative correlation)
    'pe_vol_direction': 'lower_better',
}

# BSE Thresholds (From Jan 2026 Analysis - DIFFERENT from NSE!)
BSE_THRESHOLDS = {
    'itm_pe_vol_optimal': 3.8,        # Above this is optimal (volume INCREASING for BSE!)
    'itm_ce_vol_optimal': 0.0,        # At or below this is optimal
    'itm_ce_vol_warning': 7.5,        # Above this is warning
    'itm_pe_delta_optimal_min': 3.5,  # Higher PE delta is better for BSE
    'itm_pe_delta_optimal_max': 187.4,
    'itm_ce_delta_optimal_min': -0.45,
    'itm_ce_delta_optimal_max': 1.64,
    # For BSE: Higher PE Vol is better (positive correlation +0.55!)
    'pe_vol_direction': 'higher_better',
}

def get_thresholds(exchange: Optional[str] = None) -> dict:
    """Get exchange-specific thresholds."""
    if exchange == 'BSE':
        return BSE_THRESHOLDS
    return NSE_THRESHOLDS

# Legacy global constants (for backward compatibility - uses NSE defaults)
ITM_PE_VOL_OPTIMAL = NSE_THRESHOLDS['itm_pe_vol_optimal']
ITM_CE_VOL_OPTIMAL = NSE_THRESHOLDS['itm_ce_vol_optimal']
ITM_CE_VOL_WARNING = NSE_THRESHOLDS['itm_ce_vol_warning']
ITM_PE_DELTA_OPTIMAL_MIN = NSE_THRESHOLDS['itm_pe_delta_optimal_min']
ITM_PE_DELTA_OPTIMAL_MAX = NSE_THRESHOLDS['itm_pe_delta_optimal_max']
ITM_CE_DELTA_OPTIMAL_MIN = NSE_THRESHOLDS['itm_ce_delta_optimal_min']
ITM_CE_DELTA_OPTIMAL_MAX = NSE_THRESHOLDS['itm_ce_delta_optimal_max']

# NEW: Bottom Detection Thresholds (Jan 2026 Analysis)
BOTTOM_TYPE1_CE_MIN = 5.0       # CE Δ% > 5% (call accumulation)
BOTTOM_TYPE1_PE_MAX = -5.0      # PE Δ% < -5% (put unwinding)
BOTTOM_TYPE1_DIV_MIN = 10.0     # Divergence > 10%
BOTTOM_TYPE1_DIV_VERY_STRONG = 15.0  # Very strong bottom

BOTTOM_TYPE2_PE_MIN = 5.0       # PE Δ% > 5% (put accumulation)
BOTTOM_TYPE2_CE_MAX = -5.0      # CE Δ% < -5% (call unwinding)
BOTTOM_TYPE2_DIV_MAX = -8.0     # Divergence < -8%

# Peak Detection Thresholds
PEAK_DIVERGENCE_MIN = 3.0       # Divergence > 3% for peak detection
PEAK_STRONG_DIVERGENCE = 5.0    # Strong peak detection


def evaluate_itm_features(features_dict: Dict[str, float], exchange: Optional[str] = None) -> Dict[str, Any]:
    """
    Comprehensive ITM feature evaluation based on reverse engineering analysis.
    
    This function evaluates the four key ITM features:
    1. ITM PE Vol Δ% (Most Important - 40% weight)
    2. ITM CE Vol Δ% (High Importance - 30% weight)
    3. ITM PE Δ% (Medium Importance - 20% weight)
    4. ITM CE Δ% (Low Importance - 10% weight)
    
    PLUS NEW Bottom Detection (Jan 2026 Analysis):
    - Type 1 Bottom: CE+/PE- with divergence > 10% (accumulation)
    - Type 2 Bottom: PE+/CE- with divergence < -8% (traditional bullish)
    - Bottom signals correlation with PnL: +0.399, Win Rate: +0.575
    
    EXCHANGE-SPECIFIC BEHAVIOR (Jan 2026):
    - NSE: Lower PE Vol is better (correlation -0.30)
    - BSE: Higher PE Vol is better (correlation +0.55) - OPPOSITE!
    
    Args:
        features_dict: Dictionary containing feature values
        exchange: Optional exchange code ('NSE' or 'BSE') for exchange-specific thresholds
        
    Returns:
        {
            'should_skip_trade': bool,  # True if trade should be skipped
            'confidence_multiplier': float,  # Multiplier for confidence (0.7 to 1.25)
            'position_size_multiplier': float,  # Multiplier for position size (0.5 to 1.5)
            'itm_score': float,  # Combined ITM score (0-1)
            'reasons': List[str],  # Positive reasons
            'warnings': List[str],  # Warning messages
            'raw_values': Dict[str, float],  # Raw feature values
            'bearish_signal': bool,  # BEARISH signal detected
            'bullish_signal': bool,  # BULLISH signal detected
            'peak_detection': bool,  # Peak detected (skip longs)
            'bottom_detection': bool,  # Bottom detected (enter longs)
            'bottom_type': str,  # 'type1', 'type2', or 'none'
            'market_regime': str,  # 'range_bound' or 'trending'
            'exchange': str,  # Exchange used for evaluation
        }
    """
    # Get exchange-specific thresholds
    thresholds = get_thresholds(exchange)
    is_bse = (exchange == 'BSE')
    
    # Extract ITM features
    itm_pe_vol = features_dict.get('itm_volume_pe_pct_change_3m_wavg', 0.0)
    itm_ce_vol = features_dict.get('itm_volume_ce_pct_change_3m_wavg', 0.0)
    itm_pe_delta = features_dict.get('itm_oi_pe_pct_change_3m_wavg', 0.0)
    itm_ce_delta = features_dict.get('itm_oi_ce_pct_change_3m_wavg', 0.0)
    
    # Initialize return values
    should_skip = False
    confidence_mult = 1.0
    position_mult = 1.0
    reasons = []
    warnings = []
    itm_score = 0.0
    
    # 1. ITM PE Vol Δ% (MOST IMPORTANT - 40% weight)
    # NSE: Correlation -0.30 (lower is better - put selling)
    # BSE: Correlation +0.55 (HIGHER is better - different market dynamics!)
    if is_bse:
        # BSE: Higher PE Vol is BETTER (positive correlation +0.55)
        if itm_pe_vol > thresholds['itm_pe_vol_optimal']:
            # Strong signal for BSE - volume increasing is bullish
            confidence_mult = min(1.15, confidence_mult + 0.15)
            position_mult = min(1.2, position_mult + 0.1)
            itm_score += 0.4
            reasons.append(f"✓ [BSE] ITM PE Vol Δ% = {itm_pe_vol:.2f}% (strong - volume increasing)")
        elif itm_pe_vol < -5.0:
            # For BSE: Volume decreasing is a warning (opposite of NSE)
            confidence_mult = max(0.85, confidence_mult - 0.1)
            warnings.append(f"⚠️ [BSE] ITM PE Vol Δ% = {itm_pe_vol:.2f}% (volume decreasing - caution)")
    else:
        # NSE (default): Lower PE Vol is better (negative correlation -0.30)
        if itm_pe_vol < thresholds['itm_pe_vol_optimal']:
            # Strong bullish signal - volume decreasing in puts (put selling)
            confidence_mult = min(1.15, confidence_mult + 0.15)
            position_mult = min(1.2, position_mult + 0.1)
            itm_score += 0.4
            reasons.append(f"✓ ITM PE Vol Δ% = {itm_pe_vol:.2f}% (strong bullish - volume decreasing)")
        elif itm_pe_vol > 0:
            # CRITICAL WARNING: Volume increasing in puts = bearish/hedging
            # This is the strongest negative signal for NSE
            should_skip = True
            confidence_mult = 0.7  # Set low confidence
            position_mult = 0.5     # Reduce position size
            warnings.append(f"⚠️ CRITICAL: ITM PE Vol Δ% = {itm_pe_vol:.2f}% (volume increasing - bearish)")
            # Calculate divergence for early return
            divergence_early = itm_ce_delta - itm_pe_delta
            # Calculate bearish/bullish signals for early return
            is_bearish_early = (itm_ce_delta > itm_pe_delta) and (itm_ce_delta > 0) and (itm_pe_delta < 0)
            is_bullish_early = (itm_pe_delta > itm_ce_delta) and (itm_pe_delta > 0) and (itm_ce_delta < 0)
            is_peak_early = is_bearish_early and divergence_early > PEAK_DIVERGENCE_MIN
            # Don't apply other bonuses when we should skip
            return {
                'should_skip_trade': should_skip,
                'confidence_multiplier': confidence_mult,
                'position_size_multiplier': position_mult,
                'itm_score': 0.0,
                'reasons': [],
                'warnings': warnings,
                'raw_values': {
                    'itm_pe_vol': itm_pe_vol,
                    'itm_ce_vol': itm_ce_vol,
                    'itm_pe_delta': itm_pe_delta,
                    'itm_ce_delta': itm_ce_delta,
                    'divergence': divergence_early
                },
                'bearish_signal': is_bearish_early,
                'bullish_signal': is_bullish_early,
                'peak_detection': is_peak_early,
                'bottom_detection': False,
                'bottom_type': 'none',
                'bottom_type1': False,
                'bottom_type2': False,
                'market_regime': 'trending',
                'exchange': exchange or 'NSE',
            }
    
    # 2. ITM CE Vol Δ% (HIGH IMPORTANCE - 30% weight)
    # Both NSE and BSE: Lower CE Vol is better (correlation -0.17 NSE, -0.32 BSE)
    ce_vol_optimal = thresholds['itm_ce_vol_optimal']
    ce_vol_warning = thresholds['itm_ce_vol_warning']
    
    if itm_ce_vol < ce_vol_optimal:
        # Good condition - no volume spike
        confidence_mult = min(1.1, confidence_mult + 0.1)
        itm_score += 0.2
        if is_bse:
            reasons.append(f"✓ [BSE] ITM CE Vol Δ% = {itm_ce_vol:.2f}% (optimal - below {ce_vol_optimal})")
    elif itm_ce_vol > ce_vol_warning:
        # Volume spike warning - may indicate exhaustion
        confidence_mult = max(0.8, confidence_mult - 0.1)
        position_mult = max(0.7, position_mult - 0.1)
        warnings.append(f"⚠️ ITM CE Vol Δ% = {itm_ce_vol:.2f}% (volume spike - exhaustion risk)")
    
    # 3. ITM PE Δ% (MEDIUM IMPORTANCE - 20% weight)
    # NSE: Optimal range -0.65% to 0.0% (80% win rate)
    # BSE: Optimal range 3.5% to 187% (higher is better, corr +0.21)
    pe_delta_min = thresholds['itm_pe_delta_optimal_min']
    pe_delta_max = thresholds['itm_pe_delta_optimal_max']
    
    if pe_delta_min <= itm_pe_delta <= pe_delta_max:
        # Optimal range
        confidence_mult = min(1.1, confidence_mult + 0.1)
        position_mult = min(1.1, position_mult + 0.05)
        itm_score += 0.2
        if is_bse:
            reasons.append(f"✓ [BSE] ITM PE Δ% = {itm_pe_delta:.2f}% (optimal range)")
        else:
            reasons.append(f"✓ ITM PE Δ% = {itm_pe_delta:.2f}% (optimal range - put selling)")
    elif not is_bse and itm_pe_delta > 0:
        # NSE: Put buying (bearish) - only applies to NSE
        confidence_mult = max(0.9, confidence_mult - 0.05)
        warnings.append(f"⚠️ ITM PE Δ% = {itm_pe_delta:.2f}% (put buying - bearish)")
    
    # 4. ITM CE Δ% (LOW IMPORTANCE - 10% weight)
    # NSE: Optimal range 0.3% to 2.5% (100% win rate in small sample)
    # BSE: Optimal range -0.45% to 1.64%
    ce_delta_min = thresholds['itm_ce_delta_optimal_min']
    ce_delta_max = thresholds['itm_ce_delta_optimal_max']
    
    if ce_delta_min <= itm_ce_delta <= ce_delta_max:
        # Optimal range
        confidence_mult = min(1.05, confidence_mult + 0.05)
        itm_score += 0.1
        if is_bse:
            reasons.append(f"✓ [BSE] ITM CE Δ% = {itm_ce_delta:.2f}% (optimal range)")
    elif itm_ce_delta > 18:
        # Very high - possible overbought condition
        confidence_mult = max(0.9, confidence_mult - 0.05)
        warnings.append(f"⚠️ ITM CE Δ% = {itm_ce_delta:.2f}% (very high - overbought?)")
    
    # 5. Signal Agreement (10% weight)
    # When OI and Volume signals agree, PnL is much better (+₹282 vs +₹37)
    oi_signal = 1 if itm_pe_delta < itm_ce_delta else -1
    vol_signal = 1 if itm_pe_vol < itm_ce_vol else -1
    
    if oi_signal * vol_signal > 0:
        # Signals agree
        confidence_mult = min(1.1, confidence_mult + 0.1)
        position_mult = min(1.1, position_mult + 0.05)
        itm_score += 0.1
        reasons.append("✓ Signals agree (OI and Volume)")
    else:
        # Signals disagree
        warnings.append("⚠️ Signals disagree (OI and Volume)")
    
    # 6. VALIDATED CE/PE Divergence Signals (Chart Correlation Analysis)
    # Key Finding: CE>PE (CE+, PE-) = BEARISH (predictive, contrarian)
    #              PE>CE (PE+, CE-) = BULLISH (predictive, contrarian)
    divergence = itm_ce_delta - itm_pe_delta
    
    # BEARISH Signal: CE Δ% > PE Δ% AND CE positive AND PE negative
    # This is a CONTRARIAN indicator - predicts decline at peaks
    is_bearish_signal = (itm_ce_delta > itm_pe_delta) and (itm_ce_delta > 0) and (itm_pe_delta < 0)
    
    # BULLISH Signal: PE Δ% > CE Δ% AND PE positive AND CE negative
    # This is a CONTRARIAN indicator - predicts rise at bottoms
    is_bullish_signal = (itm_pe_delta > itm_ce_delta) and (itm_pe_delta > 0) and (itm_ce_delta < 0)
    
    # Peak detection: Very strong BEARISH signal (divergence > 3%)
    # Analysis showed 77%+ BEARISH signals at price peaks
    is_peak_bearish = is_bearish_signal and divergence > 3.0
    
    if is_peak_bearish:
        # Very strong bearish signal at peak - high probability of decline
        should_skip = True  # Skip long trades
        confidence_mult = 0.6  # Very low confidence for long
        position_mult = 0.3  # Very small position if trading
        warnings.append(f"⚠️ PEAK BEARISH: CE>PE (CE+{itm_ce_delta:.2f}%, PE-{itm_pe_delta:.2f}%) - predicts decline")
        # Continue to return with all fields (don't return early)
    
    if is_bearish_signal:
        # BEARISH signal - contrarian indicator
        # Reduce confidence for long trades, boost for short trades
        if divergence > 1.0:  # Strong bearish
            confidence_mult = max(0.7, confidence_mult - 0.2)
            position_mult = max(0.5, position_mult - 0.2)
            warnings.append(f"⚠️ BEARISH: CE>PE (CE+{itm_ce_delta:.2f}%, PE-{itm_pe_delta:.2f}%) - contrarian bearish")
        else:  # Moderate bearish
            confidence_mult = max(0.8, confidence_mult - 0.1)
            position_mult = max(0.7, position_mult - 0.1)
            warnings.append(f"⚠️ BEARISH: CE>PE (CE+{itm_ce_delta:.2f}%, PE-{itm_pe_delta:.2f}%) - moderate bearish")
    
    if is_bullish_signal:
        # BULLISH signal - contrarian indicator
        # Boost confidence for long trades, reduce for short trades
        if abs(divergence) > 1.0:  # Strong bullish
            confidence_mult = min(1.15, confidence_mult + 0.15)
            position_mult = min(1.2, position_mult + 0.1)
            itm_score += 0.2
            reasons.append(f"✓ BULLISH: PE>CE (PE+{itm_pe_delta:.2f}%, CE-{itm_ce_delta:.2f}%) - contrarian bullish")
        else:  # Moderate bullish
            confidence_mult = min(1.1, confidence_mult + 0.1)
            position_mult = min(1.1, position_mult + 0.05)
            itm_score += 0.1
            reasons.append(f"✓ BULLISH: PE>CE (PE+{itm_pe_delta:.2f}%, CE-{itm_ce_delta:.2f}%) - moderate bullish")
    
    # --- NEW: Bottom Detection Logic (Jan 2026 Analysis) ---
    # Key Finding: Bottom signals have STRONG correlation with profitability
    # Bottom signals vs PnL: +0.399, vs Win Rate: +0.575
    # High bottom detection days: Avg PnL +₹27,659 vs -₹20,394 for low bottom days
    
    is_bottom_type1 = False
    is_bottom_type2 = False
    is_bottom_detected = False
    bottom_type = 'none'
    
    # Type 1 Bottom: CE+/PE- with large divergence (smart money accumulation)
    # CE OI > 5% (call accumulation) + PE OI < -5% (put unwinding) + divergence > 10%
    if (itm_ce_delta > BOTTOM_TYPE1_CE_MIN and 
        itm_pe_delta < BOTTOM_TYPE1_PE_MAX and 
        divergence > BOTTOM_TYPE1_DIV_MIN):
        is_bottom_type1 = True
        is_bottom_detected = True
        bottom_type = 'type1'
        
        # Very strong bottom if divergence > 15%
        if divergence > BOTTOM_TYPE1_DIV_VERY_STRONG:
            # Very strong bottom signal - high confidence for long entry
            should_skip = False  # Override any skip signal
            confidence_mult = min(1.25, confidence_mult + 0.25)
            position_mult = min(1.5, position_mult + 0.3)
            itm_score += 0.4
            reasons.append(f"✓ VERY STRONG BOTTOM (Type 1): CE+{itm_ce_delta:.2f}%, PE{itm_pe_delta:.2f}%, Div={divergence:.2f}% - accumulation")
        else:
            # Strong bottom signal
            should_skip = False  # Override any skip signal
            confidence_mult = min(1.20, confidence_mult + 0.20)
            position_mult = min(1.3, position_mult + 0.2)
            itm_score += 0.3
            reasons.append(f"✓ STRONG BOTTOM (Type 1): CE+{itm_ce_delta:.2f}%, PE{itm_pe_delta:.2f}%, Div={divergence:.2f}% - accumulation")
    
    # Type 2 Bottom: PE+/CE- with large negative divergence (traditional bullish)
    # PE OI > 5% (put accumulation) + CE OI < -5% (call unwinding) + divergence < -8%
    if (itm_pe_delta > BOTTOM_TYPE2_PE_MIN and 
        itm_ce_delta < BOTTOM_TYPE2_CE_MAX and 
        divergence < BOTTOM_TYPE2_DIV_MAX):
        is_bottom_type2 = True
        is_bottom_detected = True
        if bottom_type == 'none':
            bottom_type = 'type2'
        else:
            bottom_type = 'both'  # Both types detected
        
        # Strong traditional bullish bottom
        should_skip = False  # Override any skip signal
        confidence_mult = min(1.15, confidence_mult + 0.15)
        position_mult = min(1.2, position_mult + 0.15)
        itm_score += 0.25
        reasons.append(f"✓ STRONG BOTTOM (Type 2): PE+{itm_pe_delta:.2f}%, CE{itm_ce_delta:.2f}%, Div={divergence:.2f}% - oversold")
    
    # Market Regime Detection
    # Range-bound: Both peak and bottom signals present = good for trading
    # Trending: Low signal activity = caution with mean-reversion
    peak_signal_present = is_peak_bearish or (is_bearish_signal and divergence > PEAK_STRONG_DIVERGENCE)
    bottom_signal_present = is_bottom_detected
    
    if peak_signal_present or bottom_signal_present:
        market_regime = 'range_bound'  # Clear reversal points = good
    else:
        market_regime = 'trending'  # No clear signals = caution
        # In trending regime, reduce position sizes
        if not is_bottom_detected and not is_bullish_signal:
            position_mult = max(0.7, position_mult - 0.1)
            warnings.append("⚠️ Trending market regime - reduced position size")
    
    return {
        'should_skip_trade': should_skip,
        'confidence_multiplier': confidence_mult,
        'position_size_multiplier': position_mult,
        'itm_score': min(itm_score, 1.0),
        'reasons': reasons,
        'warnings': warnings,
        'raw_values': {
            'itm_pe_vol': itm_pe_vol,
            'itm_ce_vol': itm_ce_vol,
            'itm_pe_delta': itm_pe_delta,
            'itm_ce_delta': itm_ce_delta,
            'divergence': divergence
        },
        'bearish_signal': is_bearish_signal,
        'bullish_signal': is_bullish_signal,
        'peak_detection': is_peak_bearish,
        'bottom_detection': is_bottom_detected,
        'bottom_type': bottom_type,
        'exchange': exchange or 'NSE',
        'bottom_type1': is_bottom_type1,
        'bottom_type2': is_bottom_type2,
        'market_regime': market_regime,
    }


def get_itm_feature_weights() -> Dict[str, float]:
    """
    Return feature weights for model training.
    Higher weights = more important features.
    
    Based on reverse engineering analysis:
    - ITM PE Vol Δ%: 0.54 importance score
    - ITM CE Vol Δ%: 0.28 importance score
    - ITM PE Δ%: 0.18 importance score
    - ITM CE Δ%: 0.03 importance score
    
    NEW (Jan 2026 Analysis):
    - Bottom signals have STRONG correlation with profitability
    - Bottom vs PnL: +0.399, vs Win Rate: +0.575
    """
    return {
        # ITM Volume Features (HIGHEST PRIORITY)
        'itm_volume_pe_pct_change_3m_wavg': 2.0,  # Most important
        'itm_volume_ce_pct_change_3m_wavg': 1.5,  # High importance
        
        # ITM OI Features (MEDIUM PRIORITY)
        'itm_oi_pe_pct_change_3m_wavg': 1.2,      # Medium importance
        'itm_oi_ce_pct_change_3m_wavg': 1.0,     # Low importance
        
        # Derived Features
        'itm_volume_dominance_signal': 1.8,       # PE Vol - CE Vol
        'itm_combined_dominance_signal': 1.6,     # Combined OI + Volume
        'itm_signal_agreement': 1.4,              # OI and Volume agree
        
        # Optimal Range Indicators (NEW - easier for models to learn)
        'itm_pe_vol_in_optimal_range': 2.0,
        'itm_pe_vol_warning': -1.5,  # Negative weight (penalty)
        'itm_ce_vol_in_optimal_range': 1.5,
        'itm_ce_vol_warning': -1.0,  # Negative weight (penalty)
        'itm_pe_delta_in_optimal_range': 1.2,
        'itm_ce_delta_in_optimal_range': 1.0,
        'itm_signals_agree': 1.4,
        'itm_combined_score': 1.8,
        
        # NEW: Bottom Detection Features (HIGH PRIORITY)
        # Correlation with PnL: +0.399, with Win Rate: +0.575
        'itm_bottom_signal_type1': 2.5,           # Strong accumulation signal
        'itm_bottom_signal_type2': 2.0,           # Traditional bullish
        'itm_bottom_signal_strong': 2.2,          # Any strong bottom
        'itm_bottom_signal_very_strong': 3.0,     # Very strong bottom (highest)
        'itm_pe_unwinding': 1.8,                  # Put unwinding (bullish)
        'itm_ce_accumulation': 1.5,               # Call accumulation (bullish)
        'itm_enter_long_signal': 2.5,             # Combined entry signal
        
        # Peak Detection (avoid trades)
        'itm_peak_bearish_signal': -2.0,          # Negative weight (penalty)
        'itm_skip_long_signal': -2.5,             # Strong penalty
        
        # Market Regime
        'itm_market_regime': -0.5,                # Trending = penalty for mean-reversion
        'itm_reversal_potential': 1.5,            # Higher = better for trading
    }


def create_itm_optimal_range_features(features: Dict[str, float]) -> Dict[str, float]:
    """
    Create binary indicators for optimal feature ranges.
    These are easier for models to learn than continuous values.
    
    Returns new features to add to feature dictionary.
    """
    new_features = {}
    
    # ITM PE Vol Δ% optimal range indicator
    itm_pe_vol = features.get('itm_volume_pe_pct_change_3m_wavg', 0)
    new_features['itm_pe_vol_in_optimal_range'] = 1.0 if itm_pe_vol < ITM_PE_VOL_OPTIMAL else 0.0
    new_features['itm_pe_vol_warning'] = 1.0 if itm_pe_vol > 0 else 0.0
    
    # ITM CE Vol Δ% optimal range indicator
    itm_ce_vol = features.get('itm_volume_ce_pct_change_3m_wavg', 0)
    new_features['itm_ce_vol_in_optimal_range'] = 1.0 if itm_ce_vol < ITM_CE_VOL_OPTIMAL else 0.0
    new_features['itm_ce_vol_warning'] = 1.0 if itm_ce_vol > ITM_CE_VOL_WARNING else 0.0
    
    # ITM PE Δ% optimal range indicator
    itm_pe_delta = features.get('itm_oi_pe_pct_change_3m_wavg', 0)
    new_features['itm_pe_delta_in_optimal_range'] = 1.0 if (
        ITM_PE_DELTA_OPTIMAL_MIN <= itm_pe_delta <= ITM_PE_DELTA_OPTIMAL_MAX
    ) else 0.0
    
    # ITM CE Δ% optimal range indicator
    itm_ce_delta = features.get('itm_oi_ce_pct_change_3m_wavg', 0)
    new_features['itm_ce_delta_in_optimal_range'] = 1.0 if (
        ITM_CE_DELTA_OPTIMAL_MIN <= itm_ce_delta <= ITM_CE_DELTA_OPTIMAL_MAX
    ) else 0.0
    
    # Signal agreement indicator
    oi_signal = 1 if itm_pe_delta < itm_ce_delta else -1
    vol_signal = 1 if itm_pe_vol < itm_ce_vol else -1
    new_features['itm_signals_agree'] = 1.0 if oi_signal * vol_signal > 0 else 0.0
    
    # Combined ITM score (0-1 scale)
    itm_score = 0.0
    if itm_pe_vol < ITM_PE_VOL_OPTIMAL:
        itm_score += 0.4
    if itm_ce_vol < ITM_CE_VOL_OPTIMAL:
        itm_score += 0.2
    if ITM_PE_DELTA_OPTIMAL_MIN <= itm_pe_delta <= ITM_PE_DELTA_OPTIMAL_MAX:
        itm_score += 0.2
    if ITM_CE_DELTA_OPTIMAL_MIN <= itm_ce_delta <= ITM_CE_DELTA_OPTIMAL_MAX:
        itm_score += 0.1
    if oi_signal * vol_signal > 0:
        itm_score += 0.1
    
    new_features['itm_combined_score'] = min(itm_score, 1.0)
    
    # NEW: Bottom Detection Features (Jan 2026 Analysis)
    divergence = itm_ce_delta - itm_pe_delta
    
    # Type 1 Bottom: CE+/PE- with large divergence (accumulation)
    new_features['itm_bottom_signal_type1'] = 1.0 if (
        itm_ce_delta > BOTTOM_TYPE1_CE_MIN and
        itm_pe_delta < BOTTOM_TYPE1_PE_MAX and
        divergence > BOTTOM_TYPE1_DIV_MIN
    ) else 0.0
    
    # Type 2 Bottom: PE+/CE- with large negative divergence
    new_features['itm_bottom_signal_type2'] = 1.0 if (
        itm_pe_delta > BOTTOM_TYPE2_PE_MIN and
        itm_ce_delta < BOTTOM_TYPE2_CE_MAX and
        divergence < BOTTOM_TYPE2_DIV_MAX
    ) else 0.0
    
    # Any strong bottom signal
    new_features['itm_bottom_signal_strong'] = max(
        new_features['itm_bottom_signal_type1'],
        new_features['itm_bottom_signal_type2']
    )
    
    # Very strong bottom (Type 1 with divergence > 15%)
    new_features['itm_bottom_signal_very_strong'] = 1.0 if (
        itm_ce_delta > BOTTOM_TYPE1_CE_MIN and
        itm_pe_delta < BOTTOM_TYPE1_PE_MAX and
        divergence > BOTTOM_TYPE1_DIV_VERY_STRONG
    ) else 0.0
    
    # PE unwinding and CE accumulation indicators
    new_features['itm_pe_unwinding'] = 1.0 if itm_pe_delta < BOTTOM_TYPE1_PE_MAX else 0.0
    new_features['itm_ce_accumulation'] = 1.0 if itm_ce_delta > BOTTOM_TYPE1_CE_MIN else 0.0
    
    # Peak detection
    is_bearish = itm_ce_delta > itm_pe_delta and itm_ce_delta > 0 and itm_pe_delta < 0
    new_features['itm_peak_bearish_signal'] = 1.0 if is_bearish and divergence > PEAK_DIVERGENCE_MIN else 0.0
    
    # Skip long signal
    new_features['itm_skip_long_signal'] = 1.0 if (
        new_features['itm_peak_bearish_signal'] > 0 or
        (is_bearish and divergence > PEAK_STRONG_DIVERGENCE)
    ) else 0.0
    
    # Enter long signal
    is_bullish = itm_pe_delta > itm_ce_delta and itm_pe_delta > 0 and itm_ce_delta < 0
    new_features['itm_enter_long_signal'] = 1.0 if (
        new_features['itm_bottom_signal_strong'] > 0 or
        (is_bullish and divergence < -5.0)
    ) else 0.0
    
    # Market regime and reversal potential
    peak_strength = 1.0 if new_features['itm_peak_bearish_signal'] > 0 else 0.0
    new_features['itm_reversal_potential'] = peak_strength + new_features['itm_bottom_signal_strong']
    new_features['itm_market_regime'] = 0.0 if new_features['itm_reversal_potential'] > 0 else 1.0
    
    return new_features


def evaluate_for_auto_trading(features_dict: Dict[str, float], exchange: Optional[str] = None) -> Dict[str, Any]:
    """
    Simplified evaluation for auto trading decisions.
    
    Returns a clear recommendation with confidence adjustments.
    
    Args:
        features_dict: Dictionary containing feature values
        exchange: Optional exchange code ('NSE' or 'BSE') for exchange-specific thresholds
        
    Returns:
        {
            'action': str,  # 'ENTER_LONG', 'SKIP_LONG', 'NEUTRAL'
            'confidence_boost': float,  # -0.3 to +0.25
            'position_multiplier': float,  # 0.3 to 1.5
            'reason': str,  # Explanation
            'signals': Dict[str, bool],  # Signal flags
        }
    """
    eval_result = evaluate_itm_features(features_dict, exchange=exchange)
    
    # Determine action based on signals
    if eval_result.get('bottom_detection', False):
        # Bottom detected - good time to enter long
        bottom_type = eval_result.get('bottom_type', 'unknown')
        action = 'ENTER_LONG'
        
        if bottom_type == 'type1' and eval_result['raw_values']['divergence'] > 15:
            confidence_boost = 0.25
            position_mult = 1.5
            reason = f"Very strong bottom (Type 1): accumulation signal with {eval_result['raw_values']['divergence']:.1f}% divergence"
        elif bottom_type == 'type1':
            confidence_boost = 0.20
            position_mult = 1.3
            reason = f"Strong bottom (Type 1): accumulation at lows"
        else:
            confidence_boost = 0.15
            position_mult = 1.2
            reason = f"Bottom (Type 2): oversold condition"
            
    elif eval_result.get('peak_detection', False):
        # Peak detected - skip long trades
        action = 'SKIP_LONG'
        confidence_boost = -0.30
        position_mult = 0.3
        reason = f"Peak detected: high probability of decline"
        
    elif eval_result.get('should_skip_trade', False):
        # Other skip conditions
        action = 'SKIP_LONG'
        confidence_boost = -0.20
        position_mult = 0.5
        reason = "Bearish conditions detected"
        
    elif eval_result.get('bullish_signal', False):
        # Bullish signal but not strong bottom
        action = 'NEUTRAL'  # Can enter but not ideal
        confidence_boost = 0.10
        position_mult = 1.1
        reason = "Moderate bullish signal"
        
    else:
        # No clear signal
        action = 'NEUTRAL'
        confidence_boost = 0.0
        position_mult = 1.0
        reason = "No strong ITM signal"
    
    return {
        'action': action,
        'confidence_boost': confidence_boost,
        'position_multiplier': position_mult,
        'reason': reason,
        'signals': {
            'bottom_detected': eval_result.get('bottom_detection', False),
            'bottom_type': eval_result.get('bottom_type', 'none'),
            'peak_detected': eval_result.get('peak_detection', False),
            'bearish_signal': eval_result.get('bearish_signal', False),
            'bullish_signal': eval_result.get('bullish_signal', False),
            'market_regime': eval_result.get('market_regime', 'unknown'),
        },
        'raw_evaluation': eval_result,
    }
