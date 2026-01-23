"""
ITM Feature Evaluator - Enforces reverse-engineered insights from trade log analysis.

This module evaluates ITM features and provides trading decision modifiers based on
statistical analysis of winning vs losing trades.

Key Findings:
- ITM PE Vol Δ% is the most important feature (correlation: -0.30)
- ITM CE Vol Δ% is highly important (correlation: -0.17)
- Lower volume changes = Better for trades (contrarian indicator)
- Optimal ranges exist for each feature
"""

from typing import Dict, Any, List, Tuple

# Optimal ranges from reverse engineering analysis
ITM_PE_VOL_OPTIMAL = -3.0  # Below this is optimal (volume decreasing)
ITM_CE_VOL_OPTIMAL = 10.0  # Below this is optimal
ITM_CE_VOL_WARNING = 16.0  # Above this is warning (volume spike)
ITM_PE_DELTA_OPTIMAL_MIN = -0.65
ITM_PE_DELTA_OPTIMAL_MAX = 0.0
ITM_CE_DELTA_OPTIMAL_MIN = 0.3
ITM_CE_DELTA_OPTIMAL_MAX = 2.5


def evaluate_itm_features(features_dict: Dict[str, float]) -> Dict[str, Any]:
    """
    Comprehensive ITM feature evaluation based on reverse engineering analysis.
    
    This function evaluates the four key ITM features:
    1. ITM PE Vol Δ% (Most Important - 40% weight)
    2. ITM CE Vol Δ% (High Importance - 30% weight)
    3. ITM PE Δ% (Medium Importance - 20% weight)
    4. ITM CE Δ% (Low Importance - 10% weight)
    
    Args:
        features_dict: Dictionary containing feature values
        
    Returns:
        {
            'should_skip_trade': bool,  # True if trade should be skipped
            'confidence_multiplier': float,  # Multiplier for confidence (0.7 to 1.15)
            'position_size_multiplier': float,  # Multiplier for position size (0.5 to 1.2)
            'itm_score': float,  # Combined ITM score (0-1)
            'reasons': List[str],  # Positive reasons
            'warnings': List[str],  # Warning messages
            'raw_values': Dict[str, float]  # Raw feature values
        }
    """
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
    # Correlation: -0.30 (strongest negative correlation)
    # Winners average: -5.56%, Losers average: +2.18%
    if itm_pe_vol < ITM_PE_VOL_OPTIMAL:
        # Strong bullish signal - volume decreasing in puts (put selling)
        confidence_mult = min(1.15, confidence_mult + 0.15)
        position_mult = min(1.2, position_mult + 0.1)
        itm_score += 0.4
        reasons.append(f"✓ ITM PE Vol Δ% = {itm_pe_vol:.2f}% (strong bullish - volume decreasing)")
    elif itm_pe_vol > 0:
        # CRITICAL WARNING: Volume increasing in puts = bearish/hedging
        # This is the strongest negative signal
        should_skip = True
        confidence_mult = 0.7  # Set low confidence
        position_mult = 0.5     # Reduce position size
        warnings.append(f"⚠️ CRITICAL: ITM PE Vol Δ% = {itm_pe_vol:.2f}% (volume increasing - bearish)")
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
                'divergence': divergence
            },
            'bearish_signal': is_bearish_signal,
            'bullish_signal': is_bullish_signal,
            'peak_detection': is_peak_bearish
        }
    
    # 2. ITM CE Vol Δ% (HIGH IMPORTANCE - 30% weight)
    # Correlation: -0.17 (moderate negative correlation)
    # Winners average: 10.24%, Losers average: 16.23%
    if itm_ce_vol < ITM_CE_VOL_OPTIMAL:
        # Good condition - no volume spike
        confidence_mult = min(1.1, confidence_mult + 0.1)
        itm_score += 0.2
    elif itm_ce_vol > ITM_CE_VOL_WARNING:
        # Volume spike warning - may indicate exhaustion
        confidence_mult = max(0.8, confidence_mult - 0.1)
        position_mult = max(0.7, position_mult - 0.1)
        warnings.append(f"⚠️ ITM CE Vol Δ% = {itm_ce_vol:.2f}% (volume spike - exhaustion risk)")
    
    # 3. ITM PE Δ% (MEDIUM IMPORTANCE - 20% weight)
    # Optimal range: -0.65% to 0.0% (80% win rate)
    if ITM_PE_DELTA_OPTIMAL_MIN <= itm_pe_delta <= ITM_PE_DELTA_OPTIMAL_MAX:
        # Optimal range - put selling (bullish)
        confidence_mult = min(1.1, confidence_mult + 0.1)
        position_mult = min(1.1, position_mult + 0.05)
        itm_score += 0.2
        reasons.append(f"✓ ITM PE Δ% = {itm_pe_delta:.2f}% (optimal range - put selling)")
    elif itm_pe_delta > 0:
        # Put buying (bearish)
        confidence_mult = max(0.9, confidence_mult - 0.05)
        warnings.append(f"⚠️ ITM PE Δ% = {itm_pe_delta:.2f}% (put buying - bearish)")
    
    # 4. ITM CE Δ% (LOW IMPORTANCE - 10% weight)
    # Optimal range: 0.3% to 2.5% (100% win rate in small sample)
    if ITM_CE_DELTA_OPTIMAL_MIN <= itm_ce_delta <= ITM_CE_DELTA_OPTIMAL_MAX:
        # Optimal range
        confidence_mult = min(1.05, confidence_mult + 0.05)
        itm_score += 0.1
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
        'peak_detection': is_peak_bearish
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
    
    return new_features
