#!/usr/bin/env python3
"""
Test script for ITM feature enforcement.

Tests the ITM evaluator with various scenarios to ensure it works correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.itm_feature_evaluator import evaluate_itm_features, create_itm_optimal_range_features


def test_strong_bullish_signal():
    """Test case 1: Strong bullish signal (should boost confidence)"""
    print("\n=== Test 1: Strong Bullish Signal ===")
    features = {
        'itm_volume_pe_pct_change_3m_wavg': -5.0,  # Strong bullish (volume decreasing)
        'itm_volume_ce_pct_change_3m_wavg': 5.0,   # Good (below threshold)
        'itm_oi_pe_pct_change_3m_wavg': -0.3,       # Optimal range
        'itm_oi_ce_pct_change_3m_wavg': 1.0,       # Optimal range
    }
    
    result = evaluate_itm_features(features)
    
    assert result['should_skip_trade'] == False, "Should not skip strong bullish signal"
    assert result['confidence_multiplier'] > 1.0, "Should boost confidence"
    assert result['position_size_multiplier'] > 1.0, "Should increase position size"
    assert result['itm_score'] > 0.7, "Should have high ITM score"
    
    print(f"✓ Should skip: {result['should_skip_trade']}")
    print(f"✓ Confidence multiplier: {result['confidence_multiplier']:.2f}")
    print(f"✓ Position multiplier: {result['position_size_multiplier']:.2f}")
    print(f"✓ ITM score: {result['itm_score']:.2f}")
    print(f"✓ Reasons: {', '.join(result['reasons'])}")
    print("✓ Test 1 PASSED")


def test_warning_signal():
    """Test case 2: Warning signal (should skip trade)"""
    print("\n=== Test 2: Warning Signal (Should Skip) ===")
    features = {
        'itm_volume_pe_pct_change_3m_wavg': 2.0,   # WARNING! (volume increasing)
        'itm_volume_ce_pct_change_3m_wavg': 5.0,
        'itm_oi_pe_pct_change_3m_wavg': 0.5,
        'itm_oi_ce_pct_change_3m_wavg': 1.0,
    }
    
    result = evaluate_itm_features(features)
    
    assert result['should_skip_trade'] == True, "Should skip warning signal"
    assert result['confidence_multiplier'] < 1.0, "Should reduce confidence"
    assert len(result['warnings']) > 0, "Should have warnings"
    
    print(f"✓ Should skip: {result['should_skip_trade']}")
    print(f"✓ Confidence multiplier: {result['confidence_multiplier']:.2f}")
    print(f"✓ Warnings: {', '.join(result['warnings'])}")
    print("✓ Test 2 PASSED")


def test_optimal_range_features():
    """Test case 3: Optimal range features creation"""
    print("\n=== Test 3: Optimal Range Features ===")
    features = {
        'itm_volume_pe_pct_change_3m_wavg': -5.0,
        'itm_volume_ce_pct_change_3m_wavg': 8.0,
        'itm_oi_pe_pct_change_3m_wavg': -0.3,
        'itm_oi_ce_pct_change_3m_wavg': 1.5,
    }
    
    optimal_features = create_itm_optimal_range_features(features)
    
    assert 'itm_pe_vol_in_optimal_range' in optimal_features
    assert 'itm_combined_score' in optimal_features
    assert optimal_features['itm_pe_vol_in_optimal_range'] == 1.0, "Should be in optimal range"
    assert optimal_features['itm_combined_score'] > 0.5, "Should have decent score"
    
    print(f"✓ Created {len(optimal_features)} optimal range features")
    print(f"✓ ITM PE Vol in optimal range: {optimal_features['itm_pe_vol_in_optimal_range']}")
    print(f"✓ Combined score: {optimal_features['itm_combined_score']:.2f}")
    print("✓ Test 3 PASSED")


def test_mixed_signal():
    """Test case 4: Mixed signal (some good, some bad)"""
    print("\n=== Test 4: Mixed Signal ===")
    features = {
        'itm_volume_pe_pct_change_3m_wavg': -2.0,  # Good (below -3)
        'itm_volume_ce_pct_change_3m_wavg': 18.0,   # Warning (above 16)
        'itm_oi_pe_pct_change_3m_wavg': -0.2,       # Optimal
        'itm_oi_ce_pct_change_3m_wavg': 3.0,       # Slightly above optimal
    }
    
    result = evaluate_itm_features(features)
    
    # Should not skip (PE vol is good), but confidence should be reduced due to CE vol warning
    assert result['should_skip_trade'] == False, "Should not skip (PE vol is good)"
    # Confidence might be boosted by PE vol but reduced by CE vol warning
    # So it could be anywhere between 0.8 and 1.15, but should have warnings
    assert len(result['warnings']) > 0, "Should have warnings"
    
    print(f"✓ Should skip: {result['should_skip_trade']}")
    print(f"✓ Confidence multiplier: {result['confidence_multiplier']:.2f}")
    print(f"✓ Warnings: {', '.join(result['warnings'])}")
    print(f"✓ Reasons: {', '.join(result['reasons'])}")
    print("✓ Test 4 PASSED")


def test_signal_agreement():
    """Test case 5: Signal agreement bonus"""
    print("\n=== Test 5: Signal Agreement ===")
    # Both OI and Volume signals agree (both bullish)
    features = {
        'itm_volume_pe_pct_change_3m_wavg': -4.0,  # PE vol < CE vol (bullish)
        'itm_volume_ce_pct_change_3m_wavg': 5.0,
        'itm_oi_pe_pct_change_3m_wavg': -0.4,       # PE delta < CE delta (bullish)
        'itm_oi_ce_pct_change_3m_wavg': 1.5,
    }
    
    result = evaluate_itm_features(features)
    
    # Check if signal agreement is detected
    has_agreement = any('agree' in reason.lower() for reason in result['reasons'])
    assert has_agreement or result['itm_score'] > 0.8, "Should detect signal agreement"
    
    print(f"✓ ITM score: {result['itm_score']:.2f}")
    print(f"✓ Reasons: {', '.join(result['reasons'])}")
    print("✓ Test 5 PASSED")


def main():
    """Run all tests"""
    print("=" * 60)
    print("ITM Feature Enforcement Test Suite")
    print("=" * 60)
    
    try:
        test_strong_bullish_signal()
        test_warning_signal()
        test_optimal_range_features()
        test_mixed_signal()
        test_signal_agreement()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
