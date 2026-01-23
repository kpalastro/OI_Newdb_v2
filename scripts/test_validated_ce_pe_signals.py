#!/usr/bin/env python3
"""
Test Validated CE/PE Divergence Signals Implementation

Tests the implementation of validated BEARISH/BULLISH signals.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.itm_feature_evaluator import evaluate_itm_features


def test_bearish_signal():
    """Test BEARISH signal (CE>PE, CE+, PE-)"""
    print("\n=== Test 1: BEARISH Signal (CE>PE, CE+, PE-) ===")
    
    features = {
        'itm_oi_ce_pct_change_3m_wavg': 2.5,   # CE positive
        'itm_oi_pe_pct_change_3m_wavg': -1.0,  # PE negative
        'itm_volume_pe_pct_change_3m_wavg': -2.0,
        'itm_volume_ce_pct_change_3m_wavg': 5.0,
    }
    
    result = evaluate_itm_features(features)
    
    assert result['bearish_signal'] == True, "Should detect BEARISH signal"
    assert result['bullish_signal'] == False, "Should not detect BULLISH signal"
    assert result['confidence_multiplier'] < 1.0, "Should reduce confidence"
    
    print(f"✓ BEARISH signal detected: {result['bearish_signal']}")
    print(f"✓ Confidence multiplier: {result['confidence_multiplier']:.2f}")
    print(f"✓ Warnings: {', '.join(result['warnings'])}")
    print("✓ Test 1 PASSED")


def test_bullish_signal():
    """Test BULLISH signal (PE>CE, PE+, CE-)"""
    print("\n=== Test 2: BULLISH Signal (PE>CE, PE+, CE-) ===")
    
    features = {
        'itm_oi_ce_pct_change_3m_wavg': -1.5,  # CE negative
        'itm_oi_pe_pct_change_3m_wavg': 2.0,   # PE positive
        'itm_volume_pe_pct_change_3m_wavg': -3.0,
        'itm_volume_ce_pct_change_3m_wavg': 4.0,
    }
    
    result = evaluate_itm_features(features)
    
    assert result['bullish_signal'] == True, "Should detect BULLISH signal"
    assert result['bearish_signal'] == False, "Should not detect BEARISH signal"
    assert result['confidence_multiplier'] > 1.0, "Should boost confidence"
    
    print(f"✓ BULLISH signal detected: {result['bullish_signal']}")
    print(f"✓ Confidence multiplier: {result['confidence_multiplier']:.2f}")
    print(f"✓ Reasons: {', '.join(result['reasons'])}")
    print("✓ Test 2 PASSED")


def test_peak_detection():
    """Test peak detection (BEARISH + divergence > 3%)"""
    print("\n=== Test 3: Peak Detection (BEARISH + divergence > 3%) ===")
    
    features = {
        'itm_oi_ce_pct_change_3m_wavg': 5.0,   # CE positive
        'itm_oi_pe_pct_change_3m_wavg': -1.0,  # PE negative
        'itm_volume_pe_pct_change_3m_wavg': -2.0,
        'itm_volume_ce_pct_change_3m_wavg': 5.0,
    }
    
    result = evaluate_itm_features(features)
    
    assert result['peak_detection'] == True, "Should detect peak"
    assert result['should_skip_trade'] == True, "Should skip trade"
    assert result['confidence_multiplier'] <= 0.7, "Should have low confidence (0.6 or less)"
    
    print(f"✓ Peak detection: {result['peak_detection']}")
    print(f"✓ Should skip: {result['should_skip_trade']}")
    print(f"✓ Confidence multiplier: {result['confidence_multiplier']:.2f}")
    print(f"✓ Warnings: {', '.join(result['warnings'])}")
    print("✓ Test 3 PASSED")


def test_no_signal():
    """Test when no BEARISH/BULLISH signal"""
    print("\n=== Test 4: No Signal ===")
    
    features = {
        'itm_oi_ce_pct_change_3m_wavg': 1.0,   # Both positive
        'itm_oi_pe_pct_change_3m_wavg': 0.5,   # Both positive
        'itm_volume_pe_pct_change_3m_wavg': -2.0,
        'itm_volume_ce_pct_change_3m_wavg': 5.0,
    }
    
    result = evaluate_itm_features(features)
    
    assert result['bearish_signal'] == False, "Should not detect BEARISH"
    assert result['bullish_signal'] == False, "Should not detect BULLISH"
    assert result['peak_detection'] == False, "Should not detect peak"
    
    print(f"✓ No BEARISH signal: {result['bearish_signal']}")
    print(f"✓ No BULLISH signal: {result['bullish_signal']}")
    print(f"✓ No peak detection: {result['peak_detection']}")
    print("✓ Test 4 PASSED")


def main():
    """Run all tests"""
    print("=" * 70)
    print("VALIDATED CE/PE SIGNALS IMPLEMENTATION TEST")
    print("=" * 70)
    
    tests = [
        ("BEARISH Signal", test_bearish_signal),
        ("BULLISH Signal", test_bullish_signal),
        ("Peak Detection", test_peak_detection),
        ("No Signal", test_no_signal),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ TEST FAILED: {test_name}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ TEST ERROR: {test_name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED!")
        print("Validated CE/PE signals implementation is working correctly.")
        return 0
    else:
        print(f"\n✗ {failed} TEST(S) FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
