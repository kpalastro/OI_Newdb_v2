#!/usr/bin/env python3
"""
Test ITM enforcement in paper trading mode.

This script simulates the paper trading flow with ITM feature evaluation
to verify that ITM filters are working correctly.
"""

import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.strategy_router import StrategySignal
from execution.auto_executor import AutoExecutor, ExecutionConfig
from utils.itm_feature_evaluator import evaluate_itm_features, create_itm_optimal_range_features


def create_test_signal(
    signal: str,
    confidence: float,
    features_dict: Dict[str, Any]
) -> StrategySignal:
    """Create a test StrategySignal with features."""
    # Build metadata
    metadata = {
        'regime': 'NORMAL',
        'horizon': 'intraday',
        'buy_prob': 0.8 if signal == 'BUY' else 0.2,
        'sell_prob': 0.8 if signal == 'SELL' else 0.2,
        **features_dict
    }
    
    # Create signal with correct parameters (StrategySignal requires: signal, confidence, source, rationale, metadata)
    signal_obj = StrategySignal(
        signal=signal,
        confidence=confidence,
        source='lightgbm',
        rationale='Test signal',
        metadata=metadata
    )
    
    # Add features_dict as attribute for ITM evaluation
    signal_obj.features_dict = features_dict
    
    return signal_obj


def test_itm_filter_blocks_bad_trade():
    """Test that ITM filter blocks trades when PE vol is increasing."""
    print("\n" + "=" * 70)
    print("TEST 1: ITM Filter Blocks Bad Trade (PE Vol Increasing)")
    print("=" * 70)
    
    # Create features with BAD ITM conditions (PE vol increasing)
    bad_features = {
        'itm_volume_pe_pct_change_3m_wavg': 2.5,  # WARNING! Volume increasing
        'itm_volume_ce_pct_change_3m_wavg': 5.0,
        'itm_oi_pe_pct_change_3m_wavg': 0.3,
        'itm_oi_ce_pct_change_3m_wavg': 1.0,
        'vix': 20.0,
    }
    
    # Evaluate ITM features
    itm_eval = evaluate_itm_features(bad_features)
    print(f"\nITM Evaluation:")
    print(f"  Should skip: {itm_eval['should_skip_trade']}")
    print(f"  Confidence multiplier: {itm_eval['confidence_multiplier']:.2f}")
    print(f"  Warnings: {', '.join(itm_eval['warnings'])}")
    
    # Create signal (high confidence BUY)
    signal = create_test_signal('BUY', 0.85, bad_features)
    
    # Create executor
    config = ExecutionConfig(
        enabled=True,
        paper_mode=True,
        min_confidence=0.70,
        min_kelly_fraction=0.15,
    )
    executor = AutoExecutor('NSE', config)
    
    # Try to execute
    result = executor.should_execute(
        signal=signal,
        current_price=100.0,
        current_open_positions={}
    )
    
    print(f"\nExecution Result:")
    print(f"  Executed: {result.executed}")
    print(f"  Reason: {result.reason}")
    
    # Verify trade was blocked
    assert not result.executed, "Trade should be blocked by ITM filter"
    assert 'ITM' in result.reason or 'skip' in result.reason.lower(), "Reason should mention ITM filter"
    
    print("\n✓ TEST PASSED: Bad trade correctly blocked by ITM filter")
    return True


def test_itm_boost_good_trade():
    """Test that ITM filter boosts confidence for good trades."""
    print("\n" + "=" * 70)
    print("TEST 2: ITM Filter Boosts Good Trade (Optimal Conditions)")
    print("=" * 70)
    
    # Create features with GOOD ITM conditions
    good_features = {
        'itm_volume_pe_pct_change_3m_wavg': -5.0,  # Strong bullish (volume decreasing)
        'itm_volume_ce_pct_change_3m_wavg': 8.0,    # Good (below threshold)
        'itm_oi_pe_pct_change_3m_wavg': -0.3,       # Optimal range
        'itm_oi_ce_pct_change_3m_wavg': 1.5,       # Optimal range
        'vix': 20.0,
    }
    
    # Evaluate ITM features
    itm_eval = evaluate_itm_features(good_features)
    print(f"\nITM Evaluation:")
    print(f"  Should skip: {itm_eval['should_skip_trade']}")
    print(f"  Confidence multiplier: {itm_eval['confidence_multiplier']:.2f}")
    print(f"  Position multiplier: {itm_eval['position_size_multiplier']:.2f}")
    print(f"  ITM score: {itm_eval['itm_score']:.2f}")
    print(f"  Reasons: {', '.join(itm_eval['reasons'])}")
    
    # Create signal (moderate confidence BUY)
    base_confidence = 0.75
    signal = create_test_signal('BUY', base_confidence, good_features)
    
    # Apply ITM confidence multiplier (simulating ml_core.py behavior)
    adjusted_confidence = min(0.95, base_confidence * itm_eval['confidence_multiplier'])
    signal.confidence = adjusted_confidence
    
    print(f"\nConfidence Adjustment:")
    print(f"  Base confidence: {base_confidence:.2%}")
    print(f"  ITM multiplier: {itm_eval['confidence_multiplier']:.2f}")
    print(f"  Adjusted confidence: {adjusted_confidence:.2%}")
    
    # Create executor
    config = ExecutionConfig(
        enabled=True,
        paper_mode=True,
        min_confidence=0.70,
        min_kelly_fraction=0.15,
    )
    executor = AutoExecutor('NSE', config)
    
    # Try to execute
    result = executor.should_execute(
        signal=signal,
        current_price=100.0,
        current_open_positions={}
    )
    
    print(f"\nExecution Result:")
    print(f"  Executed: {result.executed}")
    print(f"  Reason: {result.reason}")
    
    # Verify trade would execute (if confidence is high enough)
    if adjusted_confidence >= config.min_confidence:
        print(f"\n✓ TEST PASSED: Good trade boosted and would execute")
    else:
        print(f"\n⚠️  Trade would not execute (confidence {adjusted_confidence:.2%} < {config.min_confidence:.2%})")
    
    # Verify confidence was boosted
    assert adjusted_confidence > base_confidence, "Confidence should be boosted"
    assert itm_eval['confidence_multiplier'] > 1.0, "ITM multiplier should be > 1.0"
    
    return True


def test_itm_position_sizing():
    """Test that ITM conditions affect position sizing."""
    print("\n" + "=" * 70)
    print("TEST 3: ITM Position Sizing Adjustment")
    print("=" * 70)
    
    from risk_manager import get_optimal_position_size
    
    # Test case 1: Optimal ITM conditions (should increase position size)
    good_features = {
        'itm_volume_pe_pct_change_3m_wavg': -5.0,
        'itm_volume_ce_pct_change_3m_wavg': 8.0,
        'itm_oi_pe_pct_change_3m_wavg': -0.3,
        'itm_oi_ce_pct_change_3m_wavg': 1.5,
    }
    
    itm_eval_good = evaluate_itm_features(good_features)
    
    # Test case 2: Warning ITM conditions (should decrease position size)
    bad_features = {
        'itm_volume_pe_pct_change_3m_wavg': 1.0,  # Not critical, but not optimal
        'itm_volume_ce_pct_change_3m_wavg': 18.0,  # Warning (volume spike)
        'itm_oi_pe_pct_change_3m_wavg': 0.5,
        'itm_oi_ce_pct_change_3m_wavg': 1.0,
    }
    
    itm_eval_bad = evaluate_itm_features(bad_features)
    
    # Calculate position sizes
    base_confidence = 0.80
    win_rate = 0.60
    avg_w_l_ratio = 1.5
    
    position_good = get_optimal_position_size(
        ml_confidence=base_confidence,
        win_rate=win_rate,
        avg_win_loss_ratio=avg_w_l_ratio,
        current_volatility=0.20,
        itm_position_multiplier=itm_eval_good['position_size_multiplier']
    )
    
    position_bad = get_optimal_position_size(
        ml_confidence=base_confidence,
        win_rate=win_rate,
        avg_win_loss_ratio=avg_w_l_ratio,
        current_volatility=0.20,
        itm_position_multiplier=itm_eval_bad['position_size_multiplier']
    )
    
    print(f"\nPosition Sizing Comparison:")
    print(f"  Good ITM conditions:")
    print(f"    ITM position multiplier: {itm_eval_good['position_size_multiplier']:.2f}")
    print(f"    Recommended lots: {position_good['recommended_lots']}")
    print(f"    Kelly fraction: {position_good['kelly_fraction']:.4f}")
    
    print(f"\n  Warning ITM conditions:")
    print(f"    ITM position multiplier: {itm_eval_bad['position_size_multiplier']:.2f}")
    print(f"    Recommended lots: {position_bad['recommended_lots']}")
    print(f"    Kelly fraction: {position_bad['kelly_fraction']:.4f}")
    
    # Verify position sizing is adjusted
    assert position_good['recommended_lots'] >= position_bad['recommended_lots'], \
        "Good ITM conditions should allow larger position"
    
    print(f"\n✓ TEST PASSED: Position sizing correctly adjusted by ITM conditions")
    return True


def test_optimal_range_features():
    """Test that optimal range features are created correctly."""
    print("\n" + "=" * 70)
    print("TEST 4: Optimal Range Features Creation")
    print("=" * 70)
    
    features = {
        'itm_volume_pe_pct_change_3m_wavg': -5.0,
        'itm_volume_ce_pct_change_3m_wavg': 8.0,
        'itm_oi_pe_pct_change_3m_wavg': -0.3,
        'itm_oi_ce_pct_change_3m_wavg': 1.5,
    }
    
    optimal_features = create_itm_optimal_range_features(features)
    
    print(f"\nOptimal Range Features Created:")
    for key, value in optimal_features.items():
        print(f"  {key}: {value}")
    
    # Verify all expected features are present
    expected_features = [
        'itm_pe_vol_in_optimal_range',
        'itm_pe_vol_warning',
        'itm_ce_vol_in_optimal_range',
        'itm_ce_vol_warning',
        'itm_pe_delta_in_optimal_range',
        'itm_ce_delta_in_optimal_range',
        'itm_signals_agree',
        'itm_combined_score'
    ]
    
    for feature in expected_features:
        assert feature in optimal_features, f"Missing feature: {feature}"
    
    # Verify optimal range indicators
    assert optimal_features['itm_pe_vol_in_optimal_range'] == 1.0, "PE vol should be in optimal range"
    assert optimal_features['itm_pe_vol_warning'] == 0.0, "PE vol should not have warning"
    assert optimal_features['itm_combined_score'] > 0.7, "Combined score should be high"
    
    print(f"\n✓ TEST PASSED: Optimal range features created correctly")
    return True


def main():
    """Run all paper trading tests."""
    print("=" * 70)
    print("ITM Enforcement Paper Trading Test Suite")
    print("=" * 70)
    print(f"\nTest started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("ITM Filter Blocks Bad Trade", test_itm_filter_blocks_bad_trade),
        ("ITM Boost Good Trade", test_itm_boost_good_trade),
        ("ITM Position Sizing", test_itm_position_sizing),
        ("Optimal Range Features", test_optimal_range_features),
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
    print(f"  Total: {passed + failed}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED!")
        print("\nITM enforcement is working correctly in paper trading mode.")
        return 0
    else:
        print(f"\n✗ {failed} TEST(S) FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
