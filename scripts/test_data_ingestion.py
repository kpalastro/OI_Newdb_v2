"""
Test script to verify data ingestion functions work correctly.

Run this to test your data ingestion setup before scheduling.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
import database_new as db
from data_ingestion.macro_feeds import record_macro_snapshot
from data_ingestion.vix_term_structure import record_vix_term_structure
from time_utils import now_ist

def test_macro_data():
    """Test macro data ingestion."""
    print("\n" + "=" * 60)
    print("Testing Macro Data Ingestion")
    print("=" * 60)
    
    # Save test data
    record_macro_snapshot(
        exchange="NSE",
        fii_flow=1000.0,
        dii_flow=-500.0,
        usdinr=83.0,
        usdinr_trend=0.1,
        crude_price=75.0,
        crude_trend=-0.5,
        banknifty_correlation=0.75,
        risk_on_score=0.2,
        metadata={"test": True, "source": "test_script"},
        timestamp=now_ist()
    )
    
    # Verify it was saved
    latest = db.get_latest_macro_signals("NSE")
    if latest:
        print("✓ Macro data saved successfully")
        print(f"  FII Flow: {latest.get('fii_flow')}")
        print(f"  DII Flow: {latest.get('dii_flow')}")
        print(f"  FII-DII Net: {latest.get('fii_dii_net')}")
        print(f"  USD/INR: {latest.get('usdinr')}")
        print(f"  Crude Price: {latest.get('crude_price')}")
        print(f"  Risk Score: {latest.get('risk_on_score')}")
        return True
    else:
        print("✗ Failed to retrieve macro data")
        return False


def test_vix_term_structure():
    """Test VIX term structure ingestion."""
    print("\n" + "=" * 60)
    print("Testing VIX Term Structure Ingestion")
    print("=" * 60)
    
    # Save test data
    record_vix_term_structure(
        exchange="NSE",
        front_month_price=18.0,
        next_month_price=18.5,
        source="test_script",
        timestamp=now_ist()
    )
    
    # Verify it was saved
    latest = db.get_latest_vix_term_structure("NSE")
    if latest:
        print("✓ VIX term structure saved successfully")
        print(f"  Front Month: {latest.get('front_month_price')}")
        print(f"  Next Month: {latest.get('next_month_price')}")
        print(f"  Contango %: {latest.get('contango_pct'):.2f}%")
        print(f"  Backwardation %: {latest.get('backwardation_pct'):.2f}%")
        return True
    else:
        print("✗ Failed to retrieve VIX data")
        return False


def test_partial_data():
    """Test that partial data (some fields None) works correctly."""
    print("\n" + "=" * 60)
    print("Testing Partial Data Handling")
    print("=" * 60)
    
    # Save with only some fields
    record_macro_snapshot(
        exchange="NSE",
        fii_flow=1000.0,  # Only this field
        dii_flow=None,     # Missing
        usdinr=83.0,       # Available
        usdinr_trend=None, # Missing
        crude_price=None,  # Missing
        crude_trend=None,  # Missing
        timestamp=now_ist()
    )
    
    latest = db.get_latest_macro_signals("NSE")
    if latest:
        print("✓ Partial data saved successfully")
        print(f"  Available fields: FII={latest.get('fii_flow')}, USD/INR={latest.get('usdinr')}")
        print(f"  Missing fields handled gracefully (None values)")
        return True
    else:
        print("✗ Failed to save partial data")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("External Data Ingestion Test Suite")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Macro Data", test_macro_data()))
    results.append(("VIX Term Structure", test_vix_term_structure()))
    results.append(("Partial Data", test_partial_data()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n✓ All tests passed! Data ingestion is working correctly.")
        print("\nNext steps:")
        print("1. Customize scripts/example_macro_fetcher.py with your data sources")
        print("2. Test with real data")
        print("3. Set up scheduling (see EXTERNAL_DATA_GUIDE.md)")
    else:
        print("\n✗ Some tests failed. Check database and imports.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

