"""
Simple test runner - executes all test suites without pytest.
Run with: python run_all_tests.py
"""

import subprocess
import sys
from pathlib import Path


def run_test_file(test_file):
    """Run a single test file and return results."""
    result = subprocess.run(
        [sys.executable, test_file],
        capture_output=False,
        text=True
    )
    return result.returncode == 0


def main():
    """Run all test suites."""
    test_dir = Path(__file__).parent
    
    print("\n" + "="*70)
    print("DRONE SECURITY ANALYST - TEST SUITE RUNNER")
    print("="*70)
    
    test_files = [
        ("Telemetry Generation", "simple_test_telemetry.py"),
        ("Alert System", "simple_test_alerts.py"),
        ("Frame Indexing", "simple_test_indexing.py")
    ]
    
    results = {}
    
    for name, test_file in test_files:
        test_path = test_dir / test_file
        
        if not test_path.exists():
            print(f"\n⚠️  Test file not found: {test_file}")
            results[name] = False
            continue
        
        print(f"\n{'='*70}")
        print(f"Running: {name}")
        print(f"{'='*70}")
        
        passed = run_test_file(str(test_path))
        results[name] = passed
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
