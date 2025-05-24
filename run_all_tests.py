#!/usr/bin/env python3
"""
Comprehensive test runner for the Frai project.
Runs all tests with proper organization and reporting.
"""
import sys
import subprocess
import time
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    print(f"Exit code: {result.returncode}")
    print(f"Duration: {end_time - start_time:.2f} seconds")
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    return result.returncode == 0


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Run Frai project tests")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--performance", action="store_true", help="Run only performance tests")
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage reporting")
    
    args = parser.parse_args()
    
    # Default to all tests if no specific type is selected
    if not any([args.unit, args.integration, args.performance]):
        args.all = True
    
    tests_dir = Path(__file__).parent
    results = []
    
    print("Frai Project Test Suite")
    print("=" * 60)
    print(f"Test directory: {tests_dir}")
    print(f"Project root: {project_root}")
      # Prepare pytest args - use current Python executable
    pytest_args = [sys.executable, "-m", "pytest"]
    
    if args.verbose:
        pytest_args.append("-v")
    else:
        pytest_args.extend(["-q", "--tb=short"])
    
    if args.coverage:
        pytest_args.extend(["--cov=.", "--cov-report=html", "--cov-report=term"])
    
    # Add colored output
    pytest_args.append("--color=yes")
      # Run unit tests
    if args.unit or args.all:
        unit_cmd = pytest_args + [str(tests_dir / "tests" / "unit")]
        success = run_command(unit_cmd, "Unit Tests")
        results.append(("Unit Tests", success))
    
    # Run integration tests
    if args.integration or args.all:
        integration_cmd = pytest_args + [str(tests_dir / "tests" / "integration")]
        success = run_command(integration_cmd, "Integration Tests")
        results.append(("Integration Tests", success))
    
    # Run performance tests
    if args.performance or args.all:
        performance_cmd = pytest_args + [str(tests_dir / "tests" / "performance")]
        success = run_command(performance_cmd, "Performance Tests")
        results.append(("Performance Tests", success))
    
    # Run end-to-end tests
    if args.all:
        blackbox_cmd = pytest_args + [str(tests_dir / "tests" / "blackbox")]
        success = run_command(blackbox_cmd, "End-to-End/Blackbox Tests")
        results.append(("End-to-End Tests", success))
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    all_passed = True
    for test_type, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_type:<25} {status}")
        if not success:
            all_passed = False
    
    print(f"{'='*60}")
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("💥 SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
