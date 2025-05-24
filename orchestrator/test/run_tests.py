#!/usr/bin/env python3
"""
Test runner for orchestrator chat components.
Runs all tests for ChatModerator and ChatOrchestrator with proper configuration.
"""

import sys
import subprocess
import pytest
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def run_moderator_tests(verbose=False):
    """Run ChatModerator tests."""
    test_file = Path(__file__).parent / "chatmod" / "test_chat_moderator.py"
    
    args = [str(test_file)]
    if verbose:
        args.extend(["-v", "-s"])
    
    print("=" * 60)
    print("Running ChatModerator Tests")
    print("=" * 60)
    
    result = pytest.main(args)
    return result == 0


def run_orchestrator_tests(verbose=False):
    """Run ChatOrchestrator tests."""
    test_file = Path(__file__).parent / "chat" / "test_chat_orchestrator.py"
    
    args = [str(test_file)]
    if verbose:
        args.extend(["-v", "-s"])
    
    print("=" * 60)
    print("Running ChatOrchestrator Tests")
    print("=" * 60)
    
    result = pytest.main(args)
    return result == 0


def run_integration_tests(verbose=False):
    """Run integration tests."""
    test_file = Path(__file__).parent / "test_integration.py"
    
    args = [str(test_file)]
    if verbose:
        args.extend(["-v", "-s"])
    
    print("=" * 60)
    print("Running Integration Tests")
    print("=" * 60)
    
    result = pytest.main(args)
    return result == 0


def run_all_tests(verbose=False):
    """Run all orchestrator tests."""
    print("Running All Orchestrator Chat Tests")
    print("=" * 80)
    
    results = []
    
    # Run individual test suites
    results.append(run_moderator_tests(verbose))
    results.append(run_orchestrator_tests(verbose))
    results.append(run_integration_tests(verbose))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    test_names = ["ChatModerator", "ChatOrchestrator", "Integration"]
    for name, passed in zip(test_names, results):
        status = "PASSED" if passed else "FAILED"
        print(f"{name:20} : {status}")
    
    all_passed = all(results)
    print(f"\nOverall Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return all_passed


def run_quick_tests():
    """Run quick smoke tests."""
    print("Running Quick Smoke Tests")
    print("=" * 60)
    
    # Run just a few key tests from each suite
    test_args = [
        str(Path(__file__).parent),
        "-k", "test_init or test_create_chat_session or test_complete_chat_workflow",
        "-v"
    ]
    
    result = pytest.main(test_args)
    return result == 0


def run_performance_tests():
    """Run performance-focused tests."""
    print("Running Performance Tests")
    print("=" * 60)
    
    test_args = [
        str(Path(__file__).parent),
        "-k", "performance or concurrent or large",
        "-v"
    ]
    
    result = pytest.main(test_args)
    return result == 0


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run orchestrator chat tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", "-q", action="store_true", help="Run quick smoke tests")
    parser.add_argument("--performance", "-p", action="store_true", help="Run performance tests")
    parser.add_argument("--moderator", "-m", action="store_true", help="Run only moderator tests")
    parser.add_argument("--orchestrator", "-o", action="store_true", help="Run only orchestrator tests")
    parser.add_argument("--integration", "-i", action="store_true", help="Run only integration tests")
    
    args = parser.parse_args()
    
    success = True
    
    if args.quick:
        success = run_quick_tests()
    elif args.performance:
        success = run_performance_tests()
    elif args.moderator:
        success = run_moderator_tests(args.verbose)
    elif args.orchestrator:
        success = run_orchestrator_tests(args.verbose)
    elif args.integration:
        success = run_integration_tests(args.verbose)
    else:
        success = run_all_tests(args.verbose)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
