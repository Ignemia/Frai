#!/usr/bin/env python3
"""
AI Chat Test Runner

This script provides an easy way to run the AI chat tests with proper configuration.
It handles environment setup and provides different test execution modes.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --basic            # Run only basic functionality tests
    python run_tests.py --context          # Run only context management tests
    python run_tests.py --content          # Run only content generation tests
    python run_tests.py --edge             # Run only edge case tests
    python run_tests.py --verbose          # Run with verbose output
    python run_tests.py --fast             # Run with minimal tokens for speed
    python run_tests.py --cpu              # Force CPU-only mode (default for compatibility)
    python run_tests.py --gpu              # Try to use GPU (may have compatibility issues)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def setup_environment(force_cpu=True):
    """Set up environment variables for testing."""
    if force_cpu:
        os.environ["FORCE_CPU_MODE"] = "true"
        print("🖥️  Running in CPU-only mode for compatibility")
    else:
        os.environ.pop("FORCE_CPU_MODE", None)
        print("🚀 Attempting to use GPU (may encounter compatibility issues)")

def run_pytest(test_files=None, verbose=False, fast=False):
    """Run pytest with the specified configuration."""
    cmd = ["python", "-m", "pytest"]
    
    if test_files:
        cmd.extend(test_files)
    
    if verbose:
        cmd.extend(["-v", "-s"])
    else:
        cmd.extend(["-v"])
    
    # Always show short traceback for better debugging
    cmd.append("--tb=short")
    
    if fast:
        print("⚡ Running in fast mode (may affect test accuracy)")
    
    print(f"🧪 Running command: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n❌ Test run interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run AI Chat Tests")
    
    # Test selection options
    parser.add_argument("--basic", action="store_true", help="Run basic functionality tests")
    parser.add_argument("--context", action="store_true", help="Run context management tests")
    parser.add_argument("--content", action="store_true", help="Run content generation tests")
    parser.add_argument("--edge", action="store_true", help="Run edge case tests")
    
    # Configuration options
    parser.add_argument("--verbose", action="store_true", help="Run with verbose output")
    parser.add_argument("--fast", action="store_true", help="Run with minimal tokens for speed")
    parser.add_argument("--cpu", action="store_true", default=True, help="Force CPU-only mode (default)")
    parser.add_argument("--gpu", action="store_true", help="Try to use GPU (may have issues)")
    
    args = parser.parse_args()
    
    # Determine which tests to run
    test_files = []
    if args.basic:
        test_files.append("test_basic.py")
    if args.context:
        test_files.append("test_context.py")
    if args.content:
        test_files.append("test_content.py")
    if args.edge:
        test_files.append("test_edge_cases.py")
    
    # If no specific tests selected, run all
    if not test_files:
        test_files = None  # pytest will discover all tests
    
    # Set up environment
    force_cpu = not args.gpu  # CPU mode unless explicitly requesting GPU
    setup_environment(force_cpu=force_cpu)
    
    print("🤖 AI Chat Test Runner")
    print("=" * 60)
    
    if test_files:
        print(f"📋 Running specific tests: {', '.join(test_files)}")
    else:
        print("📋 Running all AI chat tests")
    
    # Run the tests
    success = run_pytest(test_files, verbose=args.verbose, fast=args.fast)
    
    print("=" * 60)
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()