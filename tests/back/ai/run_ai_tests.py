#!/usr/bin/env python3
"""
AI Module Test Runner

This script demonstrates how to run tests at different levels:
- Individual AI modules (sentiment, img2img, text2img, voicein, voiceout)
- All AI modules together
- Specific test types (unit, accuracy, consistency, performance, edge_cases)

Usage:
    python run_ai_tests.py --help
    python run_ai_tests.py --module sentiment
    python run_ai_tests.py --module all
    python run_ai_tests.py --type unit
    python run_ai_tests.py --module sentiment --type accuracy
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

def run_pytest_command(args_list, description):
    """Run pytest with given arguments and handle results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: pytest {' '.join(args_list)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            ['pytest'] + args_list,
            cwd=project_root,
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n✅ {description} - PASSED")
        else:
            print(f"\n❌ {description} - FAILED (exit code: {result.returncode})")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"\n💥 Error running {description}: {e}")
        return False

def run_module_tests(module_name, test_type=None, verbose=False):
    """Run tests for a specific AI module."""
    if module_name == "all":
        test_path = "Frai/tests/back/ai/"
        description = "All AI modules"
    else:
        test_path = f"Frai/tests/back/ai/{module_name}/"
        description = f"AI module: {module_name}"
    
    args = [test_path]
    
    # Add test type filter if specified
    if test_type:
        args.extend(["-m", test_type])
        description += f" ({test_type} tests)"
    
    # Add verbosity
    if verbose:
        args.append("-vv")
    else:
        args.append("-v")
    
    # Add other useful options
    args.extend([
        "--tb=short",
        "--color=yes",
        "--durations=10"
    ])
    
    return run_pytest_command(args, description)

def run_test_type_across_modules(test_type, verbose=False):
    """Run a specific test type across all AI modules."""
    args = [
        "Frai/tests/back/ai/",
        "-m", test_type
    ]
    
    if verbose:
        args.append("-vv")
    else:
        args.append("-v")
    
    args.extend([
        "--tb=short",
        "--color=yes",
        "--durations=10"
    ])
    
    description = f"All AI modules - {test_type} tests only"
    return run_pytest_command(args, description)

def run_discovery_test():
    """Run test discovery to show what tests are available."""
    args = [
        "Frai/tests/back/ai/",
        "--collect-only",
        "-q"
    ]
    
    description = "Test discovery (show available tests)"
    return run_pytest_command(args, description)

def main():
    parser = argparse.ArgumentParser(
        description="Run AI module tests at different levels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all tests for sentiment analysis module
    python run_ai_tests.py --module sentiment
    
    # Run only unit tests for img2img module
    python run_ai_tests.py --module img2img --type unit
    
    # Run all AI module tests
    python run_ai_tests.py --module all
    
    # Run only accuracy tests across all AI modules
    python run_ai_tests.py --type accuracy
    
    # Run performance tests with verbose output
    python run_ai_tests.py --type performance --verbose
    
    # Discover all available tests
    python run_ai_tests.py --discover
        """
    )
    
    parser.add_argument(
        "--module", "-m",
        choices=["sentiment", "img2img", "text2img", "voicein", "voiceout", "all"],
        help="AI module to test"
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=["unit", "accuracy", "consistency", "performance", "edge_case", "tdd"],
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--discover", "-d",
        action="store_true",
        help="Discover and list available tests without running them"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run only fast tests (skip performance tests)"
    )
    
    parser.add_argument(
        "--tdd-only",
        action="store_true",
        help="Run only TDD tests (tests that may fail until implemented)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.module, args.type, args.discover, args.tdd_only]):
        parser.error("Must specify at least one of: --module, --type, --discover, or --tdd-only")
    
    if args.module and args.type and args.module != "all":
        # Run specific test type for specific module
        success = run_module_tests(args.module, args.type, args.verbose)
    elif args.module:
        # Run all tests for specific module(s)
        if args.fast:
            success = run_module_tests(args.module, "unit", args.verbose)
        else:
            success = run_module_tests(args.module, None, args.verbose)
    elif args.type:
        # Run specific test type across all modules
        success = run_test_type_across_modules(args.type, args.verbose)
    elif args.discover:
        # Discover tests
        success = run_discovery_test()
    elif args.tdd_only:
        # Run only TDD tests
        success = run_test_type_across_modules("tdd", args.verbose)
    else:
        parser.error("Invalid combination of arguments")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()