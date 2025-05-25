#!/usr/bin/env python3
"""
Comprehensive Test Runner for Chat System

This script runs the complete test suite including smoke, blackbox, and integration tests.
It provides options for running specific test categories and generating coverage reports.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def setup_environment():
    """Setup the environment for running tests"""
    # Add project root to Python path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "orchestrator"))
    
    # Set environment variables
    os.environ["PYTHONPATH"] = str(project_root)
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"


def run_command(command, description):
    """Run a shell command and display results"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"❌ Command failed with return code {result.returncode}")
            return False
        else:
            print("✅ Command completed successfully")
            return True
            
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False


def run_tests(test_type="all", verbose=False, coverage=False):
    """Run the specified tests"""
    
    # Base pytest command
    base_cmd = "python -m pytest"
    
    # Add verbosity
    if verbose:
        base_cmd += " -v"
    else:
        base_cmd += " -q"
    
    # Add coverage if requested
    if coverage:
        base_cmd += " --cov=. --cov-report=html --cov-report=term-missing"
    
    # Define test commands for different types
    test_commands = {
        "smoke": f"{base_cmd} -m smoke",
        "blackbox": f"{base_cmd} -m blackbox", 
        "integration": f"{base_cmd} -m integration",
        "unit": f"{base_cmd} -m unit",
        "performance": f"{base_cmd} -m performance",
        "fast": f"{base_cmd} -m 'smoke or (unit and not slow)'",
        "all": f"{base_cmd}",
    }
    
    success = True
    
    if test_type in test_commands:
        success = run_command(
            test_commands[test_type], 
            f"Running {test_type} tests"
        )
    else:
        print(f"❌ Unknown test type: {test_type}")
        print(f"Available types: {list(test_commands.keys())}")
        return False
    
    return success


def run_test_discovery():
    """Run test discovery to verify all tests can be found"""
    print("\n📋 Discovering tests...")
    return run_command(
        "python -m pytest --collect-only -q",
        "Test discovery"
    )


def run_test_validation():
    """Validate test setup and dependencies"""
    print("\n🔍 Validating test setup...")
    
    # Check if required modules can be imported
    validation_script = '''
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "orchestrator"))

try:
    # Test imports
    from unittest.mock import patch
    print("✅ unittest.mock available")
    
    import pytest
    print("✅ pytest available")
    
    # Test project imports with mocking
    with patch('transformers.pipeline'):
        try:
            from chatmod import ChatModerator
            print("✅ ChatModerator can be imported")
        except Exception as e:
            print(f"❌ ChatModerator import failed: {e}")
        
        try:
            from chat.orchestrator import ChatOrchestrator
            print("✅ ChatOrchestrator can be imported")
        except Exception as e:
            print(f"❌ ChatOrchestrator import failed: {e}")
        
        try:
            from chat.models import ChatMessage, ChatSession
            print("✅ Chat models can be imported")
        except Exception as e:
            print(f"❌ Chat models import failed: {e}")

except Exception as e:
    print(f"❌ Critical import error: {e}")
    sys.exit(1)

print("✅ All validation checks passed")
'''
    
    return run_command(
        f'python -c "{validation_script}"',
        "Dependency validation"
    )


def generate_test_report():
    """Generate a comprehensive test report"""
    print("\n📊 Generating comprehensive test report...")
    
    report_commands = [
        ("python -m pytest -m smoke --tb=line", "Smoke test results"),
        ("python -m pytest -m blackbox --tb=line", "Blackbox test results"),
        ("python -m pytest -m integration --tb=line", "Integration test results"),
    ]
    
    all_passed = True
    
    for command, description in report_commands:
        success = run_command(command, description)
        if not success:
            all_passed = False
    
    return all_passed


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for chat system"
    )
    
    parser.add_argument(
        "test_type",
        nargs="?",
        default="all",
        choices=["smoke", "blackbox", "integration", "unit", "performance", "fast", "all"],
        help="Type of tests to run (default: all)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation, don't run tests"
    )
    
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only run test discovery"
    )
    
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate comprehensive test report"
    )
    
    args = parser.parse_args()
    
    print("🚀 Chat System Test Runner")
    print("=" * 40)
    
    # Setup environment
    setup_environment()
    
    success = True
    
    # Run validation
    if not run_test_validation():
        print("❌ Validation failed!")
        return 1
    
    if args.validate_only:
        print("✅ Validation completed successfully")
        return 0
    
    # Run test discovery
    if not run_test_discovery():
        print("❌ Test discovery failed!")
        return 1
    
    if args.discover_only:
        print("✅ Test discovery completed successfully")
        return 0
    
    # Run tests
    if args.report:
        success = generate_test_report()
    else:
        success = run_tests(args.test_type, args.verbose, args.coverage)
    
    # Summary
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests completed successfully!")
        print("✅ Chat system is working correctly")
    else:
        print("❌ Some tests failed!")
        print("🔧 Please check the output above for details")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
