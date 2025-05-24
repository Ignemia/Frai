#!/usr/bin/env python3
"""
Test runner script for the personal-chatter application.

This script provides convenient commands for running different types of tests
with appropriate configurations and reporting.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import os


def run_command(command, description):
    """Run a command and handle the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    result = subprocess.run(command, shell=True, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n❌ {description} failed with exit code {result.returncode}")
        return False
    else:
        print(f"\n✅ {description} completed successfully")
        return True


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Test runner for personal-chatter")
    parser.add_argument(
        "test_type",
        choices=["unit", "integration", "implementation", "blackbox", "performance", "all"],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--parallel",
        "-n",
        type=int,
        default=1,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--fail-fast",
        "-x",
        action="store_true",
        help="Stop on first failure"
    )
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML coverage report"
    )
    
    args = parser.parse_args()
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Base pytest command
    base_cmd = ["python", "-m", "pytest"]
    
    # Add test directory based on type
    if args.test_type == "all":
        test_dirs = ["tests/"]
    else:
        test_dirs = [f"tests/{args.test_type}/"]
    
    # Build command
    cmd_parts = base_cmd + test_dirs
    
    # Add options
    if args.verbose:
        cmd_parts.append("-v")
    
    if args.fail_fast:
        cmd_parts.append("-x")
    
    if args.parallel > 1:
        cmd_parts.extend(["-n", str(args.parallel)])
    
    if args.coverage:
        cmd_parts.extend([
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=xml"
        ])
        
        if args.html_report:
            cmd_parts.append("--cov-report=html")
    
    # Add configuration file
    cmd_parts.extend(["-c", "tests/pytest.ini"])
    
    # Run the tests
    command = " ".join(cmd_parts)
    success = run_command(command, f"{args.test_type.title()} Tests")
    
    if success:
        print(f"\n🎉 All {args.test_type} tests passed!")
        
        if args.coverage:
            print("\n📊 Coverage report generated:")
            print("  - Terminal: Above")
            print("  - XML: coverage.xml")
            if args.html_report:
                print("  - HTML: htmlcov/index.html")
    else:
        print(f"\n💥 {args.test_type.title()} tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
