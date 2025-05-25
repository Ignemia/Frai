#!/usr/bin/env python3
"""
Test runner script for chat moderation and orchestrator tests.
"""

import sys
import pytest
from pathlib import Path

def main():
    """Run the tests and save output to a file."""
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Add orchestrator module to path
    orchestrator_path = project_root / "orchestrator"
    sys.path.insert(0, str(orchestrator_path))
    
    # Run tests
    test_path = "test/ai/chat/test_integration.py"
    args = [
        test_path,
        "-v",
        "--tb=short",
        "--capture=tee-sys"
    ]
    
    # Run pytest and capture output
    result = pytest.main(args)
    
    # Print result code
    print(f"\nTest execution completed with result code: {result}")
    return result

if __name__ == "__main__":
    sys.exit(main()) 