#!/usr/bin/env python3
"""
Backend test runner script for AI chat components.
Runs comprehensive tests for ChatModerator, ChatOrchestrator, and integration tests.
"""

import sys
import subprocess
import os
from pathlib import Path

def run_tests():
    """Run the backend AI chat test suite."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    # Test files to run (excluding the broken original test file)
    test_files = [
        "back/test/ai/chat/test_chat_moderator_fixed.py",
        "back/test/ai/chat/test_chat_orchestrator.py", 
        "back/test/ai/chat/test_integration.py"
    ]
    
    print("=" * 70)
    print("BACKEND AI CHAT TEST SUITE")
    print("=" * 70)
    print("Running comprehensive tests for:")
    print("• ChatModerator (message validation, spam detection, toxicity filtering)")
    print("• ChatOrchestrator (session management, message handling)")
    print("• Integration tests (complete backend AI chat workflow)")
    print("• Uses local sentiment analysis model from models/multilingual-sentiment-analysis")
    print("=" * 70)
    
    # Build pytest command
    cmd = [
        sys.executable, "-m", "pytest"
    ] + test_files + [
        "-v",                    # verbose output
        "--tb=short",           # short traceback format
        "--color=yes",          # colored output
        "-x",                   # stop on first failure
        "--durations=10"        # show 10 slowest tests
    ]
    
    # Run the tests
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        return 130
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

def run_quick_tests():
    """Run a quick subset of tests for development."""
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    print("Running quick backend AI chat tests...")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "back/test/ai/chat/test_chat_moderator_fixed.py::TestChatModerator::test_moderate_message_valid",
        "back/test/ai/chat/test_chat_orchestrator.py::TestChatOrchestrator::test_create_chat_session_backend",
        "back/test/ai/chat/test_integration.py::TestBackendChatIntegration::test_backend_ai_chat_workflow",
        "-v"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode
    except Exception as e:
        print(f"Error running quick tests: {e}")
        return 1

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        exit_code = run_quick_tests()
    else:
        exit_code = run_tests()
    
    print("\n" + "=" * 70)
    if exit_code == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 70)
    
    sys.exit(exit_code)
