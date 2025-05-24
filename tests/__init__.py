# Tests package for personal-chatter application
"""
Test suite for the personal-chatter application.

This package contains:
- unit: Unit tests for individual components
- integration: Integration tests for component interactions
- implementation: Implementation-specific tests
- blackbox: End-to-end blackbox tests
- performance: Performance and load tests
- fixtures: Test data and fixtures
- utils: Test utilities and helpers
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
