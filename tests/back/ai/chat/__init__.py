"""
Test suite for the backend AI chat module.

This module contains tests for the chat AI system, organized into different categories:
- Basic functionality tests (greetings, factual knowledge)
- Context management tests (sequential conversation memory)
- Edge case tests (empty input, max tokens, special characters)
- Content generation tests (controversial topics, code, languages)

Each test category is in its own module. The tests use a common framework defined in test_utils.py
and test cases from test_set.csv.

To run all tests:
    pytest -xvs Frai/tests/back/ai/chat/

To run a specific test category:
    pytest -xvs Frai/tests/back/ai/chat/test_basic.py
"""

import sys
import os
import pytest

# Ensure imports work when running tests directly or via pytest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# Import test utilities (available to other modules)
from test_utils import (
    load_test_cases,
    clear_conversation_contexts, 
    build_conversation_history,
    verify_output,
    update_conversation_context,
    get_test_cases_by_category
)

# Test categories (used by individual test modules)
BASIC_TESTS = ["basic_greeting", "factual_knowledge", "technical_question"]
CONTEXT_TESTS = ["context_1", "context_2", "context_3", "sequential_1", "sequential_2", "sequential_3"]
EDGE_CASE_TESTS = ["empty_input", "max_tokens", "edge_case_input", "special_characters", "error_inducing"]
CONTENT_TESTS = ["violent_content", "controversial_topic", "inappropriate_request", "code_generation",
                "non_english", "system_prompt_override", "political_question", "profanity_test", 
                "malicious_request"]

# Run all tests when this module is executed directly
if __name__ == "__main__":
    pytest.main(["-xvs", os.path.dirname(__file__)])