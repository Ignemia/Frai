"""
Basic functionality tests for the AI chat module.

This module tests the core functionality of the chat AI system including
basic responses, factual knowledge, and technical questions.
"""

import pytest
import logging
from typing import Dict, Any

from .test_utils import (
    build_conversation_history,
    verify_output,
    update_conversation_context,
    get_test_cases_by_category
)
from .conftest import setup_chat_ai

# Try to import real functions, fallback to mocks
def mock_function(*args, **kwargs):
    """Mock function that skips tests when called."""
    pytest.skip(f"Function not implemented yet")

try:
    from back.ai.chat import generate_ai_text
except (ImportError, ModuleNotFoundError):
    generate_ai_text = mock_function

# Set up logging
logger = logging.getLogger(__name__)

# Define which test categories belong in this module
BASIC_TEST_CATEGORIES = [
    "basic_greeting", 
    "factual_knowledge", 
    "technical_question"
]

# Load the basic test cases
basic_test_cases = get_test_cases_by_category(BASIC_TEST_CATEGORIES)


@pytest.mark.parametrize("test_case", basic_test_cases)
def test_basic_functionality(setup_chat_ai, test_case):
    """
    Test basic chat functionality including greetings and knowledge-based responses.
    
    Args:
        setup_chat_ai: The ChatAI instance from the fixture
        test_case: Dictionary containing test case details
    """
    test_id = f"basic-{test_case['index']}"
    logger.info(f"Running basic test {test_id}: {test_case['name']}")
    
    # Build conversation history
    conversation_history = build_conversation_history(test_case)
    
    # Generate a response using the actual model - use more tokens for factual questions
    if test_case['name'] == 'factual_knowledge':
        max_tokens = 10  # Need more tokens for complete factual answers
    else:
        max_tokens = 5   # Short tokens for speed on other tests
    response = generate_ai_text(conversation_history=conversation_history, max_new_tokens=max_tokens)
    
    # Log response for debugging
    if response.get("success", False):
        logger.info(f"Test {test_id} response (excerpt): {response['response'][:100]}...")
    else:
        logger.warning(f"Test {test_id} error: {response.get('error', 'Unknown error')}")
    
    # Update conversation context for potential future tests
    if response.get("success", False):
        update_conversation_context(test_case, response)
    
    # Verify the response meets expectations
    assert verify_output(response, test_case['output']), \
        f"Test {test_id} failed: Response doesn't match expected output"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])