"""
Edge case tests for the AI chat module.

This module tests the chat AI's behavior with unusual inputs and edge cases
such as empty input, very long inputs, special characters, and inputs designed
to trigger errors.
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
from back.ai.chat import generate_ai_text

# Set up logging
logger = logging.getLogger(__name__)

# Define which test categories belong in this module
EDGE_CASE_CATEGORIES = [
    "empty_input", 
    "max_tokens", 
    "special_characters", 
    "error_inducing",
    "edge_case_input"
]

# Load the edge case test cases
edge_case_tests = get_test_cases_by_category(EDGE_CASE_CATEGORIES)


def test_empty_input():
    """
    Test with empty input to verify proper error handling.
    This test is handled separately because it doesn't use the normal flow.
    """
    logger.info("Running empty input test")
    
    # Empty conversation history
    empty_conversation = []
    
    # Generate response
    response = generate_ai_text(conversation_history=empty_conversation)
    
    # Should fail with an error
    assert not response.get("success", False), "Empty input should result in an error"
    assert response.get("error") is not None, "Empty input should have an error message"
    
    logger.info(f"Empty input test passed with error: {response.get('error')}")


@pytest.mark.parametrize("test_case", [t for t in edge_case_tests if t["name"] != "empty_input"])
def test_edge_cases(setup_chat_ai, test_case):
    """
    Test the AI's behavior with edge case inputs.
    
    Args:
        setup_chat_ai: The ChatAI instance from the fixture
        test_case: Dictionary containing test case details
    """
    test_id = f"edge-{test_case['index']}"
    logger.info(f"Running edge case test {test_id}: {test_case['name']}")
    
    # Build conversation history
    conversation_history = build_conversation_history(test_case)
    
    # Generate a response using the actual model with short tokens for speed
    response = generate_ai_text(conversation_history=conversation_history, max_new_tokens=5)
    
    # Log response for debugging
    if response.get("success", False):
        logger.info(f"Test {test_id} response (excerpt): {response['response'][:100]}...")
    else:
        logger.warning(f"Test {test_id} error: {response.get('error', 'Unknown error')}")
    
    # For successful responses, update context
    if response.get("success", False):
        update_conversation_context(test_case, response)
    
    # Verify the response meets expectations
    assert verify_output(response, test_case['output']), \
        f"Test {test_id} failed: Response doesn't meet expectations for this edge case"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])