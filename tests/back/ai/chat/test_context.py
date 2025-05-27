"""
Context management tests for the AI chat module.

This module tests the chat AI's ability to maintain context across 
sequential messages in a conversation.
"""

import pytest
import logging
from typing import Dict, Any

from test_utils import (
    build_conversation_history,
    verify_output,
    update_conversation_context,
    get_test_cases_by_category
)
from conftest import setup_chat_ai
from Frai.back.ai.chat import generate_ai_text

# Set up logging
logger = logging.getLogger(__name__)

# Define which test categories belong in this module
CONTEXT_TEST_CATEGORIES = [
    "context_1", 
    "context_2", 
    "context_3", 
    "sequential_1", 
    "sequential_2", 
    "sequential_3"
]

# Load the context test cases
context_test_cases = get_test_cases_by_category(CONTEXT_TEST_CATEGORIES)


@pytest.mark.parametrize("test_case", context_test_cases)
def test_context_management(setup_chat_ai, test_case):
    """
    Test the AI's ability to maintain context across messages.
    
    Args:
        setup_chat_ai: The ChatAI instance from the fixture
        test_case: Dictionary containing test case details
    """
    test_id = f"context-{test_case['index']}"
    logger.info(f"Running context test {test_id}: {test_case['name']}")
    
    # Build conversation history, including any partial conversations
    # This is critical for context tests, as they rely on previous messages
    conversation_history = build_conversation_history(test_case)
    
    # Skip if the conversation history couldn't be built properly
    if not conversation_history and test_case['partials']:
        pytest.skip(f"Test {test_id} skipped: Required previous tests not available in context")
    
    # Generate a response using the actual model
    response = generate_ai_text(conversation_history=conversation_history)
    
    # Log response for debugging
    if response.get("success", False):
        logger.info(f"Test {test_id} response (excerpt): {response['response'][:100]}...")
    else:
        logger.warning(f"Test {test_id} error: {response.get('error', 'Unknown error')}")
    
    # Update conversation context for future tests
    if response.get("success", False):
        update_conversation_context(test_case, response)
    
    # Verify the response meets expectations
    # For context tests, this often means checking if certain information from previous
    # messages is present in the response
    assert verify_output(response, test_case['output']), \
        f"Test {test_id} failed: Response doesn't demonstrate required context retention"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])