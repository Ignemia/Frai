"""
Context management tests for the AI chat module.

This module tests the chat AI's ability to maintain context across 
sequential messages in a conversation.
"""

import pytest
import logging
from typing import Dict, Any

from .test_utils import (
    build_conversation_history,
    verify_output,
    update_conversation_context,
    get_test_cases_by_category,
    load_test_cases
)
from .conftest import setup_chat_ai
from back.ai.chat import generate_ai_text

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


def _run_prerequisite_tests(test_case: Dict[str, str], setup_chat_ai):
    """Run prerequisite tests to build conversation context."""
    if not test_case['partials']:
        return
    
    all_test_cases = load_test_cases()
    test_dict = {tc['index']: tc for tc in all_test_cases}
    
    partial_indices = test_case['partials'].split(';')
    for index in partial_indices:
        partial_id = index.strip()
        if partial_id not in test_dict:
            continue
            
        prerequisite_test = test_dict[partial_id]
        logger.info(f"Running prerequisite test {partial_id}: {prerequisite_test['name']}")
        
        # Build conversation for prerequisite
        prerequisite_conversation = build_conversation_history(prerequisite_test)
        
        # Generate response for prerequisite
        prereq_response = generate_ai_text(
            conversation_history=prerequisite_conversation, 
            max_new_tokens=5
        )
        
        # Update context if successful
        if prereq_response.get("success", False):
            update_conversation_context(prerequisite_test, prereq_response)
            logger.info(f"Prerequisite {partial_id} completed, context updated")
        else:
            logger.warning(f"Prerequisite {partial_id} failed: {prereq_response.get('error', 'Unknown')}")


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
    
    # Run prerequisite tests to build context if needed
    _run_prerequisite_tests(test_case, setup_chat_ai)
    
    # Build conversation history, including any partial conversations
    # This is critical for context tests, as they rely on previous messages
    conversation_history = build_conversation_history(test_case)
    
    # Skip if the conversation history couldn't be built properly
    if not conversation_history and test_case['partials']:
        pytest.skip(f"Test {test_id} skipped: Required previous tests not available in context")
    
    # Generate a response using the actual model - use more tokens for context tests
    # that need complete answers or reasoning
    if test_case['name'] in ['context_3', 'sequential_3']:
        max_tokens = 20  # Need more tokens for reasoning and complete answers
    elif test_case['name'].startswith('sequential'):
        max_tokens = 15  # Sequential tests need reasoning space
    else:
        max_tokens = 5
    response = generate_ai_text(conversation_history=conversation_history, max_new_tokens=max_tokens)
    
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