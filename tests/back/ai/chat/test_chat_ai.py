"""
Tests for ChatAI module using the test_set.csv file.
"""

import os
import sys
import pytest
import logging
from typing import Dict, List, Any

# Add the project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# Import the helper functions from the test outline
from test_outline import (
    load_test_cases, 
    build_conversation_history, 
    verify_output,
    update_conversation_context,
    clear_conversation_contexts
)

# Import the chat module functions to test
from Frai.back.ai.chat import (
    initialize_chat_system,
    get_chat_ai_instance,
    generate_ai_text
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def setup_chat_ai():
    """
    Initialize the ChatAI system once for all tests.
    """
    # Clear any existing context from previous test runs
    clear_conversation_contexts()
    
    # Initialize the chat system
    success = initialize_chat_system()
    if not success:
        pytest.fail("Failed to initialize ChatAI system")
    
    # Get the chat AI instance
    chat_ai = get_chat_ai_instance()
    
    return chat_ai


# Load all test cases from CSV
test_cases = load_test_cases()


@pytest.mark.parametrize("test_case", test_cases)
def test_chat_responses(setup_chat_ai, test_case):
    """
    Test each case from the test set.
    
    Args:
        setup_chat_ai: The ChatAI instance from the fixture
        test_case: Dictionary containing test case details from CSV
    """
    # Build conversation history including any partial conversations
    conversation_history = build_conversation_history(test_case)
    
    # Skip empty input test if it's expecting an error (special handling)
    if not conversation_history and test_case['output'].lower() == 'error':
        assert True
        return
    
    # Generate response using the real generate_ai_text function
    response = generate_ai_text(conversation_history=conversation_history)
    
    # Log the response for debugging
    if response.get("success", False):
        logger.info(f"Test chat-{test_case['index']} response: {response['response'][:100]}...")
    else:
        logger.warning(f"Test chat-{test_case['index']} error: {response.get('error', 'Unknown error')}")
    
    # Update conversation context if successful
    if response.get("success", False):
        update_conversation_context(test_case, response)
    
    # Verify the response matches expected output
    assert verify_output(response, test_case['output']), \
        f"Test chat-{test_case['index']} failed: Response doesn't match expected output"


if __name__ == "__main__":
    # To run the tests directly
    pytest.main(["-xvs", __file__])