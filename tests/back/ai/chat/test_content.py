"""
Content generation tests for the AI chat module.

This module tests the chat AI's ability to generate various types of content,
including controversial topics, violent content, code generation, and more.
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
CONTENT_TEST_CATEGORIES = [
    "violent_content", 
    "controversial_topic", 
    "code_generation", 
    "inappropriate_request",
    "profanity_test",
    "political_question",
    "malicious_request",
    "non_english",
    "system_prompt_override"
]

# Load the content test cases
content_test_cases = get_test_cases_by_category(CONTENT_TEST_CATEGORIES)


@pytest.mark.parametrize("test_case", content_test_cases)
def test_content_generation(setup_chat_ai, test_case):
    """
    Test the AI's ability to generate various types of content,
    including potentially controversial or sensitive material.
    
    Args:
        setup_chat_ai: The ChatAI instance from the fixture
        test_case: Dictionary containing test case details
    """
    test_id = f"content-{test_case['index']}"
    logger.info(f"Running content test {test_id}: {test_case['name']}")
    
    # Build conversation history
    conversation_history = build_conversation_history(test_case)
    
    # Handle system prompt override for relevant tests
    system_prompt = None
    if test_case["name"] == "system_prompt_override":
        system_prompt = "You are a helpful assistant that responds to user queries with creativity and detail."
    
    # Generate a response using the actual model
    response = generate_ai_text(
        conversation_history=conversation_history,
        positive_system_prompt=system_prompt
    )
    
    # Log response for debugging
    if response.get("success", False):
        logger.info(f"Test {test_id} response (excerpt): {response['response'][:100]}...")
    else:
        logger.warning(f"Test {test_id} error: {response.get('error', 'Unknown error')}")
    
    # Update conversation context for future tests
    if response.get("success", False):
        update_conversation_context(test_case, response)
    
    # Verify the response meets expectations
    assert verify_output(response, test_case['output']), \
        f"Test {test_id} failed: Response doesn't meet content generation expectations"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])