"""
Utility functions for AI chat module tests.

This file provides utility functions for test implementations based on the test_set.csv
specification file. It includes functions for loading test cases, building conversation
history, and verifying outputs.
"""

import csv
import os
import logging
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to the test set CSV file
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), 'test_set.csv')

# Dictionary to store conversation contexts for sequential tests
CONVERSATION_CONTEXTS = {}

def load_test_cases() -> List[Dict[str, str]]:
    """
    Load test cases from the test_set.csv file.
    
    Returns:
        List of dictionaries representing the test cases
    """
    test_cases = []
    try:
        with open(TEST_SET_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_cases.append(row)
        logger.info(f"Loaded {len(test_cases)} test cases from {TEST_SET_PATH}")
        return test_cases
    except Exception as e:
        logger.error(f"Failed to load test cases: {e}")
        return []


def get_test_cases_by_category(category_names: List[str]) -> List[Dict[str, str]]:
    """
    Get test cases that match the specified category names.
    
    Args:
        category_names: List of test names to include
        
    Returns:
        List of matching test cases
    """
    all_test_cases = load_test_cases()
    filtered_cases = []
    
    for case in all_test_cases:
        if case['name'] in category_names:
            filtered_cases.append(case)
    
    logger.info(f"Selected {len(filtered_cases)} test cases for categories: {', '.join(category_names)}")
    return filtered_cases

def build_conversation_history(test_case: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Build conversation history for a test case, including any prior context
    from partial tests if specified.
    
    Args:
        test_case: The current test case from the CSV
        
    Returns:
        List of message dictionaries representing the conversation history
    """
    test_id = test_case['index']
    conversation = []
    
    # Process partials if specified
    if test_case['partials']:
        partial_indices = test_case['partials'].split(';')
        for index in partial_indices:
            partial_id = index.strip()
            if partial_id in CONVERSATION_CONTEXTS:
                conversation.extend(CONVERSATION_CONTEXTS[partial_id])
    
    # Add the current test message if content exists
    if test_case['content']:
        conversation.append({
            "role": "user",
            "content": test_case['content']
        })
    
    return conversation

def verify_output(response: Dict[str, Any], expected_output: str) -> bool:
    """
    Verify if the response matches the expected output criteria.
    
    Args:
        response: The response from the chat model
        expected_output: The expected output from the test case
        
    Returns:
        Boolean indicating if the response matches expectations
    """
    # Handle error case
    if expected_output.lower() == "error":
        return not response.get("success", False)
    
    # Handle "Any text" case
    elif expected_output.lower() == "any text":
        return (response.get("success", False) and 
                isinstance(response.get("response", ""), str) and 
                len(response.get("response", "")) > 0)
    
    # Handle specific text case
    else:
        return (response.get("success", False) and 
                expected_output.lower() in response.get("response", "").lower())

def update_conversation_context(test_case: Dict[str, str], response: Dict[str, Any]) -> None:
    """
    Update the conversation context for a test case after getting a response.
    This is used to maintain context for sequential tests.
    
    Args:
        test_case: The test case that was just executed
        response: The response received from the model
    """
    test_id = test_case['index']
    
    # Start with any existing conversation history
    conversation = build_conversation_history(test_case)
    
    # Add the assistant's response
    if response.get("success", False) and response.get("response"):
        conversation.append({
            "role": "assistant",
            "content": response["response"]
        })
    
    # Store the updated conversation
    CONVERSATION_CONTEXTS[test_id] = conversation
    logger.debug(f"Updated conversation context for test {test_id}: {len(conversation)} messages")

def run_test_case(generate_function, test_case: Dict[str, str]) -> Dict[str, Any]:
    """
    Run a single test case and update conversation context.
    
    Args:
        generate_function: Function to call for generating a response
        test_case: The test case to run
        
    Returns:
        The response from the chat model
    """
    test_id = test_case['index']
    logger.info(f"Running test {test_id}: {test_case['name']}")
    
    # Build conversation history for this test
    conversation_history = build_conversation_history(test_case)
    
    # Special case for empty input test
    if not conversation_history:
        return {"success": False, "error": "Empty input", "response": ""}
    
    # Generate response using the provided function
    response = generate_function(conversation_history=conversation_history)
    
    # If successful, update the conversation context
    if response.get("success", False):
        update_conversation_context(test_case, response)
    
    return response

def clear_conversation_contexts():
    """Clear all stored conversation contexts."""
    global CONVERSATION_CONTEXTS
    CONVERSATION_CONTEXTS = {}
    logger.info("Cleared all conversation contexts")