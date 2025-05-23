"""
Test script for command preprocessing functionality.

This script tests the command preprocessing system by simulating
various types of user commands and verifying they're processed correctly.
"""
import logging
import os
import sys
import time
from pprint import pprint

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the necessary modules
from services.command_processor import preprocess_message, CommandIntent
from services.config import get_config, load_config

def test_command_detection():
    """Test the command detection functionality with various messages."""
    print("\n=== Testing Command Detection ===\n")
    
    test_messages = [
        "Generate an image of a cat playing with a ball of yarn",
        "Can you search online for the latest news about quantum computing?",
        "My name is John and my email is john@example.com",
        "Remember that I prefer dark chocolate and dislike coffee",
        "Search my documents for information about the quarterly report",
        "What's the weather like today?",
        "Tell me about the history of Rome",
        "How do I make a chocolate cake?",
    ]
    
    for message in test_messages:
        print(f"Message: \"{message}\"")
        
        # Test with pattern matching only
        intent, params = preprocess_message(message, use_main_model=False)
        print(f"Pattern matching result: {intent.name}")
        
        # Test with main model
        intent, params = preprocess_message(message, use_main_model=True)
        print(f"Full processing result: {intent.name}")
        
        if params:
            print("Parameters:")
            for key, value in params.items():
                if key == "original_message":
                    continue
                print(f"  {key}: {value[:50]}...")
        
        print("-" * 60)

def test_image_generation():
    """Test the image generation command."""
    from services.chat.chat_manager import _process_command_intent
    
    print("\n=== Testing Image Generation ===\n")
    
    # Create a test command
    intent = CommandIntent.GENERATE_IMAGE
    params = {
        "image_prompt": "A beautiful mountain landscape with a lake and sunset"
    }
    
    # Process the command
    response = _process_command_intent(intent, params, "test_user", "test_chat", "test_session")
    
    print("Response:")
    print(response)
    print("-" * 60)

def test_online_search():
    """Test the online search command."""
    from services.chat.chat_manager import _process_command_intent
    
    print("\n=== Testing Online Search ===\n")
    
    # Create a test command
    intent = CommandIntent.ONLINE_SEARCH
    params = {
        "search_query": "What is the capital of France?"
    }
    
    # Process the command
    response = _process_command_intent(intent, params, "test_user", "test_chat", "test_session")
    
    print("Response:")
    print(response)
    print("-" * 60)

def test_store_user_info():
    """Test the store user info command."""
    from services.chat.chat_manager import _process_command_intent
    
    print("\n=== Testing Store User Info ===\n")
    
    # Create a test command
    intent = CommandIntent.STORE_USER_INFO
    params = {
        "user_info": "My name is Alice Smith and I live in New York. My email is alice@example.com."
    }
    
    # Process the command
    response = _process_command_intent(intent, params, "test_user", "test_chat", "test_session")
    
    print("Response:")
    print(response)
    print("-" * 60)

def test_store_memory():
    """Test the store memory command."""
    from services.chat.chat_manager import _process_command_intent
    
    print("\n=== Testing Store Memory ===\n")
    
    # Create a test command
    intent = CommandIntent.STORE_MEMORY
    params = {
        "memory_content": "The office security code is 1234. Remember this for me."
    }
    
    # Process the command
    response = _process_command_intent(intent, params, "test_user", "test_chat", "test_session")
    
    print("Response:")
    print(response)
    print("-" * 60)

def test_local_search():
    """Test the local document search command."""
    from services.chat.chat_manager import _process_command_intent
    
    print("\n=== Testing Local Document Search ===\n")
    
    # Create a test command
    intent = CommandIntent.SEARCH_LOCAL
    params = {
        "local_query": "quarterly financial report"
    }
    
    # Process the command
    response = _process_command_intent(intent, params, "test_user", "test_chat", "test_session")
    
    print("Response:")
    print(response)
    print("-" * 60)

def test_configuration():
    """Test the configuration functionality."""
    print("\n=== Testing Configuration ===\n")
    
    from services.config import get_config, update_config
    
    # Get the current configuration
    config = get_config()
    
    print("Command preprocessing configuration:")
    pprint(config.get("command_preprocessing", {}))
    
    # Update a configuration value temporarily for testing
    original_value = config.get("command_preprocessing", {}).get("enabled", True)
    
    print(f"\nTemporarily changing 'enabled' from {original_value} to {not original_value}")
    update_config({"command_preprocessing": {"enabled": not original_value}})
    
    # Get the updated configuration
    updated_config = get_config()
    print("Updated configuration:")
    pprint(updated_config.get("command_preprocessing", {}))
    
    # Revert the change
    update_config({"command_preprocessing": {"enabled": original_value}})
    print(f"\nReverted 'enabled' back to {original_value}")
    
    print("-" * 60)

if __name__ == "__main__":
    print("=== Command Preprocessing System Test ===")
    
    # Load configuration
    config = load_config()
    
    # Run tests
    try:        # Simple tests first
        test_command_detection()
        test_configuration()
        
        # Feature-specific tests
        if config.get("command_preprocessing", {}).get("image_generation_enabled", True):
            test_image_generation()
        
        if config.get("command_preprocessing", {}).get("online_search_enabled", True):
            test_online_search()
        
        if config.get("command_preprocessing", {}).get("store_user_info_enabled", True):
            test_store_user_info()
        
        if config.get("command_preprocessing", {}).get("store_memory_enabled", True):
            test_store_memory()
        
        if config.get("command_preprocessing", {}).get("local_search_enabled", True):
            test_local_search()
        
        print("\nAll tests completed!")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        print(f"\nTest failed with error: {e}")
