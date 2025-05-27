import os
import sys
import pytest
import logging
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from test_utils import clear_conversation_contexts
from Frai.back.ai.chat import (
    initialize_chat_system,
    get_chat_ai_instance,
    generate_ai_text
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def setup_chat_ai():
    """
    Initialize the ChatAI system once for all tests.
    This fixture is shared across all test modules.
    """
    # Clear any existing conversation contexts
    clear_conversation_contexts()
    
    # Initialize the chat system with the real model
    logger.info("Initializing ChatAI system for tests...")
    success = initialize_chat_system()
    
    if not success:
        pytest.fail("Failed to initialize ChatAI system")
        return None
    
    # Get the chat AI instance
    chat_ai = get_chat_ai_instance()
    logger.info(f"ChatAI initialized successfully with model: {chat_ai.model_name}")
    
    yield chat_ai
    
    logger.info("Test session complete. ChatAI instance remains loaded.")


@pytest.fixture
def chat_response():
    """
    Fixture to generate responses from the chat AI.
    Provides a function that can be called with conversation history.
    """
    def _generate_response(conversation_history, system_prompt=None):
        """
        Generate a chat response for the given conversation history.
        
        Args:
            conversation_history: List of message dictionaries
            system_prompt: Optional system prompt override
            
        Returns:
            Response dictionary
        """
        response = generate_ai_text(
            conversation_history=conversation_history,
            positive_system_prompt=system_prompt
        )
        return response
    
    return _generate_response