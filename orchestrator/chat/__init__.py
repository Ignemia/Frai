# -*- coding: utf-8 -*-
"""
Chat Orchestration Package.

This package provides the core functionalities for managing chat sessions and messages.
It includes data models for chat messages and sessions, and the ChatOrchestrator
class responsabile for the main chat logic.

Public API:
  - MessageType: Enum for message types (USER, ASSISTANT, SYSTEM).
  - ChatMessage: Dataclass representing a single chat message.
  - ChatSession: Dataclass representing a chat session.
  - ChatOrchestrator: Main class for orchestrating chat operations.
  - get_chat_orchestrator(): Function to get a singleton instance of ChatOrchestrator.
  - Convenience functions: create_session, send_message, add_assistant_response, get_conversation_history.
"""

import logging
from typing import List, Optional

# Import core components from submodules to define the package's public API
from .models import MessageType, ChatMessage, ChatSession
from .orchestrator import ChatOrchestrator

logger = logging.getLogger(__name__)

# --- Singleton Instance Management for ChatOrchestrator ---
_chat_orchestrator_instance: Optional[ChatOrchestrator] = None

def get_chat_orchestrator() -> ChatOrchestrator:
    """
    Provides a singleton instance of the ChatOrchestrator.
    
    This ensures that all parts of the application share the same chat management state.
    
    Returns:
        The singleton ChatOrchestrator instance.
    """
    global _chat_orchestrator_instance
    if _chat_orchestrator_instance is None:
        logger.info("Creating new ChatOrchestrator singleton instance.")
        _chat_orchestrator_instance = ChatOrchestrator()
    return _chat_orchestrator_instance

# --- Convenience Functions (Public API) ---

def initiate_chat_orchestrator() -> bool:
    """
    Initializes the global chat orchestrator instance.
    Call this at application startup.

    Returns:
        True if initialization was successful or instance already exists, False on error.
    """
    try:
        logger.info("Initiating chat orchestrator service...")
        get_chat_orchestrator() # This will create if it doesn't exist
        logger.info("Chat orchestrator service initiated successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to initiate chat orchestrator: {e}", exc_info=True)
        return False

def create_session(user_id: str, title: Optional[str] = None) -> ChatSession:
    """Creates a new chat session for the given user."""
    return get_chat_orchestrator().create_chat_session(user_id, title)

def send_message(session_id: str, content: str, user_id: Optional[str] = None) -> ChatMessage:
    """Sends a user message to the specified chat session."""
    return get_chat_orchestrator().add_message(
        session_id, MessageType.USER, content, user_id
    )

def add_assistant_response(session_id: str, content: str) -> ChatMessage:
    """Adds an assistant's response to the specified chat session."""
    return get_chat_orchestrator().add_message(
        session_id, MessageType.ASSISTANT, content
    )

def get_conversation_history(session_id: str, limit: Optional[int] = None, offset: int = 0) -> List[ChatMessage]:
    """Retrieves the conversation history for a given session, with optional pagination."""
    return get_chat_orchestrator().get_session_messages(session_id, limit=limit, offset=offset)

# Define what gets imported with "from orchestrator.chat import *"
# It's generally better to encourage explicit imports, but __all__ can be useful.
__all__ = [
    'MessageType',
    'ChatMessage',
    'ChatSession',
    'ChatOrchestrator',
    'get_chat_orchestrator',
    'initiate_chat_orchestrator',
    'create_session',
    'send_message',
    'add_assistant_response',
    'get_conversation_history'
]