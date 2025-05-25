# -*- coding: utf-8 -*-
"""
Core implementation of the ChatOrchestrator class, responsible for managing 
chat sessions, messages, and related operations.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from .models import ChatMessage, ChatSession, MessageType

logger = logging.getLogger(__name__)

class ChatOrchestrator:
    """
    Chat orchestrator manages chat sessions and message handling.
    It provides functionalities to create, retrieve, and manage chat sessions,
    add messages, and export chat data.
    """
    
    def __init__(self):
        """Initializes the ChatOrchestrator with in-memory storage for sessions and messages."""
        self.sessions: Dict[str, ChatSession] = {}
        self.messages: Dict[str, List[ChatMessage]] = {}
        self.user_sessions: Dict[str, List[str]] = {}  # user_id -> list of session_ids
        
        logger.info("Chat orchestrator initialized")
    
    def create_chat_session(self, user_id: str, title: Optional[str] = None) -> ChatSession:
        """
        Creates a new chat session for a given user.
        
        If no title is provided, a default title is generated based on the creation timestamp.

        Args:
            user_id: The unique identifier for the user creating the session.
            title: An optional title for the chat session.
            
        Returns:
            The newly created ChatSession object.
        """
        session_id = str(uuid.uuid4())
        
        if not title:
            title = f"Chat Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        session = ChatSession(
            id=session_id,
            user_id=user_id,
            title=title,
            created_at=datetime.now()
        )
        
        self.sessions[session_id] = session
        
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []
        self.user_sessions[user_id].append(session_id)
        
        self.messages[session_id] = []
        
        logger.info(f"Created new chat session {session_id} for user {user_id} with title '{title}'")
        return session
    
    def get_chat_session(self, session_id: str) -> ChatSession:
        """
        Retrieves a specific chat session by its ID.

        Args:
            session_id: The unique identifier of the chat session to retrieve.
            
        Returns:
            The ChatSession object if found.
            
        Raises:
            KeyError: If no session is found for the given session_id.
        """
        if session_id not in self.sessions:
            raise KeyError(f"Session with ID '{session_id}' not found.")
        return self.sessions[session_id]
    
    def get_user_sessions(self, user_id: str) -> List[ChatSession]:
        """
        Retrieves all chat sessions associated with a specific user ID.

        Args:
            user_id: The unique identifier of the user.
            
        Returns:
            A list of ChatSession objects belonging to the user. Returns an empty list if the user has no sessions.
        """
        session_ids = self.user_sessions.get(user_id, [])
        return [self.sessions[sid] for sid in session_ids if sid in self.sessions]
    
    def add_message(self, chat_session_id: str, message_type: MessageType, 
                   content: str, user_id: Optional[str] = None, 
                   metadata: Optional[Dict[str, Any]] = None) -> ChatMessage:
        """
        Adds a new message to the specified chat session.
        
        Args:
            chat_session_id: The ID of the session to add the message to.
            message_type: The type of the message (USER, ASSISTANT, SYSTEM).
            content: The textual content of the message.
            user_id: The ID of the user sending the message (if applicable).
            metadata: Optional dictionary for additional message metadata.
            
        Returns:
            The newly created ChatMessage object.
            
        Raises:
            KeyError: If the specified chat_session_id does not exist.
        """
        if chat_session_id not in self.sessions: # Check against active sessions
            raise KeyError(f"Cannot add message to non-existent chat session '{chat_session_id}'")
        
        message_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        message = ChatMessage(
            id=message_id,
            chat_session_id=chat_session_id,
            message_type=message_type,
            content=content,
            timestamp=timestamp,
            user_id=user_id,
            metadata=metadata
        )
        
        self.messages[chat_session_id].append(message)
        self.sessions[chat_session_id].last_message_at = timestamp
        
        logger.info(f"Added {message_type.value} message (ID: {message_id}) to session {chat_session_id}")
        return message
    
    def get_session_messages(self, chat_session_id: str, 
                           limit: Optional[int] = None,
                           offset: int = 0) -> List[ChatMessage]:
        """
        Retrieves messages from a specified chat session, with optional pagination.

        Args:
            chat_session_id: The ID of the session from which to retrieve messages.
            limit: The maximum number of messages to return. None for no limit.
            offset: The number of messages to skip from the beginning of the list.
            
        Returns:
            A list of ChatMessage objects from the session, sorted by timestamp.
            
        Raises:
            KeyError: If the session_id does not exist or has no messages.
        """
        if chat_session_id not in self.messages:
            raise KeyError(f"Messages for session ID '{chat_session_id}' not found. Session may have been deleted or never existed.")
        
        session_messages = sorted(self.messages[chat_session_id], key=lambda m: m.timestamp)
        
        start_index = offset
        end_index = offset + limit if limit is not None else None
        
        return session_messages[start_index:end_index]
    
    def get_conversation_context(self, chat_session_id: str, 
                               max_messages: int = 20) -> str:
        """
        Generates a formatted string of the conversation context for a given session.
        This is typically used to provide context to an AI model.

        Args:
            chat_session_id: The ID of the chat session.
            max_messages: The maximum number of recent messages to include in the context.
            
        Returns:
            A newline-separated string of conversation history.
            
        Raises:
            KeyError: If the session_id does not exist.
        """
        # Ensure session exists before trying to get messages
        if chat_session_id not in self.sessions:
            raise KeyError(f"Session ID '{chat_session_id}' not found for context generation.")

        messages_list = self.get_session_messages(chat_session_id)
        
        if len(messages_list) > max_messages:
            messages_list = messages_list[-max_messages:]
        
        context_lines = []
        for message in messages_list:
            role = message.message_type.value.capitalize()
            content = message.content.strip()
            context_lines.append(f"{role}: {content}")
        
        return "\n".join(context_lines)
    
    def delete_session(self, session_id: str) -> bool:
        """
        Deletes a chat session and all associated messages.

        Args:
            session_id: The ID of the session to delete.
            
        Returns:
            True if the session was successfully deleted, False if the session was not found.
        """
        if session_id not in self.sessions:
            return False
        
        session = self.sessions.pop(session_id)
        user_id = session.user_id
        
        if session_id in self.messages:
            del self.messages[session_id]
        
        if user_id in self.user_sessions and session_id in self.user_sessions[user_id]:
            self.user_sessions[user_id].remove(session_id)
            if not self.user_sessions[user_id]:
                del self.user_sessions[user_id]
        
        logger.info(f"Deleted chat session {session_id} for user {user_id}")
        return True
    
    def update_session_title(self, session_id: str, new_title: str) -> bool:
        """
        Updates the title of an existing chat session.

        Args:
            session_id: The ID of the session to update.
            new_title: The new title for the session.
            
        Returns:
            True if the title was updated, False if the session was not found.
        """
        if session_id not in self.sessions:
            return False
        
        self.sessions[session_id].title = new_title
        logger.info(f"Updated title for session {session_id} to '{new_title}'")
        return True
    
    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves statistics for a given chat session.

        Args:
            session_id: The ID of the session.
            
        Returns:
            A dictionary containing statistics like message counts, character counts, etc., 
            or None if the session is not found.
        """
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        session_messages = self.messages.get(session_id, [])
        
        user_messages_count = sum(1 for m in session_messages if m.message_type == MessageType.USER)
        assistant_messages_count = sum(1 for m in session_messages if m.message_type == MessageType.ASSISTANT)
        total_chars = sum(len(m.content) for m in session_messages)
        
        return {
            "session_id": session_id,
            "title": session.title,
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "last_message_at": session.last_message_at.isoformat() if session.last_message_at else None,
            "total_messages": len(session_messages),
            "user_messages": user_messages_count,
            "assistant_messages": assistant_messages_count,
            "total_characters": total_chars,
            "is_active": session.is_active
        }
    
    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Exports a complete chat session, including session details and all messages, as a dictionary.

        Args:
            session_id: The ID of the session to export.
            
        Returns:
            A dictionary containing the session data and a list of messages, 
            or None if the session is not found.
        """
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        session_messages = self.messages.get(session_id, []) # Should be sorted if get_session_messages was used
        
        return {
            "session": session.to_dict(),
            "messages": [message.to_dict() for message in sorted(session_messages, key=lambda m: m.timestamp)]
        } 