# -*- coding: utf-8 -*-
# This module initializes the chat orchestrator
# Chat orchestrator deals with chat actions but does not handle chat moderation (safety checks, etc.)

import logging
import uuid
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class MessageType(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

@dataclass
class ChatMessage:
    """Represents a single chat message."""
    id: str
    chat_session_id: str
    message_type: MessageType
    content: str
    timestamp: datetime
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        """Convert message to dictionary."""
        data = asdict(self)
        data['message_type'] = self.message_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ChatMessage':
        """Create message from dictionary."""
        data['message_type'] = MessageType(data['message_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

@dataclass
class ChatSession:
    """Represents a chat session."""
    id: str
    user_id: str
    title: str
    created_at: datetime
    last_message_at: Optional[datetime] = None
    is_active: bool = True
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        """Convert session to dictionary."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        if self.last_message_at:
            data['last_message_at'] = self.last_message_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ChatSession':
        """Create session from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('last_message_at'):
            data['last_message_at'] = datetime.fromisoformat(data['last_message_at'])
        return cls(**data)

class ChatOrchestrator:
    """
    Chat orchestrator manages chat sessions and message handling.
    """
    
    def __init__(self):
        # In-memory storage for now - this would be replaced with database integration
        self.sessions: Dict[str, ChatSession] = {}
        self.messages: Dict[str, List[ChatMessage]] = {}
        self.user_sessions: Dict[str, List[str]] = {}  # user_id -> list of session_ids
        
        logger.info("Chat orchestrator initialized")
    
    def create_chat_session(self, user_id: str, title: Optional[str] = None) -> ChatSession:
        """
        Create a new chat session.
        
        Args:
            user_id: ID of the user creating the session
            title: Optional title for the session
            
        Returns:
            Created ChatSession
        """
        session_id = str(uuid.uuid4())
        
        if not title:
            # Generate a default title based on timestamp
            title = f"Chat Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        session = ChatSession(
            id=session_id,
            user_id=user_id,
            title=title,
            created_at=datetime.now()
        )
        
        self.sessions[session_id] = session
        
        # Add to user's session list
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []
        self.user_sessions[user_id].append(session_id)
        
        # Initialize empty message list for session
        self.messages[session_id] = []
        
        logger.info(f"Created new chat session {session_id} for user {user_id}")
        return session
    
    def get_chat_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a chat session by ID."""
        return self.sessions.get(session_id)
    
    def get_user_sessions(self, user_id: str) -> List[ChatSession]:
        """Get all chat sessions for a user."""
        session_ids = self.user_sessions.get(user_id, [])
        return [self.sessions[sid] for sid in session_ids if sid in self.sessions]
    
    def add_message(self, chat_session_id: str, message_type: MessageType, 
                   content: str, user_id: Optional[str] = None, 
                   metadata: Optional[Dict[str, Any]] = None) -> ChatMessage:
        """
        Add a message to a chat session.
        
        Args:
            chat_session_id: ID of the chat session
            message_type: Type of message (user, assistant, system)
            content: Message content
            user_id: Optional user ID
            metadata: Optional metadata
            
        Returns:
            Created ChatMessage
        """
        if chat_session_id not in self.sessions:
            raise ValueError(f"Chat session {chat_session_id} does not exist")
        
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
        
        # Add to message storage
        if chat_session_id not in self.messages:
            self.messages[chat_session_id] = []
        self.messages[chat_session_id].append(message)
        
        # Update session's last message time
        session = self.sessions[chat_session_id]
        session.last_message_at = timestamp
        
        logger.info(f"Added {message_type.value} message to session {chat_session_id}")
        return message
    
    def get_session_messages(self, chat_session_id: str, 
                           limit: Optional[int] = None,
                           offset: int = 0) -> List[ChatMessage]:
        """
        Get messages from a chat session.
        
        Args:
            chat_session_id: ID of the chat session
            limit: Maximum number of messages to return
            offset: Number of messages to skip
            
        Returns:
            List of ChatMessage objects
        """
        if chat_session_id not in self.messages:
            return []
        
        messages = self.messages[chat_session_id]
        
        # Sort by timestamp (oldest first)
        messages.sort(key=lambda m: m.timestamp)
        
        # Apply offset and limit
        if offset > 0:
            messages = messages[offset:]
        
        if limit is not None:
            messages = messages[:limit]
        
        return messages
    
    def get_conversation_context(self, chat_session_id: str, 
                               max_messages: int = 20) -> str:
        """
        Get conversation context as a formatted string for AI processing.
        
        Args:
            chat_session_id: ID of the chat session
            max_messages: Maximum number of recent messages to include
            
        Returns:
            Formatted conversation context
        """
        messages = self.get_session_messages(chat_session_id)
        
        # Get the most recent messages
        if len(messages) > max_messages:
            messages = messages[-max_messages:]
        
        context_lines = []
        for message in messages:
            role = message.message_type.value.capitalize()
            content = message.content.strip()
            context_lines.append(f"{role}: {content}")
        
        return "\n".join(context_lines)
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a chat session and all its messages.
        
        Args:
            session_id: ID of the session to delete
            
        Returns:
            True if deleted successfully, False if session not found
        """
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        user_id = session.user_id
        
        # Remove from sessions
        del self.sessions[session_id]
        
        # Remove messages
        if session_id in self.messages:
            del self.messages[session_id]
        
        # Remove from user's session list
        if user_id in self.user_sessions:
            self.user_sessions[user_id] = [
                sid for sid in self.user_sessions[user_id] if sid != session_id
            ]
        
        logger.info(f"Deleted chat session {session_id}")
        return True
    
    def update_session_title(self, session_id: str, new_title: str) -> bool:
        """
        Update a session's title.
        
        Args:
            session_id: ID of the session
            new_title: New title for the session
            
        Returns:
            True if updated successfully, False if session not found
        """
        if session_id not in self.sessions:
            return False
        
        self.sessions[session_id].title = new_title
        logger.info(f"Updated title for session {session_id}")
        return True
    
    def get_session_stats(self, session_id: str) -> Optional[Dict]:
        """
        Get statistics for a chat session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            Dictionary with session statistics or None if session not found
        """
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        messages = self.messages.get(session_id, [])
        
        user_messages = [m for m in messages if m.message_type == MessageType.USER]
        assistant_messages = [m for m in messages if m.message_type == MessageType.ASSISTANT]
        
        total_chars = sum(len(m.content) for m in messages)
        
        return {
            "session_id": session_id,
            "title": session.title,
            "created_at": session.created_at.isoformat(),
            "last_message_at": session.last_message_at.isoformat() if session.last_message_at else None,
            "total_messages": len(messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "total_characters": total_chars,
            "is_active": session.is_active
        }
    
    def export_session(self, session_id: str) -> Optional[Dict]:
        """
        Export a complete chat session as a dictionary.
        
        Args:
            session_id: ID of the session to export
            
        Returns:
            Dictionary containing session and all messages, or None if not found
        """
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        messages = self.messages.get(session_id, [])
        
        return {
            "session": session.to_dict(),
            "messages": [message.to_dict() for message in messages]
        }


# Global orchestrator instance
_chat_orchestrator = None

def get_chat_orchestrator() -> ChatOrchestrator:
    """Get the global chat orchestrator instance."""
    global _chat_orchestrator
    if _chat_orchestrator is None:
        _chat_orchestrator = ChatOrchestrator()
    return _chat_orchestrator

def initiate_chat_orchestrator():
    """Initialize the chat orchestrator."""
    try:
        logger.info("Initializing chat orchestrator...")
        orchestrator = get_chat_orchestrator()
        logger.info("Chat orchestrator initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize chat orchestrator: {e}")
        return False

# Convenience functions for external use
def create_session(user_id: str, title: Optional[str] = None) -> ChatSession:
    """Create a new chat session."""
    return get_chat_orchestrator().create_chat_session(user_id, title)

def send_message(session_id: str, content: str, user_id: Optional[str] = None) -> ChatMessage:
    """Send a user message to a chat session."""
    return get_chat_orchestrator().add_message(
        session_id, MessageType.USER, content, user_id
    )

def add_assistant_response(session_id: str, content: str) -> ChatMessage:
    """Add an assistant response to a chat session."""
    return get_chat_orchestrator().add_message(
        session_id, MessageType.ASSISTANT, content
    )

def get_conversation_history(session_id: str, limit: Optional[int] = None) -> List[ChatMessage]:
    """Get conversation history for a session."""
    return get_chat_orchestrator().get_session_messages(session_id, limit)