# -*- coding: utf-8 -*-
"""
Data models for the chat system, including message types, message structures, and session information.
"""

from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any

class MessageType(Enum):
    """Enumeration for the type of chat message (user, assistant, or system)."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

@dataclass
class ChatMessage:
    """Represents a single chat message with its content, type, sender, and metadata."""
    id: str
    chat_session_id: str
    message_type: MessageType
    content: str
    timestamp: datetime
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the chat message to a dictionary, serializing enum and datetime objects."""
        data = asdict(self)
        data['message_type'] = self.message_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Creates a ChatMessage instance from a dictionary, deserializing enum and datetime objects."""
        data['message_type'] = MessageType(data['message_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

@dataclass
class ChatSession:
    """Represents a chat session, including its ID, user, title, and activity status."""
    id: str
    user_id: str
    title: str
    created_at: datetime
    last_message_at: Optional[datetime] = None
    is_active: bool = True
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the chat session to a dictionary, serializing datetime objects."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        if self.last_message_at:
            data['last_message_at'] = self.last_message_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        """Creates a ChatSession instance from a dictionary, deserializing datetime objects."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('last_message_at'):
            data['last_message_at'] = datetime.fromisoformat(data['last_message_at'])
        return cls(**data) 