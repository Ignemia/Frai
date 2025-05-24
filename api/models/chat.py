"""
from typing import List
API models for chat operations.

This module defines Pydantic models for request validation and response formatting
for the chat API endpoints.
"""
from pydantic import BaseModel, Field
from typing import  Dict, Any, Optional
from datetime import datetime

class NewChatRequest(BaseModel):
    """Request model for creating a new chat."""
    chat_name: Optional[str] = None

class ChatMessage(BaseModel):
    """A chat message from user to AI."""
    message: str
    thoughts: Optional[str] = None

class ChatMessageResponse(BaseModel):
    """Response from the AI."""
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
class ChatHistoryMessage(BaseModel):
    """A message in the chat history."""
    role: str
    content: str
    timestamp: Optional[datetime] = None
    
class ChatMetadata(BaseModel):
    """Metadata for a chat."""
    chat_id: str
    chat_name: str
    created_at: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    message_count: Optional[int] = None
