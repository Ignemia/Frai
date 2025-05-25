"""
Database Backend Data Models

This module contains Pydantic models for all database backend modules.
These models ensure type safety and consistent data structures for
chat storage, user management, and security operations.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
import uuid


# ============================================================================
# Common Database Models
# ============================================================================

class DatabaseStatus(str, Enum):
    """Database connection status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class DatabaseConfig(BaseModel):
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str
    username: str
    password: str
    ssl_mode: str = "prefer"
    connection_timeout: int = 30
    max_connections: int = 20


class DatabaseStats(BaseModel):
    """Database statistics"""
    total_sessions: int = 0
    total_messages: int = 0
    total_users: int = 0
    is_connected: bool = False
    last_backup: Optional[datetime] = None
    storage_size_mb: Optional[float] = None


# ============================================================================
# Chat Database Models
# ============================================================================

class MessageType(str, Enum):
    """Types of chat messages"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    ERROR = "error"
    STATUS = "status"


class ChatMessageDB(BaseModel):
    """Chat message as stored in database"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    role: MessageType
    content: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tokens_used: Optional[int] = None
    processing_time_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role.value,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metadata": self.metadata,
            "tokens_used": self.tokens_used,
            "processing_time_ms": self.processing_time_ms
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessageDB":
        """Create from dictionary loaded from database"""
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            role=MessageType(data["role"]),
            content=data["content"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
            metadata=data.get("metadata", {}),
            tokens_used=data.get("tokens_used"),
            processing_time_ms=data.get("processing_time_ms")
        )


class SessionStatus(str, Enum):
    """Chat session status"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
    PAUSED = "paused"


class ChatSessionDB(BaseModel):
    """Chat session as stored in database"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    title: str = "New Chat"
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.now)
    last_message_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    message_count: int = 0
    total_tokens_used: int = 0
    model_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metadata": self.metadata,
            "message_count": self.message_count,
            "total_tokens_used": self.total_tokens_used,
            "model_name": self.model_name
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatSessionDB":
        """Create from dictionary loaded from database"""
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            title=data["title"],
            status=SessionStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_message_at=datetime.fromisoformat(data["last_message_at"]) if data.get("last_message_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
            metadata=data.get("metadata", {}),
            message_count=data.get("message_count", 0),
            total_tokens_used=data.get("total_tokens_used", 0),
            model_name=data.get("model_name")
        )


class MessageQueryParams(BaseModel):
    """Parameters for querying messages"""
    session_id: str
    limit: Optional[int] = None
    offset: int = 0
    role_filter: Optional[List[MessageType]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    search_content: Optional[str] = None


class SessionQueryParams(BaseModel):
    """Parameters for querying sessions"""
    user_id: str
    status_filter: Optional[List[SessionStatus]] = None
    limit: Optional[int] = None
    offset: int = 0
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    search_title: Optional[str] = None


class ChatDatabaseRequest(BaseModel):
    """Base request for chat database operations"""
    operation: str
    timestamp: datetime = Field(default_factory=datetime.now)


class SaveSessionRequest(ChatDatabaseRequest):
    """Request to save a chat session"""
    operation: str = "save_session"
    session_data: ChatSessionDB


class SaveMessageRequest(ChatDatabaseRequest):
    """Request to save a chat message"""
    operation: str = "save_message"
    message_data: ChatMessageDB


class LoadSessionRequest(ChatDatabaseRequest):
    """Request to load a chat session"""
    operation: str = "load_session"
    session_id: str


class LoadMessagesRequest(ChatDatabaseRequest):
    """Request to load session messages"""
    operation: str = "load_messages"
    query_params: MessageQueryParams


class DeleteSessionRequest(ChatDatabaseRequest):
    """Request to delete a chat session"""
    operation: str = "delete_session"
    session_id: str
    cascade_messages: bool = True


class ChatDatabaseResponse(BaseModel):
    """Base response from chat database operations"""
    success: bool
    timestamp: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None
    operation: Optional[str] = None


class SessionResponse(ChatDatabaseResponse):
    """Response containing session data"""
    session: Optional[ChatSessionDB] = None


class MessagesResponse(ChatDatabaseResponse):
    """Response containing message data"""
    messages: List[ChatMessageDB] = Field(default_factory=list)
    total_count: Optional[int] = None
    has_more: bool = False


class SessionListResponse(ChatDatabaseResponse):
    """Response containing list of sessions"""
    sessions: List[ChatSessionDB] = Field(default_factory=list)
    total_count: Optional[int] = None
    has_more: bool = False


class DatabaseStatsResponse(ChatDatabaseResponse):
    """Response containing database statistics"""
    stats: Optional[DatabaseStats] = None


# ============================================================================
# User Database Models
# ============================================================================

class UserRole(str, Enum):
    """User roles"""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"
    BANNED = "banned"


class UserStatus(str, Enum):
    """User account status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    DELETED = "deleted"
    PENDING_VERIFICATION = "pending_verification"


class UserPreferences(BaseModel):
    """User preferences and settings"""
    theme: str = "dark"
    language: str = "en"
    timezone: str = "UTC"
    notifications_enabled: bool = True
    voice_enabled: bool = False
    auto_save_sessions: bool = True
    default_model: Optional[str] = None
    max_tokens_per_request: int = 8192
    custom_system_prompt: Optional[str] = None


class UserDB(BaseModel):
    """User as stored in database"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str = Field(min_length=3, max_length=50)
    email: str = Field(regex=r'^[^@]+@[^@]+\.[^@]+$')
    password_hash: str
    role: UserRole = UserRole.USER
    status: UserStatus = UserStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    session_count: int = 0
    total_messages: int = 0
    total_tokens_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "password_hash": self.password_hash,
            "role": self.role.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "preferences": self.preferences.dict(),
            "metadata": self.metadata,
            "session_count": self.session_count,
            "total_messages": self.total_messages,
            "total_tokens_used": self.total_tokens_used
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserDB":
        """Create from dictionary loaded from database"""
        return cls(
            id=data["id"],
            username=data["username"],
            email=data["email"],
            password_hash=data["password_hash"],
            role=UserRole(data["role"]),
            status=UserStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_login=datetime.fromisoformat(data["last_login"]) if data.get("last_login") else None,
            last_activity=datetime.fromisoformat(data["last_activity"]) if data.get("last_activity") else None,
            preferences=UserPreferences(**data.get("preferences", {})),
            metadata=data.get("metadata", {}),
            session_count=data.get("session_count", 0),
            total_messages=data.get("total_messages", 0),
            total_tokens_used=data.get("total_tokens_used", 0)
        )


# ============================================================================
# Security Database Models
# ============================================================================

class SecurityEventType(str, Enum):
    """Types of security events"""
    LOGIN = "login"
    LOGOUT = "logout"
    FAILED_LOGIN = "failed_login"
    PASSWORD_CHANGE = "password_change"
    ACCOUNT_CREATED = "account_created"
    ACCOUNT_DELETED = "account_deleted"
    PERMISSION_DENIED = "permission_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"


class SecurityLevel(str, Enum):
    """Security event severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventDB(BaseModel):
    """Security event as stored in database"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    event_type: SecurityEventType
    severity: SecurityLevel = SecurityLevel.LOW
    description: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by
        }


class APIKeyDB(BaseModel):
    """API key as stored in database"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    key_hash: str
    name: str = "Default API Key"
    permissions: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True
    usage_count: int = 0
    rate_limit_per_hour: int = 1000
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "key_hash": self.key_hash,
            "name": self.name,
            "permissions": self.permissions,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
            "usage_count": self.usage_count,
            "rate_limit_per_hour": self.rate_limit_per_hour,
            "metadata": self.metadata
        }
