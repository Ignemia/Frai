"""
Core command system definitions.

This module defines the fundamental structures for the Service Command Interface,
including command objects, execution contexts, and result handling.
"""

from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid


class CommandType(Enum):
    """Enumeration of available command types."""
    # Authentication Commands
    LOGIN = "auth.login"
    REGISTER = "auth.register"
    LOGOUT = "auth.logout"
    
    # Chat Commands
    CHAT_CREATE = "chat.create"
    CHAT_OPEN = "chat.open"
    CHAT_SEND_MESSAGE = "chat.send_message"
    CHAT_GET_HISTORY = "chat.get_history"
    CHAT_LIST = "chat.list"
    CHAT_CLOSE = "chat.close"
    CHAT_DELETE = "chat.delete"    # Image Generation Commands
    IMAGE_GENERATE = "image.generate"
    IMAGE_GENERATE_I2I = "image.generate_i2i"
    IMAGE_TO_IMAGE = "image.to_image"
    IMAGE_GENERATE_WITH_FEEDBACK = "image.generate_with_feedback"
    IMAGE_GET_STYLES = "image.get_styles"
    IMAGE_VALIDATE_PROMPT = "image.validate_prompt"
    IMAGE_MODEL_STATUS = "image.model_status"
    IMAGE_MODEL_RELOAD = "image.model_reload"
      # Configuration Commands
    CONFIG_GET = "config.get"
    CONFIG_UPDATE = "config.update"
    CONFIG_RESET = "config.reset"
    CONFIG_SAVE = "config.save"
    CONFIG_RELOAD = "config.reload"
    CONFIG_VALIDATE = "config.validate"
    CONFIG_LIST_SECTIONS = "config.list_sections"
    
    # User Management Commands
    USER_GET_INFO = "user.get_info"
    USER_UPDATE_INFO = "user.update_info"
    USER_GET_PREFERENCES = "user.get_preferences"
    USER_UPDATE_PREFERENCES = "user.update_preferences"
    
    # System Commands
    SYSTEM_STATUS = "system.status"
    SYSTEM_HEALTH = "system.health"
    SYSTEM_MEMORY_STATS = "system.memory_stats"
    SYSTEM_GPU_STATUS = "system.gpu_status"
    SYSTEM_PERFORMANCE_METRICS = "system.performance_metrics"
    SYSTEM_CLEANUP_MEMORY = "system.cleanup_memory"


class ResultType(Enum):
    """Types of command results."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    ASYNC = "async"


@dataclass
class ExecutionContext:
    """Context information for command execution."""
    
    # User and session information
    user_id: Optional[str] = None
    session_token: Optional[str] = None
    username: Optional[str] = None
    
    # Request metadata
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    interface_type: str = "unknown"  # 'cli', 'api', 'web', etc.
    
    # Security and permissions
    permissions: List[str] = field(default_factory=list)
    is_authenticated: bool = False
    
    # Session state
    current_chat_id: Optional[str] = None
    app_state: Optional[str] = None
    
    # Additional context data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and setup context after initialization."""
        if self.user_id and self.session_token and not self.is_authenticated:
            self.is_authenticated = True


@dataclass 
class Command:
    """Represents a user command to be executed by the system."""
    
    # Core command information
    command_type: CommandType
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Execution context
    context: Optional[ExecutionContext] = None
    
    # Command metadata
    command_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 0  # Higher numbers = higher priority
    timeout_seconds: Optional[int] = None
    
    # Progress tracking
    supports_progress: bool = False
    progress_callback: Optional[callable] = None
    
    def __post_init__(self):
        """Validate command after initialization."""
        if self.context is None:
            self.context = ExecutionContext()
    
    def validate(self) -> List[str]:
        """
        Validate the command and return list of validation errors.
        
        Returns:
            List of validation error messages. Empty list if valid.
        """
        errors = []
        
        # Check required parameters based on command type
        required_params = self._get_required_parameters()
        for param in required_params:
            if param not in self.parameters:
                errors.append(f"Missing required parameter: {param}")
        
        # Check authentication requirements
        if self._requires_authentication() and not self.context.is_authenticated:
            errors.append("Authentication required for this command")
        
        return errors
    
    def _get_required_parameters(self) -> List[str]:
        """Get list of required parameters for this command type."""
        param_requirements = {
            CommandType.LOGIN: ["username", "password"],
            CommandType.REGISTER: ["username", "password"],
            CommandType.CHAT_CREATE: [],  # chat_name is optional
            CommandType.CHAT_OPEN: ["chat_id"],
            CommandType.CHAT_SEND_MESSAGE: ["chat_id", "message"],
            CommandType.CHAT_GET_HISTORY: ["chat_id"],
            CommandType.IMAGE_GENERATE: ["prompt"],
            CommandType.IMAGE_GENERATE_I2I: ["prompt", "input_image"],
            CommandType.CONFIG_UPDATE: ["config_data"],
            CommandType.USER_UPDATE_INFO: ["user_data"],
        }
        return param_requirements.get(self.command_type, [])
    
    def _requires_authentication(self) -> bool:
        """Check if this command type requires authentication."""
        public_commands = {
            CommandType.LOGIN,
            CommandType.REGISTER,
            CommandType.SYSTEM_STATUS,
            CommandType.SYSTEM_HEALTH
        }
        return self.command_type not in public_commands


@dataclass
class CommandResult:
    """Result of a command execution."""
    
    # Core result information
    command_id: str
    result_type: ResultType
    success: bool
    
    # Result data
    data: Any = None
    message: str = ""
    
    # Error information
    error_code: Optional[str] = None
    error_details: Optional[str] = None
    
    # Execution metadata
    execution_time_ms: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Progress and async handling
    progress: Optional[float] = None  # 0.0 to 1.0
    async_token: Optional[str] = None  # For tracking async operations
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def success(cls, command_id: str, data: Any = None, message: str = "") -> 'CommandResult':
        """Create a success result."""
        return cls(
            command_id=command_id,
            result_type=ResultType.SUCCESS,
            success=True,
            data=data,
            message=message
        )
    
    @classmethod
    def error(cls, command_id: str, error_code: str, error_details: str = "", message: str = "") -> 'CommandResult':
        """Create an error result."""
        return cls(
            command_id=command_id,
            result_type=ResultType.ERROR,
            success=False,
            error_code=error_code,
            error_details=error_details,
            message=message
        )
    
    @classmethod
    def async_started(cls, command_id: str, async_token: str, message: str = "") -> 'CommandResult':
        """Create an async operation started result."""
        return cls(
            command_id=command_id,
            result_type=ResultType.ASYNC,
            success=True,
            async_token=async_token,
            message=message
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "command_id": self.command_id,
            "result_type": self.result_type.value,
            "success": self.success,
            "data": self.data,
            "message": self.message,
            "error_code": self.error_code,
            "error_details": self.error_details,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "progress": self.progress,
            "async_token": self.async_token,
            "metadata": self.metadata
        }
