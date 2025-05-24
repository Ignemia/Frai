"""
Main Service Command Interface - The primary entry point for all system operations.

This is the central abstraction layer that provides a unified interface
for all user interactions across different frontend interfaces.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Union
from contextlib import contextmanager

from .command_system import Command, CommandResult, CommandType, ExecutionContext, ResultType
from .command_router import CommandRouter
from .response_formatter import ResponseFormatter

logger = logging.getLogger(__name__)


class ServiceCommandInterface:
    """
    Main Service Command Interface - Single point of contact for all operations.
    
    This class provides a unified interface for handling all user operations
    across different interfaces (CLI, API, Web UI, etc.). It manages:
    - Command routing and execution
    - User context and session management
    - Response formatting
    - Error handling and logging
    - Progress tracking
    """
    
    def __init__(self):
        """Initialize the Service Command Interface."""
        self.router = CommandRouter()
        self.formatter = ResponseFormatter()
        self._active_sessions: Dict[str, ExecutionContext] = {}
        self._command_history: List[Command] = []
        
        logger.info("Service Command Interface initialized")
    
    def execute_command(self, 
                       command_type: CommandType, 
                       parameters: Dict[str, Any] = None,
                       context: ExecutionContext = None,
                       **kwargs) -> CommandResult:
        """
        Execute a command through the unified interface.
        
        Args:
            command_type: Type of command to execute
            parameters: Command parameters
            context: Execution context (user, session, etc.)
            **kwargs: Additional command options
            
        Returns:
            CommandResult: Result of the command execution
        """
        start_time = time.time()
        
        # Create command object
        command = Command(
            command_type=command_type,
            parameters=parameters or {},
            context=context or ExecutionContext(),
            **kwargs
        )
        
        # Add to command history
        self._command_history.append(command)
        
        # Update session if context provided
        if context and context.session_token:
            self._active_sessions[context.session_token] = context
        
        logger.info(f"Executing command: {command_type.value} (ID: {command.command_id})")
        
        try:
            # Route and execute command
            result = self.router.route_command(command)
            
            # Calculate execution time
            execution_time = int((time.time() - start_time) * 1000)
            result.execution_time_ms = execution_time
            
            # Log result
            if result.success:
                logger.info(f"Command {command.command_id} completed successfully in {execution_time}ms")
            else:
                logger.warning(f"Command {command.command_id} failed: {result.error_code} - {result.message}")
            
            return result
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger.error(f"Command {command.command_id} failed with exception: {e}", exc_info=True)
            
            return CommandResult.error(
                command_id=command.command_id,
                error_code="EXECUTION_EXCEPTION",
                error_details=str(e),
                message="Command execution failed with exception"
            )
    
    @contextmanager
    def user_session(self, session_token: str, user_id: str = None, username: str = None):
        """
        Context manager for handling user sessions.
        
        Usage:
            with sci.user_session("token123", "user456") as context:
                result = sci.execute_command(CommandType.CHAT_CREATE, context=context)
        """
        context = ExecutionContext(
            session_token=session_token,
            user_id=user_id,
            username=username,
            is_authenticated=True
        )
        
        # Load existing session data if available
        if session_token in self._active_sessions:
            existing_context = self._active_sessions[session_token]
            context.current_chat_id = existing_context.current_chat_id
            context.app_state = existing_context.app_state
            context.metadata.update(existing_context.metadata)
        
        try:
            yield context
        finally:
            # Update session
            self._active_sessions[session_token] = context
    
    def get_session_context(self, session_token: str) -> Optional[ExecutionContext]:
        """Get the execution context for a session."""
        return self._active_sessions.get(session_token)
    
    def format_response(self, result: CommandResult, format_type: str = "json") -> Any:
        """
        Format a command result for a specific interface.
        
        Args:
            result: Command result to format
            format_type: Format type ('json', 'cli', 'api', etc.)
            
        Returns:
            Formatted response appropriate for the interface
        """
        return self.formatter.format_response(result, format_type)
    
    # Convenience methods for common operations
    
    def login(self, username: str, password: str, interface_type: str = "unknown") -> CommandResult:
        """Convenience method for user login."""
        context = ExecutionContext(interface_type=interface_type)
        return self.execute_command(
            CommandType.LOGIN,
            {"username": username, "password": password},
            context=context
        )
    
    def register(self, username: str, password: str, interface_type: str = "unknown") -> CommandResult:
        """Convenience method for user registration."""
        context = ExecutionContext(interface_type=interface_type)
        return self.execute_command(
            CommandType.REGISTER,
            {"username": username, "password": password},
            context=context
        )
    
    def create_chat(self, session_token: str, chat_name: str = None) -> CommandResult:
        """Convenience method for creating a chat."""
        context = self.get_session_context(session_token)
        if not context:
            return CommandResult.error(
                command_id="no_session",
                error_code="NO_SESSION",
                message="Session not found"
            )
        
        return self.execute_command(
            CommandType.CHAT_CREATE,
            {"chat_name": chat_name},
            context=context
        )
    
    def send_chat_message(self, session_token: str, chat_id: str, message: str) -> CommandResult:
        """Convenience method for sending a chat message."""
        context = self.get_session_context(session_token)
        if not context:
            return CommandResult.error(
                command_id="no_session",
                error_code="NO_SESSION", 
                message="Session not found"
            )
        
        return self.execute_command(
            CommandType.CHAT_SEND_MESSAGE,
            {"chat_id": chat_id, "message": message},
            context=context
        )
    
    def generate_image(self, session_token: str, prompt: str, **image_params) -> CommandResult:
        """Convenience method for image generation."""
        context = self.get_session_context(session_token)
        if not context:
            return CommandResult.error(
                command_id="no_session",
                error_code="NO_SESSION",
                message="Session not found"
            )
        
        parameters = {"prompt": prompt}
        parameters.update(image_params)
        
        return self.execute_command(
            CommandType.IMAGE_GENERATE,
            parameters,
            context=context
        )
    
    def get_system_status(self) -> CommandResult:
        """Convenience method for getting system status."""
        return self.execute_command(CommandType.SYSTEM_STATUS)
    
    # Session and state management
    
    def update_session_state(self, session_token: str, **state_updates):
        """Update session state information."""
        if session_token in self._active_sessions:
            context = self._active_sessions[session_token]
            for key, value in state_updates.items():
                if hasattr(context, key):
                    setattr(context, key, value)
                else:
                    context.metadata[key] = value
    
    def cleanup_session(self, session_token: str):
        """Clean up a user session."""
        if session_token in self._active_sessions:
            del self._active_sessions[session_token]
            logger.info(f"Cleaned up session: {session_token}")
    
    def get_command_history(self, session_token: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get command execution history."""
        if session_token:
            # Filter by session
            commands = [
                cmd for cmd in self._command_history[-limit:]
                if cmd.context and cmd.context.session_token == session_token
            ]
        else:
            commands = self._command_history[-limit:]
        
        return [
            {
                "command_id": cmd.command_id,
                "command_type": cmd.command_type.value,
                "timestamp": cmd.context.timestamp.isoformat() if cmd.context else None,
                "interface_type": cmd.context.interface_type if cmd.context else "unknown"
            }
            for cmd in commands
        ]
    
    def get_available_commands(self) -> Dict[str, Dict[str, str]]:
        """Get list of all available commands."""
        return self.router.list_available_commands()


# Global instance for easy access
_sci_instance = None


def get_service_command_interface() -> ServiceCommandInterface:
    """
    Get the global Service Command Interface instance.
    
    Returns:
        ServiceCommandInterface: The global SCI instance
    """
    global _sci_instance
    if _sci_instance is None:
        _sci_instance = ServiceCommandInterface()
    return _sci_instance
