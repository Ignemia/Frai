"""
Command Router - Routes commands to appropriate handlers based on command type.

This module manages the mapping of command types to their execution handlers
and provides the routing logic for the Service Command Interface.
"""

import logging
from typing import Dict, Callable, Awaitable, Union
from .command_system import Command, CommandResult, CommandType

logger = logging.getLogger(__name__)


class CommandRouter:
    """Routes commands to their appropriate handlers."""
    
    def __init__(self):
        """Initialize the command router."""
        self._handlers: Dict[CommandType, Callable] = {}
        self._async_handlers: Dict[CommandType, Callable] = {}
        self._setup_default_handlers()
    
    def register_handler(self, command_type: CommandType, handler: Callable, is_async: bool = False):
        """
        Register a handler for a specific command type.
        
        Args:
            command_type: The type of command this handler processes
            handler: Function that takes (Command) and returns CommandResult
            is_async: Whether this is an async handler
        """
        if is_async:
            self._async_handlers[command_type] = handler
            logger.info(f"Registered async handler for {command_type.value}")
        else:
            self._handlers[command_type] = handler
            logger.info(f"Registered sync handler for {command_type.value}")
    
    def get_handler(self, command_type: CommandType) -> Union[Callable, None]:
        """Get the handler for a command type."""
        # Prefer async handlers if available
        if command_type in self._async_handlers:
            return self._async_handlers[command_type]
        return self._handlers.get(command_type)
    
    def has_handler(self, command_type: CommandType) -> bool:
        """Check if a handler exists for the given command type."""
        return (command_type in self._handlers or 
                command_type in self._async_handlers)
    
    def route_command(self, command: Command) -> CommandResult:
        """
        Route a command to its appropriate handler.
        
        Args:
            command: The command to route and execute
            
        Returns:
            CommandResult: The result of command execution
        """
        try:
            # Validate command first
            validation_errors = command.validate()
            if validation_errors:
                return CommandResult.error(
                    command_id=command.command_id,
                    error_code="VALIDATION_ERROR",
                    error_details="; ".join(validation_errors),
                    message="Command validation failed"
                )
            
            # Find handler
            handler = self.get_handler(command.command_type)
            if not handler:
                return CommandResult.error(
                    command_id=command.command_id,
                    error_code="NO_HANDLER",
                    error_details=f"No handler registered for {command.command_type.value}",
                    message="Command type not supported"
                )
            
            # Execute command
            logger.info(f"Routing command {command.command_type.value} (ID: {command.command_id})")
            
            # Check if handler is async
            if command.command_type in self._async_handlers:
                # For async handlers, we need special handling
                # This would typically involve creating an async task
                # For now, we'll return an async started result
                import asyncio
                if asyncio.iscoroutinefunction(handler):
                    # Would need to handle async execution properly
                    # For now, return async started
                    return CommandResult.async_started(
                        command_id=command.command_id,
                        async_token=f"async_{command.command_id}",
                        message="Async command execution started"
                    )
              # Execute sync handler
            result = handler(command)
            
            if not isinstance(result, CommandResult):
                logger.warning(f"Handler for {command.command_type.value} returned non-CommandResult: {type(result)}")
                return CommandResult.error(
                    command_id=command.command_id,
                    error_code="INVALID_HANDLER_RESULT",
                    error_details=f"Handler returned {type(result)} instead of CommandResult",
                    message="Internal handler error"
                )
            
            logger.info(f"Command {command.command_id} completed with result: {result.result_type.value}")
            return result
            
        except Exception as e:
            logger.error(f"Error routing command {command.command_id}: {e}", exc_info=True)
            return CommandResult.error(
                command_id=command.command_id,
                error_code="EXECUTION_ERROR",
                error_details=str(e),
                message="Command execution failed"
            )
    
    def _setup_default_handlers(self):
        """Setup default command handlers using existing service functions."""
        # Import handlers
        from .handlers.auth_handlers import AuthHandlers
        from .handlers.chat_handlers import ChatHandlers
        from .handlers.image_handlers import ImageHandlers
        from .handlers.config_handlers import ConfigHandlers
        from .handlers.system_handlers import SystemHandlers
        from .handlers.user_handlers import UserHandlers
        
        # Register authentication handlers
        auth_handlers = AuthHandlers()
        self.register_handler(CommandType.LOGIN, auth_handlers.handle_login)
        self.register_handler(CommandType.REGISTER, auth_handlers.handle_register)
        self.register_handler(CommandType.LOGOUT, auth_handlers.handle_logout)
        
        # Register chat handlers
        chat_handlers = ChatHandlers()
        self.register_handler(CommandType.CHAT_CREATE, chat_handlers.handle_create_chat)
        self.register_handler(CommandType.CHAT_OPEN, chat_handlers.handle_open_chat)
        self.register_handler(CommandType.CHAT_SEND_MESSAGE, chat_handlers.handle_send_message)
        self.register_handler(CommandType.CHAT_GET_HISTORY, chat_handlers.handle_get_history)
        self.register_handler(CommandType.CHAT_LIST, chat_handlers.handle_list_chats)
        self.register_handler(CommandType.CHAT_CLOSE, chat_handlers.handle_close_chat)
        self.register_handler(CommandType.CHAT_DELETE, chat_handlers.handle_delete_chat)        # Register image handlers
        image_handlers = ImageHandlers()
        self.register_handler(CommandType.IMAGE_GENERATE, image_handlers.handle_generate_image)
        self.register_handler(CommandType.IMAGE_GENERATE_I2I, image_handlers.handle_generate_image_i2i)
        self.register_handler(CommandType.IMAGE_TO_IMAGE, image_handlers.handle_image_to_image)
        self.register_handler(CommandType.IMAGE_GENERATE_WITH_FEEDBACK, image_handlers.handle_image_generate_with_feedback)
        self.register_handler(CommandType.IMAGE_GET_STYLES, image_handlers.handle_get_styles)
        self.register_handler(CommandType.IMAGE_VALIDATE_PROMPT, image_handlers.handle_validate_prompt)
        self.register_handler(CommandType.IMAGE_MODEL_STATUS, image_handlers.handle_model_status)
        self.register_handler(CommandType.IMAGE_MODEL_RELOAD, image_handlers.handle_model_reload)
          # Register config handlers
        config_handlers = ConfigHandlers()
        self.register_handler(CommandType.CONFIG_GET, config_handlers.handle_get_config)
        self.register_handler(CommandType.CONFIG_UPDATE, config_handlers.handle_update_config)
        self.register_handler(CommandType.CONFIG_RESET, config_handlers.handle_reset_config)
        self.register_handler(CommandType.CONFIG_SAVE, config_handlers.handle_save_config)
        self.register_handler(CommandType.CONFIG_RELOAD, config_handlers.handle_reload_config)
        self.register_handler(CommandType.CONFIG_VALIDATE, config_handlers.handle_validate_config)
        self.register_handler(CommandType.CONFIG_LIST_SECTIONS, config_handlers.handle_list_config_sections)
        
        # Register system handlers
        system_handlers = SystemHandlers()
        self.register_handler(CommandType.SYSTEM_STATUS, system_handlers.handle_system_status)
        self.register_handler(CommandType.SYSTEM_HEALTH, system_handlers.handle_health_check)
        self.register_handler(CommandType.SYSTEM_MEMORY_STATS, system_handlers.handle_memory_stats)
        self.register_handler(CommandType.SYSTEM_GPU_STATUS, system_handlers.handle_gpu_status)
        self.register_handler(CommandType.SYSTEM_PERFORMANCE_METRICS, system_handlers.handle_performance_metrics)
        self.register_handler(CommandType.SYSTEM_CLEANUP_MEMORY, system_handlers.handle_cleanup_memory)
        
        # Register user management handlers
        user_handlers = UserHandlers()
        self.register_handler(CommandType.USER_GET_INFO, user_handlers.handle_get_user_info)
        self.register_handler(CommandType.USER_UPDATE_INFO, user_handlers.handle_update_user_info)
        self.register_handler(CommandType.USER_GET_PREFERENCES, user_handlers.handle_get_user_preferences)
        self.register_handler(CommandType.USER_UPDATE_PREFERENCES, user_handlers.handle_update_user_preferences)
    
    def list_available_commands(self) -> Dict[str, Dict[str, str]]:
        """
        Get a list of all available commands and their descriptions.
        
        Returns:
            Dictionary mapping command types to their metadata
        """
        commands = {}
        
        for command_type in CommandType:
            commands[command_type.value] = {
                "name": command_type.value,
                "has_handler": self.has_handler(command_type),
                "is_async": command_type in self._async_handlers,
                "description": self._get_command_description(command_type)
            }
        
        return commands
    
    def _get_command_description(self, command_type: CommandType) -> str:
        """Get description for a command type."""
        descriptions = {
            CommandType.LOGIN: "Authenticate user with username and password",
            CommandType.REGISTER: "Register a new user account",
            CommandType.LOGOUT: "End user session",
            CommandType.CHAT_CREATE: "Create a new chat session",
            CommandType.CHAT_OPEN: "Open an existing chat session",
            CommandType.CHAT_SEND_MESSAGE: "Send a message in a chat",
            CommandType.CHAT_GET_HISTORY: "Get chat message history",
            CommandType.CHAT_LIST: "List all user chats",
            CommandType.CHAT_CLOSE: "Close current chat session",
            CommandType.CHAT_DELETE: "Delete a chat permanently",            CommandType.IMAGE_GENERATE: "Generate an image from text prompt",
            CommandType.IMAGE_GENERATE_I2I: "Generate image from input image and prompt",
            CommandType.IMAGE_GENERATE_WITH_FEEDBACK: "Generate image with feedback analysis",
            CommandType.IMAGE_GET_STYLES: "Get available image style presets",
            CommandType.IMAGE_VALIDATE_PROMPT: "Validate an image generation prompt",
            CommandType.IMAGE_MODEL_STATUS: "Get image generation model status",
            CommandType.IMAGE_MODEL_RELOAD: "Reload the image generation model",CommandType.CONFIG_GET: "Get application configuration",
            CommandType.CONFIG_UPDATE: "Update application configuration",
            CommandType.CONFIG_RESET: "Reset configuration to defaults",
            CommandType.CONFIG_SAVE: "Save current configuration to disk",
            CommandType.CONFIG_RELOAD: "Reload configuration from disk",
            CommandType.CONFIG_VALIDATE: "Validate current configuration",
            CommandType.CONFIG_LIST_SECTIONS: "List all configuration sections",
            CommandType.USER_GET_INFO: "Get user profile information",
            CommandType.USER_UPDATE_INFO: "Update user profile information",
            CommandType.USER_GET_PREFERENCES: "Get user preferences",
            CommandType.USER_UPDATE_PREFERENCES: "Update user preferences",
            CommandType.SYSTEM_STATUS: "Get system status information",
            CommandType.SYSTEM_HEALTH: "Get system health check",
            CommandType.SYSTEM_MEMORY_STATS: "Get detailed memory statistics",
            CommandType.SYSTEM_GPU_STATUS: "Get GPU status and utilization",
            CommandType.SYSTEM_PERFORMANCE_METRICS: "Get performance metrics",
            CommandType.SYSTEM_CLEANUP_MEMORY: "Clean up and free system memory",
        }
        return descriptions.get(command_type, "No description available")
