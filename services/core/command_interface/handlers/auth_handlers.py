"""
Authentication command handlers.

This module contains handlers for authentication-related commands
such as login, register, and logout.
"""

import logging
from typing import Dict, Any
from ..command_system import Command, CommandResult

logger = logging.getLogger(__name__)


class AuthHandlers:
    """Handlers for authentication commands."""
    
    def handle_login(self, command: Command) -> CommandResult:
        """
        Handle user login command.
        
        Args:
            command: Command object with username and password
            
        Returns:
            CommandResult with session token on success
        """
        try:
            username = command.parameters.get("username")
            password = command.parameters.get("password")
            
            # Import the existing login logic
            from services.database.passwords import verify_credentials
            from services.database.users import user_exists, get_user_id
            from services.database.sessions import create_session
            from hashlib import sha256
            
            # Validate credentials
            if not user_exists(username):
                return CommandResult.error(
                    command_id=command.command_id,
                    error_code="USER_NOT_FOUND",
                    message="User does not exist"
                )
            
            # Hash password for verification
            password_hash = sha256(password.encode()).hexdigest()
            
            if not verify_credentials(username, password_hash):
                return CommandResult.error(
                    command_id=command.command_id,
                    error_code="INVALID_CREDENTIALS",
                    message="Invalid username or password"
                )
            
            # Get user ID and create session
            user_id = get_user_id(username)
            session_token = create_session(user_id)
            
            if not session_token:
                return CommandResult.error(
                    command_id=command.command_id,
                    error_code="SESSION_CREATION_FAILED",
                    message="Failed to create user session"
                )
            
            # Update command context
            command.context.user_id = str(user_id)
            command.context.username = username
            command.context.session_token = session_token
            command.context.is_authenticated = True
            
            return CommandResult.success(
                command_id=command.command_id,
                data={
                    "session_token": session_token,
                    "user_id": user_id,
                    "username": username
                },
                message=f"Login successful for user {username}"
            )
            
        except Exception as e:
            logger.error(f"Login failed for command {command.command_id}: {e}", exc_info=True)
            return CommandResult.error(
                command_id=command.command_id,
                error_code="LOGIN_ERROR",
                error_details=str(e),
                message="Login failed due to internal error"
            )
    
    def handle_register(self, command: Command) -> CommandResult:
        """
        Handle user registration command.
        
        Args:
            command: Command object with username and password
            
        Returns:
            CommandResult with session token on success
        """
        try:
            username = command.parameters.get("username")
            password = command.parameters.get("password")
            
            # Import the existing registration logic
            from services.database.users import user_exists, create_user
            from services.database.passwords import store_password
            from services.database.sessions import create_session
            from hashlib import sha256
            
            # Check if user already exists
            if user_exists(username):
                return CommandResult.error(
                    command_id=command.command_id,
                    error_code="USER_EXISTS",
                    message="Username already exists"
                )
            
            # Create user
            user_id = create_user(username)
            if not user_id:
                return CommandResult.error(
                    command_id=command.command_id,
                    error_code="USER_CREATION_FAILED",
                    message="Failed to create user account"
                )
            
            # Store password
            password_hash = sha256(password.encode()).hexdigest()
            if not store_password(user_id, password_hash):
                return CommandResult.error(
                    command_id=command.command_id,
                    error_code="PASSWORD_STORAGE_FAILED",
                    message="Failed to store user password"
                )
            
            # Create session
            session_token = create_session(user_id)
            if not session_token:
                return CommandResult.error(
                    command_id=command.command_id,
                    error_code="SESSION_CREATION_FAILED",
                    message="User created but failed to create session"
                )
            
            # Update command context
            command.context.user_id = str(user_id)
            command.context.username = username
            command.context.session_token = session_token
            command.context.is_authenticated = True
            
            return CommandResult.success(
                command_id=command.command_id,
                data={
                    "session_token": session_token,
                    "user_id": user_id,
                    "username": username
                },
                message=f"Registration successful for user {username}"
            )
            
        except Exception as e:
            logger.error(f"Registration failed for command {command.command_id}: {e}", exc_info=True)
            return CommandResult.error(
                command_id=command.command_id,
                error_code="REGISTRATION_ERROR",
                error_details=str(e),
                message="Registration failed due to internal error"
            )
    
    def handle_logout(self, command: Command) -> CommandResult:
        """
        Handle user logout command.
        
        Args:
            command: Command object with session context
            
        Returns:
            CommandResult confirming logout
        """
        try:
            session_token = command.context.session_token
            
            if not session_token:
                return CommandResult.error(
                    command_id=command.command_id,
                    error_code="NO_SESSION",
                    message="No active session to logout"
                )
            
            # Import logout logic
            from services.database.sessions import invalidate_session
            
            # Invalidate session
            success = invalidate_session(session_token)
            
            if not success:
                logger.warning(f"Failed to invalidate session {session_token}")
            
            # Clear context
            command.context.user_id = None
            command.context.username = None
            command.context.session_token = None
            command.context.is_authenticated = False
            command.context.current_chat_id = None
            
            return CommandResult.success(
                command_id=command.command_id,
                data={"logged_out": True},
                message="Logout successful"
            )
            
        except Exception as e:
            logger.error(f"Logout failed for command {command.command_id}: {e}", exc_info=True)
            return CommandResult.error(
                command_id=command.command_id,
                error_code="LOGOUT_ERROR",
                error_details=str(e),
                message="Logout failed due to internal error"
            )
