"""
User management handlers for the Service Command Interface.

Handles user information management, preferences, and profile operations.
"""
import logging
from typing import Dict, Any, Optional

from ..command_system import Command, CommandResult, ExecutionContext
from services.user_memory import get_user_information, store_user_information
from services.database.users import get_user_id, find_user_by_id
from services.database.connection import get_db_session

logger = logging.getLogger(__name__)


class UserHandlers:
    """Handlers for user management operations."""
    
    @staticmethod
    def handle_get_user_info(command: Command, context: ExecutionContext) -> CommandResult:
        """
        Get user information and profile data.
        
        Args:
            command: Command object
            context: Execution context with user authentication
            
        Returns:
            CommandResult with user information
        """
        try:
            # Require authentication
            if not context.user_id:
                return CommandResult(
                    success=False,
                    message="Authentication required to get user information",
                    error_code="AUTHENTICATION_REQUIRED"
                )
            
            # Get basic user data from database
            user_info = {}
            
            with get_db_session() as session:
                user = find_user_by_id(int(context.user_id), session)
                if user:
                    user_info["username"] = user.name
                    user_info["user_id"] = user.user_id
                    user_info["created_date"] = str(user.name)  # Placeholder - assuming creation tracking exists
                else:
                    return CommandResult(
                        success=False,
                        message="User not found in database",
                        error_code="USER_NOT_FOUND"
                    )
            
            # Get stored personal information
            personal_info = get_user_information(context.user_id)
            if personal_info:
                user_info["personal_info"] = personal_info
            
            # Get optional specific info section
            section = command.parameters.get('section')
            if section:
                if section == 'basic':
                    result_data = {
                        "username": user_info.get("username"),
                        "user_id": user_info.get("user_id")
                    }
                elif section == 'personal':
                    result_data = user_info.get("personal_info", {})
                elif section in user_info.get("personal_info", {}):
                    result_data = {section: user_info["personal_info"][section]}
                else:
                    return CommandResult(
                        success=False,
                        message=f"User information section '{section}' not found",
                        error_code="SECTION_NOT_FOUND"
                    )
            else:
                result_data = user_info
            
            return CommandResult(
                success=True,
                message="User information retrieved successfully",
                data=result_data
            )
            
        except Exception as e:
            logger.error(f"Error getting user information: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to get user information: {str(e)}",
                error_code="USER_INFO_GET_FAILED"
            )
    
    @staticmethod
    def handle_update_user_info(command: Command, context: ExecutionContext) -> CommandResult:
        """
        Update user information and profile data.
        
        Args:
            command: Command with 'updates' parameter containing new information
            context: Execution context with user authentication
            
        Returns:
            CommandResult indicating success/failure
        """
        try:
            # Require authentication
            if not context.user_id:
                return CommandResult(
                    success=False,
                    message="Authentication required to update user information",
                    error_code="AUTHENTICATION_REQUIRED"
                )
            
            updates = command.parameters.get('updates')
            if not updates or not isinstance(updates, dict):
                return CommandResult(
                    success=False,
                    message="Missing or invalid 'updates' parameter",
                    error_code="INVALID_PARAMETERS"
                )
            
            # Get existing user information
            existing_info = get_user_information(context.user_id) or {}
            
            # Track changes
            changes = []
            for key, value in updates.items():
                if key.startswith('_'):
                    # Skip internal metadata fields
                    continue
                
                old_value = existing_info.get(key)
                if old_value != value:
                    changes.append(f"{key}: {old_value} -> {value}")
                    existing_info[key] = value
            
            # Store updated information
            success = store_user_information(context.user_id, existing_info)
            
            if not success:
                return CommandResult(
                    success=False,
                    message="Failed to store updated user information",
                    error_code="USER_INFO_STORE_FAILED"
                )
            
            return CommandResult(
                success=True,
                message=f"User information updated successfully. Changes: {', '.join(changes) if changes else 'No changes'}",
                data={
                    "changes": changes,
                    "updated_info": existing_info
                }
            )
            
        except Exception as e:
            logger.error(f"Error updating user information: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to update user information: {str(e)}",
                error_code="USER_INFO_UPDATE_FAILED"
            )
    
    @staticmethod
    def handle_get_user_preferences(command: Command, context: ExecutionContext) -> CommandResult:
        """
        Get user preferences and settings.
        
        Args:
            command: Command object with optional 'category' parameter
            context: Execution context with user authentication
            
        Returns:
            CommandResult with user preferences
        """
        try:
            # Require authentication
            if not context.user_id:
                return CommandResult(
                    success=False,
                    message="Authentication required to get user preferences",
                    error_code="AUTHENTICATION_REQUIRED"
                )
            
            # Get user information which includes preferences
            user_info = get_user_information(context.user_id) or {}
            preferences = user_info.get("preferences", {})
            
            # Apply default preferences if none exist
            if not preferences:
                preferences = {
                    "interface": {
                        "theme": "default",
                        "language": "en",
                        "timezone": "UTC"
                    },
                    "chat": {
                        "show_timestamps": True,
                        "auto_scroll": True,
                        "message_limit": 100
                    },
                    "notifications": {
                        "enabled": True,
                        "sound_enabled": False,
                        "show_progress": True
                    },
                    "privacy": {
                        "store_chat_history": True,
                        "store_user_info": True,
                        "share_usage_data": False
                    }
                }
            
            # Get specific category if requested
            category = command.parameters.get('category')
            if category:
                if category in preferences:
                    result_data = {category: preferences[category]}
                    message = f"Retrieved preferences for category: {category}"
                else:
                    return CommandResult(
                        success=False,
                        message=f"Preference category '{category}' not found",
                        error_code="CATEGORY_NOT_FOUND"
                    )
            else:
                result_data = preferences
                message = "Retrieved all user preferences"
            
            return CommandResult(
                success=True,
                message=message,
                data=result_data
            )
            
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to get user preferences: {str(e)}",
                error_code="USER_PREFERENCES_GET_FAILED"
            )
    
    @staticmethod
    def handle_update_user_preferences(command: Command, context: ExecutionContext) -> CommandResult:
        """
        Update user preferences and settings.
        
        Args:
            command: Command with 'preferences' parameter containing new settings
            context: Execution context with user authentication
            
        Returns:
            CommandResult indicating success/failure
        """
        try:
            # Require authentication
            if not context.user_id:
                return CommandResult(
                    success=False,
                    message="Authentication required to update user preferences",
                    error_code="AUTHENTICATION_REQUIRED"
                )
            
            preference_updates = command.parameters.get('preferences')
            if not preference_updates or not isinstance(preference_updates, dict):
                return CommandResult(
                    success=False,
                    message="Missing or invalid 'preferences' parameter",
                    error_code="INVALID_PARAMETERS"
                )
            
            # Get existing user information
            user_info = get_user_information(context.user_id) or {}
            existing_preferences = user_info.get("preferences", {})
            
            # Validate preference updates
            valid_categories = ["interface", "chat", "notifications", "privacy"]
            changes = []
            
            for category, settings in preference_updates.items():
                if category not in valid_categories:
                    logger.warning(f"Unknown preference category: {category}")
                    continue
                
                if not isinstance(settings, dict):
                    continue
                
                # Initialize category if it doesn't exist
                if category not in existing_preferences:
                    existing_preferences[category] = {}
                
                # Update settings in this category
                for key, value in settings.items():
                    old_value = existing_preferences[category].get(key)
                    if old_value != value:
                        changes.append(f"{category}.{key}: {old_value} -> {value}")
                        existing_preferences[category][key] = value
            
            # Update user information with new preferences
            user_info["preferences"] = existing_preferences
            
            # Store updated information
            success = store_user_information(context.user_id, user_info)
            
            if not success:
                return CommandResult(
                    success=False,
                    message="Failed to store updated preferences",
                    error_code="USER_PREFERENCES_STORE_FAILED"
                )
            
            return CommandResult(
                success=True,
                message=f"User preferences updated successfully. Changes: {', '.join(changes) if changes else 'No changes'}",
                data={
                    "changes": changes,
                    "updated_preferences": existing_preferences
                }
            )
            
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to update user preferences: {str(e)}",
                error_code="USER_PREFERENCES_UPDATE_FAILED"
            )
