"""
Chat Handlers for Service Command Interface

This module contains handlers for all chat-related operations including:
- Creating new chats
- Sending messages
- Retrieving chat history
- Managing chat sessions
- Listing user chats
"""
import logging
from typing import Optional, Dict, Any
from ..command_system import CommandResult, ExecutionContext
from services.database.chats import (
    create_chat,
    open_chat, 
    add_message_to_chat,
    get_chat_history,
    list_user_chats,
    update_chat_title,
    close_chat
)
from services.chat.chat_manager import (
    start_new_chat,
    process_user_message,
    end_chat_session
)

logger = logging.getLogger(__name__)


class ChatHandlers:
    """Handlers for chat-related commands."""
    
    @staticmethod
    def create_chat(command_id: str, params: Dict[str, Any], context: ExecutionContext) -> CommandResult:
        """
        Create a new chat session.
        
        Parameters:
            - chat_name (optional): Name for the new chat
            
        Returns:
            CommandResult with chat_id in data
        """
        try:
            chat_name = params.get('chat_name')
            
            # Use the high-level chat manager function
            chat_id = start_new_chat(context.session_token, chat_name)
            
            if not chat_id:
                return CommandResult.error(
                    command_id=command_id,
                    error_code="CHAT_CREATE_FAILED",
                    message="Failed to create new chat"
                )
            
            # Update context with current chat
            context.current_chat_id = chat_id
            
            return CommandResult.success(
                command_id=command_id,
                message=f"Created new chat: {chat_name or f'Chat {chat_id}'}",
                data={
                    "chat_id": chat_id,
                    "chat_name": chat_name or f"Chat {chat_id}"
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating chat: {e}", exc_info=True)
            return CommandResult.error(
                command_id=command_id,
                error_code="CHAT_CREATE_ERROR",
                message=f"Error creating chat: {str(e)}"
            )
    
    @staticmethod
    def open_chat(command_id: str, params: Dict[str, Any], context: ExecutionContext) -> CommandResult:
        """
        Open an existing chat session.
        
        Parameters:
            - chat_id: ID of the chat to open
            
        Returns:
            CommandResult with chat content in data
        """
        try:
            chat_id = params.get('chat_id')
            if not chat_id:
                return CommandResult.error(
                    command_id=command_id,
                    error_code="MISSING_CHAT_ID",
                    message="Chat ID is required"
                )
            
            # Try to open the chat
            chat_content = open_chat(chat_id, context.session_token)
            if not chat_content:
                return CommandResult.error(
                    command_id=command_id,
                    error_code="CHAT_NOT_FOUND",
                    message=f"Chat {chat_id} not found or access denied"
                )
            
            # Update context with current chat
            context.current_chat_id = chat_id
            
            # Get chat history for returning to client
            chat_history = get_chat_history(chat_id, context.session_token)
            
            return CommandResult.success(
                command_id=command_id,
                message=f"Opened chat {chat_id}",
                data={
                    "chat_id": chat_id,
                    "chat_content": chat_content,
                    "chat_history": chat_history
                }
            )
            
        except Exception as e:
            logger.error(f"Error opening chat: {e}", exc_info=True)
            return CommandResult.error(
                command_id=command_id,
                error_code="CHAT_OPEN_ERROR",
                message=f"Error opening chat: {str(e)}"
            )
    
    @staticmethod
    def send_message(command_id: str, params: Dict[str, Any], context: ExecutionContext) -> CommandResult:
        """
        Send a message in a chat.
        
        Parameters:
            - chat_id: ID of the chat (or uses context.current_chat_id)
            - message: Message content to send
            - thoughts (optional): Additional thoughts/metadata
            
        Returns:
            CommandResult with AI response in data
        """
        try:
            chat_id = params.get('chat_id') or context.current_chat_id
            if not chat_id:
                return CommandResult.error(
                    command_id=command_id,
                    error_code="NO_ACTIVE_CHAT",
                    message="No active chat. Create or open a chat first."
                )
            
            message = params.get('message')
            if not message or not message.strip():
                return CommandResult.error(
                    command_id=command_id,
                    error_code="EMPTY_MESSAGE",
                    message="Message cannot be empty"
                )
            
            thoughts = params.get('thoughts')
            
            # Process the message using the chat manager
            ai_response = process_user_message(
                chat_id, 
                context.session_token, 
                message, 
                thoughts=thoughts
            )
            
            if not ai_response:
                return CommandResult.error(
                    command_id=command_id,
                    error_code="MESSAGE_PROCESSING_FAILED",
                    message="Failed to process message"
                )
            
            # Update context
            context.current_chat_id = chat_id
            
            return CommandResult.success(
                command_id=command_id,
                message="Message sent successfully",
                data={
                    "chat_id": chat_id,
                    "user_message": message,
                    "ai_response": ai_response,
                    "thoughts": thoughts
                }
            )
            
        except Exception as e:
            logger.error(f"Error sending message: {e}", exc_info=True)
            return CommandResult.error(
                command_id=command_id,
                error_code="MESSAGE_SEND_ERROR",
                message=f"Error sending message: {str(e)}"
            )
    
    @staticmethod
    def get_chat_history(command_id: str, params: Dict[str, Any], context: ExecutionContext) -> CommandResult:
        """
        Get chat history for a specific chat.
        
        Parameters:
            - chat_id: ID of the chat (or uses context.current_chat_id)
            
        Returns:
            CommandResult with chat history in data
        """
        try:
            chat_id = params.get('chat_id') or context.current_chat_id
            if not chat_id:
                return CommandResult.error(
                    command_id=command_id,
                    error_code="NO_ACTIVE_CHAT",
                    message="No active chat specified"
                )
            
            # Get chat history
            chat_history = get_chat_history(chat_id, context.session_token)
            
            return CommandResult.success(
                command_id=command_id,
                message=f"Retrieved {len(chat_history)} messages",
                data={
                    "chat_id": chat_id,
                    "messages": chat_history
                }
            )
            
        except Exception as e:
            logger.error(f"Error getting chat history: {e}", exc_info=True)
            return CommandResult.error(
                command_id=command_id,
                error_code="CHAT_HISTORY_ERROR",
                message=f"Error retrieving chat history: {str(e)}"
            )
    
    @staticmethod
    def list_chats(command_id: str, params: Dict[str, Any], context: ExecutionContext) -> CommandResult:
        """
        List all chats for the user.
        
        Returns:
            CommandResult with chat list in data
        """
        try:
            # Get user's chats
            chats_data = list_user_chats(context.session_token)
            if chats_data is None:
                return CommandResult.error(
                    command_id=command_id,
                    error_code="CHAT_LIST_FAILED",
                    message="Failed to retrieve chat list"
                )
            
            # Format chat data
            chats = []
            for chat_id, chat_name, last_modified in chats_data:
                chats.append({
                    "chat_id": chat_id,
                    "chat_name": chat_name,
                    "last_modified": last_modified.isoformat() if last_modified else None
                })
            
            return CommandResult.success(
                command_id=command_id,
                message=f"Retrieved {len(chats)} chats",
                data={
                    "chats": chats
                }
            )
            
        except Exception as e:
            logger.error(f"Error listing chats: {e}", exc_info=True)
            return CommandResult.error(
                command_id=command_id,
                error_code="CHAT_LIST_ERROR",
                message=f"Error listing chats: {str(e)}"
            )
    
    @staticmethod
    def update_chat_title(command_id: str, params: Dict[str, Any], context: ExecutionContext) -> CommandResult:
        """
        Update the title of a chat.
        
        Parameters:
            - chat_id: ID of the chat (or uses context.current_chat_id)
            - new_title: New title for the chat
            
        Returns:
            CommandResult confirming title update
        """
        try:
            chat_id = params.get('chat_id') or context.current_chat_id
            if not chat_id:
                return CommandResult.error(
                    command_id=command_id,
                    error_code="NO_ACTIVE_CHAT",
                    message="No active chat specified"
                )
            
            new_title = params.get('new_title')
            if not new_title or not new_title.strip():
                return CommandResult.error(
                    command_id=command_id,
                    error_code="EMPTY_TITLE",
                    message="Title cannot be empty"
                )
            
            # Update the chat title
            success = update_chat_title(chat_id, new_title.strip(), context.session_token)
            if not success:
                return CommandResult.error(
                    command_id=command_id,
                    error_code="TITLE_UPDATE_FAILED",
                    message="Failed to update chat title"
                )
            
            return CommandResult.success(
                command_id=command_id,
                message=f"Updated chat title to: {new_title}",
                data={
                    "chat_id": chat_id,
                    "new_title": new_title
                }
            )
            
        except Exception as e:
            logger.error(f"Error updating chat title: {e}", exc_info=True)
            return CommandResult.error(
                command_id=command_id,
                error_code="TITLE_UPDATE_ERROR",
                message=f"Error updating chat title: {str(e)}"
            )
    
    @staticmethod
    def close_chat(command_id: str, params: Dict[str, Any], context: ExecutionContext) -> CommandResult:
        """
        Close a chat session.
        
        Parameters:
            - chat_id: ID of the chat (or uses context.current_chat_id)
            - username: Username for verification
            - password_hash: Password hash for verification
            
        Returns:
            CommandResult confirming chat closure
        """
        try:
            chat_id = params.get('chat_id') or context.current_chat_id
            if not chat_id:
                return CommandResult.error(
                    command_id=command_id,
                    error_code="NO_ACTIVE_CHAT",
                    message="No active chat specified"
                )
            
            username = params.get('username')
            password_hash = params.get('password_hash')
            
            if not username or not password_hash:
                return CommandResult.error(
                    command_id=command_id,
                    error_code="MISSING_CREDENTIALS",
                    message="Username and password hash required for chat closure"
                )
            
            # Close the chat using the high-level function
            success = end_chat_session(chat_id, context.session_token, username, password_hash)
            if not success:
                return CommandResult.error(
                    command_id=command_id,
                    error_code="CHAT_CLOSE_FAILED",
                    message="Failed to close chat"
                )
            
            # Clear current chat from context if it was closed
            if context.current_chat_id == chat_id:
                context.current_chat_id = None
            
            return CommandResult.success(
                command_id=command_id,
                message=f"Chat {chat_id} closed successfully",
                data={
                    "chat_id": chat_id
                }
            )
            
        except Exception as e:
            logger.error(f"Error closing chat: {e}", exc_info=True)
            return CommandResult.error(
                command_id=command_id,
                error_code="CHAT_CLOSE_ERROR",
                message=f"Error closing chat: {str(e)}"
            )
    
    @staticmethod
    def add_message_to_chat(command_id: str, params: Dict[str, Any], context: ExecutionContext) -> CommandResult:
        """
        Add a message directly to a chat (without AI processing).
        
        Parameters:
            - chat_id: ID of the chat (or uses context.current_chat_id)
            - role: Message role ('user', 'agent', 'system')
            - message: Message content
            - thoughts (optional): Additional thoughts/metadata
            - sources (optional): Source information
            
        Returns:
            CommandResult confirming message addition
        """
        try:
            chat_id = params.get('chat_id') or context.current_chat_id
            if not chat_id:
                return CommandResult.error(
                    command_id=command_id,
                    error_code="NO_ACTIVE_CHAT",
                    message="No active chat specified"
                )
            
            role = params.get('role')
            message = params.get('message')
            
            if not role or not message:
                return CommandResult.error(
                    command_id=command_id,
                    error_code="MISSING_PARAMETERS",
                    message="Role and message are required"
                )
            
            if role not in ['user', 'agent', 'system']:
                return CommandResult.error(
                    command_id=command_id,
                    error_code="INVALID_ROLE",
                    message="Role must be 'user', 'agent', or 'system'"
                )
            
            thoughts = params.get('thoughts')
            sources = params.get('sources')
            
            # Add the message to the chat
            success = add_message_to_chat(
                chat_id, 
                context.session_token, 
                role, 
                message, 
                thoughts=thoughts, 
                sources=sources
            )
            
            if not success:
                return CommandResult.error(
                    command_id=command_id,
                    error_code="MESSAGE_ADD_FAILED",
                    message="Failed to add message to chat"
                )
            
            return CommandResult.success(
                command_id=command_id,
                message=f"Added {role} message to chat",
                data={
                    "chat_id": chat_id,
                    "role": role,
                    "message": message,
                    "thoughts": thoughts,
                    "sources": sources
                }
            )
            
        except Exception as e:
            logger.error(f"Error adding message to chat: {e}", exc_info=True)
            return CommandResult.error(
                command_id=command_id,
                error_code="MESSAGE_ADD_ERROR",
                message=f"Error adding message: {str(e)}"
            )
