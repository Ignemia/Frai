"""
from typing import List
Database operations for chat management.

This module provides functions for creating, opening, updating, and
managing encrypted chat conversations in the database.
"""
import logging
from services.database.sessions import verify_session_token
from services.database.connection import get_db_session
from services.database.models import Chat
from services.database.chat_database import (
    fetch_chat_and_keys, 
    fetch_chat_encryption_data,
    store_chat_in_database
)
from services.database.chat_encryption import (
    prepare_encryption_data,
    decrypt_user_rsa_keys,
    decrypt_chat_with_keys
)
from services.database.chat_xml import (
    create_empty_chat_xml,
    format_and_append_message,
    count_user_messages_in_chat,
    parse_chat_xml_to_history
)
from services.database.chat_utils import (
    complete_list_user_chats as list_user_chats,
    complete_update_chat_title as update_chat_title,
    complete_save_chat_content as save_chat_content,
    save_updated_chat_content,
    complete_close_chat as close_chat
)

logger = logging.getLogger(__name__)

def create_chat(chat_name, session_token):
    """
    Create a new chat for the user with initial encryption.
    
    Creates an empty chat XML structure, encrypts it with user-specific keys,
    and stores the encrypted data in the database.
    
    Args:
        chat_name (str): Name to identify the chat
        session_token (str): Active session token for authentication
        
    Returns:
        str: The chat_id of the created chat if successful
        None: If creation fails for any reason
    """
    user_id = verify_session_token(session_token)
    if not user_id:
        logger.error("Invalid session token when creating chat")
        return None
    
    try:
        chat_xml = create_empty_chat_xml()
        _, encrypted_data, encrypted_rsa_keys = prepare_encryption_data(chat_xml, user_id)
        return store_chat_in_database(user_id, chat_name, encrypted_data, encrypted_rsa_keys)
    except Exception as e:
        logger.error(f"Error creating chat: {e}", exc_info=True)
        return None

def open_chat(chat_id, session_token):
    """
    Open and decrypt a chat.
    Returns the decrypted chat content if successful, None otherwise.
    """
    user_id = verify_session_token(session_token)
    if not user_id:
        logger.error("Invalid session token when opening chat")
        return None
    
    try:
        chat, chat_key_data, user_encrypted_rsa_keys = fetch_chat_and_keys(chat_id, user_id)
        if not chat:
            return None
        
        user_rsa_keys = decrypt_user_rsa_keys(user_id, user_encrypted_rsa_keys.encrypted_keys)
        
        decrypted_xml = decrypt_chat_with_keys(
            chat.contents,
            chat_key_data.encrypted_key,
            chat_key_data.iv,
            user_rsa_keys['private']
        )
        
        logger.info(f"Successfully decrypted chat {chat_id} for user {user_id}")
        return decrypted_xml
    except Exception as e:
        logger.error(f"Error opening chat: {e}", exc_info=True)
        return None

def add_message_to_chat(chat_id, session_token, role, message, thoughts=None, sources=None):
    """
    Add a message to the chat.
    Returns True if successful, False otherwise.
    """
    user_id = verify_session_token(session_token)
    if not user_id:
        logger.error("Invalid session token when adding message to chat")
        return False
    
    try:
        chat_data = fetch_chat_encryption_data(chat_id, user_id)
        if not chat_data:
            return False
            
        chat, decrypted_xml, encryption_keys = chat_data
        
        updated_xml = format_and_append_message(
            decrypted_xml, 
            role, 
            message, 
            thoughts, 
            sources
        )
        
        success = save_updated_chat_content(
            chat, 
            updated_xml, 
            encryption_keys
        )
        
        logger.info(f"Added {role} message to chat {chat_id}")
        return success
    except Exception as e:
        logger.error(f"Error adding message to chat: {e}", exc_info=True)
        return False

def list_user_chats(session_token):
    """
    List all chats belonging to the user.
    Returns a list of chat details (id, name, last_modified) if successful, None otherwise.
    """
    user_id = verify_session_token(session_token)
    if not user_id:
        logger.error("Invalid session token when listing user chats")
        return None
    
    try:
        with get_db_session() as session:
            # Query specific columns and order
            chats_data = session.query(Chat.chat_id, Chat.chat_name, Chat.last_modified)\
                                .filter(Chat.user_id == user_id)\
                                .order_by(Chat.last_modified.desc())\
                                .all()
            
            # chats_data will be a list of Row objects (like named tuples)
            logger.info(f"Retrieved {len(chats_data)} chats for user ID: {user_id}")
            return chats_data # Returns list of (chat_id, chat_name, last_modified) tuples
    except Exception as e:
        logger.error(f"Error listing user chats: {e}", exc_info=True)
        return None

def update_chat_title(chat_id, new_title, session_token):
    """
    Update the title of a chat.
    Returns True if successful, False otherwise.
    """
    user_id = verify_session_token(session_token)
    if not user_id:
        logger.error("Invalid session token when updating chat title")
        return False
    
    try:
        with get_db_session() as session:
            chat = session.query(Chat)\
                          .filter(Chat.chat_id == chat_id, Chat.user_id == user_id)\
                          .first()
            
            if not chat:
                logger.warning(f"Chat {chat_id} not found or doesn't belong to user {user_id} for title update.")
                return False
                
            chat.chat_name = new_title
            # chat.last_modified will be updated automatically by onupdate=datetime.datetime.now in model
            # If not using onupdate, then: chat.last_modified = datetime.datetime.now()
            
            # session.commit() is handled by context manager
            logger.info(f"Updated title for chat {chat_id} to '{new_title}'")
            return True
    except Exception as e:
        logger.error(f"Error updating chat title: {e}", exc_info=True)
        return False

def save_chat_content(chat_id, updated_content_xml, session_token): # updated_content_xml is plain XML
    """
    Save updated chat content to the database.
    Re-encrypts the content with the existing keys.
    Returns True if successful, False otherwise.
    """
    user_id = verify_session_token(session_token)
    if not user_id:
        logger.error("Invalid session token when saving chat content")
        return False
    
    try:
        chat_data = fetch_chat_encryption_data(chat_id, user_id)
        if not chat_data:
            return False
            
        chat, _, encryption_keys = chat_data
        
        success = save_updated_chat_content(
            chat, 
            updated_content_xml, 
            encryption_keys
        )
        
        logger.info(f"Updated content for chat {chat_id}")
        return success
    except Exception as e:
        logger.error(f"Error saving chat content: {e}", exc_info=True)
        return False

def get_chat_history(chat_id, session_token):
    """
    Get the chat history for a given chat in a format ready for the LLM.
    
    This function:
    1. Verifies the user session
    2. Retrieves and decrypts the chat XML
    3. Parses the XML into a list of message dictionaries with 'role' and 'content' keys
       that can be directly passed to the LLM interface
    
    Args:
        chat_id (str): ID of the chat to retrieve history for
        session_token (str): User's session token for authentication
        
    Returns:
        list[dict]: List of message dictionaries in the format:
            [{'role': 'user'|'assistant', 'content': 'message text'}, ...]
        Empty list if retrieval fails or chat has no messages
    """
    user_id = verify_session_token(session_token)
    if not user_id:
        logger.error("Invalid session token when getting chat history")
        return []
    
    try:
        # Get the decrypted chat XML
        chat_xml = open_chat(chat_id, session_token)
        if not chat_xml:
            logger.warning(f"Could not open chat {chat_id} to get history")
            return []
        
        # Parse XML to chat history format
        chat_history = parse_chat_xml_to_history(chat_xml)
        logger.debug(f"Retrieved {len(chat_history)} messages from chat {chat_id}")
        return chat_history
        
    except Exception as e:
        logger.error(f"Error getting chat history: {e}", exc_info=True)
        return []
