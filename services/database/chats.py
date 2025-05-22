import logging
from services.database.sessions import verify_session_token
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
    count_user_messages_in_chat
)
from services.database.chat_utils import (
    complete_list_user_chats as list_user_chats,
    complete_update_chat_title as update_chat_title,
    complete_save_chat_content as save_chat_content
)

logger = logging.getLogger(__name__)

def create_chat(chat_name, session_token):
    """
    Create a new chat for the user with initial encryption.
    Returns the chat_id if successful, None otherwise.
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
