import logging
import datetime
from services.database.connection import get_db_session
from services.database.sessions import verify_session_token
from .models import Chat, ChatKey # Import SQLAlchemy models
import os
from services.encryption.chat_crypto import (
    generate_rsa_key_pair, 
    serialize_key_pair,
    encrypt_chat_content, 
    decrypt_chat_content,
    encrypt_rsa_keys_with_credentials,
    decrypt_rsa_keys_with_credentials
)

logger = logging.getLogger(__name__)

def format_chat_message(role, message, timestamp=None, thoughts=None, sources=None):
    """Format a chat message in XML format"""
    if timestamp is None:
        timestamp = datetime.datetime.now().isoformat()
    
    thoughts_attr = f' thoughts="{thoughts}"' if thoughts else ''
    sources_attr = f' sources="{sources}"' if sources else ''
    
    return f'<{role} timestamp="{timestamp}"{thoughts_attr}{sources_attr}>{message}</{role}>'

def get_system_prompt():
    """Get the system prompt for new chats"""
    return f"{os.environ.get("POSITIVE_SYSTEM_PROMPT_CHAT")}/n{os.environ.get("NEGATIVE_SYSTEM_PROMPT_CHAT")}"

def create_empty_chat_xml():
    """Create an empty chat XML structure with system prompt"""
    timestamp = datetime.datetime.now().isoformat()
    system_prompt = get_system_prompt()
    system_message = format_chat_message("system", system_prompt, timestamp=timestamp)
    return f'<chat created="{timestamp}">{system_message}</chat>'

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
        key_pair = generate_rsa_key_pair() # This generates new RSA keys for each chat
        # serialized_keys = serialize_key_pair(key_pair) # This might not be needed if only parts are stored
        
        # Encrypt the chat content using the public part of the newly generated RSA key pair
        # to encrypt a new AES key, which then encrypts the content.
        encrypted_data = encrypt_chat_content(chat_xml, key_pair['public'])
        # encrypted_data should be a dict: {'content': AES_encrypted_xml, 'encrypted_key': RSA_encrypted_AES_key, 'iv': AES_iv}
        
        with get_db_session() as session:
            # The 'contents' column in Chat model now stores the AES_encrypted_xml
            new_chat = Chat(
                user_id=user_id,
                chat_name=chat_name,
                contents=encrypted_data['content'] 
            )
            session.add(new_chat)
            session.flush() # To get new_chat.chat_id for ChatKey

            new_chat_key = ChatKey(
                chat_id=new_chat.chat_id,
                encrypted_key=encrypted_data['encrypted_key'], # This is the RSA_encrypted_AES_key
                iv=encrypted_data['iv'] # This is the AES_iv
            )
            session.add(new_chat_key)
            
            # Commit is handled by the get_db_session context manager
            logger.info(f"Created new chat '{chat_name}' (ID: {new_chat.chat_id}) for user {user_id}")
            return new_chat.chat_id
            
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
        with get_db_session() as session:
            # Fetch the chat, ensuring it belongs to the user
            chat = session.query(Chat).filter(Chat.chat_id == chat_id, Chat.user_id == user_id).first()
            
            if not chat:
                logger.warning(f"Chat {chat_id} not found or doesn't belong to user {user_id}")
                return None
            
            encrypted_content = chat.contents # This is AES_encrypted_xml
            
            # Fetch the corresponding ChatKey
            chat_key_data = session.query(ChatKey).filter(ChatKey.chat_id == chat_id).first()
            if not chat_key_data:
                logger.error(f"Encryption key data not found for chat {chat_id}")
                return None
            
            rsa_encrypted_aes_key = chat_key_data.encrypted_key
            aes_iv = chat_key_data.iv
            
            # TODO: Full decryption logic:
            # 1. Retrieve the user's private RSA key. This key corresponds to the public key
            #    used during `encrypt_chat_content` when this chat was created.
            #    This implies that `key_pair['private']` from `create_chat` needs to be stored
            #    securely, perhaps encrypted with the user's main password hash.
            #    Or, if each chat has its own RSA key pair, that private key needs to be retrieved.
            #    The current `encrypt_chat_content` seems to imply a new AES key is generated,
            #    and *that* AES key is encrypted with a provided RSA public key.
            #    The `decrypt_chat_content` function would need the corresponding RSA private key
            #    to get the AES key, then use the AES key and IV to decrypt content.

            # For now, using placeholder as per original code:
            # decrypted_xml = decrypt_chat_content(encrypted_content, rsa_encrypted_aes_key, aes_iv, user_rsa_private_key)
            # This assumes decrypt_chat_content can handle this.
            # The function `decrypt_rsa_keys_with_credentials` might be relevant if the RSA private key itself is encrypted.
            
            logger.warning(f"Chat opening for chat_id {chat_id} is returning placeholder due to pending full decryption logic.")
            return create_empty_chat_xml() # Placeholder from original code
    except Exception as e:
        logger.error(f"Error opening chat: {e}", exc_info=True)
        return None

def count_user_messages_in_chat(chat_xml):
    """
    Count the number of user messages in a chat XML.
    """
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(chat_xml)
        user_messages = root.findall("user")
        return len(user_messages)
    except Exception as e:
        logger.error(f"Error counting messages in chat XML: {e}")
        return 0

def add_message_to_chat(chat_id, session_token, role, message, thoughts=None, sources=None):
    """
    Add a message to the chat. (Placeholder - needs full decryption/encryption cycle)
    Returns True if successful, False otherwise.
    """
    user_id = verify_session_token(session_token)
    if not user_id:
        logger.error("Invalid session token when adding message to chat")
        return False
    
    try:
        with get_db_session() as session:
            chat = session.query(Chat).filter(Chat.chat_id == chat_id, Chat.user_id == user_id).first()
            
            if not chat:
                logger.error(f"Chat {chat_id} not found or doesn't belong to user {user_id}")
                return False
            
            # TODO: Full decryption/encryption cycle for adding a message:
            # 1. Decrypt current chat.contents (similar to open_chat, need user's RSA private key to get AES key).
            #    Let's say `decrypted_xml = actual_decrypt_function(...)`
            # 2. Parse `decrypted_xml`, add the new formatted message.
            #    `new_message_xml = format_chat_message(role, message, thoughts=thoughts, sources=sources)`
            #    `updated_xml = append_message_to_xml(decrypted_xml, new_message_xml)`
            # 3. Re-encrypt `updated_xml` using the *same* AES key and IV that were used for this chat.
            #    `new_encrypted_content = actual_encrypt_with_existing_aes_key(updated_xml, decrypted_aes_key, aes_iv)`
            # 4. Update chat.contents and chat.last_modified.
            #    `chat.contents = new_encrypted_content`
            
            # For now, just update last_modified as a placeholder (as in original code)
            chat.last_modified = datetime.datetime.now() # This will be updated by model's onupdate if configured
            # session.commit() is handled by context manager
            
            logger.info(f"Added {role} message to chat {chat_id} (placeholder - content not actually modified, only last_modified updated)")
            return True
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
    Re-encrypts the content with the existing keys. (Placeholder - needs full encryption cycle)
    Returns True if successful, False otherwise.
    """
    user_id = verify_session_token(session_token)
    if not user_id:
        logger.error("Invalid session token when saving chat content")
        return False
    
    try:
        with get_db_session() as session:
            chat = session.query(Chat).filter(Chat.chat_id == chat_id, Chat.user_id == user_id).first()
            if not chat:
                logger.error(f"Chat {chat_id} not found for user {user_id} when trying to save content.")
                return False

            chat_key_entry = session.query(ChatKey).filter(ChatKey.chat_id == chat_id).first()
            if not chat_key_entry:
                logger.error(f"Encryption key data not found for chat {chat_id} during save.")
                return False
            
            # TODO: Full re-encryption logic:
            # 1. Retrieve user's RSA private key (see notes in open_chat/add_message_to_chat).
            # 2. Decrypt `chat_key_entry.encrypted_key` (RSA_encrypted_AES_key) to get the plain AES key.
            #    `plain_aes_key = decrypt_aes_key_with_rsa(chat_key_entry.encrypted_key, user_rsa_private_key)`
            # 3. Encrypt the `updated_content_xml` using this `plain_aes_key` and `chat_key_entry.iv`.
            #    `new_encrypted_db_payload = actual_aes_encrypt_function(updated_content_xml, plain_aes_key, chat_key_entry.iv)`
            #    This is different from `encrypt_chat_content` which generates a *new* AES key.
            #    You'll need a function that encrypts with an *existing* AES key.
            
            # Placeholder logic (as in original code):
            logger.warning(f"save_chat_content for chat {chat_id}: Re-encryption logic is a placeholder. Content not actually re-encrypted and saved with new data.")
            # chat.contents = "placeholder_re_encrypted_content_on_save" # Placeholder update
            # chat.last_modified will be updated by model's onupdate
            
            # session.commit() is handled by context manager
            logger.info(f"Updated content for chat {chat_id} (placeholder action)")
            return True # Placeholder success
    except Exception as e:
        logger.error(f"Error saving chat content: {e}", exc_info=True)
        return False
