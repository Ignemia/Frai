import logging
import datetime
from services.database.connection import get_db_session
from services.database.sessions import verify_session_token
from .models import Chat, ChatKey, User, PasswordEntry # Import SQLAlchemy models
from services.database.users import get_user_credentials_for_key_derivation # Helper function
from services.encryption.chat_crypto import (
    generate_rsa_key_pair, 
    serialize_key_pair,
    deserialize_key_pair, # Added deserialize
    encrypt_chat_content, 
    decrypt_chat_content,
    encrypt_rsa_keys_with_credentials,
    decrypt_rsa_keys_with_credentials
)
import xml.etree.ElementTree as ET # For add_message_to_chat

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
    return """You are an AI assistant. You are helpful, creative, clever, and very friendly.
Answer questions truthfully and accurately. If you don't know the answer, say so rather than making something up.
Be concise unless asked to elaborate. Always maintain a positive and supportive tone."""

def create_empty_chat_xml():
    """Create an empty chat XML structure with system prompt"""
    timestamp = datetime.datetime.now().isoformat()
    system_prompt = get_system_prompt()
    system_message = format_chat_message("system", system_prompt, timestamp=timestamp)
    return f'<chat created="{timestamp}">{system_message}</chat>'

def create_chat(chat_name, session_token):
    """
    Create a new chat for the user with the new encryption flow.
    Returns the chat_id if successful, None otherwise.
    """
    user_id = verify_session_token(session_token)
    if not user_id:
        logger.error("Invalid session token when creating chat")
        return None
    
    try:
        with get_db_session() as session:
            user_creds = get_user_credentials_for_key_derivation(user_id, session)
            if not user_creds:
                logger.error(f"Could not retrieve credentials for user {user_id} to create chat.")
                return None
            username, password_hash = user_creds

            chat_xml = create_empty_chat_xml()
            
            # 1. Generate chat-specific RSA key pair
            chat_rsa_key_pair_obj = generate_rsa_key_pair() # Returns objects

            # 2. Encrypt chat content (generates AES key, encrypts it with chat's RSA public key)
            encrypted_chat_data = encrypt_chat_content(chat_xml, chat_rsa_key_pair_obj['public'])
            # encrypted_chat_data = {'content', 'encrypted_key', 'iv'}

            # 3. Serialize the chat's RSA private key to PEM string
            serialized_chat_rsa_private_pem = serialize_key_pair(chat_rsa_key_pair_obj)['private']

            # 4. Encrypt the serialized chat RSA private key PEM using UserDerivedKey
            encrypted_rsa_private_key_package = encrypt_rsa_keys_with_credentials(
                {'private': serialized_chat_rsa_private_pem}, # Pass as dict
                username, 
                password_hash
            )
            # encrypted_rsa_private_key_package = {'encrypted_key', 'iv'}

            # 5. Store everything
            new_chat = Chat(
                user_id=user_id,
                chat_name=chat_name,
                contents=encrypted_chat_data['content'] # AES-encrypted XML (base64)
            )
            session.add(new_chat)
            session.flush() # To get new_chat.chat_id

            new_chat_key_entry = ChatKey(
                chat_id=new_chat.chat_id,
                encrypted_key=encrypted_chat_data['encrypted_key'], # ChatAESKey, RSA-encrypted (base64)
                iv=encrypted_chat_data['iv'], # IV for content AES (base64)
                encrypted_rsa_private_key=encrypted_rsa_private_key_package['encrypted_key'], # Chat RSA private key, AES-encrypted (base64)
                user_derived_key_iv=encrypted_rsa_private_key_package['iv'] # IV for UserDerivedKey AES (base64)
            )
            session.add(new_chat_key_entry)
            
            logger.info(f"Created new chat '{chat_name}' (ID: {new_chat.chat_id}) for user {user_id}")
            return new_chat.chat_id
            
    except Exception as e:
        logger.error(f"Error creating chat: {e}", exc_info=True)
        return None

def _get_decrypted_chat_rsa_private_key(chat_key_entry: ChatKey, user_id: int, session: 'SQLAlchemySession') -> Optional[object]:
    """Helper to decrypt and return the chat's RSA private key object."""
    user_creds = get_user_credentials_for_key_derivation(user_id, session)
    if not user_creds:
        logger.error(f"Could not retrieve credentials for user {user_id} to decrypt chat RSA key.")
        return None
    username, password_hash = user_creds

    encrypted_rsa_package = {
        'encrypted_key': chat_key_entry.encrypted_rsa_private_key,
        'iv': chat_key_entry.user_derived_key_iv
    }
    
    decrypted_serialized_private_pem_dict = decrypt_rsa_keys_with_credentials(
        encrypted_rsa_package, username, password_hash
    ) # Returns {'private': pem_string}
    
    # Deserialize PEM string to RSA private key object
    # deserialize_key_pair expects {'private': pem, 'public': pem_or_none}
    # but can derive public from private if 'public' is missing.
    chat_rsa_key_pair_obj = deserialize_key_pair({'private': decrypted_serialized_private_pem_dict['private']})
    return chat_rsa_key_pair_obj['private']


def open_chat(chat_id, session_token):
    """
    Open and decrypt a chat using the new encryption flow.
    Returns the decrypted chat content if successful, None otherwise.
    """
    user_id = verify_session_token(session_token)
    if not user_id:
        logger.error("Invalid session token when opening chat")
        return None
    
    try:
        with get_db_session() as session:
            chat = session.query(Chat).filter(Chat.chat_id == chat_id, Chat.user_id == user_id).first()
            if not chat:
                logger.warning(f"Chat {chat_id} not found or doesn't belong to user {user_id}")
                return None
            
            chat_key_entry = session.query(ChatKey).filter(ChatKey.chat_id == chat_id).first()
            if not chat_key_entry:
                logger.error(f"Encryption key data not found for chat {chat_id}")
                return None

            # 1. Decrypt the chat's RSA private key
            chat_rsa_private_key_obj = _get_decrypted_chat_rsa_private_key(chat_key_entry, user_id, session)
            if not chat_rsa_private_key_obj:
                logger.error(f"Failed to decrypt chat RSA private key for chat {chat_id}.")
                return None # Or raise an error indicating decryption failure

            # 2. Prepare data for decrypt_chat_content
            encrypted_content_package = {
                'content': chat.contents, # AES-encrypted XML (base64)
                'encrypted_key': chat_key_entry.encrypted_key, # ChatAESKey, RSA-encrypted (base64)
                'iv': chat_key_entry.iv # IV for content AES (base64)
            }

            # 3. Decrypt chat content using the chat's RSA private key (which decrypts ChatAESKey)
            decrypted_xml = decrypt_chat_content(encrypted_content_package, chat_rsa_private_key_obj)
            
            logger.info(f"Successfully opened and decrypted chat {chat_id} for user {user_id}")
            return decrypted_xml
    except Exception as e:
        logger.error(f"Error opening chat {chat_id}: {e}", exc_info=True)
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
    Add a message to the chat, involving decryption and re-encryption.
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
            
            chat_key_entry = session.query(ChatKey).filter(ChatKey.chat_id == chat_id).first()
            if not chat_key_entry:
                logger.error(f"Encryption key data not found for chat {chat_id}")
                return False

            # 1. Decrypt the chat's RSA private key to get its pair (public key needed for re-encryption)
            chat_rsa_private_key_obj = _get_decrypted_chat_rsa_private_key(chat_key_entry, user_id, session)
            if not chat_rsa_private_key_obj:
                logger.error(f"Failed to decrypt chat RSA private key for chat {chat_id} to add message.")
                return False
            chat_rsa_public_key_obj = chat_rsa_private_key_obj.public_key()

            # 2. Decrypt current chat content
            encrypted_content_package = {
                'content': chat.contents,
                'encrypted_key': chat_key_entry.encrypted_key,
                'iv': chat_key_entry.iv
            }
            decrypted_xml_str = decrypt_chat_content(encrypted_content_package, chat_rsa_private_key_obj)

            # 3. Add new message to XML
            # Ensure decrypted_xml_str is not None or empty before parsing
            if not decrypted_xml_str:
                logger.error(f"Decrypted chat content is empty for chat {chat_id}. Cannot add message.")
                # Potentially initialize with empty chat structure if this happens
                decrypted_xml_str = create_empty_chat_xml()


            try:
                root = ET.fromstring(decrypted_xml_str)
                new_message_element_str = format_chat_message(role, message, thoughts=thoughts, sources=sources)
                new_message_element = ET.fromstring(new_message_element_str)
                root.append(new_message_element)
                updated_xml_str = ET.tostring(root, encoding='unicode')
            except ET.ParseError as e:
                logger.error(f"Error parsing chat XML for chat {chat_id}: {e}. Content: '{decrypted_xml_str[:100]}...'")
                # If XML is corrupted, might need a recovery strategy or error out
                # For now, let's try to re-initialize if parsing fails badly.
                logger.warning(f"Re-initializing chat content for chat {chat_id} due to parse error.")
                current_chat_xml_str = create_empty_chat_xml()
                root = ET.fromstring(current_chat_xml_str)
                new_message_element_str = format_chat_message(role, message, thoughts=thoughts, sources=sources)
                new_message_element = ET.fromstring(new_message_element_str)
                root.append(new_message_element)
                updated_xml_str = ET.tostring(root, encoding='unicode')


            # 4. Re-encrypt the updated XML content
            # encrypt_chat_content will generate a new AES key and IV for the content
            new_encrypted_chat_data = encrypt_chat_content(updated_xml_str, chat_rsa_public_key_obj)

            # 5. Update database
            chat.contents = new_encrypted_chat_data['content']
            chat.last_modified = datetime.datetime.now() # Or rely on model's onupdate

            chat_key_entry.encrypted_key = new_encrypted_chat_data['encrypted_key'] # New ChatAESKey, RSA-encrypted
            chat_key_entry.iv = new_encrypted_chat_data['iv'] # New IV for content AES
            # The chat_rsa_private_key and its encryption (user_derived_key_iv) do not change here.
            
            logger.info(f"Added {role} message to chat {chat_id}")
            return True
    except Exception as e:
        logger.error(f"Error adding message to chat {chat_id}: {e}", exc_info=True)
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
            chat.last_modified = datetime.datetime.now() # Explicitly update if model doesn't do it via onupdate
            
            logger.info(f"Updated title for chat {chat_id} to '{new_title}'")
            return True
    except Exception as e:
        logger.error(f"Error updating chat title: {e}", exc_info=True)
        return False

def save_chat_content(chat_id, updated_content_xml, session_token):
    """
    Save updated chat content to the database. Re-encrypts the content.
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
            
            # 1. Decrypt the chat's RSA private key to get its pair (public key needed for re-encryption)
            chat_rsa_private_key_obj = _get_decrypted_chat_rsa_private_key(chat_key_entry, user_id, session)
            if not chat_rsa_private_key_obj:
                logger.error(f"Failed to decrypt chat RSA private key for chat {chat_id} to save content.")
                return False
            chat_rsa_public_key_obj = chat_rsa_private_key_obj.public_key()

            # 2. Re-encrypt the provided updated_content_xml
            # encrypt_chat_content will generate a new AES key and IV.
            new_encrypted_chat_data = encrypt_chat_content(updated_content_xml, chat_rsa_public_key_obj)

            # 3. Update database
            chat.contents = new_encrypted_chat_data['content']
            chat.last_modified = datetime.datetime.now() # Or rely on model's onupdate

            chat_key_entry.encrypted_key = new_encrypted_chat_data['encrypted_key'] # New ChatAESKey, RSA-encrypted
            chat_key_entry.iv = new_encrypted_chat_data['iv'] # New IV for content AES
            # The chat_rsa_private_key and its encryption (user_derived_key_iv) do not change.

            logger.info(f"Updated and saved content for chat {chat_id}")
            return True
    except Exception as e:
        logger.error(f"Error saving chat content for chat {chat_id}: {e}", exc_info=True)
        return False
