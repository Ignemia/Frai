import logging
import datetime
from services.database.connection import get_db_session
from .models import Chat, ChatKey, UserKey

logger = logging.getLogger(__name__)

def store_chat_in_database(user_id, chat_name, encrypted_data, encrypted_rsa_keys):
    """Store a new chat and its encryption data in the database."""
    with get_db_session() as session:
        new_chat = Chat(
            user_id=user_id,
            chat_name=chat_name,
            contents=encrypted_data['content'] 
        )
        session.add(new_chat)
        session.flush()
        
        new_chat_key = ChatKey(
            chat_id=new_chat.chat_id,
            encrypted_key=encrypted_data['encrypted_key'],
            iv=encrypted_data['iv']
        )
        session.add(new_chat_key)
        
        store_user_keys_if_needed(session, user_id, encrypted_rsa_keys)
        
        logger.info(f"Created chat '{chat_name}' (ID: {new_chat.chat_id}) for user {user_id}")
        return new_chat.chat_id

def store_user_keys_if_needed(session, user_id, encrypted_rsa_keys):
    """Store user RSA keys if they don't exist already."""
    existing_keys = session.query(UserKey).filter(UserKey.user_id == user_id).first()
    if not existing_keys:
        new_user_key = UserKey(
            user_id=user_id,
            encrypted_keys=encrypted_rsa_keys
        )
        session.add(new_user_key)
    else:
        logger.info(f"User {user_id} already has stored RSA keys")

def fetch_chat_and_keys(chat_id, user_id):
    """
    Fetch chat and encryption keys from database.
    
    Returns:
        tuple: (chat, chat_key_data, user_encrypted_rsa_keys) or (None, None, None)
    """
    with get_db_session() as session:
        chat = session.query(Chat).filter(
            Chat.chat_id == chat_id, 
            Chat.user_id == user_id
        ).first()
        
        if not chat:
            logger.warning(f"Chat {chat_id} not found or doesn't belong to user {user_id}")
            return None, None, None
        
        chat_key_data = session.query(ChatKey).filter(
            ChatKey.chat_id == chat_id
        ).first()
        
        if not chat_key_data:
            logger.error(f"Encryption key data not found for chat {chat_id}")
            return None, None, None
        
        user_encrypted_rsa_keys = session.query(UserKey).filter(
            UserKey.user_id == user_id
        ).first()
        
        if not user_encrypted_rsa_keys:
            logger.error(f"RSA keys not found for user {user_id}")
            return None, None, None
            
        return chat, chat_key_data, user_encrypted_rsa_keys

def fetch_chat_encryption_data(chat_id, user_id):
    """
    Fetch chat and encryption data needed for updates.
    
    Returns:
        tuple: (chat, decrypted_xml, encryption_keys) or None if failed
    """
    from services.database.chat_encryption import decrypt_user_rsa_keys, decrypt_chat_with_keys
    
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
    
    encryption_keys = {
        'rsa_encrypted_aes_key': chat_key_data.encrypted_key,
        'aes_iv': chat_key_data.iv,
        'rsa_private_key': user_rsa_keys['private']
    }
    
    return chat, decrypted_xml, encryption_keys

def list_user_chats_query(user_id):
    """Query all chats belonging to a user."""
    with get_db_session() as session:
        chats_data = session.query(Chat.chat_id, Chat.chat_name, Chat.last_modified)\
                            .filter(Chat.user_id == user_id)\
                            .order_by(Chat.last_modified.desc())\
                            .all()
        
        logger.info(f"Retrieved {len(chats_data)} chats for user ID: {user_id}")
        return chats_data
