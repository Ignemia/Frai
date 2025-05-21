import logging
import datetime
from services.database.connection import get_db_cursor
from services.database.sessions import verify_session_token
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
    Create a new chat for the user with initial encryption.
    Returns the chat_id if successful, None otherwise.
    """
    user_id = verify_session_token(session_token)
    if not user_id:
        logger.error("Invalid session token when creating chat")
        return None
    
    try:
        # Create empty XML chat content with system prompt
        chat_xml = create_empty_chat_xml()
        
        # Generate RSA keys for the chat
        key_pair = generate_rsa_key_pair()
        serialized_keys = serialize_key_pair(key_pair)
        
        # Encrypt the chat content
        encrypted_data = encrypt_chat_content(chat_xml, key_pair['public'])
        
        with get_db_cursor() as cursor:
            # Check which columns exist in the chats table
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'chats';
            """)
            
            columns = [row[0] for row in cursor.fetchall()]
            logger.debug(f"Available columns in chats table: {columns}")
            
            # Build SQL query based on available columns
            query_parts = []
            params = []
            
            # Always include user_id
            query_parts.append("user_id = %s")
            params.append(user_id)
            
            # Add chat_name if column exists
            if 'chat_name' in columns:
                query_parts.append("chat_name = %s")
                params.append(chat_name)
            
            # Handle content (could be either 'contents' or 'encrypted_contents')
            content_added = False
            if 'contents' in columns:
                query_parts.append("contents = %s")
                params.append(encrypted_data['content'])
                content_added = True
            
            if 'encrypted_contents' in columns and not content_added:
                query_parts.append("encrypted_contents = %s")
                params.append(encrypted_data['content'])
                content_added = True
            
            # Ensure at least one content column is updated
            if not content_added:
                logger.error("Neither 'contents' nor 'encrypted_contents' column found in chats table")
                return None
            
            # Build the INSERT query using column specification
            column_names = [part.split(" = ")[0] for part in query_parts]
            value_placeholders = ["%s" for _ in column_names]
            
            query = f"""
                INSERT INTO chats ({', '.join(column_names)})
                VALUES ({', '.join(value_placeholders)})
                RETURNING chat_id;
            """
            
            # Execute the query
            cursor.execute(query, params)
            chat_id = cursor.fetchone()[0]
            
            # Insert the encrypted key
            cursor.execute(
                """
                INSERT INTO chat_keys (chat_id, encrypted_key, iv)
                VALUES (%s, %s, %s);
                """,
                (chat_id, encrypted_data['encrypted_key'], encrypted_data['iv'])
            )
            
            cursor.connection.commit()
            logger.info(f"Created new chat '{chat_name}' (ID: {chat_id}) for user {user_id}")
            
            return chat_id
    except Exception as e:
        logger.error(f"Error creating chat: {e}")
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
        with get_db_cursor() as cursor:
            # Check which columns exist in the chats table
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'chats';
            """)
            
            columns = [row[0] for row in cursor.fetchall()]
            
            # Determine which column to use for content
            content_column = "encrypted_contents" if "encrypted_contents" in columns else "contents"
            
            # Verify the chat belongs to the user
            cursor.execute(
                f"""
                SELECT {content_column} FROM chats 
                WHERE chat_id = %s AND user_id = %s;
                """,
                (chat_id, user_id)
            )
            result = cursor.fetchone()
            if not result:
                logger.warning(f"Chat {chat_id} not found or doesn't belong to user {user_id}")
                return None
            
            encrypted_content = result[0]
            
            # Get the encryption key
            cursor.execute(
                """
                SELECT encrypted_key, iv FROM chat_keys
                WHERE chat_id = %s;
                """,
                (chat_id,)
            )
            key_data = cursor.fetchone()
            if not key_data:
                logger.error(f"Encryption key not found for chat {chat_id}")
                return None
            
            encrypted_key, iv = key_data
            
            # TODO: Implement fetching the user's password hash to decrypt the key
            # For now, we'll return a placeholder empty chat XML
            return create_empty_chat_xml()
    except Exception as e:
        logger.error(f"Error opening chat: {e}")
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
    Add a message to the chat.
    Returns True if successful, False otherwise.
    """
    user_id = verify_session_token(session_token)
    if not user_id:
        logger.error("Invalid session token when adding message to chat")
        return False
    
    try:
        with get_db_cursor() as cursor:
            # Get current chat content and encryption data
            cursor.execute(
                """
                SELECT c.encrypted_contents, k.encrypted_key, k.iv
                FROM chats c
                JOIN chat_keys k ON c.chat_id = k.chat_id
                WHERE c.chat_id = %s AND c.user_id = %s;
                """,
                (chat_id, user_id)
            )
            
            result = cursor.fetchone()
            if not result:
                logger.error(f"Chat {chat_id} not found or doesn't belong to user {user_id}")
                return False
            
            encrypted_content, encrypted_key, iv = result
            
            # For now, we'll use a placeholder for the decrypted content
            # In a real implementation, you would:
            # 1. Decrypt the content
            # 2. Add the new message
            # 3. Re-encrypt the content
            # 4. Update the database
            
            # Placeholder implementation - just update timestamp
            cursor.execute(
                """
                UPDATE chats
                SET last_modified = CURRENT_TIMESTAMP
                WHERE chat_id = %s;
                """,
                (chat_id,)
            )
            
            cursor.connection.commit()
            logger.info(f"Added {role} message to chat {chat_id}")
            
            return True
    except Exception as e:
        logger.error(f"Error adding message to chat: {e}")
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
        with get_db_cursor() as cursor:
            cursor.execute(
                """
                SELECT chat_id, chat_name, last_modified 
                FROM chats 
                WHERE user_id = %s
                ORDER BY last_modified DESC;
                """,
                (user_id,)
            )
            chats = cursor.fetchall()
            
            logger.info(f"Retrieved {len(chats)} chats for user ID: {user_id}")
            return chats
    except Exception as e:
        logger.error(f"Error listing user chats: {e}")
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
        with get_db_cursor() as cursor:
            cursor.execute(
                """
                UPDATE chats 
                SET chat_name = %s, last_modified = CURRENT_TIMESTAMP 
                WHERE chat_id = %s AND user_id = %s
                RETURNING chat_id;
                """,
                (new_title, chat_id, user_id)
            )
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"Chat {chat_id} not found or doesn't belong to user {user_id}")
                return False
                
            cursor.connection.commit()
            logger.info(f"Updated title for chat {chat_id} to '{new_title}'")
            return True
    except Exception as e:
        logger.error(f"Error updating chat title: {e}")
        return False

def save_chat_content(chat_id, updated_content, session_token):
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
        with get_db_cursor() as cursor:
            # Get the current encryption key
            cursor.execute(
                """
                SELECT ck.encrypted_key, ck.iv
                FROM chat_keys ck
                JOIN chats c ON c.chat_id = ck.chat_id
                WHERE c.chat_id = %s AND c.user_id = %s;
                """,
                (chat_id, user_id)
            )
            key_data = cursor.fetchone()
            
            if not key_data:
                logger.error(f"Encryption key not found for chat {chat_id}")
                return False
            
            # TODO: Implement the re-encryption of content
            # For now, this is a placeholder
            
            cursor.execute(
                """
                UPDATE chats
                SET encrypted_contents = %s, last_modified = CURRENT_TIMESTAMP
                WHERE chat_id = %s AND user_id = %s;
                """,
                ("placeholder_encrypted_content", chat_id, user_id)
            )
            
            cursor.connection.commit()
            logger.info(f"Updated content for chat {chat_id}")
            return True
    except Exception as e:
        logger.error(f"Error saving chat content: {e}")
        return False
