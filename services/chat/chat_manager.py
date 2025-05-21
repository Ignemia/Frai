import logging
import datetime
from services.database.chats import (
    create_chat, 
    open_chat, 
    add_message_to_chat,
    close_chat
)
from services.chat.pipeline import send_query, clear_history

logger = logging.getLogger(__name__)

def start_new_chat(session_token, chat_name=None):
    """
    Start a new chat session.
    Returns chat_id if successful, None otherwise.
    """
    if not chat_name:
        chat_name = f"Chat {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    # Create the chat in the database
    chat_id = create_chat(chat_name, session_token)
    if not chat_id:
        logger.error("Failed to create new chat")
        return None
    
    # Clear the LLM history for a fresh start
    clear_history()
    
    return chat_id

def process_user_message(chat_id, session_token, message, thoughts=None):
    """
    Process a user message, add it to the chat, and get AI response.
    Returns the AI response if successful, None otherwise.
    """
    # Add the user message to the chat
    success = add_message_to_chat(
        chat_id, 
        session_token, 
        "user", 
        message, 
        thoughts=thoughts
    )
    
    if not success:
        logger.error("Failed to add user message to chat")
        return None
    
    # Generate AI response
    try:
        ai_response = send_query(message)
        
        # Add the AI response to the chat
        success = add_message_to_chat(
            chat_id,
            session_token,
            "agent",
            ai_response
        )
        
        if not success:
            logger.error("Failed to add AI response to chat")
            return None
        
        return ai_response
    except Exception as e:
        logger.error(f"Error processing message with AI: {e}")
        return None

def end_chat_session(chat_id, session_token, username, password_hash):
    """
    End a chat session, re-encrypt it, and secure the key.
    Returns True if successful, False otherwise.
    """
    return close_chat(chat_id, session_token, username, password_hash)
