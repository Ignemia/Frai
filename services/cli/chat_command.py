import logging
from services.state import get_state, set_state
from services.database.chats import add_message_to_chat, update_chat_title, get_chat_history
from services.chat.llm_interface import send_query
from services.cli.title_generator import generate_chat_title, should_update_title

logger = logging.getLogger(__name__)

def chat_command(user_message):
    """
    Process a user chat message, send it to the AI, and update the chat.
    """
    if not user_message or not user_message.strip():
        print("Your message is empty. Please type something.")
        return False
    
    # Get necessary state
    session_token = get_state('session_token')
    chat_id = get_state('current_chat_id')
    
    if not session_token:
        print("You must be logged in to chat. Use /login or /register first.")
        return False
    
    if not chat_id:
        print("No active chat. Use /new_chat to create one or /open <id> to open an existing chat.")
        return False
    
    try:
        # Add user message to the chat
        success = add_message_to_chat(chat_id, session_token, "user", user_message)
        if not success:
            print("Failed to save your message. Please try again.")
            return False
          # Get AI response
        chat_history = get_chat_history(chat_id, session_token)
        ai_response, updated_history = send_query(user_message, chat_history)
        
        # Add AI response to the chat
        success = add_message_to_chat(chat_id, session_token, "agent", ai_response)
        if not success:
            print("Failed to save AI response. Please try again.")
            return False
        
        # Print the AI response to the console
        print(f"\nAI: {ai_response}\n")
        
        # Update message count
        message_count = get_state('message_count') or 0
        message_count += 1
        set_state('message_count', message_count)
        
        # Check if we should update the chat title (every 5 messages)
        if should_update_title(message_count):
            # Generate a title based on the latest messages
            new_title = generate_chat_title([user_message])
            if new_title:
                update_chat_title(chat_id, new_title, session_token)
                print(f"Chat title updated to: '{new_title}'")
        
        return True
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        print("An error occurred while processing your message. Please try again.")
        return False
