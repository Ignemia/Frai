import datetime
from services.state import get_state, set_state, set_app_state, APP_STATE_CHAT
from services.database.chats import create_chat

def new_chat_command(args=None):
    """
    Create a new chat through the CLI.
    Args can contain the chat name if provided.
    """
    # Get the session token from state
    session_token = get_state('session_token')
    if not session_token:
        print("You must be logged in to create a new chat. Use /login or /register first.")
        return False
        
    # Determine chat name (use provided args or prompt user)
    chat_name = None
    if args and args.strip():
        chat_name = args.strip()
    else:
        chat_name = input("Enter a name for this chat (or press Enter for default): ").strip()
        
    # Use default name if none provided
    if not chat_name:
        chat_name = f"Chat {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    # Create the chat
    chat_id = create_chat(chat_name, session_token)
    if not chat_id:
        print("Failed to create a new chat. Please try again.")
        return False
    
    # Save the current chat ID in state
    set_state('current_chat_id', chat_id)
    set_state('message_count', 0)
    set_app_state(APP_STATE_CHAT)  # Set state to chat
    print(f"Created new chat: '{chat_name}' (ID: {chat_id})")
    print("You can now start chatting. Type your message or use /help for commands.")
    
    return True
