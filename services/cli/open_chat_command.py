from services.state import get_state, set_state, set_app_state, APP_STATE_CHAT
from services.database.chats import open_chat, list_user_chats, count_user_messages_in_chat

def open_chat_command(args=None):
    """
    Open an existing chat.
    Args should contain the chat ID to open.
    """
    # Get the session token from state
    session_token = get_state('session_token')
    if not session_token:
        print("You must be logged in to open a chat. Use /login or /register first.")
        return False
    
    # Get the chat ID from arguments or prompt the user
    chat_id = None
    if args and args.strip().isdigit():
        chat_id = int(args.strip())
    else:
        # If no chat ID provided, show list of chats first
        chats = list_user_chats(session_token)
        if not chats:
            print("You don't have any chats yet. Use /new_chat to create one.")
            return False
            
        print("\n=== Your Chats ===")
        print(f"{'ID':<6} {'Last Modified':<20} {'Chat Name'}")
        print("-" * 50)
        
        for c_id, c_name, c_modified in chats:
            modified_str = c_modified.strftime("%Y-%m-%d %H:%M")
            print(f"{c_id:<6} {modified_str:<20} {c_name}")
        
        # Prompt for ID
        try:
            chat_id_input = input("\nEnter the ID of the chat to open: ").strip()
            chat_id = int(chat_id_input) if chat_id_input.isdigit() else None
        except ValueError:
            print("Invalid chat ID. Please try again.")
            return False
    
    if not chat_id:
        print("No valid chat ID provided. Please try again with a valid ID.")
        return False
    
    # Try to open the chat (in a real implementation this would decrypt the content)
    chat_content = open_chat(chat_id, session_token)
    
    # Save the current chat ID in state
    set_state('current_chat_id', chat_id)
    set_app_state(APP_STATE_CHAT)  # Set state to chat
    
    # Count messages for title update tracking
    message_count = 0
    if chat_content:
        message_count = count_user_messages_in_chat(chat_content)
    set_state('message_count', message_count)
    
    # Get the chat name from the database
    chats = list_user_chats(session_token)
    chat_name = ""
    if chats:
        for c_id, c_name, _ in chats:
            if c_id == chat_id:
                chat_name = c_name
                break
    
    print(f"Opened chat: '{chat_name}' (ID: {chat_id})")
    print("You can now continue your conversation. Type your message or use /help for commands.")
    
    return True
