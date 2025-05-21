from services.state import get_state
from services.database.chats import list_user_chats

def list_chats_command():
    """
    List all chats belonging to the current user.
    """
    # Get the session token from state
    session_token = get_state('session_token')
    if not session_token:
        print("You must be logged in to list chats. Use /login or /register first.")
        return False
    
    # Get the chats from the database
    chats = list_user_chats(session_token)
    if chats is None:
        print("Failed to retrieve your chats. Please try again.")
        return False
    
    if not chats:
        print("You don't have any chats yet. Use /new_chat to create one.")
        return True
    
    # Display the chats
    print("\n=== Your Chats ===")
    print(f"{'ID':<6} {'Last Modified':<20} {'Chat Name'}")
    print("-" * 50)
    
    for chat_id, chat_name, last_modified in chats:
        last_modified_str = last_modified.strftime("%Y-%m-%d %H:%M")
        print(f"{chat_id:<6} {last_modified_str:<20} {chat_name}")
    
    print("\nTo open a chat, type: /open <chat_id>")
    return True
