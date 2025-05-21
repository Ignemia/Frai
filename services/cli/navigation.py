from services.state import (
    get_state, set_state, 
    get_current_app_state, set_app_state,
    APP_STATE_HOME, APP_STATE_LOGIN, APP_STATE_REGISTER, 
    APP_STATE_LIST, APP_STATE_CHAT
)

from services.cli.login_command import login_command
from services.cli.register_command import register_command
from services.cli.new_chat_command import new_chat_command
from services.cli.open_chat_command import open_chat_command
from services.cli.list_chats_command import list_chats_command
from services.cli.chat_command import chat_command

import logging

logger = logging.getLogger(__name__)

def is_command(message):
    if not message.startswith("/"):
        return False
    return True

def parse_command(message):
    if not is_command(message):
        return None, None
    
    parts = message[1:].split(" ", 1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    
    return command, args


def help_command():
    print("Available commands:")
    print("  /help                - Show this help message")
    print("  /back                - Go back to previous menu/state")
    print("  /quit, /exit         - Exit the application")
    print("  /login [username]    - Login with your username")
    print("  /register [username] - Register a new account")
    print("  /new_chat [name]     - Create a new chat")
    print("  /open [chat_id]      - Open an existing chat")
    print("  /list_chats          - List all your chats")
    print("  /close_chat          - Close the current chat")

def quit_command():
    print("Exiting the application. Goodbye!")
    exit(0)

def back_command():
    """Go back to the previous state or menu"""
    current_state = get_current_app_state()
    
    # If in chat, go back to home
    if current_state == APP_STATE_CHAT:
        set_state('current_chat_id', None)
        set_state('message_count', 0)
        set_app_state(APP_STATE_HOME)
        print("Chat closed. Returning to home.")
    # If in login or register, go back to home
    elif current_state in [APP_STATE_LOGIN, APP_STATE_REGISTER, APP_STATE_LIST]:
        set_app_state(APP_STATE_HOME)
        print("Returning to home.")
    else:
        print("Already at home.")

def handle_command(command, args):
    match command:
        case "help" | "h" | "?" | "commands": 
            help_command()
        case "back" | "b":
            back_command()
        case "quit" | "exit" | "q":
            quit_command()
            
        case "new_chat" | "new":
            result = new_chat_command(args)
            if result:
                set_app_state(APP_STATE_CHAT)
        case "open_chat" | "open":
            result = open_chat_command(args)
            if result:
                set_app_state(APP_STATE_CHAT)
        case "list_chats" | "list" | "chats":
            set_app_state(APP_STATE_LIST)
            list_chats_command()
        case "close_chat" | "close":
            # Close the current chat by clearing it from state
            set_state('current_chat_id', None)
            set_state('message_count', 0)
            set_app_state(APP_STATE_HOME)
            print("Chat closed.")
            
        case "login":
            set_app_state(APP_STATE_LOGIN)
            session_token = login_command(args.strip())
            if not session_token:
                logger.error("Login failed.")
                set_app_state(APP_STATE_HOME)
                return
            set_state('session_token', session_token)
            set_app_state(APP_STATE_HOME)
            
        case "register":
            set_app_state(APP_STATE_REGISTER)
            session_token = register_command(args.strip())
            if not session_token:
                logger.error("Registration failed.")
                set_app_state(APP_STATE_HOME)
                return
            set_state('session_token', session_token)
            set_app_state(APP_STATE_HOME)
            
        case _:
            logger.warning(f"Unknown command: {command}")
            help_command()

def process_input(user_input):
    """
    Process user input, distinguishing between commands and chat messages.
    """
    if is_command(user_input):
        command, args = parse_command(user_input)
        handle_command(command, args)
    else:
        # Only process as chat message if in chat state
        current_state = get_current_app_state()
        if current_state == APP_STATE_CHAT:
            # Check that there's an active chat
            chat_id = get_state('current_chat_id')
            if chat_id:
                chat_command(user_input)
            else:
                print("No active chat. Use /new_chat to create one or /open <id> to open an existing chat.")
        else:
            # Not in chat state, show help instead
            print(f"Not in a chat. Type a command like /help to see available options.")
            help_command()
