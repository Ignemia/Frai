"""
Handles the Command Line Interface interactions for the application.
"""
import os
import logging
from services.cli.navigation import help_command, process_input
from services.state import get_current_app_state

logger = logging.getLogger(__name__)

WELCOME_MESSAGE = "Hello! I'm your Personal Chatter companion, ready to chat and help. Simply type your message below to get started, or use /help to see available commands."

def start_cli_interface():
    """
    Initializes and starts the command line interface.
    Displays a welcome message, initial help, and then enters the main input loop.
    """
    try:
        os.system("cls" if os.name == "nt" else "clear")
    except Exception as e:
        logger.warning(f"Could not clear screen: {e}")
    print(WELCOME_MESSAGE)
    
    try:
        help_command() # Display initial help commands
    except Exception as e:
        logger.error(f"Error displaying initial help: {e}", exc_info=True)
        print("Could not display help commands. Please try typing /help manually.")

    while True:
        try:
            current_state = get_current_app_state()
            user_input = input(f"\n{current_state} > ").strip()
            
            if not user_input:
                continue
                
            process_input(user_input)
        except KeyboardInterrupt:
            print("\nExiting application. Goodbye!")
            logger.info("User initiated exit via KeyboardInterrupt.")
            break
        except Exception as e:
            logger.error(f"Error processing input in CLI: {e}", exc_info=True)
            print("An error occurred in the CLI. Please check logs and try again.")
