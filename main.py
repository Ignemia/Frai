import dotenv
dotenv.load_dotenv()

import logging
import os

from services.database.connection import start_engine, validate_db_config, check_database_tables_presence, initiate_tables
from services.cli.navigation import handle_command, is_command, parse_command, process_input, help_command
from services.state import get_current_app_state

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WELCOME_MESSAGE = "Hello! I'm your Personal Chatter companion, ready to chat and help. Simply type your message below to get started, or use /help to see available commands."

database_engine = None

def request_message(state):
    message = input(f"{state} > ")
    if is_command(message):
        command, args = parse_command(message) 
        handle_command(command, args)

def init_database():
    """Initialize database tables with proper error handling"""
    try:
        logger.info("Checking database tables...")
        table_check = check_database_tables_presence()
        
        if not table_check:
            logger.info("Creating necessary database tables...")
            if not initiate_tables():
                logger.error("Failed to create database tables")
                return False
            logger.info("Database tables created successfully")
        else:
            logger.info("Database tables exist, performing schema upgrades if needed...")
            # Import and call the upgrade function
            from services.database.connection import upgrade_database_schema
            if not upgrade_database_schema():
                logger.error("Failed to upgrade database schema")
                return False
            logger.info("Database schema is up to date")
        
        return True
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        return False

def main():
    global database_engine
    
    # Initialize database connection
    database_engine = start_engine()
    if not database_engine:
        print("Failed to establish database engine. Please check your configuration.")
        return 1
    
    # Initialize database tables
    if not init_database():
        print("Database initialization failed. Please check the logs for details.")
        return 1
    
    # Load the AI model
    try:
        import services.chat.pipeline as chat_pipeline
        chat_pipeline.load_model()
    except Exception as e:
        logger.error(f"Failed to load AI model: {e}")
        print("Failed to load AI model. Some features may not work properly.")
    
    # Clear screen and show welcome message
    os.system("cls" if os.name == "nt" else "clear")
    print(WELCOME_MESSAGE)
    
    help_command()
    
    while True:
        try:
            # Get the current state to show in prompt
            current_state = get_current_app_state()
            
            # Display prompt with current state
            user_input = input(f"\n{current_state} > ").strip()
            
            if not user_input:
                continue
                
            process_input(user_input)
        except KeyboardInterrupt:
            print("\nExiting application. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            print("An error occurred. Please try again.")
    
    return 0

if __name__ == "__main__":
    main()