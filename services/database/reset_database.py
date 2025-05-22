#================================
#=====THIS FILE AI GENERATED===== 
#================================


import os
import logging
import getpass
from dotenv import load_dotenv
from .connection import start_engine, get_existing_engine # Removed get_db_session as not directly used here
from services.state import get_state, set_state # Keep state management
from .models import Base # Import Base for metadata operations

logger = logging.getLogger(__name__)

def check_reset_permission():
    """
    Check if the user has permission to reset the database.
    Returns True if allowed, False otherwise.
    """
    # Load environment variables
    load_dotenv()
    
    # Check if we're in debug mode
    debug_mode = os.environ.get("DEBUG") == "true"
    if debug_mode:
        logger.info("Debug mode enabled, bypassing password verification")
        return True
    
    # Get the secret reset password from environment
    secret_reset_password = os.environ.get("DATABASE_RESET_PASSWORD")
    if not secret_reset_password:
        logger.error("DATABASE_RESET_PASSWORD not set in environment variables")
        return False
    
    # Prompt for password (hiding input)
    entered_password = getpass.getpass("Enter database reset password: ")
    
    # Check if password matches
    return entered_password == secret_reset_password

def drop_all_tables():
    """
    Drop all application tables from the database using SQLAlchemy metadata.
    Returns True if successful, False otherwise.
    """
    engine = get_existing_engine()
    if not engine:
        logger.info("Database engine not found for dropping tables. Attempting to start.")
        engine = start_engine()
        if not engine:
            logger.error("Failed to start database engine for dropping tables.")
            return False
            
    try:
        logger.info("Dropping all tables defined in SQLAlchemy metadata...")
        # Ensure all tables are dropped, considering dependencies.
        # drop_all drops tables in order respecting foreign keys.
        Base.metadata.drop_all(bind=engine)
        logger.info("All tables defined in metadata dropped successfully.")
        return True
    except Exception as e:
        logger.error(f"Error dropping tables: {e}", exc_info=True)
        return False

def recreate_tables():
    """
    Recreate all application tables using SQLAlchemy metadata.
    Returns True if successful, False otherwise.
    """
    engine = get_existing_engine()
    if not engine:
        logger.info("Database engine not found for recreating tables. Attempting to start.")
        engine = start_engine()
        if not engine:
            logger.error("Failed to start database engine for recreating tables.")
            return False
            
    try:
        logger.info("Recreating all tables defined in SQLAlchemy metadata...")
        Base.metadata.create_all(bind=engine)
        logger.info("All tables defined in metadata recreated successfully.")
        return True
    except Exception as e:
        logger.error(f"Error recreating tables: {e}", exc_info=True)
        return False

def reset_database():
    """
    Reset the database by dropping and recreating all tables.
    Requires password verification unless in debug mode.
    """
    # Ensure logging is configured (e.g., if run as a standalone script)
    if not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO)
    
    logger.info("Database reset requested.")
    
    # Check permission
    if not check_reset_permission():
        logger.error("Permission denied for database reset: Incorrect password or missing configuration.")
        return False
    
    # Initialize database engine if not already done
    # get_existing_engine and start_engine handle state internally
    engine = get_existing_engine()
    if not engine:
        logger.info("Engine not found, attempting to start for reset operation.")
        engine = start_engine() 
        if not engine:
            logger.error("Failed to initialize database engine for reset operation.")
            return False
    
    logger.info("Proceeding with dropping all tables.")
    if not drop_all_tables():
        logger.error("Failed to drop all tables during reset.")
        return False
    
    logger.info("Proceeding with recreating all tables.")
    if not recreate_tables():
        logger.error("Failed to recreate all tables during reset.")
        return False
    
    logger.info("Database reset completed successfully.")
    return True

if __name__ == '__main__':
    # Example of how to run reset if this script is executed directly
    # Ensure .env is loaded if run this way
    load_dotenv() 
    reset_success = reset_database()
    if reset_success:
        print("Database has been reset.")
    else:
        print("Database reset failed. Check logs.")
