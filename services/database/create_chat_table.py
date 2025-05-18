#================================
#=====THIS FILE AI GENERATED===== 
#================================
import logging
from typing import Optional
from .connection import start_connection_pool, get_db_cursor
# Import state management functions
from services.state import get_state, set_state

logger = logging.getLogger(__name__)

def create_chat_table() -> bool:
    """
    Create the chats table if it doesn't exist.
    Returns True if successful, False otherwise.
    """
    try:
        # Connect to database and create a cursor
        with get_db_cursor(commit=True) as cursor:
            # Check if the table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'chats'
                );
            """)
            
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                logger.info("Creating chats table...")
                # Create the table
                # Note: PostgreSQL doesn't have UNSIGNED INT type, using SERIAL instead
                cursor.execute("""
                    CREATE TABLE chats (
                        chat_id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL CHECK (user_id > 0),
                        contents TEXT NOT NULL
                    );
                """)
                logger.info("Chats table created successfully")
            else:
                logger.info("Chats table already exists")
                
        return True
    except Exception as e:
        logger.error(f"Error creating chats table: {e}")
        return False

def initialize_database() -> bool:
    """
    Initialize the database connection and create necessary tables.
    Returns True if successful, False otherwise.
    """
    try:
        # Check if pool already exists in state
        pool = get_state('db_connection_pool')
        
        if not pool:
            # Start the connection pool
            pool = start_connection_pool()
            if not pool:
                logger.error("Failed to initialize database connection pool")
                return False
            
            # Store the pool in global state
            set_state('db_connection_pool', pool)
            
        # Create the chats table
        if not create_chat_table():
            return False
            
        logger.info("Database initialization completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False
