#================================
#=====THIS FILE AI GENERATED===== 
#================================
import logging
from typing import Optional
from .connection import start_connection_pool, get_db_cursor
from services.state import get_state, set_state
from .passwords import create_passwords_table

logger = logging.getLogger(__name__)

def create_users_table() -> bool:
    """
    Create the users table if it doesn't exist.
    Returns True if successful, False otherwise.
    """
    try:
        # Connect to database and create a cursor
        with get_db_cursor(commit=True) as cursor:
            # Check if the table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'users'
                );
            """)
            
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                logger.info("Creating users table...")
                # Create the table with CHECK constraint to ensure positive values (unsigned)
                cursor.execute("""
                    CREATE TABLE users (
                        user_id SERIAL PRIMARY KEY CHECK (user_id > 0),
                        name TEXT NOT NULL
                    );
                """)
                logger.info("Users table created successfully")
            else:
                logger.info("Users table already exists")
                
        return True
    except Exception as e:
        logger.error(f"Error creating users table: {e}")
        return False

def initialize_users_database() -> bool:
    """
    Initialize the database connection and create users table.
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
            
        # Create the users table
        if not create_users_table():
            return False
            
        # Create the passwords table (which depends on users table)
        if not create_passwords_table():
            logger.error("Failed to create passwords table")
            return False
            
        logger.info("Users and passwords database initialization completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing users database: {e}")
        return False
