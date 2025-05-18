#================================
#=====THIS FILE AI GENERATED===== 
#================================
import logging
from typing import Optional
from .connection import start_connection_pool, get_db_cursor
from services.state import get_state, set_state

logger = logging.getLogger(__name__)

def create_passwords_table() -> bool:
    """
    Create the passwords table if it doesn't exist.
    Returns True if successful, False otherwise.
    """
    try:
        # Connect to database and create a cursor
        with get_db_cursor(commit=True) as cursor:
            # Check if the table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'passwords'
                );
            """)
            
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                logger.info("Creating passwords table...")
                # Create the table
                # SHA-256 hashes are 64 characters long in hex representation
                cursor.execute("""
                    CREATE TABLE passwords (
                        user_id INTEGER PRIMARY KEY,
                        hashed_password VARCHAR(64) NOT NULL,
                        expire_date TIMESTAMP NOT NULL,
                        FOREIGN KEY (user_id) REFERENCES users(user_id)
                    );
                """)
                logger.info("Passwords table created successfully")
            else:
                logger.info("Passwords table already exists")
                
        return True
    except Exception as e:
        logger.error(f"Error creating passwords table: {e}")
        return False

def initialize_passwords_database() -> bool:
    """
    Initialize the database connection and create passwords table.
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
            
        # Create the passwords table
        if not create_passwords_table():
            return False
            
        logger.info("Passwords database initialization completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing passwords database: {e}")
        return False
