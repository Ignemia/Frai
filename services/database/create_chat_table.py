#================================
#=====THIS FILE AI GENERATED===== 
#================================
import logging

logger = logging.getLogger(__name__)

def create_chat_table(cursor) -> bool:
    """
    Create the chats table if it doesn't exist.
    Returns True if successful, False otherwise.
    """
    try:
        # Connect to database and create a cursor
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

def initialize_chats_database(cursor) -> bool:
    """
    Initialize the database connection and create necessary tables.
    Returns True if successful, False otherwise.
    """
    try:
        # Check if pool already exists in state
        if not create_chat_table(cursor):
            return False
            
        logger.info("Database initialization completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False
