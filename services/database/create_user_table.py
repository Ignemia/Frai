#================================
#=====THIS FILE AI GENERATED===== 
#================================
import logging

logger = logging.getLogger(__name__)

def create_users_table(cursor) -> bool:
    """
    Create the users table if it doesn't exist.
    Returns True if successful, False otherwise.
    """
    try:
        # Connect to database and create a cursor
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

def initialize_users_database(cursor) -> bool:
    """
    Initialize the database connection and create users table.
    Returns True if successful, False otherwise.
    """
    try:
        # Create the users table
        if not create_users_table(cursor):
            return False
            
        logger.info("Users database initialization completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing users database: {e}")
        return False
