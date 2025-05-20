#================================
#=====THIS FILE AI GENERATED===== 
#================================
import logging

logger = logging.getLogger(__name__)

def create_passwords_table(cursor) -> bool:
    """
    Create the passwords table if it doesn't exist.
    Returns True if successful, False otherwise.
    """
    try:
        # Connect to database and create a cursor
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

def initialize_passwords_database(cursor) -> bool:
    """
    Initialize the database connection and create passwords table.
    Returns True if successful, False otherwise.
    """
    try:
            
        # Create the passwords table
        if not create_passwords_table(cursor):
            return False
            
        logger.info("Passwords database initialization completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing passwords database: {e}")
        return False
