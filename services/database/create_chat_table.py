#================================
#=====THIS FILE AI GENERATED===== 
#================================
import logging

logger = logging.getLogger(__name__)

def check_column_exists(cursor, table, column):
    """Check if a column exists in a table"""
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.columns 
            WHERE table_name = %s AND column_name = %s
        );
    """, (table, column))
    return cursor.fetchone()[0]

def create_chat_table(cursor) -> bool:
    """
    Create the chats table if it doesn't exist.
    Returns True if successful, False otherwise.
    """
    try:
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
            # Create the table with the full schema
            cursor.execute("""
                CREATE TABLE chats (
                    chat_id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL CHECK (user_id > 0),
                    chat_name TEXT NOT NULL,
                    contents TEXT NOT NULL,
                    last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            logger.info("Chats table created successfully")
        else:
            logger.info("Chats table already exists")
            
            # Check if chat_name column exists and add it if missing
            if not check_column_exists(cursor, 'chats', 'chat_name'):
                logger.info("Adding missing chat_name column to chats table")
                cursor.execute("""
                    ALTER TABLE chats 
                    ADD COLUMN chat_name TEXT NOT NULL DEFAULT 'Untitled Chat';
                """)
                logger.info("Added chat_name column successfully")
            
            # Check if contents column exists and add it if missing
            if not check_column_exists(cursor, 'chats', 'contents'):
                logger.info("Adding missing contents column to chats table")
                cursor.execute("""
                    ALTER TABLE chats 
                    ADD COLUMN contents TEXT NOT NULL DEFAULT '';
                """)
                logger.info("Added contents column successfully")
            
            # Check if last_modified column exists and add it if missing
            if not check_column_exists(cursor, 'chats', 'last_modified'):
                logger.info("Adding missing last_modified column to chats table")
                cursor.execute("""
                    ALTER TABLE chats 
                    ADD COLUMN last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
                """)
                logger.info("Added last_modified column successfully")
                
        return True
    except Exception as e:
        logger.error(f"Error creating or updating chats table: {e}")
        return False

def create_chat_keys_table(cursor) -> bool:
    """
    Create the chat_keys table if it doesn't exist.
    Returns True if successful, False otherwise.
    """
    try:
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'chat_keys'
            );
        """)
        
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            logger.info("Creating chat_keys table...")
            cursor.execute("""
                CREATE TABLE chat_keys (
                    chat_id INTEGER PRIMARY KEY,
                    encrypted_key TEXT NOT NULL,
                    iv TEXT NOT NULL,
                    FOREIGN KEY (chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE
                );
            """)
            logger.info("Chat_keys table created successfully")
        else:
            logger.info("Chat_keys table already exists")
            
        return True
    except Exception as e:
        logger.error(f"Error creating chat_keys table: {e}")
        return False

def initialize_chats_database(cursor) -> bool:
    """
    Initialize the database connection and create necessary tables.
    Returns True if successful, False otherwise.
    """
    try:
        if not create_chat_table(cursor):
            return False
            
        if not create_chat_keys_table(cursor):
            return False
            
        logger.info("Chats database initialization completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing chats database: {e}")
        return False
