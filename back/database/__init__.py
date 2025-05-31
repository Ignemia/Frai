from back.database.chats import initiate_chat_database, test_chat_database

def initiate_database_connection():
    """Initialize the database connection."""
    print("Database connection initialized.")
    
    # Initialize chat database
    if not initiate_chat_database():
        print("Failed to initialize chat database.")
        return False
    
    return True

def test_database_connection():
    """Test the database connection."""
    print("Database connection tested successfully.")
    
    # Test chat database
    if not test_chat_database():
        print("Chat database test failed.")
        return False
    
    return True