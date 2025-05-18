#================================
#=====THIS FILE AI GENERATED===== 
#================================


import os
import logging
import getpass
from dotenv import load_dotenv
from .connection import start_connection_pool, get_db_cursor
from services.state import get_state, set_state

# Import initialization functions to recreate tables after reset
from .create_user_table import create_users_table
from .create_chat_database import create_chat_table
from .passwords import create_passwords_table

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
    Drop all application tables from the database.
    Returns True if successful, False otherwise.
    """
    try:
        database_name = os.environ.get("POSTGRES_DATABASE")
        if not database_name:
            logger.error("POSTGRES_DATABASE not set in environment variables")
            return False
            
        with get_db_cursor(commit=True) as cursor:
            logger.info("Dropping all tables...")
            
            # First, get a list of all tables in our database
            cursor.execute("""
                SELECT tablename FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY tablename;
            """)
            
            tables = [row[0] for row in cursor.fetchall()]
            logger.info(f"Found tables: {', '.join(tables)}")
            
            # Then get table dependencies through foreign keys
            cursor.execute("""
                SELECT
                    tc.table_name,
                    ccu.table_name AS foreign_table_name
                FROM 
                    information_schema.table_constraints AS tc 
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY';
            """)
            
            dependencies = {}
            for row in cursor.fetchall():
                table, foreign_table = row
                if table not in dependencies:
                    dependencies[table] = []
                dependencies[table].append(foreign_table)
                
            # Sort tables to drop them in proper order
            ordered_tables = []
            visited = set()
            
            def visit(table):
                if table in visited:
                    return
                visited.add(table)
                # First visit tables that depend on this one
                for t in tables:
                    if t in dependencies and table in dependencies[t]:
                        visit(t)
                ordered_tables.append(table)
            
            for table in tables:
                visit(table)
                
            # Drop tables in proper order
            for table in ordered_tables:
                logger.info(f"Dropping table: {table}")
                cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
                
            logger.info("All tables dropped successfully")
        return True
    except Exception as e:
        logger.error(f"Error dropping tables: {e}")
        return False

def recreate_tables():
    """
    Recreate all application tables.
    Returns True if successful, False otherwise.
    """
    try:
        # Recreate tables in proper order
        if not create_users_table():
            return False
            
        if not create_passwords_table():
            return False
            
        if not create_chat_table():
            return False
            
        logger.info("All tables recreated successfully")
        return True
    except Exception as e:
        logger.error(f"Error recreating tables: {e}")
        return False

def reset_database():
    """
    Reset the database by dropping and recreating all tables.
    Requires password verification unless in debug mode.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Database reset requested")
    
    # Check permission
    if not check_reset_permission():
        logger.error("Permission denied: Incorrect password or missing configuration")
        return False
    
    # Initialize database connection
    pool = get_state('db_connection_pool')
    if not pool:
        pool = start_connection_pool()
        if not pool:
            logger.error("Failed to connect to database")
            return False
        set_state('db_connection_pool', pool)
    
    # Drop all tables
    if not drop_all_tables():
        return False
    
    # Recreate tables
    if not recreate_tables():
        return False
    
    logger.info("Database reset completed successfully")
    return True
