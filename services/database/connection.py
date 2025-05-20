import os
import time
import psycopg2
import logging
from typing import Optional
from psycopg2.pool import ThreadedConnectionPool
from contextlib import contextmanager
from services.state import get_state, set_state
from .create_chat_table import initialize_chats_database
from .create_password_table import initialize_passwords_database
from .create_user_table import initialize_users_database

logger = logging.getLogger(__name__)

def validate_db_config() -> bool:
    required_vars = ["POSTGRES_DATABASE", "POSTGRES_HOST", "POSTGRES_USER", "POSTGRES_PASSWORD"]
    missing = [var for var in required_vars if not os.environ.get(var)]
   
    for var in required_vars:
        logger.info(f"Environment variable '{var}' is set to: {os.environ.get(var)}")

    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        return False
    return True

def get_existing_pool() -> Optional[ThreadedConnectionPool]:
    # Get from global state instead of module variable
    return get_state('db_connection_pool')

def create_connection_pool() -> Optional[ThreadedConnectionPool]:
    try:
        validate_db_config()
        
        pool = ThreadedConnectionPool(
            minconn=1, 
            maxconn=20,
            database=os.environ.get("POSTGRES_DATABASE"),
            host=os.environ.get("POSTGRES_HOST"),
            user=os.environ.get("POSTGRES_USER"),
            password=os.environ.get("POSTGRES_PASSWORD"),
            port=5432
        )
        logger.info("Database connection pool created successfully.")
        # Save to global state
        set_state('db_connection_pool', pool)
        return pool
    except psycopg2.Error as e:
        logger.error(f"Failed to create connection pool: {e}")
        return None

def retry_connection_creation(max_retries: int, retry_delay: int) -> Optional[ThreadedConnectionPool]:
    for attempt in range(1, max_retries + 1):
        logger.info(f"Attempting to create connection pool (attempt {attempt}/{max_retries})...")
        
        pool = create_connection_pool()
        if pool:
            return pool
        
        # Exit early on last attempt
        if attempt >= max_retries:
            break
            
        logger.warning(f"Connection attempt failed. Retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)
        
    logger.error(f"Database connection pool creation failed after {max_retries} attempts")
    return None

def start_connection_pool(max_retries: int = 3, retry_delay: int = 5) -> Optional[ThreadedConnectionPool]:
    existing_pool = get_existing_pool()
    if existing_pool is not None:
        logger.info("Using existing database connection pool.")
        return existing_pool
        
    if not validate_db_config():
        logger.error("Database configuration validation failed. Cannot start connection pool.")
        return None
    
    return retry_connection_creation(max_retries, retry_delay)

@contextmanager
def get_db_connection():
    _pool = get_existing_pool()
    
    if _pool is None:
        logger.info("No existing connection pool found. Starting a new one...")
        _pool = create_connection_pool()
        
        
    conn = None
    try:
        conn = _pool.getconn()
        if conn is None:
            raise psycopg2.Error("Failed to get a connection from the pool")
        yield conn
    finally:
        if conn and _pool:
            logger.info("Returning connection to the pool")
            _pool.putconn(conn)

@contextmanager
def get_db_cursor(commit=False):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            yield cursor
            if commit:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
            
def check_database_table_presence() -> bool:
    with get_db_cursor() as cursor:
        cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'users');")
        users_table_exists = cursor.fetchone()[0]
        
        cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'passwords');")
        passwords_table_exists = cursor.fetchone()[0]
        
        cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'chats');")
        chats_table_exists = cursor.fetchone()[0]

        return users_table_exists and passwords_table_exists and chats_table_exists
    
def initiate_tables():
    try:
        with get_db_cursor(commit=True) as cursor:
            if not initialize_users_database(cursor):
                logger.error("Failed to create users table.")
                return False
            if not initialize_passwords_database(cursor):
                logger.error("Failed to create passwords table.")
                return False
            if not initialize_chats_database(cursor):
                logger.error("Failed to create chats table.")
                return False
            logger.info("All tables created successfully.")
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        return False
    return True