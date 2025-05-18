import os
import time
import psycopg2
import logging
from typing import Optional
from psycopg2.pool import ThreadedConnectionPool
from contextlib import contextmanager
from services.state import get_state, set_state

logger = logging.getLogger(__name__)

def validate_db_config() -> bool:
    required_vars = ["POSTGRES_DATABASE", "POSTGRES_HOST", "POSTGRES_USER", "POSTGRES_PASSWORD"]
    missing = [var for var in required_vars if not os.environ.get(var)]
    
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        return False
    return True

def get_existing_pool() -> Optional[ThreadedConnectionPool]:
    # Get from global state instead of module variable
    return get_state('db_connection_pool')

def create_connection_pool() -> Optional[ThreadedConnectionPool]:
    try:
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
        return existing_pool
        
    if not validate_db_config():
        return None
    
    return retry_connection_creation(max_retries, retry_delay)

@contextmanager
def get_db_connection():
    _pool = get_existing_pool()
    
    if _pool is None:
        raise RuntimeError("Database connection pool not initialized")
        
    conn = None
    try:
        conn = _pool.getconn()
        yield conn
    finally:
        if conn and _pool:
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