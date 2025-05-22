import os
import time
import psycopg2 # Keep for specific psycopg2.Error if needed, though SQLAlchemy has its own
import logging
from typing import Optional, Iterator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session as SQLAlchemySession # Alias Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
from services.state import get_state, set_state
from .models import Base, User, PasswordEntry, Chat, ChatKey, Session # Import all models and Base

logger = logging.getLogger(__name__)

def get_database_url() -> str:
    return f"postgresql+psycopg2://{os.environ.get('POSTGRES_USER')}:{os.environ.get('POSTGRES_PASSWORD')}@{os.environ.get('POSTGRES_HOST')}:5432/{os.environ.get('POSTGRES_DATABASE')}"

def validate_db_config() -> bool:
    required_vars = ["POSTGRES_DATABASE", "POSTGRES_HOST", "POSTGRES_USER", "POSTGRES_PASSWORD"]
    missing = [var for var in required_vars if not os.environ.get(var)]
   
    for var in required_vars:
        logger.info(f"Environment variable '{var}' is set to: {os.environ.get(var)}")

    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        return False
    return True

def select_all_from_table(table_name: str) -> Optional[list]:
    """
    Select all rows from a given table using SQLAlchemy.
    Returns a list of rows if successful, None otherwise.
    """
    with get_db_session() as session:
        try:
            # For a generic function, raw SQL is often simplest.
            # Alternatively, map table_name to Model class if a limited set of tables.
            result = session.execute(text(f"SELECT * FROM {table_name};"))
            rows = result.fetchall() # Returns list of Row objects
            logger.info(f"Fetched {len(rows)} rows from {table_name}.")
            return [row._asdict() for row in rows] # Convert to list of dicts for easier use
        except SQLAlchemyError as e:
            logger.error(f"Error selecting from {table_name}: {e}")
            return None

def get_existing_engine() -> Optional[any]: # sqlalchemy.engine.Engine
    return get_state('db_engine')

def create_db_engine() -> Optional[any]: # sqlalchemy.engine.Engine
    if not validate_db_config():
        logger.error("Database configuration validation failed. Cannot create engine.")
        return None
    try:
        db_url = get_database_url()
        engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=5,        # Default for QueuePool
            max_overflow=10,    # Default for QueuePool
            pool_timeout=30,    # Default for QueuePool
            echo=os.environ.get("SQLALCHEMY_ECHO", "False").lower() == "true" # Optional: echo SQL
        )
        logger.info("Database engine created successfully.")
        set_state('db_engine', engine)
        return engine
    except Exception as e: # Catch broader exceptions during create_engine
        logger.error(f"Failed to create database engine: {e}")
        return None

def retry_engine_creation(max_retries: int, retry_delay: int) -> Optional[any]: # sqlalchemy.engine.Engine
    for attempt in range(1, max_retries + 1):
        logger.info(f"Attempting to create database engine (attempt {attempt}/{max_retries})...")
        
        engine = create_db_engine()
        if engine:
            try:
                # Test the connection
                with engine.connect() as connection:
                    logger.info("Database engine connection test successful.")
                return engine
            except SQLAlchemyError as e: # Catch SQLAlchemy specific connection errors
                logger.warning(f"Engine created, but connection test failed: {e}")
                set_state('db_engine', None) # Clear potentially bad engine

        if attempt >= max_retries:
            break
            
        logger.warning(f"Engine creation attempt failed. Retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)
        
    logger.error(f"Database engine creation failed after {max_retries} attempts")
    return None

def start_engine(max_retries: int = 3, retry_delay: int = 5) -> Optional[any]: # sqlalchemy.engine.Engine
    existing_engine = get_existing_engine()
    if existing_engine is not None:
        logger.info("Using existing database engine.")
        return existing_engine
    
    return retry_engine_creation(max_retries, retry_delay)

@contextmanager
def get_db_session() -> Iterator[SQLAlchemySession]: # Correct type hint for generator
    engine = get_existing_engine()
    if engine is None:
        logger.info("No existing engine found. Attempting to start a new one...")
        engine = start_engine()
        if engine is None:
            # Log and raise a more specific error if engine cannot be started
            err_msg = "Failed to start database engine for session."
            logger.critical(err_msg)
            raise RuntimeError(err_msg)

    # Create a SessionLocal class if it doesn't exist or engine changed.
    # This is typically done once.
    SessionLocal = get_state('db_session_local')
    if SessionLocal is None or SessionLocal.kw['bind'] != engine:
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        set_state('db_session_local', SessionLocal)
        
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except SQLAlchemyError: # Catch SQLAlchemy specific errors
        session.rollback()
        logger.error("SQLAlchemy error occurred, transaction rolled back.", exc_info=True)
        raise
    except Exception: # Catch other exceptions
        session.rollback()
        logger.error("Generic error occurred, transaction rolled back.", exc_info=True)
        raise
    finally:
        session.close()
            
def check_database_tables_presence() -> bool: # Renamed for clarity
    engine = get_existing_engine()
    if not engine:
        logger.error("Database engine not available for table check.")
        # Attempt to start engine if not available
        engine = start_engine()
        if not engine:
            logger.error("Failed to start database engine for table check.")
            return False
    
    from sqlalchemy import inspect # Local import
    inspector = inspect(engine)
    
    # All tables defined in models.py Base
    required_tables = [table.name for table in Base.metadata.tables.values()]
    
    try:
        existing_tables = inspector.get_table_names()
        all_present = True
        for table_name in required_tables:
            if table_name not in existing_tables:
                logger.warning(f"Table '{table_name}' not found in the database.")
                all_present = False
        
        if all_present:
            logger.info("All required tables are present.")
        return all_present
    except SQLAlchemyError as e: # Catch SQLAlchemy specific errors
        logger.error(f"Error checking table presence: {e}")
        return False
    
def initiate_tables():
    engine = get_existing_engine()
    if not engine:
        logger.error("Database engine not available for table initialization.")
        engine = start_engine()
        if not engine:
            logger.error("Failed to start database engine for table initialization.")
            return False
    try:
        logger.info("Initializing database tables (creating if they don't exist)...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully.")
        return True
    except SQLAlchemyError as e: # Catch SQLAlchemy specific errors
        logger.error(f"Error creating tables: {e}")
        return False
    except Exception as e: # Catch other potential errors
        logger.error(f"Unexpected error creating tables: {e}")
        return False

# The fix_chats_table_schema is complex with ORM. Schema migrations (e.g. Alembic)
# are preferred for altering existing tables. Base.metadata.create_all only creates.
# For simplicity, this function is removed. `upgrade_database_schema` will just ensure creation.

def upgrade_database_schema():
    """
    Ensures all tables defined in models are created in the database.
    This does not handle complex migrations (e.g., altering columns on existing tables).
    For schema changes, a migration tool like Alembic is recommended.
    """
    logger.info("Checking and creating database tables if needed (does not alter existing tables)...")
    if not initiate_tables():
        logger.error("Failed to ensure database tables are created.")
        return False
    
    # Optionally, verify presence again, though initiate_tables should suffice
    if not check_database_tables_presence():
        logger.warning("Some tables might still be missing after initialization attempt.")
        return False
        
    logger.info("Database schema check/creation completed.")
    return True