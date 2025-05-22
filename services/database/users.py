from typing import Optional, Tuple
from sqlalchemy.orm import Session as SQLAlchemySession
from services.database.connection import get_db_session, select_all_from_table
from .models import User, PasswordEntry
import datetime
import logging

logger = logging.getLogger(__name__)


def find_user_by_username(username: str, session: SQLAlchemySession) -> Optional[User]:
    """Fetch a user by username."""
    logger.debug(f"Executing query to find user: {username}")
    return session.query(User).filter(User.name == username).first()


def find_user_by_id(user_id: int, session: SQLAlchemySession) -> Optional[User]:
    """Fetch a user by ID."""
    logger.debug(f"Executing query to find user_id: {user_id}")
    return session.query(User).filter(User.user_id == user_id).first()


def find_password_entry(user_id: int, session: SQLAlchemySession) -> Optional[PasswordEntry]:
    """Fetch the password entry for a user."""
    logger.debug(f"Executing query to find password for user_id: {user_id}")
    return session.query(PasswordEntry).filter(PasswordEntry.user_id == user_id).first()


def user_exists(username: str) -> bool:
    """Check if a user exists in the database."""
    logger.info(f"Checking if user {username} exists")
    
    with get_db_session() as session:
        user = find_user_by_username(username, session)
        exists = user is not None
        logger.debug(f"User exists: {exists}")
        return exists


def get_user_id(username: str) -> Optional[int]:
    """Retrieve user ID for a given username."""
    logger.info(f"Fetching user ID for username: {username}")
    
    with get_db_session() as session:
        user = find_user_by_username(username, session)
        
        if not user:
            logger.error(f"User {username} does not exist")
            return None
            
        logger.debug(f"User {username} found with ID {user.user_id}")
        return user.user_id


def create_user(username: str, session: SQLAlchemySession) -> Optional[User]:
    """Create a new user record."""
    logger.debug(f"Creating new user: {username}")
    
    new_user = User(name=username)
    session.add(new_user)
    
    try:
        session.flush()
        logger.debug(f"Created user with ID: {new_user.user_id}")
        return new_user
    except Exception as e:
        logger.error(f"Error creating user {username}: {e}")
        session.rollback()
        return None


def create_password_entry(
    user_id: int, 
    hashed_password: str, 
    expire_days: int, 
    session: SQLAlchemySession
) -> Optional[PasswordEntry]:
    """Create a new password entry for a user."""
    logger.debug(f"Creating password entry for user_id: {user_id}")
    
    expire_date = datetime.datetime.now() + datetime.timedelta(days=expire_days)
    logger.debug(f"Setting password expiration date to {expire_date}")
    
    password_entry = PasswordEntry(
        user_id=user_id,
        hashed_password=hashed_password,
        expire_date=expire_date
    )
    
    session.add(password_entry)
    return password_entry


def register_user(username: str, hashed_password: str, expire_days: int = 90) -> Optional[int]:
    """Register a new user with a password."""
    logger.info(f"Registering user: {username}")
    
    with get_db_session() as session:
        # Check if user already exists
        if find_user_by_username(username, session):
            logger.error(f"User {username} already exists")
            return None
            
        # Create user
        new_user = create_user(username, session)
        if not new_user:
            return None
            
        # Create password entry
        password_entry = create_password_entry(
            new_user.user_id, 
            hashed_password, 
            expire_days, 
            session
        )
        if not password_entry:
            logger.error(f"Failed to create password entry for user {username}")
            return None
            
        logger.info(f"User {username} registered successfully with ID {new_user.user_id}")
        return new_user.user_id


def get_user_credentials_for_key_derivation(user_id: int, session: SQLAlchemySession) -> Optional[Tuple[str, str]]:
    """
    Fetches the username and hashed password for a given user_id.
    Used for deriving encryption keys based on user credentials.
    """
    logger.debug(f"Fetching credentials for key derivation for user_id: {user_id}")
    
    user = find_user_by_id(user_id, session)
    if not user:
        logger.error(f"User not found for user_id: {user_id}")
        return None

    password_entry = find_password_entry(user_id, session)
    if not password_entry:
        logger.error(f"Password entry not found for user_id: {user_id}")
        return None
    
    logger.debug(f"Successfully fetched username and password hash for user_id: {user_id}")
    return user.name, password_entry.hashed_password
