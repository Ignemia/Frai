from services.database.connection import get_db_session, select_all_from_table # select_all_from_table might be used for debugging
from .models import User, PasswordEntry
import datetime
import logging

logger = logging.getLogger(__name__)


def user_exists(username):
    logger.info(f"Checking if user {username} exists.")
    logger.debug(f"Attempting to get a database session.")
    with get_db_session() as session:
        logger.debug(f"Got a database session. Executing query to check user existence.")
        user = session.query(User).filter(User.name == username).first()
        
        logger.debug(f"Executed query to check user existence. User found: {user is not None}.")
        return user is not None


def get_user_id(username):
    logger.info(f"Fetching user ID for username: {username}")
    # No need to call user_exists separately, query directly
    logger.debug(f"Attempting to get a database session.")
    with get_db_session() as session:
        logger.debug(f"Got a database session. Executing query to fetch user.")
        user = session.query(User).filter(User.name == username).first()
        
        if user:
            logger.debug(f"User {username} exists with ID {user.user_id}.")
            return user.user_id
        else:
            logger.error(f"User {username} does not exist.")
            return None


def register_user(username, password, expire_days=90): # 'password' is assumed to be pre-hashed
    logger.info(f"Registering user: {username}")
    
    with get_db_session() as session:
        logger.debug(f"Got a database session. Checking if user {username} already exists.")
        existing_user = session.query(User).filter(User.name == username).first()
        if existing_user:
            logger.error(f"User {username} already exists.")
            return None

        logger.info(f"User {username} does not exist. Proceeding with registration.")

        new_user = User(name=username)
        session.add(new_user)
        
        # Flush to get the auto-generated user_id before creating PasswordEntry
        # This is necessary because PasswordEntry needs user_id
        try:
            session.flush() 
        except Exception as e:
            logger.error(f"Error flushing session to get user_id for {username}: {e}")
            session.rollback() # Rollback on flush error
            return None

        logger.debug(
            f"User {username} added to session, user_id is {new_user.user_id}. Proceeding to insert password."
        )
        expire_date = datetime.datetime.now() + datetime.timedelta(days=expire_days)
        
        logger.debug(f"Setting password expiration date to {expire_date}.")
        new_password_entry = PasswordEntry(
            user_id=new_user.user_id, 
            hashed_password=password, # 'password' is the hashed password
            expire_date=expire_date
        )
        session.add(new_password_entry)
        
        # Commit is handled by the get_db_session context manager
        
        logger.info(f"User {username} registered successfully with ID {new_user.user_id}.")
        # Debugging calls, ensure select_all_from_table is SQLAlchemy compatible if used
        # select_all_from_table("users")
        # select_all_from_table("passwords")
        return new_user.user_id


def get_user_credentials_for_key_derivation(user_id: int, session: 'SQLAlchemySession') -> Optional[tuple[str, str]]:
    """
    Fetches the username and hashed password for a given user_id.
    Used for deriving encryption keys based on user credentials.
    """
    logger.debug(f"Fetching credentials for key derivation for user_id: {user_id}")
    user = session.query(User).filter(User.user_id == user_id).first()
    if not user:
        logger.error(f"User not found for user_id: {user_id}")
        return None

    password_entry = session.query(PasswordEntry).filter(PasswordEntry.user_id == user_id).first()
    if not password_entry:
        logger.error(f"Password entry not found for user_id: {user_id}")
        return None
    
    logger.debug(f"Successfully fetched username and password hash for user_id: {user_id}")
    return user.name, password_entry.hashed_password
