from services.database.connection import get_db_session
from .models import User, PasswordEntry # Import SQLAlchemy models
import datetime
import logging

logger = logging.getLogger(__name__)


def get_password_hash(user_id):
    logger.info(f"Fetching password hash for user ID: {user_id}")
    logger.debug(f"Attempting to get a database session.")
    with get_db_session() as session:
        logger.debug(f"Got a database session. Executing query to fetch password entry.")
        password_entry = session.query(PasswordEntry).filter(PasswordEntry.user_id == user_id).first()
        
        if password_entry:
            logger.debug(f"Password hash found for user ID: {user_id}")
            return password_entry.hashed_password
        else:
            logger.warning(f"No password hash found for user ID: {user_id}")
            return None


def update_password(user_id, new_password_hash, expire_days=90):
    logger.info(f"Updating password for user ID: {user_id}")
    expire_date = datetime.datetime.now() + datetime.timedelta(days=expire_days)
    logger.debug(f"Setting password expiration date to {expire_date}")
    
    logger.debug(f"Attempting to get a database session.")
    with get_db_session() as session:
        logger.debug(f"Got a database session. Checking if password entry exists for user ID: {user_id}")
        password_entry = session.query(PasswordEntry).filter(PasswordEntry.user_id == user_id).first()
        
        if password_entry:
            logger.debug(f"Password entry exists for user ID: {user_id}. Updating password.")
            password_entry.hashed_password = new_password_hash
            password_entry.expire_date = expire_date
            logger.debug(f"Password updated for user ID: {user_id}")
        else:
            logger.debug(f"No password entry exists for user ID: {user_id}. Inserting new password entry.")
            new_password_entry = PasswordEntry(
                user_id=user_id, 
                hashed_password=new_password_hash, 
                expire_date=expire_date
            )
            session.add(new_password_entry)
            logger.debug(f"New password entry inserted for user ID: {user_id}")
        
        # Commit is handled by the get_db_session context manager
        # session.commit() is called by the context manager on successful exit
        
        # SQLAlchemy ORM operations don't return rowcount directly like cursor.rowcount.
        # Success is typically assumed if no exception is raised.
        logger.info(f"Password successfully updated/inserted for user ID: {user_id}")
        return True # Return True on success


def verify_credentials(username, password_hash):
    logger.info(f"Verifying credentials for username: {username}")
    # No need to import get_user_id from users, can query User model directly
    
    logger.debug(f"Attempting to get a database session.")
    with get_db_session() as session:
        logger.debug(f"Got a database session. Fetching user by username: {username}")
        user = session.query(User).filter(User.name == username).first()
        
        if not user:
            logger.warning(f"User not found for username: {username}")
            return False
        
        logger.debug(f"User ID for username {username} is {user.user_id}. Verifying password.")
        
        password_entry = session.query(PasswordEntry).filter(PasswordEntry.user_id == user.user_id).first()
        
        if not password_entry:
            logger.warning(f"No password found for user ID: {user.user_id}")
            return False
            
        stored_hash = password_entry.hashed_password
        expire_date = password_entry.expire_date
        logger.debug(f"Found password for user ID: {user.user_id}. Stored hash: {stored_hash[:5]}..., Expire date: {expire_date}")
        
        logger.debug(f"Checking if password is expired. Current time: {datetime.datetime.now()}")
        if expire_date < datetime.datetime.now():
            logger.warning(f"Password for user ID: {user.user_id} is expired (Expired: {expire_date})")
            return False
        
        logger.debug(f"Password is not expired. Comparing provided hash with stored hash.")
        matches = stored_hash == password_hash
        
        if matches:
            logger.debug(f"Password hash matches for user ID: {user.user_id}")
            logger.info(f"Credentials verified successfully for username: {username}")
        else:
            logger.debug(f"Password hash does not match for user ID: {user.user_id}")
            logger.warning(f"Invalid password provided for username: {username}")
        return matches


def is_password_expired(user_id):
    logger.info(f"Checking if password is expired for user ID: {user_id}")
    logger.debug(f"Attempting to get a database session.")
    with get_db_session() as session:
        logger.debug(f"Got a database session. Executing query to fetch password entry.")
        password_entry = session.query(PasswordEntry).filter(PasswordEntry.user_id == user_id).first()
        
        if not password_entry:
            logger.warning(f"No password found for user ID: {user_id}. Considering as expired.")
            return True # If no password record, treat as expired or invalid
            
        expire_date = password_entry.expire_date
        is_expired = expire_date < datetime.datetime.now()
        
        if is_expired:
            logger.warning(f"Password for user ID: {user_id} is expired (Expired: {expire_date})")
        else:
            logger.debug(f"Password for user ID: {user_id} is valid until {expire_date}")
        
        return is_expired


def request_password():
    logger.info("Requesting password from user")
    import getpass
    password = getpass.getpass("Please enter your password: ")
    
    if not password:
        logger.error("Empty password provided")
        raise ValueError("Password cannot be empty")
    
    logger.debug("Password successfully received from user")
    return password
