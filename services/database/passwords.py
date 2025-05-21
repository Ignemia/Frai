from services.database.connection import get_db_cursor
import datetime
import logging

logger = logging.getLogger(__name__)


def get_password_hash(user_id):
    logger.info(f"Fetching password hash for user ID: {user_id}")
    logger.debug(f"Attempting to get a database cursor.")
    with get_db_cursor() as cursor:
        logger.debug(f"Got a database cursor. Executing query to fetch password hash.")
        cursor.execute("SELECT hashed_password FROM passwords WHERE user_id = %s;", (user_id,))
        
        logger.debug(f"Executed query to fetch password hash. Fetching result.")
        result = cursor.fetchone()
        
        if result:
            logger.debug(f"Password hash found for user ID: {user_id}")
            return result[0]
        else:
            logger.warning(f"No password hash found for user ID: {user_id}")
            return None


def update_password(user_id, new_password_hash, expire_days=90):
    logger.info(f"Updating password for user ID: {user_id}")
    expire_date = datetime.datetime.now() + datetime.timedelta(days=expire_days)
    logger.debug(f"Setting password expiration date to {expire_date}")
    
    logger.debug(f"Attempting to get a database cursor.")
    with get_db_cursor() as cursor:
        logger.debug(f"Got a database cursor. Checking if password already exists for user ID: {user_id}")
        cursor.execute("SELECT 1 FROM passwords WHERE user_id = %s", (user_id,))
        exists = cursor.fetchone() is not None
        
        if exists:
            logger.debug(f"Password exists for user ID: {user_id}. Updating password.")
            cursor.execute(
                "UPDATE passwords SET hashed_password = %s, expire_date = %s WHERE user_id = %s;", 
                (new_password_hash, expire_date, user_id)
            )
            logger.debug(f"Password updated for user ID: {user_id}")
        else:
            logger.debug(f"No password exists for user ID: {user_id}. Inserting new password.")
            cursor.execute(
                "INSERT INTO passwords (user_id, hashed_password, expire_date) VALUES (%s, %s, %s);",
                (user_id, new_password_hash, expire_date)
            )
            logger.debug(f"New password inserted for user ID: {user_id}")
        
        # Commit the transaction
        cursor.connection.commit()
        logger.debug(f"Committed transaction to the database.")
        
        success = cursor.rowcount > 0
        if success:
            logger.info(f"Password successfully updated for user ID: {user_id}")
        else:
            logger.warning(f"Password update had no effect for user ID: {user_id}")
        return success


def verify_credentials(username, password_hash):
    logger.info(f"Verifying credentials for username: {username}")
    from services.database.users import get_user_id
    
    logger.debug(f"Fetching user ID for username: {username}")
    user_id = get_user_id(username)
    if not user_id:
        logger.warning(f"User ID not found for username: {username}")
        return False
    
    logger.debug(f"User ID for username {username} is {user_id}. Verifying password.")
    logger.debug(f"Attempting to get a database cursor.")
    with get_db_cursor() as cursor:
        logger.debug(f"Got a database cursor. Executing query to fetch password details.")
        cursor.execute(
            "SELECT hashed_password, expire_date FROM passwords WHERE user_id = %s;", 
            (user_id,)
        )
        
        logger.debug(f"Executed query to fetch password details. Fetching result.")
        result = cursor.fetchone()
        
        if not result:
            logger.warning(f"No password found for user ID: {user_id}")
            return False
            
        stored_hash, expire_date = result
        logger.debug(f"Found password for user ID: {user_id}. Checking expiration.")
        
        logger.debug(f"Checking if password is expired. Expiration date: {expire_date}")
        if expire_date < datetime.datetime.now():
            logger.warning(f"Password for user ID: {user_id} is expired (Expired: {expire_date})")
            return False
        
        logger.debug(f"Password is valid. Comparing hashes.")
        matches = stored_hash == password_hash
        if matches:
            logger.debug(f"Password hash matches for user ID: {user_id}")
        else:
            logger.debug(f"Password hash does not match for user ID: {user_id}")
        if matches:
            logger.info(f"Credentials verified successfully for username: {username}")
        else:
            logger.warning(f"Invalid password provided for username: {username}")
        return matches


def is_password_expired(user_id):
    logger.info(f"Checking if password is expired for user ID: {user_id}")
    logger.debug(f"Attempting to get a database cursor.")
    with get_db_cursor() as cursor:
        logger.debug(f"Got a database cursor. Executing query to fetch password expiration date.")
        cursor.execute("SELECT expire_date FROM passwords WHERE user_id = %s;", (user_id,))
        
        logger.debug(f"Executed query to fetch password expiration date. Fetching result.")
        result = cursor.fetchone()
        
        if not result:
            logger.warning(f"No password found for user ID: {user_id}")
            return True
            
        expire_date = result[0]
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
