import uuid
import datetime
import logging
from services.database.connection import get_db_cursor

logger = logging.getLogger(__name__)

def generate_session_token(user_id, session_duration_hours=24):
    """
    Generate a new UUID session token for the user and store it in the database.
    Returns the session token as plain text.
    """
    logger.info(f"Generating new session token for user ID: {user_id}")
    session_id = uuid.uuid4()
    expires_at = datetime.datetime.now() + datetime.timedelta(hours=session_duration_hours)
    
    logger.debug(f"Attempting to store session token in database")
    with get_db_cursor() as cursor:
        cursor.execute(
            "INSERT INTO sessions (session_id, user_id, expires_at) VALUES (%s, %s, %s);",
            (str(session_id), user_id, expires_at)  # Convert UUID to string
        )
        cursor.connection.commit()
        logger.debug(f"Session token stored successfully")
    
    logger.info(f"Session token generated for user ID: {user_id}")
    return str(session_id)  # Return session_id as string

def verify_session_token(session_token):
    """
    Verify if a session token is valid and not expired.
    Returns the user_id if valid, None otherwise.
    """
    logger.info(f"Verifying session token")
    try:
        # Convert string to UUID object for comparison, but use string in query
        session_id_str = str(session_token)
        
        with get_db_cursor() as cursor:
            cursor.execute(
                "SELECT user_id, expires_at FROM sessions WHERE session_id = %s;",
                (session_id_str,)  # Use string representation
            )
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"Session token not found in database")
                return None
                
            user_id, expires_at = result
            
            if expires_at < datetime.datetime.now():
                logger.warning(f"Session token has expired")
                return None
                
            logger.info(f"Session token is valid for user ID: {user_id}")
            return user_id
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid session token format: {e}")
        return None
