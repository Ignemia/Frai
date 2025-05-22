import uuid # Standard Python UUID library
import datetime
import logging
from services.database.connection import get_db_session
from .models import Session as SessionModel # Alias to avoid name collision

logger = logging.getLogger(__name__)

def generate_session_token(user_id, session_duration_hours=24):
    """
    Generate a new UUID session token for the user and store it in the database.
    Returns the session token as a string.
    """
    logger.info(f"Generating new session token for user ID: {user_id}")
    session_uuid = uuid.uuid4() # This is a UUID object
    expires_at = datetime.datetime.now() + datetime.timedelta(hours=session_duration_hours)
    
    logger.debug(f"Attempting to store session token in database. Session ID: {session_uuid}, Expires: {expires_at}")
    with get_db_session() as db_session: # Renamed session variable
        new_session_record = SessionModel(
            session_id=session_uuid, # SQLAlchemy handles UUID object correctly for PG_UUID column
            user_id=user_id,
            expires_at=expires_at
            # created_at is default in model
        )
        db_session.add(new_session_record)
        # Commit is handled by the get_db_session context manager
        logger.debug(f"Session token stored successfully in database.")
    
    logger.info(f"Session token generated for user ID: {user_id}, Token: {str(session_uuid)}")
    return str(session_uuid) # Return as string

def verify_session_token(session_token):
    """
    Verify if a session token is valid and not expired.
    Returns the user_id if valid, None otherwise.
    """
    logger.info(f"Verifying session token: {session_token}")
    try:
        # Ensure session_token is a UUID object for querying if your column type expects it
        # If session_token is already a string, convert it.
        if isinstance(session_token, str):
            session_uuid = uuid.UUID(session_token)
        elif isinstance(session_token, uuid.UUID):
            session_uuid = session_token
        else:
            logger.error(f"Invalid session token type: {type(session_token)}")
            return None
            
        with get_db_session() as db_session: # Renamed session variable
            logger.debug(f"Querying for session ID: {session_uuid}")
            session_record = db_session.query(SessionModel).filter(SessionModel.session_id == session_uuid).first()
            
            if not session_record:
                logger.warning(f"Session token not found in database: {session_uuid}")
                return None
                
            user_id = session_record.user_id
            expires_at = session_record.expires_at
            
            logger.debug(f"Session record found for user {user_id}. Expires at: {expires_at}. Current time: {datetime.datetime.now()}")
            if expires_at < datetime.datetime.now():
                logger.warning(f"Session token has expired: {session_uuid} (Expired at: {expires_at})")
                # Optionally, delete expired session from DB
                # db_session.delete(session_record)
                return None
                
            logger.info(f"Session token is valid for user ID: {user_id}")
            return user_id
    except ValueError: # Handles error from uuid.UUID(session_token) if token is invalid format
        logger.error(f"Invalid session token format: {session_token}")
        return None
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred during session verification: {e}", exc_info=True)
        return None
