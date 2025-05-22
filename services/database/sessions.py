import uuid
import datetime
import logging
from typing import Optional, Union
from sqlalchemy.orm import Session as SQLAlchemySession
from services.database.connection import get_db_session
from .models import Session as SessionModel

logger = logging.getLogger(__name__)


def convert_to_uuid(session_token: Union[str, uuid.UUID]) -> Optional[uuid.UUID]:
    """Convert a session token string to a UUID object."""
    try:
        if isinstance(session_token, str):
            return uuid.UUID(session_token)
        elif isinstance(session_token, uuid.UUID):
            return session_token
        else:
            logger.error(f"Invalid session token type: {type(session_token)}")
            return None
    except ValueError:
        logger.error(f"Invalid session token format: {session_token}")
        return None


def find_session_by_uuid(session_uuid: uuid.UUID, db_session: SQLAlchemySession) -> Optional[SessionModel]:
    """Find a session record by UUID."""
    logger.debug(f"Querying for session ID: {session_uuid}")
    return db_session.query(SessionModel).filter(SessionModel.session_id == session_uuid).first()


def is_session_expired(session_record: SessionModel) -> bool:
    """Check if a session has expired."""
    current_time = datetime.datetime.now()
    expires_at = session_record.expires_at
    is_expired = expires_at < current_time
    
    if is_expired:
        logger.warning(f"Session token has expired: {session_record.session_id} (Expired at: {expires_at})")
    
    return is_expired


def generate_session_token(user_id: int, session_duration_hours: int = 24) -> str:
    """
    Generate a new UUID session token for the user and store it in the database.
    Returns the session token as a string.
    """
    logger.info(f"Generating new session token for user ID: {user_id}")
    session_uuid = uuid.uuid4()
    expires_at = datetime.datetime.now() + datetime.timedelta(hours=session_duration_hours)
    
    logger.debug(f"Attempting to store session token in database. Session ID: {session_uuid}, Expires: {expires_at}")
    with get_db_session() as db_session:
        new_session_record = SessionModel(
            session_id=session_uuid,
            user_id=user_id,
            expires_at=expires_at
        )
        db_session.add(new_session_record)
        logger.debug(f"Session token stored successfully in database.")
    
    logger.info(f"Session token generated for user ID: {user_id}, Token: {str(session_uuid)}")
    return str(session_uuid)


def verify_session_token(session_token: Union[str, uuid.UUID]) -> Optional[int]:
    """
    Verify if a session token is valid and not expired.
    Returns the user_id if valid, None otherwise.
    """
    logger.info(f"Verifying session token: {session_token}")
    
    try:
        session_uuid = convert_to_uuid(session_token)
        if not session_uuid:
            return None
            
        with get_db_session() as db_session:
            session_record = find_session_by_uuid(session_uuid, db_session)
            
            if not session_record:
                logger.warning(f"Session token not found in database: {session_uuid}")
                return None
                
            if is_session_expired(session_record):
                # Optionally, delete expired session from DB
                # db_session.delete(session_record)
                return None
                
            user_id = session_record.user_id
            logger.info(f"Session token is valid for user ID: {user_id}")
            return user_id
    except Exception as e:
        logger.error(f"An unexpected error occurred during session verification: {e}", exc_info=True)
        return None
