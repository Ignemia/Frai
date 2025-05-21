import logging

logger = logging.getLogger(__name__)

def initialize_sessions_database(cursor):
    try:
        logger.info("Creating sessions table if it doesn't exist")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id UUID PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                CONSTRAINT fk_user
                    FOREIGN KEY(user_id)
                    REFERENCES users(user_id)
            );
        """)
        logger.info("Sessions table created or already exists")
        return True
    except Exception as e:
        logger.error(f"Error creating sessions table: {e}")
        return False
