from services.database.connection import get_db_cursor, select_all_from_table
import datetime
import logging

logger = logging.getLogger(__name__)


def user_exists(username):
    logger.info(f"Checking if user {username} exists.")
    logger.debug(f"Attempting to get a database cursor.")
    with get_db_cursor() as cursor:
        logger.debug(f"Got a database cursor. Executing query to check user existence.")
        cursor.execute(
            "SELECT EXISTS(SELECT 1 FROM users WHERE name = %s);", (username,)
        )
        
        logger.debug(f"Executed query to check user existence. Fetching result.")
        return cursor.fetchone()[0]


def get_user_id(username):
    logger.info(f"Fetching user ID for username: {username}")
    if not user_exists(username):
        logger.error(f"User {username} does not exist.")
        return None
    logger.debug(f"User {username} exists. Proceeding to fetch user ID.")
    logger.debug(f"Attempting to get a database cursor.")
    with get_db_cursor() as cursor:
        logger.debug(f"Got a database cursor. Executing query to fetch user ID.")
        cursor.execute("SELECT user_id FROM users WHERE name = %s;", (username,))
        
        logger.debug(f"Executed query to fetch user ID. Fetching result.")
        result = cursor.fetchone()
        
        logger.debug(f"Fetched result: {result}.")
        return result[0] if result else None


def register_user(username, password, expire_days=90):
    logger.info(f"Registering user: {username}")
    if user_exists(username):
        logger.error(f"User {username} already exists.")
        return None

    logger.info(f"User {username} does not exist. Proceeding with registration.")

    logger.debug(f"Attempting to get a database cursor.")
    with get_db_cursor() as cursor:
        logger.debug(
            f"Got a database cursor. Inserting user {username} into the database."
        )
        cursor.execute(
            "INSERT INTO users (name) VALUES (%s) RETURNING user_id;", (username,)
        )
        
        logger.debug(f"Inserted user {username} into the database. Fetching user ID.")
        user_id = cursor.fetchone()[0]
        logger.debug(
            f"User ID for {username} is {user_id}. Proceeding to insert password."
        )
        expire_date = datetime.datetime.now() + datetime.timedelta(days=expire_days)
        
        logger.debug(f"Setting password expiration date to {expire_date}.")
        logger.debug(f"Inserting password for user {username} into the database.")
        cursor.execute(
            "INSERT INTO passwords (user_id, hashed_password, expire_date) VALUES (%s, %s, %s);",
            (user_id, password, expire_date),
        )
        logger.debug(f"Inserted password for user {username} into the database.")
        
        # Commit the transaction to ensure the data is persisted
        cursor.connection.commit()
        logger.debug(f"Committed transaction to the database.")
        
        logger.info(f"User {username} registered successfully with ID {user_id}.")
        select_all_from_table("users")
        select_all_from_table("passwords")
        return user_id
