from services.database.connection import get_db_cursor
import datetime


def user_exists(username):
    with get_db_cursor() as cursor:
        cursor.execute("SELECT EXISTS(SELECT 1 FROM users WHERE name = %s);", (username,))
        return cursor.fetchone()[0]


def get_user_id(username):
    with get_db_cursor() as cursor:
        cursor.execute("SELECT user_id FROM users WHERE name = %s;", (username,))
        result = cursor.fetchone()
        return result[0] if result else None


def register_user(username, password, expire_days=90):
    with get_db_cursor() as cursor:
        # Insert the user first
        cursor.execute("INSERT INTO users (name) VALUES (%s) RETURNING user_id;", (username,))
        user_id = cursor.fetchone()[0]

        expire_date = datetime.datetime.now() + datetime.timedelta(days=expire_days)

        cursor.execute(
            "INSERT INTO passwords (user_id, hashed_password, expire_date) VALUES (%s, %s, %s);",
            (user_id, password, expire_date),
        )

        return user_id