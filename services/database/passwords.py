from services.database.connection import get_db_cursor
import datetime


def get_password_hash(user_id):
    with get_db_cursor() as cursor:
        cursor.execute("SELECT hashed_password FROM passwords WHERE user_id = %s;", (user_id,))
        result = cursor.fetchone()
        return result[0] if result else None


def update_password(user_id, new_password_hash, expire_days=90):
    expire_date = datetime.datetime.now() + datetime.timedelta(days=expire_days)
    
    with get_db_cursor() as cursor:

        cursor.execute("SELECT 1 FROM passwords WHERE user_id = %s", (user_id,))
        exists = cursor.fetchone() is not None
        
        if exists:
            cursor.execute(
                "UPDATE passwords SET hashed_password = %s, expire_date = %s WHERE user_id = %s;", 
                (new_password_hash, expire_date, user_id)
            )
        else:
            cursor.execute(
                "INSERT INTO passwords (user_id, hashed_password, expire_date) VALUES (%s, %s, %s);",
                (user_id, new_password_hash, expire_date)
            )
        
        return cursor.rowcount > 0


def verify_credentials(username, password_hash):
    from services.database.users import get_user_id
    
    user_id = get_user_id(username)
    if not user_id:
        return False
        
    with get_db_cursor() as cursor:
        cursor.execute(
            "SELECT hashed_password, expire_date FROM passwords WHERE user_id = %s;", 
            (user_id,)
        )
        result = cursor.fetchone()
        
        if not result:
            return False
            
        stored_hash, expire_date = result
        
        if expire_date < datetime.datetime.now():
            return False
            
        return stored_hash == password_hash


def is_password_expired(user_id):
    with get_db_cursor() as cursor:
        cursor.execute("SELECT expire_date FROM passwords WHERE user_id = %s;", (user_id,))
        result = cursor.fetchone()
        
        if not result:
            return True
            
        expire_date = result[0]
        return expire_date < datetime.datetime.now()


def request_password():
    import getpass
    password = getpass.getpass("Please enter your password: ")
    
    if not password:
        raise ValueError("Password cannot be empty")
    
    return password
