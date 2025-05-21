from hashlib import sha256

from services.state import set_state

from services.database.users import register_user, user_exists
from services.database.sessions import generate_session_token

def request_password(message="Please enter your password: "):
    import getpass
    password = getpass.getpass(message)
    if not password:
        print("Password cannot be empty.")
        return None
    if len(password) < 6:
        print("Password must be at least 6 characters long.")
        return None
    return password

def register_command(username):
    if not username:
        username = input("Please enter your username: ").strip()
    if user_exists(username):
        print(f"User '{username}' already exists. Please choose a different username.")
        return None

    set_password = request_password()
    if not set_password:
        print(f"You need to set a password.")
        return None

    verify_password = request_password("Please re-enter your password for verification: ")
    if set_password != verify_password:
        print(f"Passwords do not match. Please try again.")
        return None

    user_id = register_user(username, sha256(set_password.encode()).hexdigest())
    if not user_id:
        print("Registration failed. Please try again.")
        return None

    set_state('current_user', username)
    
    # Generate session token
    session_token = generate_session_token(user_id)
    
    print(f"Registration successful. Your session token is: {session_token}")
    return session_token


