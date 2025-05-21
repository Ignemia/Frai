import hashlib
from services.state import set_state
from services.database.users import user_exists, get_user_id
from services.database.passwords import request_password, verify_credentials
from services.database.sessions import generate_session_token



def login_command(username):
    if not username:
        username = input("Please enter your username: ").strip()
    if not user_exists(username):
        print(f"User '{username}' does not exist. Please register first.")
        return None
    
    if not verify_credentials(username, hashlib.sha256(request_password().encode()).hexdigest()):
        print(f"Password verification for user '{username}' failed. Please try again.")
        return None
    
    set_state('current_user', username)
    
    # Generate session token
    user_id = get_user_id(username)
    session_token = generate_session_token(user_id)
    
    print(f"Login successful. Your session token is: {session_token}")
    return session_token