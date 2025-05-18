from hashlib import sha256

from services.state import set_state

from services.database.users import register_user, user_exists

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
        return

    set_password = request_password()
    if not set_password:
        print(f"You need to set a password.")
        return

    verify_password = request_password("Please re-enter your password for verification: ")
    if set_password != verify_password:
        print(f"Passwords do not match. Please try again.")
        return

    register_user(username, sha256(set_password.encode()).hexdigest())

    set_state('current_user', username)
    
    
    