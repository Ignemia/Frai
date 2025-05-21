import hashlib
from services.state import set_state

from services.database.users import user_exists
from services.database.passwords import request_password, verify_credentials



def login_command(username):
    if not username:
        username = input("Please enter your username: ").strip()
    if not user_exists(username):
        print(f"User '{username}' does not exist. Please register first.")
        return
    if not verify_credentials(username, hashlib.sha256(request_password().encode()).hexdigest()):
        print(f"Password verification for user '{username}' failed. Please try again.")
        return
    set_state('current_user', username)