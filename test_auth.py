from services.database.users import user_exists, get_user_id
from services.database.passwords import verify_credentials, get_password_hash
from services.database.connection import get_db_session
from services.database.models import User, PasswordEntry
from hashlib import sha256

# Test the authentication process step by step
username = "testuser"
password = "testpassword"

print(f"Testing authentication for username: {username}")
print(f"Password: {password}")

# Step 1: Check if user exists
user_exists_result = user_exists(username)
print(f"User exists: {user_exists_result}")

if user_exists_result:
    # Step 2: Get user ID
    user_id = get_user_id(username)
    print(f"User ID: {user_id}")
    
    # Step 3: Check password hash in database
    stored_hash = get_password_hash(user_id)
    print(f"Stored password hash: {stored_hash}")
    
    # Step 4: Calculate the expected hash
    expected_hash = sha256(password.encode()).hexdigest()
    print(f"Expected password hash: {expected_hash}")
    
    # Step 5: Test verify_credentials function
    auth_result = verify_credentials(username, expected_hash)
    print(f"Authentication result: {auth_result}")
    
    # Step 6: Check password entry details
    with get_db_session() as session:
        user = session.query(User).filter(User.name == username).first()
        if user:
            password_entry = session.query(PasswordEntry).filter(PasswordEntry.user_id == user.user_id).first()
            if password_entry:
                print(f"Password entry exists:")
                print(f"  - Hashed password: {password_entry.hashed_password}")
                print(f"  - Expire date: {password_entry.expire_date}")
            else:
                print("No password entry found")
        else:
            print("User not found in second check")
else:
    print("User does not exist, authentication failed")
