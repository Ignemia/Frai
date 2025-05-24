from services.database.users import *
from services.database.connection import get_db_session
from services.database.models import User

with get_db_session() as session:
    user = session.query(User).filter(User.user_id == 6).first()
    if user:
        print(f"Username: {user.name}")
    else:
        print("User not found")
