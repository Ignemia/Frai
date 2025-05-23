"""
Temporary Pydantic models for backward compatibility.
"""
from pydantic import BaseModel

# Models that need to be kept for backward compatibility
class UserTokenData(BaseModel): 
    token: str
    
class UserLogoutData(BaseModel): 
    username: str
    
class UserDetails(BaseModel):
    username: str
    email: str
    is_active: bool = True
