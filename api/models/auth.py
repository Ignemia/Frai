"""
API models for authentication and user data.

This module defines Pydantic models for request validation and response formatting
for the authentication API endpoints.
"""
from pydantic import BaseModel, Field, EmailStr
from typing import Optional

class UserRegisterRequest(BaseModel):
    """Model for user registration request."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    email: EmailStr
    
class UserLoginRequest(BaseModel):
    """Model for user login request."""
    username: str
    password: str
    
class TokenResponse(BaseModel):
    """Model for token response."""
    access_token: str
    token_type: str = "bearer"
    
class UserResponse(BaseModel):
    """Model for user data response."""
    username: str
    email: EmailStr
    is_active: bool
    
class TokenData(BaseModel):
    """Model for JWT token data."""
    username: Optional[str] = None
