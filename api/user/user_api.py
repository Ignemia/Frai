from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

# Import Pydantic models
from api.models.auth import (
    UserRegisterRequest as UserRegisterData,
    UserLoginRequest as UserLoginData, 
    TokenResponse,
    UserResponse,
    TokenData
)
from api.auth import (
    verify_password, 
    get_password_hash, 
    create_access_token, 
    get_current_user,
    oauth2_scheme
)
# Backward compatibility models
from .compatibility_models import UserTokenData, UserLogoutData, UserDetails


# Assume these service functions are defined in a services.py or similar
# from ..services.user_services import (
#     validate_user_data,
#     user_data_exists,
#     create_user,
#     authenticate_user,
#     validate_token_data,
#     user_data_exists_by_token,
#     refresh_user_token,
#     logout_user as service_logout_user, # Renamed to avoid conflict if necessary
#     get_user_by_username,
#     validate_user_token,
# )

# Placeholder service functions:
async def validate_user_data(data): return True
async def user_data_exists(username): return False
async def create_user(data): return True
async def authenticate_user(data): return True
async def validate_token_data(data): return True
async def user_data_exists_by_token(token): return True
async def refresh_user_token(token): return "new_mock_token"
async def service_logout_user(data): return True # Renamed to avoid conflict
async def get_user_by_username(username): 
    return {"username": username, "email": f"{username}@example.com", "is_active": True} # Placeholder
async def validate_user_token(token: str = Depends(lambda: None)): return True # Placeholder, adjust as needed


user_router = APIRouter(prefix="/user", tags=["user"])


@user_router.post(
    "/register",
    response_model=dict,
    summary="Register a new user",
    status_code=201,
)
async def register_user(in_user: UserRegisterData):
    """
    Register a new user.
    """
    try:
        if not await validate_user_data(in_user):
            raise HTTPException(status_code=400, detail="Invalid user data.")

        if await user_data_exists(in_user.username):
            raise HTTPException(status_code=400, detail="User already exists.")

        if not await create_user(in_user):
            raise HTTPException(status_code=400, detail="Error creating user.")
        return {"message": "User registered successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@user_router.post(
    "/login",
    response_model=TokenResponse,
    summary="Log in an existing user and get access token",
    status_code=200,
)
async def login_user(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Log in an existing user and return a JWT token.
    
    This endpoint conforms to the OAuth2 password flow, accepting 
    form data with username and password fields.
    
    Returns:
        TokenResponse: Access token and token type
    """
    try:
        # In a real implementation, you would verify credentials against your database
        if not await user_data_exists(form_data.username):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        # In a real implementation, you would verify the password hash
        # Here we'll assume authentication is successful for any existing user
        if not await authenticate_user({"username": form_data.username, "password": form_data.password}):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token with the username as the subject
        access_token = create_access_token(data={"sub": form_data.username})
        
        # Return token response
        return TokenResponse(access_token=access_token, token_type="bearer")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login error: {str(e)}")


@user_router.post(
    "/login-by-token",
    response_model=dict,
    summary="Log in a user using a token",
    status_code=200,
)
async def login_by_token(in_token: UserTokenData):
    """
    Log in a user using a token.
    """
    try:
        if not await validate_token_data(in_token):
            raise HTTPException(status_code=400, detail="Invalid token data.")

        if not await user_data_exists_by_token(in_token.token):
            raise HTTPException(
                status_code=400, detail="User does not exist or token is invalid."
            )

        return {"message": "User logged in successfully with token."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@user_router.post(
    "/refresh-token",
    response_model=dict,
    summary="Refresh user token",
    status_code=200,
)
async def refresh_token(in_token: UserTokenData):
    """
    Refresh user token.
    """
    try:
        if not await validate_token_data(in_token):
            raise HTTPException(status_code=400, detail="Invalid token data.")

        if not await user_data_exists_by_token(in_token.token):
            raise HTTPException(
                status_code=400, detail="User does not exist or token is invalid."
            )

        new_token = await refresh_user_token(in_token.token)
        return {"token": new_token}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@user_router.post(
    "/logout",
    response_model=dict,
    summary="Log out a user",
    status_code=200,
)
async def logout_user(in_user: UserLogoutData):
    """
    Log out a user.
    """
    try:
        if not await validate_user_data(in_user):
            raise HTTPException(status_code=400, detail="Invalid user data.")

        if not await user_data_exists(in_user.username):
            raise HTTPException(status_code=400, detail="User does not exist.")
        if not await service_logout_user(in_user): 
            raise HTTPException(status_code=400, detail="Error logging out user.")

        return {"message": "User logged out successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@user_router.get(
    "/{username}",
    response_model=UserDetails,
    summary="Get user details by username",
    status_code=200,
)
async def get_user(username: str, user_token: bool = Depends(validate_user_token)):
    """
    Get user details by username.
    Requires valid token.
    """
    # user_token dependency is used for authentication/authorization
    # The actual token value might be extracted by validate_user_token if needed
    try:
        user = await get_user_by_username(username)
        if not user:
            raise HTTPException(status_code=404, detail="User not found.")
        return user
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

