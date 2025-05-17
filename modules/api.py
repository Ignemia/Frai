from fastapi import APIRouter, Depends, HTTPException

api = APIRouter()


@api.get(
    "/status",
    response_model=dict,
    tags=["status"],
    summary="Check API status",
    status_code=200,
)
async def status_check():
    """
    Status check endpoint to verify if the API is running.
    """
    return {"status": "running", "health": "ok", "version": "alpha-0.0.1"}


user_router = APIRouter(prefix="/user")


@user_router.post(
    "/register",
    response_model=dict,
    tags=["user"],
    summary="Register a new user",
    status_code=201,
)
async def register_user(in_user: UserRegisterData):
    """
    Register a new user.
    """
    try:
        if not validate_user_data(in_user):
            raise HTTPException(status_code=400, detail="Invalid user data.")

        if user_data_exists(in_user.username):
            raise HTTPException(status_code=400, detail="User already exists.")

        if not create_user(in_user):
            raise HTTPException(status_code=400, detail="Error creating user.")
        return {"message": "User registered successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@user_router.post(
    "/login",
    response_model=dict,
    tags=["user"],
    summary="Log in an existing user",
    status_code=200,
)
async def login_user(in_user: UserLoginData):
    """
    Log in an existing user.
    """
    try:
        if not validate_user_data(in_user):
            raise HTTPException(status_code=400, detail="Invalid user data.")

        if not user_data_exists(in_user.username):
            raise HTTPException(status_code=400, detail="User does not exist.")

        if not authenticate_user(in_user):
            raise HTTPException(status_code=401, detail="Invalid credentials.")

        return {"message": "User logged in successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@user_router.post(
    "/login-by-token",
    response_model=dict,
    tags=["user"],
    summary="Log in a user using a token",
    status_code=200,
)
async def login_by_token(in_token: UserTokenData):
    """
    Log in a user using a token.
    """
    try:
        if not validate_token_data(in_token):
            raise HTTPException(status_code=400, detail="Invalid token data.")

        if not user_data_exists_by_token(in_token.token):
            raise HTTPException(
                status_code=400, detail="User does not exist or token is invalid."
            )

        return {"message": "User logged in successfully with token."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@user_router.post(
    "/refresh-token",
    response_model=dict,
    tags=["user"],
    summary="Refresh user token",
    status_code=200,
)
async def refresh_token(in_token: UserTokenData):
    """
    Refresh user token.
    """
    try:
        if not validate_token_data(in_token):
            raise HTTPException(status_code=400, detail="Invalid token data.")

        if not user_data_exists_by_token(in_token.token):
            raise HTTPException(
                status_code=400, detail="User does not exist or token is invalid."
            )

        new_token = refresh_user_token(in_token.token)
        return {"token": new_token}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@user_router.post(
    "/logout",
    response_model=dict,
    tags=["user"],
    summary="Log out a user",
    status_code=200,
)
async def logout_user(in_user: UserLogoutData):
    """
    Log out a user.
    """
    try:
        if not validate_user_data(in_user):
            raise HTTPException(status_code=400, detail="Invalid user data.")

        if not user_data_exists(in_user.username):
            raise HTTPException(status_code=400, detail="User does not exist.")

        if not logout_user(in_user):
            raise HTTPException(status_code=400, detail="Error logging out user.")

        return {"message": "User logged out successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@api.get(
    "/user/{username}",
    response_model=UserDetails,
    tags=["user"],
    summary="Get user details by username",
    status_code=200,
)
async def get_user(user_token: Depends(validate_user_token), username: str):
    """
    Get user details by username.
    """
    try:
        user = get_user_by_username(username)
        if not user:
            raise HTTPException(status_code=404, detail="User not found.")
        return user
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
