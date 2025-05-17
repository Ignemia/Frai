from fastapi import APIRouter, FastAPI
import uvicorn  # Add uvicorn import

# Import the user router
from user.user_api import user_router

app = FastAPI(
    title="Personal Chatter API",
    description="API for Personal Chatter application",
    version="0.1.0",
)

api = APIRouter(prefix="/server", tags=["server"])


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


app.include_router(api, tags=["api"])
app.include_router(user_router, tags=["user"])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
