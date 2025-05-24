import sys
from fastapi import APIRouter, FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Import the API routers
from api.user.user_api import user_router
from api.chat.endpoints import chat_router
from api.image import image_router
from api.config import config_router

app = FastAPI(
    title="Frai API",
    description="API for Frai application",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - should be restricted in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
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
    
    Returns:
        dict: A dictionary containing service status information
            - status: Current operational status (running/degraded/down)
            - health: Health status (ok/warning/error)
            - version: Current API version string
            - timestamp: Server timestamp when the request was processed
            - preprocessing: Whether command preprocessing is enabled
    """
    from datetime import datetime
    from services.config import get_config
    
    # Check if command preprocessing is enabled in configuration
    config = get_config()
    preprocessing_enabled = config.get("command_preprocessing", {}).get("enabled", False)
    
    return {
        "status": "running", 
        "health": "ok", 
        "version": "alpha-0.0.1",
        "timestamp": datetime.now().isoformat(),
        "preprocessing": preprocessing_enabled
    }


app.include_router(api, tags=["api"])
app.include_router(user_router, tags=["user"])
app.include_router(chat_router, tags=["chat"])
app.include_router(image_router, tags=["image"])
app.include_router(config_router, tags=["config"])


def start_backend_api():
    """
    Start the FastAPI application using Uvicorn server.
    
    This function starts a Uvicorn server to serve the FastAPI application.
    It binds to all available network interfaces (0.0.0.0) on port 8000.
    When run in the main process, this function will block until the server
    is shut down. In a threading context, the server runs in that thread.
    """
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to start API server: {e}", exc_info=True)
        raise


def start_backend_api_cli():
    """
    CLI entry point function for starting just the API server.
    
    This function is used when the user wants to start only the API server
    without the CLI interface. It handles initialization of required components
    and then starts the API server.
    
    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    import logging
    import sys
    import dotenv
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Load environment variables
    dotenv.load_dotenv()
    
    logger.info("Starting API-only mode")
    
    # Initialize database
    from services.database.connection import start_engine, init_database
    
    logger.info("Initializing database...")
    if not start_engine():
        logger.critical("Failed to establish database engine. Exiting.")
        print("Critical: Database engine could not be initialized. API cannot start.")
        return 1
    
    if not init_database():
        logger.critical("Database initialization failed. Exiting.")
        print("Critical: Database could not be initialized. API cannot start.")
        return 1
    
    # Load the AI model
    from services.chat.model_loader import load_model
    
    logger.info("Loading AI model...")
    try:
        load_model()
        logger.info("AI model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load AI model: {e}", exc_info=True)
        print("Warning: Failed to load AI model. Some API endpoints may not work properly.")
    
    # Run basic runtime tests
    from services.runtime_tests import run_api_tests
    
    logger.info("Running API runtime tests...")
    try:
        if not run_api_tests():
            logger.warning("API runtime tests failed. API might have limited functionality.")
            print("Warning: Some API tests failed. The API may have limited functionality.")
        else:
            logger.info("API runtime tests passed.")
    except Exception as e:
        logger.error(f"Error during API runtime tests: {e}", exc_info=True)
        print("Warning: Could not complete API runtime tests.")
    
    # Start the API server
    print("Starting API server at http://0.0.0.0:8000")
    logger.info("Starting API server in main process")
    
    try:
        start_backend_api()  # This will block until the server is shut down
        return 0
    except Exception as e:
        logger.critical(f"API server failed: {e}", exc_info=True)
        print(f"Critical: API server encountered an error and had to close: {e}")
        return 1


if __name__ == "__main__":
    # Allow running the API directly with python -m api.api
    sys.exit(start_backend_api_cli())
