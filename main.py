import dotenv
dotenv.load_dotenv()

import logging
import sys
import threading

from services.database.connection import start_engine, init_database
from services.chat.model_loader import load_model
from services.cli.cli_handler import start_cli_interface
from services.runtime_tests import run_all_tests
from api.api import start_backend_api

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main orchestration function for the Personal Chatter application.
    Initializes services, loads components, and starts interfaces.
    """
    logger.info("Starting Personal Chatter application...")

    # Initialize database
    logger.info("Initializing database...")
    if not start_engine():  # start_engine now also handles init_database logic or should be separate
        logger.critical("Failed to establish database engine. Exiting.")
        print("Critical: Database engine could not be initialized. Application cannot start.")
        return 1  # Use return codes for errors
    
    if not init_database():
        logger.critical("Database initialization failed. Exiting.")
        print("Critical: Database could not be initialized. Application cannot start.")
        return 1

    # Load the AI model
    logger.info("Loading AI model...")
    try:
        load_model()
        logger.info("AI model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load AI model: {e}", exc_info=True)
        # Depending on criticality, you might choose to exit or continue with limited functionality
        print("Warning: Failed to load AI model. Some features may not work properly.")
        # return 1 # Uncomment if model is critical for startup    # Run runtime tests
    logger.info("Running runtime tests...")
    try:
        if not run_all_tests():
            logger.warning("One or more runtime tests failed. Check logs for details.")
            print("Warning: Some runtime tests failed. Application may have limited functionality.")
        else:
            logger.info("All runtime tests passed.")
    except Exception as e:
        logger.error(f"Error during runtime tests: {e}", exc_info=True)
        print("Warning: Could not complete runtime tests.")    # Start API interface in a separate thread
    logger.info("Starting API interface...")
    try:
        # Start API in a separate thread so it doesn't block the CLI
        api_thread = threading.Thread(target=start_backend_api, daemon=True)
        api_thread.start()
        logger.info("API interface started successfully in background thread.")
        print("API server running at http://0.0.0.0:8000")
    except Exception as e:
        logger.error(f"Failed to start API interface: {e}", exc_info=True)
        print("Warning: API interface could not be started.")

    # Start CLI interface
    logger.info("Starting CLI interface...")
    try:
        start_cli_interface()
    except Exception as e:
        logger.critical(f"CLI interface failed to start or crashed: {e}", exc_info=True)
        print("Critical: The command line interface encountered an error and had to close.")
        return 1

    logger.info("Personal Chatter application finished.")
    return 0

if __name__ == "__main__":
    return_code = main()
    sys.exit(return_code)  # Exit with the return code from main