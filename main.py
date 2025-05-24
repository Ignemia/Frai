import dotenv
dotenv.load_dotenv()

import logging
import sys
import threading
import argparse
from pathlib import Path

from services.database.connection import start_engine, init_database
from services.chat.model_loader import load_model
from services.cli.cli_handler import start_cli_interface
from api.api import start_backend_api

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_tests(test_args: list = None):
    """
    Run the comprehensive test suite using the test orchestrator.
    
    Args:
        test_args: Additional arguments to pass to the test orchestrator
    """
    logger.info("Starting comprehensive test suite...")
    
    try:
        # Import and run test orchestrator
        from tests.test_orchestrator import TestOrchestrator
        
        project_root = Path(__file__).parent
        orchestrator = TestOrchestrator(project_root)
        
        # Parse test arguments
        test_type = "all"
        extra_args = []
        
        if test_args:
            if test_args[0] in ["all", "unit", "integration", "implementation", 
                               "blackbox", "performance", "demo", "quick", "ci"]:
                test_type = test_args[0]
                extra_args = test_args[1:]
            else:
                extra_args = test_args
        
        # Execute tests
        success = orchestrator.execute_test_suite(test_type, extra_args)
        
        if success:
            logger.info("All tests completed successfully!")
            print("\n🎉 All tests passed! The system is ready for use.")
            return 0
        else:
            logger.error("Some tests failed. Check the output above for details.")
            print("\n❌ Some tests failed. Please review the results above.")
            return 1
            
    except ImportError as e:
        logger.error(f"Could not import test orchestrator: {e}")
        print("Error: Test system not properly installed. Please check your installation.")
        return 1
    except Exception as e:
        logger.error(f"Error during test execution: {e}", exc_info=True)
        print(f"Error: Test execution failed: {e}")
        return 1

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Personal Chatter - AI-powered chat and image generation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                    # Start the application normally
    python main.py tests              # Run all tests and demos
    python main.py tests unit         # Run only unit tests
    python main.py tests quick        # Run quick smoke tests
    python main.py tests demo         # Run demonstration workflows
    python main.py tests --coverage   # Run tests with coverage reporting
        """
    )
    
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        help="Command to execute: 'run' (default) or 'tests'"
    )
    
    parser.add_argument(
        "test_args",
        nargs="*",
        help="Additional arguments for test command"
    )
    
    return parser.parse_args()

def main():
    """
    Main orchestration function for the Personal Chatter application.
    Handles both normal application startup and test execution.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Handle test command
    if args.command == "tests":
        logger.info("Test mode requested...")
        return run_tests(args.test_args)
    
    # Normal application startup
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
        # return 1 # Uncomment if model is critical for startup

    # Start API interface in a separate thread
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