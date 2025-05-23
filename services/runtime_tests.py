"""
from test_mock_helper import List
Runtime tests for Personal Chatter application.

This module contains tests that verify critical components 
of the system are operational at startup time.
"""
import logging
from typing import  Tuple, Dict
import os.path

logger = logging.getLogger(__name__)

def test_model_availability() -> bool:
    """
    Verify that the required model files exist in the expected locations.
    
    Returns:
        bool: True if all required model files are available, False otherwise.
    """
    model_paths = [
        os.path.join("models", "gemma-3-4b-it"),
        os.path.join("models", "multilingual-sentiment-analysis"),
        # Add other critical models as needed
    ]
    
    missing_models = []
    for path in model_paths:
        if not os.path.exists(path):
            missing_models.append(path)
            logger.error(f"Required model not found: {path}")
    
    if missing_models:
        logger.error(f"Missing {len(missing_models)} required model(s)")
        return False
    
    logger.info("All required model files are available")
    return True

def test_database_connectivity() -> bool:
    """
    Test if the database connection is working properly.
    
    Returns:
        bool: True if database connection is successful, False otherwise.
    """
    from services.database.connection import get_db_session
    
    try:
        with get_db_session() as session:
            # Simple query to test connection
            result = session.execute("SELECT 1").scalar()
            if result == 1:
                logger.info("Database connectivity test passed")
                return True
            logger.error("Database connectivity test failed: unexpected result")
            return False
    except Exception as e:
        logger.error(f"Database connectivity test failed: {e}")
        return False

def run_all_tests() -> bool:
    """
    Run all runtime tests and return overall result.
    
    Returns:
        bool: True if all tests pass, False if any test fails.
    """
    logger.info("Running runtime tests...")
    
    test_results = {
        "model_availability": test_model_availability(),
        "database_connectivity": test_database_connectivity(),
        # Add more tests as needed
    }
    
    # Log individual test results
    for test_name, passed in test_results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"Test {test_name}: {status}")
    
    # Overall result
    all_passed = all(test_results.values())
    if all_passed:
        logger.info("All runtime tests passed successfully")
    else:
        logger.warning(f"Runtime tests completed with {list(test_results.values()).count(False)} failures")
    
    return all_passed


def test_api_components() -> bool:
    """
    Test specific components needed for the API server.
    
    Returns:
        bool: True if all API-specific components are available, False otherwise.
    """
    try:
        # Test if FastAPI is properly installed
        import fastapi
        import uvicorn
        logger.info("FastAPI framework available")
        
        # Test JWT functionality
        import jwt
        test_token = jwt.encode({"test": "data"}, "secret", algorithm="HS256")
        decoded = jwt.decode(test_token, "secret", algorithms=["HS256"])
        if decoded.get("test") != "data":
            raise ValueError("JWT token encoding/decoding failed")
        logger.info("JWT authentication components working")
        
        # Test WebSocket support
        from fastapi import WebSocket, WebSocketDisconnect
        logger.info("WebSocket components available")
        
        return True
    except ImportError as e:
        logger.error(f"API component test failed - missing package: {e}")
        return False
    except Exception as e:
        logger.error(f"API component test failed: {e}")
        return False


def run_api_tests() -> bool:
    """
    Run tests specifically for the API server functionality.
    
    Returns:
        bool: True if all API-related tests pass, False if any test fails.
    """
    logger.info("Running API-specific runtime tests...")
    
    test_results = {
        "database_connectivity": test_database_connectivity(),
        "api_components": test_api_components(),
        # Additional API-specific tests can be added here
    }
    
    # Log individual test results
    for test_name, passed in test_results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"API Test {test_name}: {status}")
    
    # Overall result
    all_passed = all(test_results.values())
    if all_passed:
        logger.info("All API runtime tests passed successfully")
    else:
        logger.warning(f"API runtime tests completed with {list(test_results.values()).count(False)} failures")
    
    return all_passed
