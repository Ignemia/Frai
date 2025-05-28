import pytest
import logging
import os



# Setup logging for all AI tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configure pytest
def pytest_configure(config):
    """Configure pytest for AI module tests."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "accuracy: mark test as an accuracy test"
    )
    config.addinivalue_line(
        "markers", "consistency: mark test as a consistency test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "edge_case: mark test as an edge case test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test file names
        if "test_unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "test_accuracy" in str(item.fspath):
            item.add_marker(pytest.mark.accuracy)
        elif "test_consistency" in str(item.fspath):
            item.add_marker(pytest.mark.consistency)
        elif "test_performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        elif "test_edge" in str(item.fspath):
            item.add_marker(pytest.mark.edge_case)

@pytest.fixture(scope="session", autouse=True)
def ai_test_session():
    """Session-wide fixture for AI tests."""
    logger = logging.getLogger(__name__)
    logger.info("Starting AI test session")
    
    # Setup session-wide test data directories
    test_data_dirs = [
        "Frai/tests/back/ai/img2img/inputs",
        "Frai/tests/back/ai/voicein/test_data",
        "Frai/tests/back/ai/voiceout/test_data"
    ]
    
    for directory in test_data_dirs:
        os.makedirs(directory, exist_ok=True)
    
    yield
    
    logger.info("AI test session complete")

@pytest.fixture
def ai_module_config():
    """Configuration for AI module tests."""
    return {
        'test_timeout': 30,
        'max_retries': 3,
        'test_data_path': 'Frai/tests/back/ai',
        'mock_responses': True,
        'skip_slow_tests': False
    }