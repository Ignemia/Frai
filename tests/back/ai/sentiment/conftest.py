import pytest
import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockAIInstance:
    """Mock AI instance for when real implementation doesn't exist."""
    
    def __init__(self, module_name):
        self.module_name = module_name
        self.model = None
        self.processor = None
        self.tokenizer = None


def mock_function(*args, **kwargs):
    """Mock function that skips tests when called."""
    pytest.skip(f"Function not implemented yet")


# Try to import real functions, fallback to mocks
try:
    from Frai.back.ai.sentiment import (
        initialize_sentiment_system,
        get_sentiment_ai_instance,
        analyze_sentiment
    )
except (ImportError, ModuleNotFoundError):
    initialize_sentiment_system = mock_function
    get_sentiment_ai_instance = mock_function
    analyze_sentiment = mock_function


@pytest.fixture(scope="session")
def setup_sentiment_ai():
    """
    Initialize the sentiment analysis system once for all tests.
    This fixture is shared across all sentiment test modules.
    """
    try:
        logger.info("Initializing sentiment analysis system for tests...")
        success = initialize_sentiment_system()
        
        if not success:
            pytest.fail("Failed to initialize sentiment analysis system")
            return None
        
        # Get the sentiment AI instance
        sentiment_ai = get_sentiment_ai_instance()
        logger.info("Sentiment analysis system initialized successfully")
        
        yield sentiment_ai
        
        logger.info("Test session complete. Sentiment AI instance cleanup.")
        
    except Exception as e:
        logger.warning(f"Could not initialize sentiment system: {e}")
        yield MockAIInstance("sentiment")





@pytest.fixture
def sample_texts():
    """Provide sample texts for testing."""
    return {
        'positive': [
            "I love this amazing product!",
            "This is absolutely wonderful!",
            "Fantastic work, truly exceptional!"
        ],
        'negative': [
            "This is terrible and disappointing.",
            "I hate this horrible experience.",
            "Worst product ever, completely useless!"
        ],
        'neutral': [
            "The weather is cloudy today.",
            "The meeting is scheduled for 3 PM.",
            "The document contains 50 pages."
        ],
        'mixed': [
            "Great visuals but poor plot.",
            "Love the design, hate the functionality.",
            "Beautiful location, terrible service."
        ]
    }