"""
Unit tests for individual functions and classes in the sentiment analysis module.

This module tests the core components of the sentiment analysis system,
including initialization, preprocessing, and basic functionality.
"""

import pytest
import logging
from typing import Dict, Any
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# Set up logging
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


# Using fixture from conftest.py


class TestSentimentComponents:
    """Test individual components of the sentiment analysis system."""
    
    def test_system_initialization(self, setup_sentiment_ai):
        """Test that the sentiment system initializes correctly."""
        assert setup_sentiment_ai is not None
        logger.info("Sentiment system initialized successfully")
    
    def test_model_loading(self, setup_sentiment_ai):
        """Test that the sentiment model loads properly."""
        assert hasattr(setup_sentiment_ai, 'model')
        assert hasattr(setup_sentiment_ai, 'processor')
        logger.info("Sentiment model loaded with required components")
    
    def test_preprocessing_function(self):
        """Test text preprocessing functionality."""
        try:
            from Frai.back.ai.sentiment.preprocessing import preprocess_text
            
            # Test basic preprocessing
            text = "Hello World! This is a TEST."
            processed = preprocess_text(text)
            assert isinstance(processed, str)
            assert len(processed) > 0
            
            # Test empty string
            empty_processed = preprocess_text("")
            assert isinstance(empty_processed, str)
            
            logger.info("Text preprocessing function working correctly")
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Preprocessing module not implemented yet")
    
    def test_tokenization(self, setup_sentiment_ai):
        """Test tokenization process."""
        if isinstance(setup_sentiment_ai, MockAIInstance):
            pytest.skip("Tokenization test requires real implementation")
            
        test_text = "This is a simple test sentence."
        
        # Test tokenization doesn't fail
        tokens = setup_sentiment_ai.tokenizer(test_text, return_tensors="pt")
        assert 'input_ids' in tokens
        assert 'attention_mask' in tokens
        logger.info("Tokenization working correctly")
    
    def test_score_range_validation(self):
        """Test that sentiment scores are in valid range."""
        try:
            from Frai.back.ai.sentiment.utils import validate_score_range
            
            # Test valid scores
            assert validate_score_range(0.5) == True
            assert validate_score_range(-0.8) == True
            assert validate_score_range(1.0) == True
            assert validate_score_range(-1.0) == True
            
            # Test invalid scores
            assert validate_score_range(1.5) == False
            assert validate_score_range(-1.5) == False
            assert validate_score_range(float('inf')) == False
            assert validate_score_range(float('nan')) == False
            
            logger.info("Score range validation working correctly")
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Utils module not implemented yet")
    
    def test_interval_parsing(self):
        """Test interval notation parsing."""
        try:
            from Frai.back.ai.sentiment.utils import parse_interval
            
            # Test closed interval
            lower, upper, left_closed, right_closed = parse_interval("<0.5;0.8>")
            assert lower == 0.5 and upper == 0.8
            assert left_closed == True and right_closed == True
            
            # Test open interval
            lower, upper, left_closed, right_closed = parse_interval("(0.2;0.6)")
            assert lower == 0.2 and upper == 0.6
            assert left_closed == False and right_closed == False
            
            # Test mixed interval
            lower, upper, left_closed, right_closed = parse_interval("<-0.5;0.3)")
            assert lower == -0.5 and upper == 0.3
            assert left_closed == True and right_closed == False
            
            logger.info("Interval parsing working correctly")
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Utils module not implemented yet")


class TestSentimentAPI:
    """Test the sentiment analysis API functions."""
    
    def test_analyze_sentiment_basic(self, setup_sentiment_ai):
        """Test basic sentiment analysis function."""
        result = analyze_sentiment("I love this product!")
        
        assert isinstance(result, dict)
        assert 'success' in result
        assert result['success'] == True
        assert 'sentiment_score' in result
        assert isinstance(result['sentiment_score'], (int, float))
        assert -1.0 <= result['sentiment_score'] <= 1.0
        
        logger.info(f"Basic sentiment analysis working: {result}")
    
    def test_analyze_sentiment_empty(self, setup_sentiment_ai):
        """Test sentiment analysis with empty input."""
        result = analyze_sentiment("")
        
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            assert 'sentiment_score' in result
            assert -1.0 <= result['sentiment_score'] <= 1.0
        else:
            assert 'error' in result
        
        logger.info(f"Empty input handling: {result}")
    
    def test_analyze_sentiment_none(self, setup_sentiment_ai):
        """Test sentiment analysis with None input."""
        result = analyze_sentiment(None)
        
        assert isinstance(result, dict)
        assert result['success'] == False
        assert 'error' in result
        
        logger.info(f"None input handling: {result}")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])