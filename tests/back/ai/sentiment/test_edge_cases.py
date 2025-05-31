"""
Edge case tests for the sentiment analysis module.

This module tests the sentiment analysis system's behavior with unusual inputs,
including neutral sentiment, mixed sentiment, sarcasm, and domain-specific language.
"""

import pytest
import logging
import os
import sys
from typing import Dict, Any, List

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# Try to import real functions, fallback to mocks
def mock_function(*args, **kwargs):
    """Mock function that skips tests when called."""
    import pytest
    pytest.skip(f"Function not implemented yet")

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

class MockAIInstance:
    """Mock AI instance for when real implementation doesn't exist."""
    
    def __init__(self, module_name):
        self.module_name = module_name
        self.model = None
        self.processor = None
        self.tokenizer = None


# Set up logging
logger = logging.getLogger(__name__)


# Using fixture from conftest.py


class TestSentimentEdgeCases:
    """Test sentiment analysis with edge cases and unusual inputs."""
    
    def test_empty_string(self, setup_sentiment_ai):
        """Test sentiment analysis with empty string input."""
        result = analyze_sentiment("")
        
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            # If successful, score should be neutral
            score = result.get('sentiment_score')
            assert score is not None
            assert -0.1 <= score <= 0.1, f"Empty string should be neutral, got {score}"
        else:
            # Or it should return an appropriate error
            assert 'error' in result
        
        logger.info(f"Empty string test result: {result}")
    
    def test_whitespace_only(self, setup_sentiment_ai):
        """Test sentiment analysis with whitespace-only input."""
        whitespace_inputs = ["   ", "\t\t", "\n\n", " \t\n "]
        
        for text in whitespace_inputs:
            result = analyze_sentiment(text)
            assert isinstance(result, dict)
            assert 'success' in result
            
            if result['success']:
                score = result.get('sentiment_score')
                assert score is not None
                assert -0.1 <= score <= 0.1, f"Whitespace should be neutral, got {score}"
        
        logger.info("Whitespace-only input test passed")
    
    def test_mixed_sentiment(self, setup_sentiment_ai):
        """Test sentiment analysis with mixed positive and negative content."""
        mixed_texts = [
            "The movie had great visuals but the plot was disappointing.",
            "I love the design but hate the functionality.",
            "Beautiful location, terrible service.",
            "Amazing food, awful atmosphere.",
            "Good price but poor quality."
        ]
        
        for text in mixed_texts:
            result = analyze_sentiment(text)
            assert result.get('success', False), f"Failed to analyze: {text}"
            
            score = result.get('sentiment_score')
            # Mixed sentiment should result in moderate scores
            assert -0.5 <= score <= 0.5, f"Mixed sentiment should be moderate for: {text}, got {score}"
        
        logger.info("Mixed sentiment test passed")
    
    def test_sarcasm_detection(self, setup_sentiment_ai):
        """Test sentiment analysis with sarcastic statements."""
        sarcastic_texts = [
            "Oh great! Another traffic jam. Just what I needed today.",
            "Well that went perfectly! Not like I expected it to fail anyway.",
            "Fantastic! My computer crashed right before the deadline.",
            "Amazing weather for a picnic! (during a thunderstorm)",
            "Just wonderful! The elevator is broken again."
        ]
        
        for text in sarcastic_texts:
            result = analyze_sentiment(text)
            assert result.get('success', False), f"Failed to analyze: {text}"
            
            score = result.get('sentiment_score')
            # Sarcasm should ideally be detected as negative despite positive words
            # But this is challenging, so we'll accept a range
            assert -1.0 <= score <= 0.3, f"Sarcastic text analysis for: {text}, got {score}"
        
        logger.info("Sarcasm detection test passed")
    
    def test_neutral_factual_statements(self, setup_sentiment_ai):
        """Test sentiment analysis with purely factual, neutral statements."""
        factual_texts = [
            "The meeting is scheduled for 3 PM on Tuesday.",
            "Water boils at 100 degrees Celsius.",
            "The document contains 50 pages.",
            "There are 24 hours in a day.",
            "The building has 10 floors.",
            "The project deadline is next Friday."
        ]
        
        for text in factual_texts:
            result = analyze_sentiment(text)
            assert result.get('success', False), f"Failed to analyze: {text}"
            
            score = result.get('sentiment_score')
            assert -0.2 <= score <= 0.2, f"Factual statement should be neutral for: {text}, got {score}"
        
        logger.info("Neutral factual statements test passed")
    
    def test_domain_specific_language(self, setup_sentiment_ai):
        """Test sentiment analysis with domain-specific language."""
        # Medical domain
        medical_texts = [
            "The patient's recovery is progressing remarkably well.",
            "The patient's condition has deteriorated significantly.",
            "The surgery was successful with no complications.",
            "The diagnosis indicates a serious underlying condition."
        ]
        
        # Financial domain
        financial_texts = [
            "The stock market showed strong gains today.",
            "The company reported massive losses this quarter.",
            "Investment returns exceeded expectations.",
            "The market crash wiped out investor confidence."
        ]
        
        # Technical domain
        technical_texts = [
            "The system performance has improved dramatically.",
            "Critical errors detected in the application.",
            "Database optimization completed successfully.",
            "Server downtime resulted in data loss."
        ]
        
        all_domain_texts = medical_texts + financial_texts + technical_texts
        
        for text in all_domain_texts:
            result = analyze_sentiment(text)
            assert result.get('success', False), f"Failed to analyze: {text}"
            
            score = result.get('sentiment_score')
            assert -1.0 <= score <= 1.0, f"Domain-specific text analysis for: {text}, got {score}"
            
            # Basic sentiment should still be detectable
            if any(word in text.lower() for word in ['well', 'successful', 'gains', 'improved', 'exceeded']):
                assert score >= -0.2, f"Expected non-negative sentiment for: {text}, got {score}"
            elif any(word in text.lower() for word in ['deteriorated', 'losses', 'crash', 'errors', 'downtime']):
                assert score <= 0.2, f"Expected non-positive sentiment for: {text}, got {score}"
        
        logger.info("Domain-specific language test passed")
    
    def test_special_characters_and_symbols(self, setup_sentiment_ai):
        """Test sentiment analysis with special characters and symbols."""
        special_texts = [
            "@#$% this is so frustrating! Why can't anything work properly?",
            "Amazing!!! ❤️❤️❤️ Love it so much!!!",
            "Really??? 🤔 Not sure about this...",
            "Perfect 👍 10/10 would recommend! ⭐⭐⭐⭐⭐",
            "Terrible 👎 0/10 complete waste of time ❌❌❌"
        ]
        
        for text in special_texts:
            result = analyze_sentiment(text)
            assert result.get('success', False), f"Failed to analyze: {text}"
            
            score = result.get('sentiment_score')
            assert -1.0 <= score <= 1.0, f"Special characters text analysis for: {text}, got {score}"
        
        logger.info("Special characters and symbols test passed")
    
    def test_very_long_text(self, setup_sentiment_ai):
        """Test sentiment analysis with very long text input."""
        # Create a long positive text
        long_positive = "This is an absolutely amazing experience! " * 100
        
        # Create a long negative text
        long_negative = "This is completely terrible and disappointing! " * 100
        
        # Test long positive
        result = analyze_sentiment(long_positive)
        assert result.get('success', False), "Failed to analyze long positive text"
        score = result.get('sentiment_score')
        assert score > 0.3, f"Expected positive sentiment for long positive text, got {score}"
        
        # Test long negative
        result = analyze_sentiment(long_negative)
        assert result.get('success', False), "Failed to analyze long negative text"
        score = result.get('sentiment_score')
        assert score < -0.3, f"Expected negative sentiment for long negative text, got {score}"
        
        logger.info("Very long text test passed")
    
    def test_numbers_and_measurements(self, setup_sentiment_ai):
        """Test sentiment analysis with numbers and measurements."""
        numerical_texts = [
            "I rate this 10 out of 10! Absolutely perfect score.",
            "This gets a 1/10 rating. Completely awful.",
            "The temperature is 25°C today.",
            "The package weighs 5.2 kilograms.",
            "The distance is approximately 150 meters.",
            "Sales increased by 50% this quarter - excellent results!",
            "Productivity dropped by 75% - this is terrible."
        ]
        
        for text in numerical_texts:
            result = analyze_sentiment(text)
            assert result.get('success', False), f"Failed to analyze: {text}"
            
            score = result.get('sentiment_score')
            assert -1.0 <= score <= 1.0, f"Numerical text analysis for: {text}, got {score}"
        
        logger.info("Numbers and measurements test passed")
    
    def test_multiple_languages_mixed(self, setup_sentiment_ai):
        """Test sentiment analysis with mixed language content."""
        mixed_language_texts = [
            "This is très bien! Really good stuff.",
            "The food was delicious! Molto bene!",
            "¡Excelente! This is amazing trabajo.",
            "Das ist gut, very impressive work indeed.",
            "Magnifique! This is absolutely wonderful."
        ]
        
        for text in mixed_language_texts:
            result = analyze_sentiment(text)
            assert result.get('success', False), f"Failed to analyze: {text}"
            
            score = result.get('sentiment_score')
            assert -1.0 <= score <= 1.0, f"Mixed language text analysis for: {text}, got {score}"
            # These should generally be positive due to positive English words
            assert score >= -0.3, f"Expected non-negative sentiment for: {text}, got {score}"
        
        logger.info("Multiple languages mixed test passed")
    
    def test_contradictory_statements(self, setup_sentiment_ai):
        """Test sentiment analysis with contradictory statements."""
        contradictory_texts = [
            "I love it but I hate it at the same time.",
            "It's the best worst thing I've ever seen.",
            "Beautifully ugly and perfectly imperfect.",
            "I'm happily disappointed with the results.",
            "This is a successful failure of epic proportions."
        ]
        
        for text in contradictory_texts:
            result = analyze_sentiment(text)
            assert result.get('success', False), f"Failed to analyze: {text}"
            
            score = result.get('sentiment_score')
            # Contradictory statements should result in neutral or mixed scores
            assert -0.6 <= score <= 0.6, f"Contradictory statement should be moderate for: {text}, got {score}"
        
        logger.info("Contradictory statements test passed")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])