#!/usr/bin/env python3
"""
Comprehensive tests for ChatModerator class.
Tests message validation, spam detection, toxicity filtering, and sentiment analysis.
"""

import pytest
import unittest.mock as mock
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Add orchestrator module to path
orchestrator_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(orchestrator_path))

from chatmod import ChatModerator


class TestChatModerator:
    """Test suite for ChatModerator functionality."""
    
    @pytest.fixture
    def moderator(self):
        """Create a ChatModerator instance for testing."""
        with patch('chatmod.pipeline') as mock_pipeline:
            mock_pipeline.return_value = MagicMock()
            return ChatModerator()
    
    @pytest.fixture
    def moderator_no_sentiment(self):
        """Create a ChatModerator instance without sentiment analysis."""
        with patch('chatmod.pipeline', side_effect=Exception("Model not available")):
            return ChatModerator()

    def test_init_with_sentiment_analyzer(self):
        """Test ChatModerator initialization with sentiment analyzer."""
        with patch('chatmod.pipeline') as mock_pipeline:
            mock_analyzer = MagicMock()
            mock_pipeline.return_value = mock_analyzer
            
            moderator = ChatModerator()
            
            assert moderator.sentiment_analyzer is not None
            mock_pipeline.assert_called_once_with(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=mock.ANY
            )

    def test_init_without_sentiment_analyzer(self):
        """Test ChatModerator initialization when sentiment analyzer fails to load."""
        with patch('chatmod.pipeline', side_effect=Exception("Model not available")):
            moderator = ChatModerator()
            assert moderator.sentiment_analyzer is None

    def test_validate_message_format_valid(self, moderator):
        """Test validation of valid messages."""
        valid_messages = [
            "Hello, how are you?",
            "This is a normal message with some text.",
            "Short msg",
            "A" * 100,  # Long but within limits
        ]
        
        for message in valid_messages:
            is_valid, error = moderator.validate_message_format(message)
            assert is_valid, f"Message should be valid: {message}"
            assert error is None

    def test_validate_message_format_invalid(self, moderator):
        """Test validation of invalid messages."""
        # Test empty message
        is_valid, error = moderator.validate_message_format("")
        assert not is_valid
        assert "too short" in error.lower()

        # Test message too long
        long_message = "A" * (moderator.max_message_length + 1)
        is_valid, error = moderator.validate_message_format(long_message)
        assert not is_valid
        assert "too long" in error.lower()

        # Test None message
        is_valid, error = moderator.validate_message_format(None)
        assert not is_valid
        assert "must be a string" in error.lower()

    def test_check_spam_patterns_valid(self, moderator):
        """Test spam detection with valid messages."""
        valid_messages = [
            "Hello, how are you today?",
            "This is a normal conversation.",
            "Yes, I agree with that point.",
            "Thanks for your help!",
        ]
        
        for message in valid_messages:
            is_spam, reason = moderator.check_spam_patterns(message)
            assert not is_spam, f"Message should not be spam: {message}"
            assert reason is None

    def test_check_spam_patterns_repeated_chars(self, moderator):
        """Test spam detection for repeated characters."""
        spam_messages = [
            "Helloooooooooooo",  # More than 10 repeated chars
            "Yesssssssssssss",
            "Nooooooooooooooo",
        ]
        
        for message in spam_messages:
            is_spam, reason = moderator.check_spam_patterns(message)
            assert is_spam, f"Message should be detected as spam: {message}"
            assert "repeated characters" in reason.lower()

    def test_check_toxic_content_clean(self, moderator):
        """Test toxicity detection with clean messages."""
        clean_messages = [
            "Hello, how are you?",
            "I love this beautiful day.",
            "Thank you for your help.",
            "Let's discuss this topic.",
        ]
        
        for message in clean_messages:
            is_toxic, reason = moderator.check_toxic_content(message)
            assert not is_toxic, f"Message should not be toxic: {message}"
            assert reason is None

    def test_check_toxic_content_toxic(self, moderator):
        """Test toxicity detection with toxic keywords."""
        toxic_messages = [
            "I hate this situation",
            "This is harmful content",
            "Violence is not the answer",
        ]
        
        for message in toxic_messages:
            is_toxic, reason = moderator.check_toxic_content(message)
            assert is_toxic, f"Message should be detected as toxic: {message}"
            assert "toxic content detected" in reason.lower()

    def test_analyze_sentiment_with_analyzer(self, moderator):
        """Test sentiment analysis when analyzer is available."""
        # Mock the sentiment analyzer
        mock_result = [{"label": "NEGATIVE", "score": 0.8}]
        moderator.sentiment_analyzer = MagicMock(return_value=mock_result)
        
        message = "I hate this"
        sentiment, confidence = moderator.analyze_sentiment(message)
        
        assert sentiment == "NEGATIVE"
        assert confidence == 0.8
        moderator.sentiment_analyzer.assert_called_once_with(message)

    def test_analyze_sentiment_without_analyzer(self, moderator_no_sentiment):
        """Test sentiment analysis when analyzer is not available."""
        message = "I love this"
        sentiment, confidence = moderator_no_sentiment.analyze_sentiment(message)
        
        assert sentiment == "NEUTRAL"
        assert confidence == 0.0

    def test_moderate_message_valid(self, moderator):
        """Test complete moderation of valid user messages."""
        moderator.sentiment_analyzer = MagicMock(return_value=[{"label": "POSITIVE", "score": 0.9}])
        
        valid_message = "Hello, how can I help you today?"
        result = moderator.moderate_message(valid_message)
        
        assert result["approved"] is True
        assert result["message"] == valid_message
        assert result["sentiment"]["label"] == "POSITIVE"
        assert result["sentiment"]["score"] == 0.9
        assert len(result["filters_triggered"]) == 0

    def test_moderate_message_spam(self, moderator):
        """Test moderation rejection due to spam."""
        spam_message = "HELLO WORLD ATTENTION"
        result = moderator.moderate_message(spam_message)
        
        assert result["approved"] is False
        assert "spam" in result["filters_triggered"]

    def test_moderate_message_toxic(self, moderator):
        """Test moderation rejection due to toxic content."""
        toxic_message = "I hate everything about this"
        result = moderator.moderate_message(toxic_message)
        
        assert result["approved"] is False
        assert "toxic_content" in result["filters_triggered"]

    def test_filter_response_valid(self, moderator):
        """Test AI response filtering with valid content."""
        valid_response = "I can help you with that. Here's some information."
        result = moderator.filter_response(valid_response)
        
        assert result["approved"] is True
        assert result["filtered_response"] == valid_response
        assert len(result["filters_applied"]) == 0

    def test_filter_response_with_urls(self, moderator):
        """Test AI response filtering with URL removal."""
        response_with_urls = "Check this out: https://example.com and http://test.org for more info."
        result = moderator.filter_response(response_with_urls)
        
        assert result["approved"] is True
        assert "https://example.com" not in result["filtered_response"]
        assert "http://test.org" not in result["filtered_response"]
        assert "url_removal" in result["filters_applied"]


if __name__ == "__main__":
    pytest.main([__file__])
