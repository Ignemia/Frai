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

        # Test non-string message
        is_valid, error = moderator.validate_message_format(123)
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

    def test_check_spam_patterns_all_caps(self, moderator):
        """Test spam detection for all caps sequences."""
        spam_messages = [
            "HELLO WORLD",  # 5+ consecutive caps
            "THIS IS SPAM",
            "ATTENTION PLEASE",
        ]
        
        for message in spam_messages:
            is_spam, reason = moderator.check_spam_patterns(message)
            assert is_spam, f"Message should be detected as spam: {message}"
            assert "excessive caps" in reason.lower()

    def test_check_spam_patterns_excessive_repetition(self, moderator):
        """Test spam detection for excessive word repetition."""
        # Create a message with 20+ repeated words
        repeated_message = "spam " * 25
        is_spam, reason = moderator.check_spam_patterns(repeated_message)
        assert is_spam
        assert "excessive repetition" in reason.lower()

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
            "This is illegal activity",
            "Stop the abuse now",
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

    def test_analyze_sentiment_exception_handling(self, moderator):
        """Test sentiment analysis exception handling."""
        moderator.sentiment_analyzer = MagicMock(side_effect=Exception("Analysis failed"))
        
        message = "Test message"
        sentiment, confidence = moderator.analyze_sentiment(message)
        
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

    def test_moderate_message_format_invalid(self, moderator):
        """Test moderation rejection due to format issues."""
        invalid_message = ""
        result = moderator.moderate_message(invalid_message)
        
        assert result["approved"] is False
        assert result["error"] is not None

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

    def test_filter_response_toxic(self, moderator):
        """Test AI response filtering with toxic content."""
        toxic_response = "This response contains hate and violence."
        result = moderator.filter_response(toxic_response)
        
        assert result["approved"] is False
        assert len(result["warnings"]) > 0

    def test_filter_response_empty(self, moderator):
        """Test AI response filtering with empty content."""
        empty_response = ""
        result = moderator.filter_response(empty_response)
        
        assert result["approved"] is False
        assert len(result["warnings"]) > 0

    def test_get_moderation_stats(self, moderator):
        """Test moderation statistics tracking."""
        # Perform some moderation actions
        moderator.moderate_user_message("Hello world")
        moderator.moderate_user_message("SPAM MESSAGE")
        moderator.moderate_ai_response("Good response")
        
        stats = moderator.get_moderation_stats()
        
        assert "user_messages_processed" in stats
        assert "user_messages_blocked" in stats
        assert "ai_responses_processed" in stats
        assert "ai_responses_blocked" in stats
        assert stats["user_messages_processed"] == 2
        assert stats["user_messages_blocked"] == 1
        assert stats["ai_responses_processed"] == 1

    @pytest.mark.parametrize("message,expected_spam", [
        ("Normal message", False),
        ("Helloooooooooooo", True),
        ("CAPS LOCK MESSAGE", True),
        ("spam " * 25, True),
        ("Good morning!", False),
    ])
    def test_spam_detection_parametrized(self, moderator, message, expected_spam):
        """Parametrized test for spam detection."""
        is_spam, _ = moderator.check_spam_patterns(message)
        assert is_spam == expected_spam

    @pytest.mark.parametrize("message,expected_toxic", [
        ("Hello there", False),
        ("I hate this", True),
        ("Beautiful day", False),
        ("Violence is bad", True),
        ("Thank you", False),
    ])
    def test_toxic_detection_parametrized(self, moderator, message, expected_toxic):
        """Parametrized test for toxic content detection."""
        is_toxic, _ = moderator.check_toxic_content(message)
        assert is_toxic == expected_toxic


class TestChatModeratorIntegration:
    """Integration tests for ChatModerator with real components."""
    
    def test_end_to_end_moderation_flow(self):
        """Test complete moderation workflow."""
        with patch('transformers.pipeline') as mock_pipeline:
            mock_analyzer = MagicMock()
            mock_analyzer.return_value = [{"label": "POSITIVE", "score": 0.95}]
            mock_pipeline.return_value = mock_analyzer
            
            moderator = ChatModerator()
            
            # Test valid message flow
            message = "Hello, I need help with my account."
            result = moderator.moderate_user_message(message)
            
            assert result["is_allowed"] is True
            assert result["sentiment"] == "POSITIVE"
            assert result["confidence"] == 0.95
            
            # Test AI response moderation
            ai_response = "I can help you with your account. Please provide more details."
            ai_result = moderator.moderate_ai_response(ai_response)
            
            assert ai_result["is_safe"] is True
            assert ai_result["filtered_response"] == ai_response

    def test_performance_with_large_messages(self):
        """Test moderation performance with large messages."""
        with patch('transformers.pipeline') as mock_pipeline:
            mock_pipeline.return_value = MagicMock()
            moderator = ChatModerator()
            
            # Test with maximum allowed message size
            large_message = "a" * moderator.max_message_length
            result = moderator.moderate_user_message(large_message)
            
            assert result["is_allowed"] is True

    def test_concurrent_moderation(self):
        """Test moderation with concurrent requests simulation."""
        import threading
        import time
        
        with patch('transformers.pipeline') as mock_pipeline:
            mock_pipeline.return_value = MagicMock()
            moderator = ChatModerator()
            
            results = []
            
            def moderate_message(msg):
                result = moderator.moderate_user_message(f"Message {msg}")
                results.append(result)
            
            # Simulate concurrent requests
            threads = []
            for i in range(10):
                thread = threading.Thread(target=moderate_message, args=(i,))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            assert len(results) == 10
            assert all(result["is_allowed"] for result in results)


if __name__ == "__main__":
    pytest.main([__file__])
