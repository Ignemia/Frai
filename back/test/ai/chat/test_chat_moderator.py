#!/usr/bin/env python3
"""
Comprehensive tests for ChatModerator class.
Tests message validation, spam detection, toxicity filtering, and sentiment analysis.
This is part of the backend AI chat functionality.
"""

import pytest
import unittest.mock as mock
from unittest.mock import patch, MagicMock
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Add orchestrator module to path
orchestrator_path = project_root / "orchestrator"
sys.path.insert(0, str(orchestrator_path))

try:
    from orchestrator.chatmod import ChatModerator
except ImportError:
    # Fallback import path
    sys.path.insert(0, str(project_root / "orchestrator" / "chatmod"))
    from __init__ import ChatModerator


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
    
    def test_init_with_local_sentiment_analyzer(self):
        """Test ChatModerator initialization with local sentiment analyzer."""
        # Mock the local model path to exist
        with patch('os.path.exists', return_value=True), \
             patch('chatmod.pipeline') as mock_pipeline:
            mock_analyzer = MagicMock()
            mock_pipeline.return_value = mock_analyzer
            
            moderator = ChatModerator()
            
            assert moderator.sentiment_analyzer is not None
            # Verify it tries to use the local model first
            mock_pipeline.assert_called_once()
            call_args = mock_pipeline.call_args
            assert "multilingual-sentiment-analysis" in str(call_args[1]['model'])

    def test_init_with_fallback_sentiment_analyzer(self):
        """Test ChatModerator initialization falling back to online model."""
        # Mock the local model path to not exist
        with patch('os.path.exists', return_value=False), \
             patch('chatmod.pipeline') as mock_pipeline:
            mock_analyzer = MagicMock()
            mock_pipeline.return_value = mock_analyzer
            
            moderator = ChatModerator()
            
            assert moderator.sentiment_analyzer is not None
            # Verify it falls back to online model
            mock_pipeline.assert_called_once()
            call_args = mock_pipeline.call_args
            assert "cardiffnlp/twitter-roberta-base-sentiment-latest" in str(call_args[1]['model'])

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
            assert "spam pattern" in reason.lower()

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
            assert "harmful content" in reason.lower()

    def test_analyze_sentiment_with_analyzer(self, moderator):
        """Test sentiment analysis when analyzer is available."""
        # Mock the sentiment analyzer
        mock_result = [{"label": "NEGATIVE", "score": 0.8}]
        moderator.sentiment_analyzer = MagicMock(return_value=mock_result)
        
        message = "I hate this"
        sentiment = moderator.analyze_sentiment(message)
        
        assert sentiment["label"] == "NEGATIVE"
        assert sentiment["score"] == 0.8
        moderator.sentiment_analyzer.assert_called_once_with(message)   
        
    def test_analyze_sentiment_without_analyzer(self, moderator_no_sentiment):
        """Test sentiment analysis when analyzer is not available."""
        message = "I love this"
        sentiment = moderator_no_sentiment.analyze_sentiment(message)
        
        assert sentiment is None

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
        spam_message = "HELLOOOOOOOOOOOO"  # More than 10 repeated characters
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
    """Integration tests for ChatModerator with backend AI components."""
    def test_local_sentiment_model_integration(self):
        """Test integration with local sentiment analysis model."""
        # Test that the moderator properly checks for and uses local model
        with patch('os.path.exists') as mock_exists, \
             patch('chatmod.pipeline') as mock_pipeline:
            
            # First call checks for local model
            mock_exists.return_value = True
            mock_pipeline.return_value = MagicMock()
            
            moderator = ChatModerator()
            
            # Verify local model path was checked
            mock_exists.assert_called_once()
            local_path_checked = mock_exists.call_args[0][0]
            assert "multilingual-sentiment-analysis" in local_path_checked
            
            # Verify pipeline was called with local model
            mock_pipeline.assert_called_once()
            call_args = mock_pipeline.call_args
            assert "multilingual-sentiment-analysis" in str(call_args[1]['model'])

    def test_end_to_end_moderation_workflow(self):
        """Test complete moderation workflow for backend AI chat."""
        with patch('chatmod.pipeline') as mock_pipeline:
            mock_analyzer = MagicMock()
            mock_analyzer.return_value = [{"label": "POSITIVE", "score": 0.95}]
            mock_pipeline.return_value = mock_analyzer
            
            moderator = ChatModerator()
            
            # Test user message moderation
            message = "Hello, I need help with my account."
            result = moderator.moderate_message(message)
            
            assert result["approved"] is True
            assert result["sentiment"]["label"] == "POSITIVE"
            assert result["sentiment"]["score"] == 0.95
            
            # Test AI response filtering
            ai_response = "I can help you with your account. Please provide more details."
            ai_result = moderator.filter_response(ai_response)
            
            assert ai_result["approved"] is True
            assert ai_result["filtered_response"] == ai_response

    def test_performance_with_backend_models(self):
        """Test moderation performance suitable for backend operations."""
        with patch('chatmod.pipeline') as mock_pipeline:
            mock_pipeline.return_value = MagicMock()
            moderator = ChatModerator()
            
            # Test with maximum allowed message size
            large_message = "a" * moderator.max_message_length
            result = moderator.moderate_message(large_message)
            
            assert result["approved"] is True

    def test_concurrent_backend_operations(self):
        """Test moderation with concurrent backend requests."""
        import threading
        
        with patch('chatmod.pipeline') as mock_pipeline:
            mock_pipeline.return_value = MagicMock()
            moderator = ChatModerator()
            
            results = []
            
            def moderate_message(msg):
                result = moderator.moderate_message(f"Backend message {msg}")
                results.append(result)
            
            # Simulate concurrent backend requests
            threads = []
            for i in range(10):
                thread = threading.Thread(target=moderate_message, args=(i,))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            assert len(results) == 10
            assert all(result["approved"] for result in results)


if __name__ == "__main__":
    pytest.main([__file__])
