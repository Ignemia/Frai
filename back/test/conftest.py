#!/usr/bin/env python3
"""
Test utilities and fixtures for backend AI chat tests.
Provides common test data, mocks, and helper functions.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import uuid
import random
import string

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add orchestrator module to path
orchestrator_path = project_root / "orchestrator"
sys.path.insert(0, str(orchestrator_path))

try:
    from orchestrator.chat import MessageType
except ImportError:
    # Fallback import path
    sys.path.insert(0, str(project_root / "orchestrator" / "chat"))
    from __init__ import MessageType


class TestDataGenerator:
    """Generates test data for chat components."""
    
    @staticmethod
    def random_string(length=10):
        """Generate a random string of specified length."""
        return ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=length))
    
    @staticmethod
    def random_user_id():
        """Generate a random user ID."""
        return f"user_{uuid.uuid4().hex[:8]}"
    
    @staticmethod
    def generate_clean_messages():
        """Generate clean, appropriate messages for testing."""
        return [
            "Hello, how are you today?",
            "Can you help me with my question?",
            "Thank you for the assistance",
            "I appreciate your help",
            "Have a wonderful day!",
            "This is very informative",
            "Great explanation, thanks!",
        ]
    
    @staticmethod
    def generate_spam_messages():
        """Generate various types of spam messages for testing."""
        return [
            "a" * 15,  # Repeated characters
            "URGENT URGENT URGENT URGENT URGENT",  # All caps
            "spam " * 30,  # Excessive repetition
            "BUY NOW " * 20,  # Commercial spam
            "!!!!!!!!!!!!!!!!!",  # Excessive punctuation
        ]
    
    @staticmethod
    def generate_toxic_messages():
        """Generate messages containing toxic keywords for testing."""
        return [
            "I hate this stupid system",
            "This is harmful and dangerous",
            "Violence is the only solution",
            "This is illegal activity",
            "Stop the abuse immediately",
        ]


class MockSentimentAnalyzer:
    """Mock sentiment analyzer for testing."""
    
    def __init__(self, default_sentiment="POSITIVE", default_confidence=0.8):
        self.default_sentiment = default_sentiment
        self.default_confidence = default_confidence
        self.call_count = 0
        self.last_input = None
    
    def __call__(self, text):
        """Mock sentiment analysis call."""
        self.call_count += 1
        self.last_input = text
        
        # Return different sentiments based on text content
        if any(word in text.lower() for word in ["hate", "angry", "frustrated", "terrible"]):
            return [{"label": "NEGATIVE", "score": 0.9}]
        elif any(word in text.lower() for word in ["love", "great", "wonderful", "excellent"]):
            return [{"label": "POSITIVE", "score": 0.95}]
        else:
            return [{"label": self.default_sentiment, "score": self.default_confidence}]


class TestAssertions:
    """Custom assertions for chat component testing."""
    
    @staticmethod
    def assert_valid_message(message, expected_type=None, expected_content=None):
        """Assert that a message object is valid."""
        assert hasattr(message, 'id')
        assert hasattr(message, 'chat_session_id')
        assert hasattr(message, 'message_type')
        assert hasattr(message, 'content')
        assert hasattr(message, 'timestamp')
        
        assert message.id is not None
        assert message.chat_session_id is not None
        assert isinstance(message.message_type, MessageType)
        assert isinstance(message.content, str)
        assert isinstance(message.timestamp, datetime)
        
        if expected_type:
            assert message.message_type == expected_type
        
        if expected_content:
            assert message.content == expected_content
    
    @staticmethod
    def assert_moderation_result(result, expected_approved=None, expected_filters=None):
        """Assert that a moderation result is valid."""
        assert isinstance(result, dict)
        assert "approved" in result
        assert "message" in result
        assert "filters_triggered" in result
        assert "sentiment" in result
        
        assert isinstance(result["approved"], bool)
        assert isinstance(result["filters_triggered"], list)
        assert isinstance(result["sentiment"], dict)
        
        if expected_approved is not None:
            assert result["approved"] == expected_approved
        
        if expected_filters:
            for filter_name in expected_filters:
                assert filter_name in result["filters_triggered"]


def create_mock_moderator(sentiment_enabled=True):
    """Create a mock moderator for testing."""
    try:
        from orchestrator.chatmod import ChatModerator
    except ImportError:
        # Fallback import path
        sys.path.insert(0, str(project_root / "orchestrator" / "chatmod"))
        from __init__ import ChatModerator

    with patch('orchestrator.chatmod.pipeline') as mock_pipeline:
        if sentiment_enabled:
            mock_analyzer = MockSentimentAnalyzer()
            mock_pipeline.return_value = mock_analyzer
        else:
            mock_pipeline.side_effect = Exception("Model not available")
        
        return ChatModerator()


# Export commonly used components
__all__ = [
    'TestDataGenerator',
    'MockSentimentAnalyzer', 
    'TestAssertions',
    'create_mock_moderator'
]
