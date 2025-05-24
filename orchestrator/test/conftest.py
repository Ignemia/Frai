#!/usr/bin/env python3
"""
Test utilities and fixtures for orchestrator chat tests.
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
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Add orchestrator module to path
orchestrator_path = Path(__file__).parent.parent
sys.path.insert(0, str(orchestrator_path))

from chat import MessageType


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
    def random_session_id():
        """Generate a random session ID."""
        return str(uuid.uuid4())
    
    @staticmethod
    def random_message_id():
        """Generate a random message ID."""
        return str(uuid.uuid4())
    
    @staticmethod
    def generate_conversation_messages(count=5):
        """Generate a list of conversation messages."""
        messages = []
        for i in range(count):
            message_type = MessageType.USER if i % 2 == 0 else MessageType.ASSISTANT
            content = f"Test message {i+1}: {TestDataGenerator.random_string(20)}"
            messages.append((message_type, content))
        return messages
    
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
    def generate_ai_responses_with_urls():
        """Generate AI responses containing URLs for filtering tests."""
        return [
            "Check out https://example.com for more information",
            "Visit http://docs.example.org and https://support.example.net",
            "You can find resources at https://www.example.com/help",
            "Documentation: https://github.com/example/repo",
            "See https://stackoverflow.com/questions/123456 for details",
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


class ChatTestFixtures:
    """Common test fixtures for chat components."""
    
    @staticmethod
    @pytest.fixture
    def mock_sentiment_analyzer():
        """Fixture providing a mock sentiment analyzer."""
        return MockSentimentAnalyzer()
    
    @staticmethod
    @pytest.fixture
    def test_data_generator():
        """Fixture providing test data generator."""
        return TestDataGenerator()
    
    @staticmethod
    @pytest.fixture
    def sample_user_id():
        """Fixture providing a sample user ID."""
        return "test_user_123"
    
    @staticmethod
    @pytest.fixture
    def sample_session_data():
        """Fixture providing sample session data."""
        return {
            "user_id": "test_user_123",
            "title": "Test Chat Session",
            "created_at": datetime.now(),
            "is_active": True,
            "metadata": {"test": True, "environment": "testing"}
        }
    
    @staticmethod
    @pytest.fixture
    def sample_messages():
        """Fixture providing sample message data."""
        return [
            {
                "type": MessageType.USER,
                "content": "Hello, I need help with my account",
                "user_id": "test_user_123",
                "metadata": {"source": "web"}
            },
            {
                "type": MessageType.ASSISTANT,
                "content": "I'd be happy to help you with your account. What specific issue are you experiencing?",
                "metadata": {"model": "test_model"}
            },
            {
                "type": MessageType.USER,
                "content": "I can't log in to my account",
                "user_id": "test_user_123",
                "metadata": {"source": "web"}
            }
        ]


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
    def assert_valid_session(session, expected_user_id=None, expected_title=None):
        """Assert that a session object is valid."""
        assert hasattr(session, 'id')
        assert hasattr(session, 'user_id')
        assert hasattr(session, 'title')
        assert hasattr(session, 'created_at')
        assert hasattr(session, 'is_active')
        
        assert session.id is not None
        assert session.user_id is not None
        assert isinstance(session.title, str)
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.is_active, bool)
        
        if expected_user_id:
            assert session.user_id == expected_user_id
        
        if expected_title:
            assert session.title == expected_title
    
    @staticmethod
    def assert_moderation_result(result, expected_allowed=None, expected_flags=None):
        """Assert that a moderation result is valid."""
        assert isinstance(result, dict)
        assert "is_allowed" in result
        assert "message" in result
        assert "flags" in result
        assert "sentiment" in result
        assert "confidence" in result
        
        assert isinstance(result["is_allowed"], bool)
        assert isinstance(result["flags"], list)
        assert isinstance(result["sentiment"], str)
        assert isinstance(result["confidence"], (int, float))
        
        if expected_allowed is not None:
            assert result["is_allowed"] == expected_allowed
        
        if expected_flags:
            for flag in expected_flags:
                assert any(flag in f for f in result["flags"])
    
    @staticmethod
    def assert_ai_moderation_result(result, expected_safe=None):
        """Assert that an AI moderation result is valid."""
        assert isinstance(result, dict)
        assert "is_safe" in result
        assert "filtered_response" in result
        assert "removed_elements" in result
        assert "flags" in result
        
        assert isinstance(result["is_safe"], bool)
        assert isinstance(result["filtered_response"], str)
        assert isinstance(result["removed_elements"], list)
        assert isinstance(result["flags"], list)
        
        if expected_safe is not None:
            assert result["is_safe"] == expected_safe


class PerformanceTimer:
    """Context manager for measuring performance in tests."""
    
    def __init__(self, max_duration=None):
        self.max_duration = max_duration
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        
        if self.max_duration and self.duration > self.max_duration:
            raise AssertionError(f"Operation took {self.duration:.2f}s, expected < {self.max_duration}s")


def create_mock_moderator(sentiment_enabled=True):
    """Create a mock moderator for testing."""
    from chatmod import ChatModerator
    
    with patch('transformers.pipeline') as mock_pipeline:
        if sentiment_enabled:
            mock_analyzer = MockSentimentAnalyzer()
            mock_pipeline.return_value = mock_analyzer
        else:
            mock_pipeline.side_effect = Exception("Model not available")
        
        return ChatModerator()


def create_test_conversation(orchestrator, user_id, message_count=5):
    """Create a test conversation with specified number of messages."""
    session = orchestrator.create_chat_session(user_id, f"Test Conversation {message_count}")
    
    for i in range(message_count):
        if i % 2 == 0:
            # User message
            orchestrator.add_message(
                session.id, 
                MessageType.USER, 
                f"User message {i+1}", 
                user_id
            )
        else:
            # AI response
            orchestrator.add_message(
                session.id, 
                MessageType.ASSISTANT, 
                f"AI response {i+1}"
            )
    
    return session


# Export commonly used components
__all__ = [
    'TestDataGenerator',
    'MockSentimentAnalyzer', 
    'ChatTestFixtures',
    'TestAssertions',
    'PerformanceTimer',
    'create_mock_moderator',
    'create_test_conversation'
]
