"""
Test Configuration and Fixtures

Provides common test fixtures, mocks, and utilities for the comprehensive test suite.
"""

import sys
import pytest
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockSentimentAnalyzer:
    """Mock sentiment analyzer that doesn't require transformers library"""
    
    def __init__(self, default_sentiment="POSITIVE", default_score=0.8):
        self.default_sentiment = default_sentiment
        self.default_score = default_score
        self.call_count = 0
        
    def __call__(self, text):
        self.call_count += 1
        
        # Return different sentiments based on text content
        if any(word in text.lower() for word in ["hate", "angry", "bad", "terrible", "awful"]):
            return [{"label": "NEGATIVE", "score": 0.95}]
        elif any(word in text.lower() for word in ["love", "great", "good", "excellent", "amazing"]):
            return [{"label": "POSITIVE", "score": 0.95}]
        else:
            return [{"label": self.default_sentiment, "score": self.default_score}]


class TestDataFactory:
    """Factory for generating test data"""
    
    @staticmethod
    def valid_messages():
        return [
            "Hello, how are you today?",
            "I need help with my account settings.",
            "Can you explain how this feature works?",
            "Thank you for your assistance.",
            "What are the available options?",
        ]
    
    @staticmethod
    def spam_messages():
        return [
            "URGENT URGENT URGENT ATTENTION PLEASE",
            "aaaaaaaaaaaaaaaaaaaaaa",
            "BUY NOW " * 20,
            "!!!!!!!!!!!!!!!!!!!!!!!",
            "spam spam spam spam spam spam spam",
        ]
    
    @staticmethod
    def toxic_messages():
        return [
            "I hate this stupid system",
            "This is harmful and dangerous content",
            "Violence is the answer to everything",
            "Stop the abuse immediately",
            "This is completely illegal activity",
        ]
    
    @staticmethod
    def edge_case_messages():
        return [
            "",  # Empty
            " ",  # Whitespace only
            "a",  # Single character
            "a" * 10000,  # Very long
            "\n\t\r",  # Special characters
            "Hello\x00World",  # Null bytes
            "🔥💯🚀" * 50,  # Emojis
        ]


@pytest.fixture
def mock_sentiment_analyzer():
    """Provides a mock sentiment analyzer"""
    return MockSentimentAnalyzer()


@pytest.fixture
def mock_transformers_pipeline():
    """Mock the transformers pipeline to avoid dependency issues"""
    with patch('transformers.pipeline') as mock_pipeline:
        mock_analyzer = MockSentimentAnalyzer()
        mock_pipeline.return_value = mock_analyzer
        yield mock_pipeline


@pytest.fixture
def test_data():
    """Provides test data factory"""
    return TestDataFactory()


@pytest.fixture
def chat_moderator_mock():
    """Create a mocked ChatModerator instance"""
    with patch('transformers.pipeline'):
        # Import here to avoid early import issues
        sys.path.insert(0, str(project_root / "orchestrator"))
        from chatmod import ChatModerator
        
        moderator = ChatModerator()
        moderator.sentiment_analyzer = MockSentimentAnalyzer()
        return moderator


@pytest.fixture
def chat_orchestrator():
    """Create a ChatOrchestrator instance"""
    sys.path.insert(0, str(project_root / "orchestrator"))
    from chat.orchestrator import ChatOrchestrator
    return ChatOrchestrator()


@pytest.fixture
def integrated_chat_system(chat_moderator_mock, chat_orchestrator):
    """Create an integrated chat system for testing"""
    return {
        "moderator": chat_moderator_mock,
        "orchestrator": chat_orchestrator
    }


def assert_valid_message_result(result: Dict[str, Any], should_pass: bool = True):
    """Helper function to assert message moderation results"""
    assert isinstance(result, dict)
    assert "message" in result
    assert "filters_triggered" in result
    
    if should_pass:
        assert len(result["filters_triggered"]) == 0 or all(
            filter_name not in ["toxic_content", "spam"] 
            for filter_name in result["filters_triggered"]
        )
    else:
        assert len(result["filters_triggered"]) > 0


def assert_valid_session(session, expected_user_id: str, expected_title: Optional[str] = None):
    """Helper function to assert session validity"""
    assert hasattr(session, 'id')
    assert hasattr(session, 'user_id')
    assert hasattr(session, 'title')
    assert hasattr(session, 'created_at')
    assert hasattr(session, 'is_active')
    
    assert session.user_id == expected_user_id
    assert session.is_active is True
    assert isinstance(session.created_at, datetime)
    
    if expected_title:
        assert session.title == expected_title


def assert_valid_message(message, expected_content: str, expected_type=None):
    """Helper function to assert message validity"""
    assert hasattr(message, 'id')
    assert hasattr(message, 'chat_session_id')
    assert hasattr(message, 'content')
    assert hasattr(message, 'timestamp')
    assert hasattr(message, 'message_type')
    
    assert message.content == expected_content
    assert isinstance(message.timestamp, datetime)
    
    if expected_type:
        assert message.message_type == expected_type
