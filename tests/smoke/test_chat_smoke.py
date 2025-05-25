"""
Smoke Tests for Chat System

Basic functionality tests to ensure core components can be imported and instantiated.
These tests catch critical failures early and run quickly.
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "orchestrator"))


@pytest.mark.smoke
class TestChatSystemSmoke:
    """Smoke tests for basic chat system functionality"""
    
    def test_chat_moderator_import(self, mock_transformers_pipeline):
        """Test that ChatModerator can be imported and instantiated"""
        from chatmod import ChatModerator
        
        moderator = ChatModerator()
        assert moderator is not None
        assert hasattr(moderator, 'moderate_message')
        assert hasattr(moderator, 'sentiment_analyzer')
    
    def test_chat_orchestrator_import(self):
        """Test that ChatOrchestrator can be imported and instantiated"""
        from chat.orchestrator import ChatOrchestrator
        
        orchestrator = ChatOrchestrator()
        assert orchestrator is not None
        assert hasattr(orchestrator, 'create_session')
        assert hasattr(orchestrator, 'add_message')
        assert hasattr(orchestrator, 'get_session')
    
    def test_chat_models_import(self):
        """Test that chat models can be imported"""
        from chat.models import ChatMessage, ChatSession
        
        # Test model classes exist
        assert ChatMessage is not None
        assert ChatSession is not None
        
        # Test basic instantiation (without actual data)
        assert hasattr(ChatMessage, '__init__')
        assert hasattr(ChatSession, '__init__')
    
    def test_basic_message_moderation(self, chat_moderator_mock, test_data):
        """Test basic message moderation functionality"""
        moderator = chat_moderator_mock
        
        # Test with a simple valid message
        valid_message = "Hello, how are you?"
        result = moderator.moderate_message(valid_message)
        
        assert result is not None
        assert isinstance(result, dict)
        assert "message" in result
        assert "filters_triggered" in result
    
    def test_basic_session_creation(self, chat_orchestrator):
        """Test basic session creation functionality"""
        orchestrator = chat_orchestrator
        
        # Test session creation
        session = orchestrator.create_session(user_id="test_user_123")
        
        assert session is not None
        assert hasattr(session, 'id')
        assert hasattr(session, 'user_id')
        assert session.user_id == "test_user_123"
    
    def test_basic_message_handling(self, chat_orchestrator):
        """Test basic message adding functionality"""
        orchestrator = chat_orchestrator
        
        # Create session first
        session = orchestrator.create_session(user_id="test_user_123")
        
        # Add a message
        message = orchestrator.add_message(
            session_id=session.id,
            content="Test message",
            message_type="user"
        )
        
        assert message is not None
        assert hasattr(message, 'id')
        assert hasattr(message, 'content')
        assert message.content == "Test message"
    
    def test_session_retrieval(self, chat_orchestrator):
        """Test basic session retrieval functionality"""
        orchestrator = chat_orchestrator
        
        # Create session
        original_session = orchestrator.create_session(user_id="test_user_123")
        
        # Retrieve session
        retrieved_session = orchestrator.get_session(original_session.id)
        
        assert retrieved_session is not None
        assert retrieved_session.id == original_session.id
        assert retrieved_session.user_id == original_session.user_id


@pytest.mark.smoke 
class TestChatSystemConfiguration:
    """Smoke tests for system configuration and dependencies"""
    
    def test_logging_configuration(self):
        """Test that logging is properly configured"""
        import logging
        
        logger = logging.getLogger("test_logger")
        assert logger is not None
        
        # Test that we can log without errors
        logger.info("Test log message")
        logger.debug("Test debug message")
    
    def test_dependency_mocking(self, mock_transformers_pipeline):
        """Test that dependency mocking works"""
        assert mock_transformers_pipeline is not None
        
        # Verify the mock was called when ChatModerator is instantiated
        from chatmod import ChatModerator
        moderator = ChatModerator()
        
        # The mock should have been called during instantiation
        assert mock_transformers_pipeline.called
    
    def test_test_data_factory(self, test_data):
        """Test that test data factory works"""
        assert test_data is not None
        
        # Test valid messages
        valid_msgs = test_data.valid_messages()
        assert isinstance(valid_msgs, list)
        assert len(valid_msgs) > 0
        assert all(isinstance(msg, str) for msg in valid_msgs)
        
        # Test spam messages
        spam_msgs = test_data.spam_messages()
        assert isinstance(spam_msgs, list)
        assert len(spam_msgs) > 0
        
        # Test toxic messages
        toxic_msgs = test_data.toxic_messages()
        assert isinstance(toxic_msgs, list)
        assert len(toxic_msgs) > 0
    
    def test_test_fixtures(self, chat_moderator_mock, chat_orchestrator, integrated_chat_system):
        """Test that all fixtures are properly configured"""
        assert chat_moderator_mock is not None
        assert chat_orchestrator is not None
        assert integrated_chat_system is not None
        
        assert "moderator" in integrated_chat_system
        assert "orchestrator" in integrated_chat_system
        assert integrated_chat_system["moderator"] == chat_moderator_mock
        assert integrated_chat_system["orchestrator"] == chat_orchestrator


@pytest.mark.smoke
class TestChatSystemPerformance:
    """Basic performance smoke tests"""
    
    def test_moderator_response_time(self, chat_moderator_mock):
        """Test that moderation doesn't take too long"""
        import time
        
        moderator = chat_moderator_mock
        message = "This is a test message for performance testing"
        
        start_time = time.time()
        result = moderator.moderate_message(message)
        end_time = time.time()
        
        # Should complete within reasonable time (1 second)
        assert (end_time - start_time) < 1.0
        assert result is not None
    
    def test_session_creation_performance(self, chat_orchestrator):
        """Test that session creation is fast"""
        import time
        
        orchestrator = chat_orchestrator
        
        start_time = time.time()
        session = orchestrator.create_session(user_id="perf_test_user")
        end_time = time.time()
        
        # Should complete within reasonable time (0.1 seconds)
        assert (end_time - start_time) < 0.1
        assert session is not None
    
    def test_multiple_operations_performance(self, integrated_chat_system):
        """Test performance of multiple operations"""
        import time
        
        moderator = integrated_chat_system["moderator"]
        orchestrator = integrated_chat_system["orchestrator"]
        
        start_time = time.time()
        
        # Create session
        session = orchestrator.create_session(user_id="bulk_test_user")
        
        # Add multiple messages
        for i in range(10):
            message_content = f"Test message {i}"
            
            # Moderate message
            moderation_result = moderator.moderate_message(message_content)
            
            # Add message if it passes moderation
            if not moderation_result.get("filters_triggered"):
                orchestrator.add_message(
                    session_id=session.id,
                    content=message_content,
                    message_type="user"
                )
        
        end_time = time.time()
        
        # All operations should complete within reasonable time (2 seconds)
        assert (end_time - start_time) < 2.0
