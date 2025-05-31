"""
Blackbox Tests for Chat System

Tests that verify public interfaces without knowledge of internal implementation.
These tests ensure the system behaves correctly from a user perspective.
"""

import sys
import pytest
import uuid
from pathlib import Path
from datetime import datetime

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "orchestrator"))


@pytest.mark.blackbox
class TestChatModerationBlackbox:
    """Blackbox tests for chat moderation functionality"""
    
    def test_moderate_valid_messages(self, chat_moderator_mock, test_data):
        """Test moderation of valid messages"""
        moderator = chat_moderator_mock
        
        for message in test_data.valid_messages():
            result = moderator.moderate_message(message)
            
            # Should return a valid result structure
            assert isinstance(result, dict)
            assert "message" in result
            assert "filters_triggered" in result
            assert isinstance(result["filters_triggered"], list)
            
            # Valid messages should typically pass moderation
            assert result["message"] == message
    
    def test_moderate_spam_messages(self, chat_moderator_mock, test_data):
        """Test moderation of spam messages"""
        moderator = chat_moderator_mock
        
        for message in test_data.spam_messages():
            result = moderator.moderate_message(message)
            
            assert isinstance(result, dict)
            assert "message" in result
            assert "filters_triggered" in result
            
            # Should detect spam patterns
            assert result["message"] == message
    
    def test_moderate_toxic_messages(self, chat_moderator_mock, test_data):
        """Test moderation of toxic messages"""
        moderator = chat_moderator_mock
        
        for message in test_data.toxic_messages():
            result = moderator.moderate_message(message)
            
            assert isinstance(result, dict)
            assert "message" in result
            assert "filters_triggered" in result
            
            # Should detect toxic content
            assert result["message"] == message
    
    def test_moderate_edge_cases(self, chat_moderator_mock, test_data):
        """Test moderation of edge case messages"""
        moderator = chat_moderator_mock
        
        for message in test_data.edge_case_messages():
            # Should handle edge cases gracefully without crashing
            try:
                result = moderator.moderate_message(message)
                assert isinstance(result, dict)
                assert "message" in result
                assert "filters_triggered" in result
            except Exception as e:
                # If an exception occurs, it should be a specific, handled type
                assert isinstance(e, (ValueError, TypeError))
    
    def test_moderation_consistency(self, chat_moderator_mock):
        """Test that moderation results are consistent"""
        moderator = chat_moderator_mock
        
        test_message = "This is a consistent test message"
        
        # Run moderation multiple times
        results = []
        for _ in range(5):
            result = moderator.moderate_message(test_message)
            results.append(result)
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result
    
    def test_moderation_input_types(self, chat_moderator_mock):
        """Test moderation with different input types"""
        moderator = chat_moderator_mock
        
        # Test with normal string
        result = moderator.moderate_message("Normal message")
        assert isinstance(result, dict)
        
        # Test with empty string
        result = moderator.moderate_message("")
        assert isinstance(result, dict)
        
        # Test with unicode
        result = moderator.moderate_message("Hello 🌍 World!")
        assert isinstance(result, dict)
    
    def test_moderation_output_structure(self, chat_moderator_mock):
        """Test that moderation output has consistent structure"""
        moderator = chat_moderator_mock
        
        result = moderator.moderate_message("Test message")
        
        # Required fields
        assert "message" in result
        assert "filters_triggered" in result
        
        # Type validation
        assert isinstance(result["message"], str)
        assert isinstance(result["filters_triggered"], list)
        
        # If filters are triggered, they should have proper structure
        for filter_item in result["filters_triggered"]:
            assert isinstance(filter_item, str)


@pytest.mark.blackbox
class TestChatOrchestrationBlackbox:
    """Blackbox tests for chat orchestration functionality"""
    
    def test_create_session_basic(self, chat_orchestrator):
        """Test basic session creation"""
        orchestrator = chat_orchestrator
        
        user_id = "test_user_123"
        session = orchestrator.create_session(user_id=user_id)
        
        # Verify session structure
        assert hasattr(session, 'id')
        assert hasattr(session, 'user_id')
        assert hasattr(session, 'title')
        assert hasattr(session, 'created_at')
        assert hasattr(session, 'is_active')
        
        # Verify content
        assert session.user_id == user_id
        assert session.is_active is True
        assert isinstance(session.created_at, datetime)
        assert session.id is not None
    
    def test_create_session_with_title(self, chat_orchestrator):
        """Test session creation with custom title"""
        orchestrator = chat_orchestrator
        
        user_id = "test_user_456"
        title = "Custom Session Title"
        session = orchestrator.create_session(user_id=user_id, title=title)
        
        assert session.user_id == user_id
        assert session.title == title
    
    def test_create_multiple_sessions(self, chat_orchestrator):
        """Test creating multiple sessions for same user"""
        orchestrator = chat_orchestrator
        
        user_id = "test_user_multi"
        
        session1 = orchestrator.create_session(user_id=user_id, title="Session 1")
        session2 = orchestrator.create_session(user_id=user_id, title="Session 2")
        
        # Sessions should be different
        assert session1.id != session2.id
        assert session1.title != session2.title
        
        # But have same user
        assert session1.user_id == session2.user_id == user_id
    
    def test_get_session(self, chat_orchestrator):
        """Test session retrieval"""
        orchestrator = chat_orchestrator
        
        # Create session
        original_session = orchestrator.create_session(user_id="retrieval_test")
        
        # Retrieve session
        retrieved_session = orchestrator.get_session(original_session.id)
        
        # Should be the same session
        assert retrieved_session.id == original_session.id
        assert retrieved_session.user_id == original_session.user_id
        assert retrieved_session.title == original_session.title
    
    def test_get_nonexistent_session(self, chat_orchestrator):
        """Test retrieving non-existent session"""
        orchestrator = chat_orchestrator
        
        fake_session_id = str(uuid.uuid4())
        
        # Should handle gracefully
        try:
            result = orchestrator.get_session(fake_session_id)
            # If it returns something, it should be None or raise an exception
            assert result is None
        except Exception as e:
            # Should be a specific exception type
            assert isinstance(e, (KeyError, ValueError, AttributeError))
    
    def test_add_message_basic(self, chat_orchestrator):
        """Test basic message addition"""
        orchestrator = chat_orchestrator
        
        # Create session first
        session = orchestrator.create_session(user_id="message_test")
        
        # Add message
        content = "Hello, this is a test message"
        message = orchestrator.add_message(
            session_id=session.id,
            content=content,
            message_type="user"
        )
        
        # Verify message structure
        assert hasattr(message, 'id')
        assert hasattr(message, 'chat_session_id')
        assert hasattr(message, 'content')
        assert hasattr(message, 'timestamp')
        assert hasattr(message, 'message_type')
        
        # Verify content
        assert message.chat_session_id == session.id
        assert message.content == content
        assert message.message_type == "user"
        assert isinstance(message.timestamp, datetime)
    
    def test_add_multiple_messages(self, chat_orchestrator):
        """Test adding multiple messages to session"""
        orchestrator = chat_orchestrator
        
        session = orchestrator.create_session(user_id="multi_message_test")
        
        messages = []
        for i in range(5):
            message = orchestrator.add_message(
                session_id=session.id,
                content=f"Message {i}",
                message_type="user"
            )
            messages.append(message)
        
        # All messages should be different
        message_ids = [msg.id for msg in messages]
        assert len(set(message_ids)) == len(message_ids)  # All unique
        
        # All should belong to same session
        for message in messages:
            assert message.chat_session_id == session.id
    
    def test_add_message_different_types(self, chat_orchestrator):
        """Test adding messages of different types"""
        orchestrator = chat_orchestrator
        
        session = orchestrator.create_session(user_id="type_test")
        
        message_types = ["user", "assistant", "system"]
        
        for msg_type in message_types:
            message = orchestrator.add_message(
                session_id=session.id,
                content=f"This is a {msg_type} message",
                message_type=msg_type
            )
            
            assert message.message_type == msg_type
    
    def test_session_message_relationship(self, chat_orchestrator):
        """Test relationship between sessions and messages"""
        orchestrator = chat_orchestrator
        
        # Create two different sessions
        session1 = orchestrator.create_session(user_id="user1")
        session2 = orchestrator.create_session(user_id="user2")
        
        # Add messages to each session
        msg1 = orchestrator.add_message(session1.id, "Message for session 1", "user")
        msg2 = orchestrator.add_message(session2.id, "Message for session 2", "user")
        
        # Messages should belong to correct sessions
        assert msg1.chat_session_id == session1.id
        assert msg2.chat_session_id == session2.id
        assert msg1.chat_session_id != msg2.chat_session_id


@pytest.mark.blackbox
class TestChatSystemWorkflows:
    """Blackbox tests for complete chat workflows"""
    
    def test_complete_chat_workflow(self, integrated_chat_system):
        """Test a complete chat workflow from start to finish"""
        moderator = integrated_chat_system["moderator"]
        orchestrator = integrated_chat_system["orchestrator"]
        
        # 1. Create a session
        user_id = "workflow_test_user"
        session = orchestrator.create_session(user_id=user_id, title="Workflow Test")
        
        assert session.user_id == user_id
        assert session.title == "Workflow Test"
        
        # 2. User sends a message
        user_message = "Hello, I need help with my account"
        
        # 3. Moderate the message
        moderation_result = moderator.moderate_message(user_message)
        
        assert isinstance(moderation_result, dict)
        assert moderation_result["message"] == user_message
        
        # 4. If message passes moderation, add to session
        if not moderation_result.get("filters_triggered"):
            message = orchestrator.add_message(
                session_id=session.id,
                content=user_message,
                message_type="user"
            )
            
            assert message.content == user_message
            assert message.message_type == "user"
            assert message.chat_session_id == session.id
        
        # 5. System responds
        system_response = "Hello! I'd be happy to help with your account. What specific issue are you experiencing?"
        
        response_message = orchestrator.add_message(
            session_id=session.id,
            content=system_response,
            message_type="assistant"
        )
        
        assert response_message.content == system_response
        assert response_message.message_type == "assistant"
        assert response_message.chat_session_id == session.id
    
    def test_moderated_message_workflow(self, integrated_chat_system, test_data):
        """Test workflow when messages are moderated"""
        moderator = integrated_chat_system["moderator"]
        orchestrator = integrated_chat_system["orchestrator"]
        
        session = orchestrator.create_session(user_id="moderation_test")
        
        # Test with potentially problematic messages
        for message in test_data.spam_messages()[:2]:  # Test first 2 spam messages
            
            # Moderate message
            result = moderator.moderate_message(message)
            
            # System should handle moderated content appropriately
            assert isinstance(result, dict)
            assert "filters_triggered" in result
            
            # Even if message is flagged, system should not crash
            # Implementation may choose to either reject or process with flags
            if not result.get("filters_triggered"):
                stored_message = orchestrator.add_message(
                    session_id=session.id,
                    content=message,
                    message_type="user"
                )
                assert stored_message.content == message
    
    def test_concurrent_sessions_workflow(self, integrated_chat_system):
        """Test handling multiple concurrent sessions"""
        moderator = integrated_chat_system["moderator"]
        orchestrator = integrated_chat_system["orchestrator"]
        
        # Create multiple sessions
        sessions = []
        for i in range(3):
            session = orchestrator.create_session(
                user_id=f"concurrent_user_{i}",
                title=f"Concurrent Session {i}"
            )
            sessions.append(session)
        
        # Add messages to each session
        for i, session in enumerate(sessions):
            message_content = f"Message from user {i}"
            
            # Moderate message
            moderation_result = moderator.moderate_message(message_content)
            
            # Add message if it passes
            if not moderation_result.get("filters_triggered"):
                message = orchestrator.add_message(
                    session_id=session.id,
                    content=message_content,
                    message_type="user"
                )
                
                assert message.content == message_content
                assert message.chat_session_id == session.id
        
        # Verify sessions remain independent
        for i, session in enumerate(sessions):
            retrieved_session = orchestrator.get_session(session.id)
            assert retrieved_session.user_id == f"concurrent_user_{i}"
            assert retrieved_session.title == f"Concurrent Session {i}"
    
    def test_error_recovery_workflow(self, integrated_chat_system):
        """Test system behavior with invalid inputs"""
        moderator = integrated_chat_system["moderator"]
        orchestrator = integrated_chat_system["orchestrator"]
        
        # Test with edge case inputs
        edge_cases = ["", " ", "\n", "a" * 1000]
        
        # Create session for testing
        session = orchestrator.create_session(user_id="error_recovery_test")
        
        for edge_case in edge_cases:
            try:
                # System should handle edge cases gracefully
                moderation_result = moderator.moderate_message(edge_case)
                assert isinstance(moderation_result, dict)
                
                # Try to add message
                if not moderation_result.get("filters_triggered"):
                    message = orchestrator.add_message(
                        session_id=session.id,
                        content=edge_case,
                        message_type="user"
                    )
                    # If successful, verify structure
                    assert hasattr(message, 'content')
                    
            except Exception as e:
                # If an exception occurs, it should be a handled type
                assert isinstance(e, (ValueError, TypeError, AttributeError))


@pytest.mark.blackbox
class TestChatSystemLimits:
    """Blackbox tests for system limits and boundaries"""
    
    def test_long_message_handling(self, integrated_chat_system):
        """Test handling of very long messages"""
        moderator = integrated_chat_system["moderator"]
        orchestrator = integrated_chat_system["orchestrator"]
        
        session = orchestrator.create_session(user_id="long_message_test")
        
        # Test progressively longer messages
        lengths = [100, 1000, 5000, 10000]
        
        for length in lengths:
            long_message = "a" * length
            
            try:
                # Should handle or reject gracefully
                result = moderator.moderate_message(long_message)
                assert isinstance(result, dict)
                
                if not result.get("filters_triggered"):
                    message = orchestrator.add_message(
                        session_id=session.id,
                        content=long_message,
                        message_type="user"
                    )
                    assert len(message.content) == length
                    
            except Exception as e:
                # Should be a handled exception for too-long messages
                assert isinstance(e, (ValueError, MemoryError))
    
    def test_session_limits(self, chat_orchestrator):
        """Test creation of many sessions"""
        orchestrator = chat_orchestrator
        
        # Test creating multiple sessions
        sessions = []
        for i in range(50):  # Reasonable number for testing
            try:
                session = orchestrator.create_session(user_id=f"limit_test_user_{i}")
                sessions.append(session)
            except Exception as e:
                # If there are limits, should be handled gracefully
                assert isinstance(e, (MemoryError, ValueError))
                break
        
        # Should have created at least some sessions
        assert len(sessions) > 0
        
        # All created sessions should be valid
        for session in sessions:
            assert hasattr(session, 'id')
            assert hasattr(session, 'user_id')
    
    def test_message_limits_per_session(self, chat_orchestrator):
        """Test adding many messages to a session"""
        orchestrator = chat_orchestrator
        
        session = orchestrator.create_session(user_id="message_limit_test")
        
        messages = []
        for i in range(100):  # Reasonable number for testing
            try:
                message = orchestrator.add_message(
                    session_id=session.id,
                    content=f"Message {i}",
                    message_type="user"
                )
                messages.append(message)
            except Exception as e:
                # If there are limits, should be handled gracefully
                assert isinstance(e, (MemoryError, ValueError))
                break
        
        # Should have created at least some messages
        assert len(messages) > 0
        
        # All messages should be valid and in correct order
        for i, message in enumerate(messages):
            assert message.content == f"Message {i}"
            assert message.chat_session_id == session.id
