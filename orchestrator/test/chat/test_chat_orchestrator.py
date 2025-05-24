#!/usr/bin/env python3
"""
Comprehensive tests for ChatOrchestrator class.
Tests session management, message handling, and conversation history.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import uuid
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Add orchestrator module to path
orchestrator_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(orchestrator_path))

from chat import ChatOrchestrator, ChatSession, ChatMessage, MessageType


class TestChatMessage:
    """Test suite for ChatMessage dataclass."""
    
    def test_chat_message_creation(self):
        """Test creating a ChatMessage instance."""
        message_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        message = ChatMessage(
            id=message_id,
            chat_session_id=session_id,
            message_type=MessageType.USER,
            content="Hello world",
            timestamp=timestamp,
            user_id="user123",
            metadata={"source": "web"}
        )
        
        assert message.id == message_id
        assert message.chat_session_id == session_id
        assert message.message_type == MessageType.USER
        assert message.content == "Hello world"
        assert message.timestamp == timestamp
        assert message.user_id == "user123"
        assert message.metadata == {"source": "web"}

    def test_chat_message_to_dict(self):
        """Test converting ChatMessage to dictionary."""
        timestamp = datetime.now()
        message = ChatMessage(
            id="msg1",
            chat_session_id="session1",
            message_type=MessageType.ASSISTANT,
            content="AI response",
            timestamp=timestamp
        )
        
        data = message.to_dict()
        
        assert data["id"] == "msg1"
        assert data["chat_session_id"] == "session1"
        assert data["message_type"] == "assistant"
        assert data["content"] == "AI response"
        assert data["timestamp"] == timestamp.isoformat()

    def test_chat_message_from_dict(self):
        """Test creating ChatMessage from dictionary."""
        timestamp = datetime.now()
        data = {
            "id": "msg1",
            "chat_session_id": "session1",
            "message_type": "user",
            "content": "User message",
            "timestamp": timestamp.isoformat(),
            "user_id": "user123",
            "metadata": {"test": "data"}
        }
        
        message = ChatMessage.from_dict(data)
        
        assert message.id == "msg1"
        assert message.message_type == MessageType.USER
        assert message.timestamp == timestamp
        assert message.user_id == "user123"
        assert message.metadata == {"test": "data"}

    def test_chat_message_serialization_roundtrip(self):
        """Test complete serialization and deserialization."""
        original = ChatMessage(
            id="test_id",
            chat_session_id="test_session",
            message_type=MessageType.SYSTEM,
            content="System message",
            timestamp=datetime.now(),
            user_id="system",
            metadata={"priority": "high"}
        )
        
        # Convert to dict and back
        data = original.to_dict()
        restored = ChatMessage.from_dict(data)
        
        assert restored.id == original.id
        assert restored.message_type == original.message_type
        assert restored.content == original.content
        assert restored.user_id == original.user_id
        assert restored.metadata == original.metadata


class TestChatSession:
    """Test suite for ChatSession dataclass."""
    
    def test_chat_session_creation(self):
        """Test creating a ChatSession instance."""
        session_id = str(uuid.uuid4())
        created_at = datetime.now()
        last_message_at = datetime.now()
        
        session = ChatSession(
            id=session_id,
            user_id="user123",
            title="Test Chat",
            created_at=created_at,
            last_message_at=last_message_at,
            is_active=True,
            metadata={"language": "en"}
        )
        
        assert session.id == session_id
        assert session.user_id == "user123"
        assert session.title == "Test Chat"
        assert session.created_at == created_at
        assert session.last_message_at == last_message_at
        assert session.is_active is True
        assert session.metadata == {"language": "en"}

    def test_chat_session_to_dict(self):
        """Test converting ChatSession to dictionary."""
        created_at = datetime.now()
        last_message_at = datetime.now()
        
        session = ChatSession(
            id="session1",
            user_id="user1",
            title="My Chat",
            created_at=created_at,
            last_message_at=last_message_at
        )
        
        data = session.to_dict()
        
        assert data["id"] == "session1"
        assert data["user_id"] == "user1"
        assert data["title"] == "My Chat"
        assert data["created_at"] == created_at.isoformat()
        assert data["last_message_at"] == last_message_at.isoformat()

    def test_chat_session_from_dict(self):
        """Test creating ChatSession from dictionary."""
        created_at = datetime.now()
        data = {
            "id": "session1",
            "user_id": "user1",
            "title": "Test Session",
            "created_at": created_at.isoformat(),
            "last_message_at": None,
            "is_active": True,
            "metadata": {"test": True}
        }
        
        session = ChatSession.from_dict(data)
        
        assert session.id == "session1"
        assert session.user_id == "user1"
        assert session.title == "Test Session"
        assert session.created_at == created_at
        assert session.last_message_at is None
        assert session.is_active is True

    def test_chat_session_serialization_roundtrip(self):
        """Test complete serialization and deserialization."""
        original = ChatSession(
            id="test_session",
            user_id="test_user",
            title="Test Title",
            created_at=datetime.now(),
            last_message_at=datetime.now(),
            is_active=False,
            metadata={"type": "test"}
        )
        
        data = original.to_dict()
        restored = ChatSession.from_dict(data)
        
        assert restored.id == original.id
        assert restored.user_id == original.user_id
        assert restored.title == original.title
        assert restored.is_active == original.is_active
        assert restored.metadata == original.metadata


class TestChatOrchestrator:
    """Test suite for ChatOrchestrator functionality."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create a ChatOrchestrator instance for testing."""
        return ChatOrchestrator()

    def test_orchestrator_initialization(self, orchestrator):
        """Test ChatOrchestrator initialization."""
        assert isinstance(orchestrator.sessions, dict)
        assert isinstance(orchestrator.messages, dict)
        assert isinstance(orchestrator.user_sessions, dict)
        assert len(orchestrator.sessions) == 0
        assert len(orchestrator.messages) == 0
        assert len(orchestrator.user_sessions) == 0

    def test_create_chat_session_with_title(self, orchestrator):
        """Test creating a chat session with custom title."""
        user_id = "user123"
        title = "My Custom Chat"
        
        session = orchestrator.create_chat_session(user_id, title)
        
        assert session.user_id == user_id
        assert session.title == title
        assert session.is_active is True
        assert session.id in orchestrator.sessions
        assert session.id in orchestrator.user_sessions[user_id]
        assert orchestrator.messages[session.id] == []

    def test_create_chat_session_without_title(self, orchestrator):
        """Test creating a chat session without title (auto-generated)."""
        user_id = "user456"
        
        session = orchestrator.create_chat_session(user_id)
        
        assert session.user_id == user_id
        assert "Chat Session" in session.title
        assert session.is_active is True

    def test_get_chat_session_existing(self, orchestrator):
        """Test retrieving an existing chat session."""
        user_id = "user123"
        session = orchestrator.create_chat_session(user_id, "Test Session")
        
        retrieved = orchestrator.get_chat_session(session.id)
        
        assert retrieved is not None
        assert retrieved.id == session.id
        assert retrieved.user_id == user_id
        assert retrieved.title == "Test Session"

    def test_get_chat_session_nonexistent(self, orchestrator):
        """Test retrieving a non-existent chat session."""
        fake_id = str(uuid.uuid4())
        retrieved = orchestrator.get_chat_session(fake_id)
        assert retrieved is None

    def test_delete_chat_session_existing(self, orchestrator):
        """Test deleting an existing chat session."""
        user_id = "user123"
        session = orchestrator.create_chat_session(user_id, "To Delete")
        session_id = session.id
        
        result = orchestrator.delete_chat_session(session_id)
        
        assert result is True
        assert session_id not in orchestrator.sessions
        assert session_id not in orchestrator.messages
        assert session_id not in orchestrator.user_sessions[user_id]

    def test_delete_chat_session_nonexistent(self, orchestrator):
        """Test deleting a non-existent chat session."""
        fake_id = str(uuid.uuid4())
        result = orchestrator.delete_chat_session(fake_id)
        assert result is False

    def test_add_message_to_session(self, orchestrator):
        """Test adding messages to a chat session."""
        user_id = "user123"
        session = orchestrator.create_chat_session(user_id, "Test Chat")
        
        # Add user message
        user_message = orchestrator.add_message(
            session.id, MessageType.USER, "Hello AI", user_id
        )
        
        assert user_message.chat_session_id == session.id
        assert user_message.message_type == MessageType.USER
        assert user_message.content == "Hello AI"
        assert user_message.user_id == user_id
        
        # Add AI response
        ai_message = orchestrator.add_message(
            session.id, MessageType.ASSISTANT, "Hello user!"
        )
        
        assert ai_message.message_type == MessageType.ASSISTANT
        assert ai_message.content == "Hello user!"
        
        # Check messages are stored
        messages = orchestrator.messages[session.id]
        assert len(messages) == 2
        assert messages[0] == user_message
        assert messages[1] == ai_message

    def test_add_message_nonexistent_session(self, orchestrator):
        """Test adding message to non-existent session."""
        fake_id = str(uuid.uuid4())
        
        with pytest.raises(ValueError, match="Session .* not found"):
            orchestrator.add_message(fake_id, MessageType.USER, "Test message")

    def test_get_conversation_history(self, orchestrator):
        """Test retrieving conversation history."""
        user_id = "user123"
        session = orchestrator.create_chat_session(user_id, "Test Chat")
        
        # Add multiple messages
        orchestrator.add_message(session.id, MessageType.USER, "Message 1", user_id)
        orchestrator.add_message(session.id, MessageType.ASSISTANT, "Response 1")
        orchestrator.add_message(session.id, MessageType.USER, "Message 2", user_id)
        
        # Get all history
        history = orchestrator.get_conversation_history(session.id)
        assert len(history) == 3
        assert history[0].content == "Message 1"
        assert history[1].content == "Response 1"
        assert history[2].content == "Message 2"
        
        # Get limited history
        limited_history = orchestrator.get_conversation_history(session.id, limit=2)
        assert len(limited_history) == 2
        assert limited_history[0].content == "Response 1"  # Most recent 2
        assert limited_history[1].content == "Message 2"

    def test_get_conversation_history_nonexistent_session(self, orchestrator):
        """Test getting history for non-existent session."""
        fake_id = str(uuid.uuid4())
        history = orchestrator.get_conversation_history(fake_id)
        assert history == []

    def test_get_user_sessions(self, orchestrator):
        """Test retrieving all sessions for a user."""
        user_id = "user123"
        
        # Create multiple sessions
        session1 = orchestrator.create_chat_session(user_id, "Session 1")
        session2 = orchestrator.create_chat_session(user_id, "Session 2")
        session3 = orchestrator.create_chat_session("other_user", "Other Session")
        
        user_sessions = orchestrator.get_user_sessions(user_id)
        
        assert len(user_sessions) == 2
        session_ids = [s.id for s in user_sessions]
        assert session1.id in session_ids
        assert session2.id in session_ids
        assert session3.id not in session_ids

    def test_get_user_sessions_nonexistent_user(self, orchestrator):
        """Test getting sessions for user with no sessions."""
        sessions = orchestrator.get_user_sessions("nonexistent_user")
        assert sessions == []

    def test_update_session_title(self, orchestrator):
        """Test updating session title."""
        user_id = "user123"
        session = orchestrator.create_chat_session(user_id, "Original Title")
        
        updated = orchestrator.update_session_title(session.id, "New Title")
        
        assert updated is True
        retrieved = orchestrator.get_chat_session(session.id)
        assert retrieved.title == "New Title"

    def test_update_session_title_nonexistent(self, orchestrator):
        """Test updating title for non-existent session."""
        fake_id = str(uuid.uuid4())
        result = orchestrator.update_session_title(fake_id, "New Title")
        assert result is False

    def test_get_session_stats(self, orchestrator):
        """Test getting session statistics."""
        user_id = "user123"
        session = orchestrator.create_chat_session(user_id, "Test Session")
        
        # Add some messages
        orchestrator.add_message(session.id, MessageType.USER, "Message 1", user_id)
        orchestrator.add_message(session.id, MessageType.ASSISTANT, "Response 1")
        orchestrator.add_message(session.id, MessageType.USER, "Message 2", user_id)
        
        stats = orchestrator.get_session_stats(session.id)
        
        assert stats["total_messages"] == 3
        assert stats["user_messages"] == 2
        assert stats["assistant_messages"] == 1
        assert stats["system_messages"] == 0
        assert "session_duration" in stats
        assert "last_activity" in stats

    def test_get_session_stats_nonexistent(self, orchestrator):
        """Test getting stats for non-existent session."""
        fake_id = str(uuid.uuid4())
        stats = orchestrator.get_session_stats(fake_id)
        assert stats is None

    def test_export_session_data(self, orchestrator):
        """Test exporting session data."""
        user_id = "user123"
        session = orchestrator.create_chat_session(user_id, "Export Test")
        
        # Add messages
        orchestrator.add_message(session.id, MessageType.USER, "Hello", user_id)
        orchestrator.add_message(session.id, MessageType.ASSISTANT, "Hi there!")
        
        exported = orchestrator.export_session_data(session.id)
        
        assert "session" in exported
        assert "messages" in exported
        assert "stats" in exported
        
        assert exported["session"]["id"] == session.id
        assert exported["session"]["title"] == "Export Test"
        assert len(exported["messages"]) == 2
        assert exported["stats"]["total_messages"] == 2

    def test_export_session_data_nonexistent(self, orchestrator):
        """Test exporting data for non-existent session."""
        fake_id = str(uuid.uuid4())
        exported = orchestrator.export_session_data(fake_id)
        assert exported is None

    def test_session_last_message_update(self, orchestrator):
        """Test that session last_message_at is updated when messages are added."""
        user_id = "user123"
        session = orchestrator.create_chat_session(user_id, "Test Session")
        
        original_last_message = session.last_message_at
        
        # Add a message
        orchestrator.add_message(session.id, MessageType.USER, "Test message", user_id)
        
        # Check that last_message_at was updated
        updated_session = orchestrator.get_chat_session(session.id)
        assert updated_session.last_message_at != original_last_message
        assert updated_session.last_message_at is not None

    def test_message_metadata_handling(self, orchestrator):
        """Test handling of message metadata."""
        user_id = "user123"
        session = orchestrator.create_chat_session(user_id, "Metadata Test")
        
        metadata = {"source": "web", "ip": "127.0.0.1", "priority": "high"}
        
        message = orchestrator.add_message(
            session.id, MessageType.USER, "Test with metadata", user_id, metadata
        )
        
        assert message.metadata == metadata
        
        # Retrieve and verify
        history = orchestrator.get_conversation_history(session.id)
        assert history[0].metadata == metadata

    def test_concurrent_session_operations(self, orchestrator):
        """Test concurrent operations on sessions."""
        import threading
        import time
        
        user_id = "user123"
        results = []
        
        def create_session_and_add_message(index):
            session = orchestrator.create_chat_session(user_id, f"Session {index}")
            message = orchestrator.add_message(
                session.id, MessageType.USER, f"Message {index}", user_id
            )
            results.append((session, message))
        
        # Create threads for concurrent operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_session_and_add_message, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all operations completed successfully
        assert len(results) == 5
        assert len(orchestrator.sessions) == 5
        assert len(orchestrator.user_sessions[user_id]) == 5
        
        # Verify each session has one message
        for session, message in results:
            assert len(orchestrator.messages[session.id]) == 1

    @pytest.mark.parametrize("message_type,expected_type", [
        (MessageType.USER, MessageType.USER),
        (MessageType.ASSISTANT, MessageType.ASSISTANT),
        (MessageType.SYSTEM, MessageType.SYSTEM),
    ])
    def test_message_type_handling(self, orchestrator, message_type, expected_type):
        """Parametrized test for different message types."""
        user_id = "user123"
        session = orchestrator.create_chat_session(user_id, "Type Test")
        
        message = orchestrator.add_message(
            session.id, message_type, f"Test {message_type.value} message", user_id
        )
        
        assert message.message_type == expected_type

    def test_large_conversation_handling(self, orchestrator):
        """Test handling of large conversations."""
        user_id = "user123"
        session = orchestrator.create_chat_session(user_id, "Large Conversation")
        
        # Add many messages
        num_messages = 1000
        for i in range(num_messages):
            message_type = MessageType.USER if i % 2 == 0 else MessageType.ASSISTANT
            orchestrator.add_message(
                session.id, message_type, f"Message {i}", user_id if message_type == MessageType.USER else None
            )
        
        # Test retrieval
        all_messages = orchestrator.get_conversation_history(session.id)
        assert len(all_messages) == num_messages
        
        # Test limited retrieval
        recent_messages = orchestrator.get_conversation_history(session.id, limit=10)
        assert len(recent_messages) == 10
        
        # Test stats
        stats = orchestrator.get_session_stats(session.id)
        assert stats["total_messages"] == num_messages


class TestChatOrchestratorIntegration:
    """Integration tests for ChatOrchestrator."""
    
    def test_complete_conversation_flow(self):
        """Test complete conversation workflow."""
        orchestrator = ChatOrchestrator()
        user_id = "user123"
        
        # Create session
        session = orchestrator.create_chat_session(user_id, "Integration Test")
        
        # Simulate conversation
        conversation_steps = [
            (MessageType.USER, "Hello, I need help"),
            (MessageType.ASSISTANT, "Hello! I'm here to help. What do you need assistance with?"),
            (MessageType.USER, "I want to learn about AI"),
            (MessageType.ASSISTANT, "AI is a fascinating field! Let me explain some key concepts..."),
            (MessageType.USER, "Thank you, that was helpful"),
            (MessageType.ASSISTANT, "You're welcome! Feel free to ask if you have more questions."),
        ]
        
        # Add all messages
        for msg_type, content in conversation_steps:
            orchestrator.add_message(
                session.id, msg_type, content, 
                user_id if msg_type == MessageType.USER else None
            )
        
        # Verify conversation
        history = orchestrator.get_conversation_history(session.id)
        assert len(history) == 6
        
        # Verify stats
        stats = orchestrator.get_session_stats(session.id)
        assert stats["total_messages"] == 6
        assert stats["user_messages"] == 3
        assert stats["assistant_messages"] == 3
        
        # Test export
        exported = orchestrator.export_session_data(session.id)
        assert len(exported["messages"]) == 6
        assert exported["stats"]["total_messages"] == 6

    def test_multi_user_scenario(self):
        """Test scenario with multiple users and sessions."""
        orchestrator = ChatOrchestrator()
        
        users = ["user1", "user2", "user3"]
        sessions_per_user = 3
        
        created_sessions = {}
        
        # Create multiple sessions for each user
        for user in users:
            created_sessions[user] = []
            for i in range(sessions_per_user):
                session = orchestrator.create_chat_session(user, f"Session {i+1}")
                created_sessions[user].append(session)
                
                # Add some messages to each session
                orchestrator.add_message(session.id, MessageType.USER, f"Message from {user}", user)
                orchestrator.add_message(session.id, MessageType.ASSISTANT, f"Response to {user}")
        
        # Verify user sessions
        for user in users:
            user_sessions = orchestrator.get_user_sessions(user)
            assert len(user_sessions) == sessions_per_user
            
            # Verify each session has messages
            for session in user_sessions:
                history = orchestrator.get_conversation_history(session.id)
                assert len(history) == 2

    def test_session_lifecycle_management(self):
        """Test complete session lifecycle."""
        orchestrator = ChatOrchestrator()
        user_id = "lifecycle_user"
        
        # Create session
        session = orchestrator.create_chat_session(user_id, "Lifecycle Test")
        original_title = session.title
        
        # Add messages
        orchestrator.add_message(session.id, MessageType.USER, "Initial message", user_id)
        orchestrator.add_message(session.id, MessageType.ASSISTANT, "Initial response")
        
        # Update session
        new_title = "Updated Lifecycle Test"
        orchestrator.update_session_title(session.id, new_title)
        
        # Verify update
        updated_session = orchestrator.get_chat_session(session.id)
        assert updated_session.title == new_title
        assert updated_session.title != original_title
        
        # Export session data
        exported = orchestrator.export_session_data(session.id)
        assert exported["session"]["title"] == new_title
        
        # Delete session
        result = orchestrator.delete_chat_session(session.id)
        assert result is True
        
        # Verify deletion
        deleted_session = orchestrator.get_chat_session(session.id)
        assert deleted_session is None
        
        # Verify user sessions updated
        user_sessions = orchestrator.get_user_sessions(user_id)
        assert len(user_sessions) == 0


if __name__ == "__main__":
    pytest.main([__file__])
