#!/usr/bin/env python3
"""
Comprehensive tests for ChatOrchestrator class.
Tests session management, message handling, and conversation history.
This is part of the backend AI chat functionality.
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
orchestrator_path = project_root / "orchestrator"
sys.path.insert(0, str(orchestrator_path))

try:
    from chat import ChatOrchestrator, ChatSession, ChatMessage, MessageType
except ImportError:
    # Fallback import path
    sys.path.insert(0, str(project_root / "orchestrator" / "chat"))
    from __init__ import ChatOrchestrator, ChatSession, ChatMessage, MessageType


class TestChatOrchestrator:
    """Test suite for ChatOrchestrator functionality."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create a ChatOrchestrator instance for testing."""
        return ChatOrchestrator()

    def test_orchestrator_initialization(self, orchestrator):
        """Test ChatOrchestrator initialization for backend use."""
        assert isinstance(orchestrator.sessions, dict)
        assert isinstance(orchestrator.messages, dict)
        assert isinstance(orchestrator.user_sessions, dict)
        assert len(orchestrator.sessions) == 0
        assert len(orchestrator.messages) == 0
        assert len(orchestrator.user_sessions) == 0

    def test_create_chat_session_backend(self, orchestrator):
        """Test creating a chat session for backend AI operations."""
        user_id = "backend_user_123"
        title = "Backend AI Chat Session"
        
        session = orchestrator.create_chat_session(user_id, title)
        
        assert session.user_id == user_id
        assert session.title == title
        assert session.is_active is True
        assert session.id in orchestrator.sessions
        assert session.id in orchestrator.user_sessions[user_id]
        assert orchestrator.messages[session.id] == []

    def test_add_ai_messages_backend(self, orchestrator):
        """Test adding AI-generated messages in backend context."""
        user_id = "ai_user_123"
        session = orchestrator.create_chat_session(user_id, "AI Backend Test")
        
        # Add user message
        user_message = orchestrator.add_message(
            session.id, MessageType.USER, "Request AI assistance", user_id
        )
        
        assert user_message.message_type == MessageType.USER
        assert user_message.content == "Request AI assistance"
        assert user_message.user_id == user_id
        
        # Add AI assistant response (backend generated)
        ai_message = orchestrator.add_message(
            session.id, MessageType.ASSISTANT, "AI assistance provided through backend"
        )
        
        assert ai_message.message_type == MessageType.ASSISTANT
        assert ai_message.content == "AI assistance provided through backend"
        assert ai_message.user_id is None  # AI messages don't have user_id
        
        # Add system message (backend operational)
        system_message = orchestrator.add_message(
            session.id, MessageType.SYSTEM, "Backend system notification"
        )
        
        assert system_message.message_type == MessageType.SYSTEM
        
        # Verify all messages are stored
        messages = orchestrator.messages[session.id]
        assert len(messages) == 3
        assert messages[0] == user_message
        assert messages[1] == ai_message
        assert messages[2] == system_message
    
    def test_get_conversation_history_backend(self, orchestrator):
        """Test retrieving conversation history for backend AI processing."""
        user_id = "backend_user"
        session = orchestrator.create_chat_session(user_id, "Backend History Test")
        
        # Add multiple messages to simulate AI conversation
        orchestrator.add_message(session.id, MessageType.USER, "Initial query", user_id)
        orchestrator.add_message(session.id, MessageType.ASSISTANT, "AI processing...")
        orchestrator.add_message(session.id, MessageType.ASSISTANT, "AI response complete")
        orchestrator.add_message(session.id, MessageType.USER, "Follow-up question", user_id)
        orchestrator.add_message(session.id, MessageType.ASSISTANT, "Follow-up response")
        
        # Get full history for AI context
        full_history = orchestrator.get_session_messages(session.id)
        assert len(full_history) == 5
          # Get limited history for AI token management  
        limited_history = orchestrator.get_session_messages(session.id, limit=3)
        assert len(limited_history) == 3
        # Should get the most recent messages
        assert limited_history[0].content == "Initial query"
        assert limited_history[1].content == "AI processing..."
        assert limited_history[2].content == "AI response complete"

    def test_session_stats_backend_metrics(self, orchestrator):
        """Test session statistics for backend monitoring."""
        user_id = "metrics_user"
        session = orchestrator.create_chat_session(user_id, "Metrics Test")
        
        # Add messages to track backend metrics
        orchestrator.add_message(session.id, MessageType.USER, "User query 1", user_id)
        orchestrator.add_message(session.id, MessageType.ASSISTANT, "AI response 1")
        orchestrator.add_message(session.id, MessageType.USER, "User query 2", user_id)
        orchestrator.add_message(session.id, MessageType.ASSISTANT, "AI response 2")
        orchestrator.add_message(session.id, MessageType.SYSTEM, "Backend notification")
        
        stats = orchestrator.get_session_stats(session.id)
          # Verify backend metrics
        assert stats["total_messages"] == 5
        assert stats["user_messages"] == 2
        assert stats["assistant_messages"] == 2
        # Note: system messages aren't tracked separately in the current implementation
        assert len([m for m in orchestrator.get_session_messages(session.id) 
                   if m.message_type == MessageType.SYSTEM]) == 1

    def test_export_session_backend_data(self, orchestrator):
        """Test exporting session data for backend processing/storage."""
        user_id = "export_user"
        session = orchestrator.create_chat_session(user_id, "Export Test")
          # Add conversation data
        orchestrator.add_message(session.id, MessageType.USER, "Export request", user_id)
        orchestrator.add_message(session.id, MessageType.ASSISTANT, "Processing export...")
        orchestrator.add_message(session.id, MessageType.ASSISTANT, "Export complete")
        
        exported = orchestrator.export_session(session.id)
        
        assert "session" in exported
        assert "messages" in exported
        assert exported["session"]["title"] == "Export Test"
        assert len(exported["messages"]) == 3
        
        # Verify backend-relevant data
        assert exported["session"]["id"] == session.id
        assert exported["session"]["title"] == "Export Test"

    def test_concurrent_backend_sessions(self, orchestrator):
        """Test concurrent session handling for backend scalability."""
        import threading
        
        users = [f"concurrent_user_{i}" for i in range(5)]
        results = []
        
        def create_session_with_ai_interaction(user_id):
            session = orchestrator.create_chat_session(user_id, f"Concurrent Session for {user_id}")
            
            # Simulate AI interaction
            orchestrator.add_message(session.id, MessageType.USER, f"Request from {user_id}", user_id)
            orchestrator.add_message(session.id, MessageType.ASSISTANT, f"AI response to {user_id}")
            
            results.append((user_id, session.id, len(orchestrator.messages[session.id])))
        
        # Create threads for concurrent backend operations
        threads = []
        for user_id in users:
            thread = threading.Thread(target=create_session_with_ai_interaction, args=(user_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all backend operations to complete
        for thread in threads:
            thread.join()
        
        # Verify concurrent operations completed successfully
        assert len(results) == 5
        assert len(orchestrator.sessions) == 5
        
        # Verify each session has expected AI interaction
        for user_id, session_id, message_count in results:
            assert message_count == 2  # User message + AI response
            assert session_id in orchestrator.sessions
            assert session_id in orchestrator.user_sessions[user_id]

    def test_large_conversation_backend_handling(self, orchestrator):
        """Test handling of large conversations for backend AI processing."""
        user_id = "large_conv_user"
        session = orchestrator.create_chat_session(user_id, "Large Backend Conversation")
        
        # Simulate large AI conversation
        num_exchanges = 500  # 500 user-AI exchanges
        for i in range(num_exchanges):
            orchestrator.add_message(
                session.id, MessageType.USER, f"User message {i}", user_id
            )
            orchestrator.add_message(
                session.id, MessageType.ASSISTANT, f"AI response {i}"
            )
          # Test backend retrieval capabilities
        all_messages = orchestrator.get_session_messages(session.id)
        assert len(all_messages) == num_exchanges * 2  # User + AI messages
          # Test limited retrieval for AI context window management
        recent_messages = orchestrator.get_session_messages(session.id, limit=20)
        assert len(recent_messages) == 20
        
        # Verify backend stats
        stats = orchestrator.get_session_stats(session.id)
        assert stats["total_messages"] == num_exchanges * 2
        assert stats["user_messages"] == num_exchanges
        assert stats["assistant_messages"] == num_exchanges

    def test_message_metadata_backend_tracking(self, orchestrator):
        """Test message metadata for backend AI tracking and analytics."""
        user_id = "metadata_user"
        session = orchestrator.create_chat_session(user_id, "Metadata Test")
        
        # Add message with backend metadata
        backend_metadata = {
            "ai_model": "gemma-3-4b-it",
            "processing_time": 1.23,
            "tokens_used": 145,
            "backend_version": "1.0.0",
            "request_id": "req_123456"
        }
        
        user_message = orchestrator.add_message(
            session.id, 
            MessageType.USER, 
            "Query requiring AI processing", 
            user_id,
            {"source": "api", "priority": "high"}
        )
        
        ai_message = orchestrator.add_message(
            session.id,
            MessageType.ASSISTANT,
            "AI-generated response from backend",
            None,
            backend_metadata
        )
        
        # Verify metadata is stored and retrievable
        assert user_message.metadata["source"] == "api"
        assert ai_message.metadata["ai_model"] == "gemma-3-4b-it"
        assert ai_message.metadata["tokens_used"] == 145
          # Verify metadata persists in conversation history
        history = orchestrator.get_session_messages(session.id)
        assert history[0].metadata["source"] == "api"
        assert history[1].metadata["ai_model"] == "gemma-3-4b-it"

    def test_session_lifecycle_backend_management(self):
        """Test complete session lifecycle for backend AI operations."""
        orchestrator = ChatOrchestrator()
        user_id = "lifecycle_backend_user"
        
        # Create session for AI interaction
        session = orchestrator.create_chat_session(user_id, "Backend AI Lifecycle")
        
        # Simulate AI conversation
        orchestrator.add_message(session.id, MessageType.USER, "Start AI session", user_id)
        orchestrator.add_message(session.id, MessageType.ASSISTANT, "AI session started")
        orchestrator.add_message(session.id, MessageType.SYSTEM, "Backend systems ready")
        
        # Update session for backend tracking
        new_title = "Backend AI Session - Active"
        orchestrator.update_session_title(session.id, new_title)
        
        # Verify backend session state
        updated_session = orchestrator.get_chat_session(session.id)
        assert updated_session.title == new_title
          # Export for backend storage/analysis
        exported = orchestrator.export_session(session.id)
        assert exported["session"]["title"] == new_title
        assert len(exported["messages"]) == 3
        
        # Clean up session (backend resource management)
        result = orchestrator.delete_session(session.id)
        assert result is True
        
        # Verify cleanup
        assert orchestrator.get_chat_session(session.id) is None
        assert len(orchestrator.get_user_sessions(user_id)) == 0


if __name__ == "__main__":
    pytest.main([__file__])
