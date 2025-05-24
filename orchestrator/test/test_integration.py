#!/usr/bin/env python3
"""
Integration tests for ChatModerator and ChatOrchestrator working together.
Tests the complete chat workflow including moderation and orchestration.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch
import uuid

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Add orchestrator module to path
orchestrator_path = Path(__file__).parent.parent
sys.path.insert(0, str(orchestrator_path))

from chatmod import ChatModerator
from chat import ChatOrchestrator, MessageType


class TestChatModerationIntegration:
    """Integration tests for chat moderation and orchestration."""
        
    @pytest.fixture
    def moderator(self):
        """Create a ChatModerator instance for testing."""
        with patch('transformers.pipeline') as mock_pipeline:
            mock_analyzer = MagicMock()
            mock_analyzer.return_value = [{"label": "POSITIVE", "score": 0.8}]
            mock_pipeline.return_value = mock_analyzer
            return ChatModerator()

    @pytest.fixture
    def orchestrator(self):
        """Create a ChatOrchestrator instance for testing."""
        return ChatOrchestrator()
    
    @pytest.fixture
    def chat_system(self, moderator, orchestrator):
        """Create a complete chat system with moderator and orchestrator."""
        return {"moderator": moderator, "orchestrator": orchestrator}
    
    def test_complete_chat_workflow_valid_messages(self, chat_system):
        """Test complete chat workflow with valid messages."""
        moderator = chat_system["moderator"]
        orchestrator = chat_system["orchestrator"]
        
        user_id = "user123"
        session = orchestrator.create_chat_session(user_id, "Integration Test")
        
        # Test user message flow
        user_message = "Hello, I need help with my account settings."
        
        # Moderate user message using the class method
        moderation_result = moderator.moderate_message(user_message, user_id)
        assert moderation_result["approved"] is True
        
        # If moderation passes, add to orchestrator
        if moderation_result["approved"]:
            message = orchestrator.add_message(
                session.id, MessageType.USER, user_message, user_id
            )
            assert message.content == user_message
        
        # Simulate AI response
        ai_response = "I can help you with your account settings. What specifically would you like to change?"
        
        # Moderate AI response using the class method
        ai_moderation = moderator.filter_response(ai_response)
        assert ai_moderation["approved"] is True
        
        # If AI response is safe, add to orchestrator
        if ai_moderation["approved"]:
            ai_message = orchestrator.add_message(
                session.id, MessageType.ASSISTANT, ai_moderation["filtered_response"]
            )
            assert ai_message.content == ai_response
        
        # Verify conversation history
        history = orchestrator.get_conversation_history(session.id)
        assert len(history) == 2
        assert history[0].message_type == MessageType.USER
        assert history[1].message_type == MessageType.ASSISTANT

    def test_chat_workflow_blocked_user_message(self, chat_system):
        """Test chat workflow when user message is blocked by moderation."""
        moderator = chat_system["moderator"]
        orchestrator = chat_system["orchestrator"]
        
        user_id = "user123"
        session = orchestrator.create_chat_session(user_id, "Blocked Message Test")
        
        # Test spam message
        spam_message = "URGENT URGENT URGENT ATTENTION PLEASE"
        
        # Moderate user message
        moderation_result = moderator.moderate_user_message(spam_message)
        assert moderation_result["is_allowed"] is False
        assert "spam" in moderation_result["flags"]
        
        # Message should not be added to orchestrator
        initial_count = len(orchestrator.get_conversation_history(session.id))
        
        # Don't add blocked message
        if not moderation_result["is_allowed"]:
            # Could log the blocked attempt or send warning to user
            pass
        
        # Verify no message was added
        final_count = len(orchestrator.get_conversation_history(session.id))
        assert final_count == initial_count

    def test_chat_workflow_unsafe_ai_response(self, chat_system):
        """Test chat workflow when AI response is unsafe."""
        moderator = chat_system["moderator"]
        orchestrator = chat_system["orchestrator"]
        
        user_id = "user123"
        session = orchestrator.create_chat_session(user_id, "Unsafe AI Test")
        
        # Add valid user message
        user_message = "Tell me about security"
        moderation_result = moderator.moderate_user_message(user_message)
        assert moderation_result["is_allowed"] is True
        
        orchestrator.add_message(session.id, MessageType.USER, user_message, user_id)
        
        # Simulate unsafe AI response
        unsafe_response = "Here's how to hack systems and cause violence to others"
        
        # Moderate AI response
        ai_moderation = moderator.moderate_ai_response(unsafe_response)
        assert ai_moderation["is_safe"] is False
        assert "toxic_content" in ai_moderation["flags"]
        
        # Should not add unsafe response
        if not ai_moderation["is_safe"]:
            # Could generate alternative response or error message
            safe_response = "I can help you with security best practices for protecting your accounts."
            orchestrator.add_message(session.id, MessageType.ASSISTANT, safe_response)
        
        # Verify conversation
        history = orchestrator.get_conversation_history(session.id)
        assert len(history) == 2
        assert "hack" not in history[1].content
        assert "violence" not in history[1].content

    def test_moderation_statistics_tracking(self, chat_system):
        """Test that moderation statistics are properly tracked."""
        moderator = chat_system["moderator"]
        orchestrator = chat_system["orchestrator"]
        
        user_id = "user123"
        session = orchestrator.create_chat_session(user_id, "Stats Test")
        
        # Process multiple messages with different outcomes
        messages = [
            ("Hello there", True),  # Valid
            ("SPAM SPAM SPAM SPAM SPAM", False),  # Spam
            ("This is a normal message", True),  # Valid
            ("I hate everything", False),  # Toxic
            ("Thank you for your help", True),  # Valid
        ]
        
        for message, expected_allowed in messages:
            result = moderator.moderate_user_message(message)
            assert result["is_allowed"] == expected_allowed
            
            if result["is_allowed"]:
                orchestrator.add_message(session.id, MessageType.USER, message, user_id)
        
        # Check moderation stats
        stats = moderator.get_moderation_stats()
        assert stats["user_messages_processed"] == 5
        assert stats["user_messages_blocked"] == 2
        
        # Check orchestrator has only valid messages
        history = orchestrator.get_conversation_history(session.id)
        assert len(history) == 3  # Only the 3 allowed messages

    def test_sentiment_analysis_integration(self, chat_system):
        """Test sentiment analysis integration with chat orchestration."""
        moderator = chat_system["moderator"]
        orchestrator = chat_system["orchestrator"]
        
        user_id = "user123"
        session = orchestrator.create_chat_session(user_id, "Sentiment Test")
        
        # Mock different sentiment responses
        sentiment_cases = [
            ("I love this service!", "POSITIVE", 0.95),
            ("This is okay I guess", "NEUTRAL", 0.6),
            ("I'm frustrated with this", "NEGATIVE", 0.8),
        ]
        
        for message, expected_sentiment, expected_confidence in sentiment_cases:
            # Mock the sentiment analyzer for this specific case
            moderator.sentiment_analyzer.return_value = [
                {"label": expected_sentiment, "score": expected_confidence}
            ]
            
            result = moderator.moderate_user_message(message)
            assert result["is_allowed"] is True
            assert result["sentiment"] == expected_sentiment
            assert result["confidence"] == expected_confidence
            
            # Add message with sentiment metadata
            metadata = {
                "sentiment": result["sentiment"],
                "confidence": result["confidence"]
            }
            orchestrator.add_message(
                session.id, MessageType.USER, message, user_id, metadata
            )
        
        # Verify messages stored with sentiment data
        history = orchestrator.get_conversation_history(session.id)
        assert len(history) == 3
        
        for i, (_, expected_sentiment, expected_confidence) in enumerate(sentiment_cases):
            assert history[i].metadata["sentiment"] == expected_sentiment
            assert history[i].metadata["confidence"] == expected_confidence

    def test_url_filtering_in_ai_responses(self, chat_system):
        """Test URL filtering in AI responses with orchestration."""
        moderator = chat_system["moderator"]
        orchestrator = chat_system["orchestrator"]
        
        user_id = "user123"
        session = orchestrator.create_chat_session(user_id, "URL Filter Test")
        
        # Add user message
        user_message = "Can you provide some resources?"
        moderator_result = moderator.moderate_user_message(user_message)
        assert moderator_result["is_allowed"] is True
        
        orchestrator.add_message(session.id, MessageType.USER, user_message, user_id)
        
        # AI response with URLs
        ai_response_with_urls = (
            "Sure! Check out https://example.com for tutorials and "
            "http://docs.example.org for documentation. Also visit "
            "https://community.example.net for community support."
        )
        
        # Moderate AI response
        ai_moderation = moderator.moderate_ai_response(ai_response_with_urls)
        assert ai_moderation["is_safe"] is True
        assert "urls" in ai_moderation["removed_elements"]
        
        # Verify URLs were removed
        filtered_response = ai_moderation["filtered_response"]
        assert "https://example.com" not in filtered_response
        assert "http://docs.example.org" not in filtered_response
        assert "https://community.example.net" not in filtered_response
        
        # Add filtered response to orchestrator
        orchestrator.add_message(session.id, MessageType.ASSISTANT, filtered_response)
        
        # Verify conversation
        history = orchestrator.get_conversation_history(session.id)
        assert len(history) == 2
        assert "example.com" not in history[1].content

    def test_conversation_export_with_moderation_data(self, chat_system):
        """Test exporting conversation data including moderation information."""
        moderator = chat_system["moderator"]
        orchestrator = chat_system["orchestrator"]
        
        user_id = "user123"
        session = orchestrator.create_chat_session(user_id, "Export Test")
        
        # Add moderated messages with metadata
        messages = [
            "Hello, I need assistance",
            "Can you help me with my account?",
            "Thank you for the information"
        ]
        
        for message in messages:
            # Moderate and add metadata
            moderation_result = moderator.moderate_user_message(message)
            assert moderation_result["is_allowed"] is True
            
            metadata = {
                "sentiment": moderation_result["sentiment"],
                "confidence": moderation_result["confidence"],
                "moderation_flags": moderation_result["flags"],
                "moderation_timestamp": datetime.now().isoformat()
            }
            
            orchestrator.add_message(session.id, MessageType.USER, message, user_id, metadata)
            
            # Add AI response
            ai_response = f"Response to: {message}"
            ai_moderation = moderator.moderate_ai_response(ai_response)
            assert ai_moderation["is_safe"] is True
            
            ai_metadata = {
                "moderation_safe": ai_moderation["is_safe"],
                "removed_elements": ai_moderation["removed_elements"]
            }
            
            orchestrator.add_message(session.id, MessageType.ASSISTANT, ai_response, metadata=ai_metadata)
        
        # Export session data
        exported = orchestrator.export_session_data(session.id)
        
        assert exported is not None
        assert len(exported["messages"]) == 6  # 3 user + 3 assistant
        
        # Verify moderation metadata is preserved
        user_messages = [msg for msg in exported["messages"] if msg["message_type"] == "user"]
        for msg in user_messages:
            assert "sentiment" in msg["metadata"]
            assert "confidence" in msg["metadata"]
            assert "moderation_flags" in msg["metadata"]
        
        ai_messages = [msg for msg in exported["messages"] if msg["message_type"] == "assistant"]
        for msg in ai_messages:
            assert "moderation_safe" in msg["metadata"]
            assert "removed_elements" in msg["metadata"]

    def test_concurrent_moderation_and_orchestration(self, chat_system):
        """Test concurrent moderation and orchestration operations."""
        import threading
        import time
        
        moderator = chat_system["moderator"]
        orchestrator = chat_system["orchestrator"]
        
        user_id = "concurrent_user"
        session = orchestrator.create_chat_session(user_id, "Concurrent Test")
        
        results = []
        
        def process_message(message_index):
            message = f"Concurrent message {message_index}"
            
            # Moderate message
            moderation_result = moderator.moderate_user_message(message)
            
            # Add to orchestrator if allowed
            if moderation_result["is_allowed"]:
                orchestrated_message = orchestrator.add_message(
                    session.id, MessageType.USER, message, user_id
                )
                results.append((moderation_result, orchestrated_message))
        
        # Create and start threads
        threads = []
        num_threads = 10
        
        for i in range(num_threads):
            thread = threading.Thread(target=process_message, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(results) == num_threads
        
        # Verify all messages were added to orchestrator
        history = orchestrator.get_conversation_history(session.id)
        assert len(history) == num_threads
        
        # Verify moderation stats
        stats = moderator.get_moderation_stats()
        assert stats["user_messages_processed"] >= num_threads

    def test_error_handling_integration(self, chat_system):
        """Test error handling in integrated moderation and orchestration."""
        moderator = chat_system["moderator"]
        orchestrator = chat_system["orchestrator"]
        
        user_id = "error_test_user"
        session = orchestrator.create_chat_session(user_id, "Error Test")
        
        # Test moderation with None message
        try:
            result = moderator.moderate_user_message(None)
            assert result["is_allowed"] is False
        except Exception:
            pass  # Expected to handle gracefully
        
        # Test orchestration with invalid session
        fake_session_id = str(uuid.uuid4())
        
        with pytest.raises(ValueError):
            orchestrator.add_message(fake_session_id, MessageType.USER, "Test message", user_id)
        
        # Test with very long message
        very_long_message = "A" * 10000
        moderation_result = moderator.moderate_user_message(very_long_message)
        assert moderation_result["is_allowed"] is False
        
        # Verify session remains valid after errors
        valid_message = "This is a valid message"
        valid_result = moderator.moderate_user_message(valid_message)
        assert valid_result["is_allowed"] is True
        
        orchestrator.add_message(session.id, MessageType.USER, valid_message, user_id)
        history = orchestrator.get_conversation_history(session.id)
        assert len(history) == 1

    def test_performance_with_large_conversations(self, chat_system):
        """Test performance of moderation and orchestration with large conversations."""
        moderator = chat_system["moderator"]
        orchestrator = chat_system["orchestrator"]
        
        user_id = "performance_user"
        session = orchestrator.create_chat_session(user_id, "Performance Test")
        
        num_messages = 100
        start_time = datetime.now()
        
        for i in range(num_messages):
            message = f"Performance test message number {i} with some content to analyze"
            
            # Moderate message
            moderation_result = moderator.moderate_user_message(message)
            assert moderation_result["is_allowed"] is True
            
            # Add to orchestrator
            orchestrator.add_message(session.id, MessageType.USER, message, user_id)
            
            # Add AI response
            ai_response = f"Response to message {i}"
            ai_moderation = moderator.moderate_ai_response(ai_response)
            assert ai_moderation["is_safe"] is True
            
            orchestrator.add_message(session.id, MessageType.ASSISTANT, ai_response)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Verify all messages processed
        history = orchestrator.get_conversation_history(session.id)
        assert len(history) == num_messages * 2  # User + AI messages
        
        # Check reasonable performance (adjust threshold as needed)
        assert processing_time < 30.0  # Should process 200 messages in under 30 seconds
        
        # Verify statistics
        stats = moderator.get_moderation_stats()
        assert stats["user_messages_processed"] >= num_messages
        assert stats["ai_responses_processed"] >= num_messages


if __name__ == "__main__":
    pytest.main([__file__])
