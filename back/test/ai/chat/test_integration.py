#!/usr/bin/env python3
"""
Integration tests for ChatModerator and ChatOrchestrator working together.
Tests the complete backend AI chat workflow including moderation and orchestration.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch
import uuid

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Add orchestrator module to path
orchestrator_path = project_root / "orchestrator"
sys.path.insert(0, str(orchestrator_path))

try:
    from chatmod import ChatModerator
    from chat import ChatOrchestrator, MessageType
except ImportError:
    # Fallback import paths
    sys.path.insert(0, str(project_root / "orchestrator" / "chatmod"))
    sys.path.insert(0, str(project_root / "orchestrator" / "chat"))
    from __init__ import ChatModerator
    from chat.__init__ import ChatOrchestrator, MessageType


class TestBackendChatIntegration:
    """Integration tests for backend AI chat moderation and orchestration."""
    
    @pytest.fixture
    def moderator(self):
        """Create a ChatModerator instance for backend testing."""
        with patch('chatmod.pipeline') as mock_pipeline:
            mock_analyzer = MagicMock()
            mock_analyzer.return_value = [{"label": "POSITIVE", "score": 0.8}]
            mock_pipeline.return_value = mock_analyzer
            return ChatModerator()
    
    @pytest.fixture
    def orchestrator(self):
        """Create a ChatOrchestrator instance for backend testing."""
        return ChatOrchestrator()
    
    @pytest.fixture
    def backend_chat_system(self, moderator, orchestrator):
        """Create a complete backend chat system with moderator and orchestrator."""
        return {"moderator": moderator, "orchestrator": orchestrator}

    def test_backend_ai_chat_workflow(self, backend_chat_system):
        """Test complete backend AI chat workflow with moderation."""
        moderator = backend_chat_system["moderator"]
        orchestrator = backend_chat_system["orchestrator"]
        
        user_id = "backend_ai_user_123"
        session = orchestrator.create_chat_session(user_id, "Backend AI Chat Integration")
        
        # Test user message flow through backend moderation
        user_message_content = "Hello, I need help with machine learning concepts."
        
        # Step 1: Moderate user message
        moderation_result = moderator.moderate_message(user_message_content)
        assert moderation_result["approved"] is True
        assert moderation_result["sentiment"]["label"] == "POSITIVE"
        
        # Step 2: Add approved message to orchestrator
        if moderation_result["approved"]:
            user_message = orchestrator.add_message(
                session.id, 
                MessageType.USER, 
                moderation_result["message"], 
                user_id,
                {"moderation": "passed", "sentiment": moderation_result["sentiment"]}
            )
            assert user_message.content == user_message_content
            assert user_message.metadata["moderation"] == "passed"
        
        # Step 3: Generate AI response (simulated backend AI processing)
        ai_response_content = "I'd be happy to help you with machine learning concepts. What specific area would you like to explore?"
        
        # Step 4: Filter AI response through moderator
        filter_result = moderator.filter_response(ai_response_content)
        assert filter_result["approved"] is True
        assert filter_result["filtered_response"] == ai_response_content
        
        # Step 5: Add filtered AI response to orchestrator
        if filter_result["approved"]:
            ai_message = orchestrator.add_message(
                session.id,
                MessageType.ASSISTANT,
                filter_result["filtered_response"],
                None,
                {
                    "ai_model": "gemma-3-4b-it",
                    "filtering": "passed",
                    "backend_processing": True
                }
            )
            assert ai_message.content == ai_response_content
            assert ai_message.metadata["backend_processing"] is True
          # Verify complete backend conversation
        conversation = orchestrator.get_session_messages(session.id)
        assert len(conversation) == 2
        assert conversation[0].message_type == MessageType.USER
        assert conversation[1].message_type == MessageType.ASSISTANT

    def test_backend_moderation_rejection_workflow(self, backend_chat_system):
        """Test backend workflow when moderation rejects content."""
        moderator = backend_chat_system["moderator"]
        orchestrator = backend_chat_system["orchestrator"]
        
        user_id = "test_user_rejection"
        session = orchestrator.create_chat_session(user_id, "Moderation Rejection Test")
        
        # Test toxic user message
        toxic_message = "I hate this stupid AI system"
        moderation_result = moderator.moderate_message(toxic_message)
        
        assert moderation_result["approved"] is False
        assert "toxic_content" in moderation_result["filters_triggered"]
        
        # Backend should log rejection but not add to conversation
        system_message = orchestrator.add_message(
            session.id,
            MessageType.SYSTEM,
            "Message rejected by content moderation",
            None,
            {
                "original_message_rejected": True,
                "rejection_reason": moderation_result["filters_triggered"],
                "backend_action": "message_blocked"
            }
        )
          # Verify only system message exists (user message was blocked)
        conversation = orchestrator.get_session_messages(session.id)
        assert len(conversation) == 1
        assert conversation[0].message_type == MessageType.SYSTEM
        assert conversation[0].metadata["original_message_rejected"] is True

    def test_backend_ai_response_filtering(self, backend_chat_system):
        """Test backend AI response filtering workflow."""
        moderator = backend_chat_system["moderator"]
        orchestrator = backend_chat_system["orchestrator"]
        
        user_id = "ai_filtering_user"
        session = orchestrator.create_chat_session(user_id, "AI Response Filtering")
        
        # Add approved user message
        user_message = orchestrator.add_message(
            session.id, MessageType.USER, "Can you provide some resources?", user_id
        )
        
        # Test AI response with URLs that need filtering
        ai_response_with_urls = "Sure! Check out these resources: https://example.com/ml-guide and https://research.paper.org/study"
        
        filter_result = moderator.filter_response(ai_response_with_urls)
        
        assert filter_result["approved"] is True
        assert "url_removal" in filter_result["filters_applied"]
        assert "https://example.com" not in filter_result["filtered_response"]
        assert "https://research.paper.org" not in filter_result["filtered_response"]
        
        # Add filtered response to conversation
        ai_message = orchestrator.add_message(
            session.id,
            MessageType.ASSISTANT,
            filter_result["filtered_response"],
            None,
            {
                "original_response": ai_response_with_urls,
                "filters_applied": filter_result["filters_applied"],
                "backend_filtering": True
            }
        )
          # Verify filtering metadata is preserved
        conversation = orchestrator.get_session_messages(session.id)
        assert len(conversation) == 2
        ai_msg = conversation[1]
        assert ai_msg.metadata["backend_filtering"] is True
        assert "url_removal" in ai_msg.metadata["filters_applied"]

    def test_backend_concurrent_chat_operations(self, backend_chat_system):
        """Test concurrent backend chat operations with moderation."""
        import threading
        import time
        
        moderator = backend_chat_system["moderator"]
        orchestrator = backend_chat_system["orchestrator"]
        
        results = []
        
        def backend_chat_operation(user_index):
            user_id = f"concurrent_user_{user_index}"
            session = orchestrator.create_chat_session(user_id, f"Concurrent Backend Chat {user_index}")
            
            # User message
            user_msg = f"Backend test message {user_index}"
            mod_result = moderator.moderate_message(user_msg)
            
            if mod_result["approved"]:
                user_message = orchestrator.add_message(
                    session.id, MessageType.USER, user_msg, user_id
                )
                
                # AI response
                ai_response = f"Backend AI response {user_index}"
                filter_result = moderator.filter_response(ai_response)
                
                if filter_result["approved"]:
                    ai_message = orchestrator.add_message(
                        session.id, MessageType.ASSISTANT, ai_response
                    )
                
                results.append((user_id, session.id, len(orchestrator.messages[session.id])))
        
        # Create concurrent backend operations
        threads = []
        for i in range(10):
            thread = threading.Thread(target=backend_chat_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all backend operations
        for thread in threads:
            thread.join()
        
        # Verify all concurrent operations completed successfully
        assert len(results) == 10
        assert len(orchestrator.sessions) == 10
          # Verify each session has complete conversation
        for user_id, session_id, message_count in results:
            assert message_count == 2  # User + AI message
            conversation = orchestrator.get_session_messages(session_id)
            assert len(conversation) == 2

    def test_backend_performance_monitoring(self, backend_chat_system):
        """Test backend performance monitoring with chat operations."""
        moderator = backend_chat_system["moderator"]
        orchestrator = backend_chat_system["orchestrator"]
        
        user_id = "performance_user"
        session = orchestrator.create_chat_session(user_id, "Performance Monitoring")
        
        start_time = datetime.now()
        
        # Simulate multiple backend operations
        for i in range(50):
            user_msg = f"Performance test message {i}"
            mod_result = moderator.moderate_message(user_msg)
            
            if mod_result["approved"]:
                orchestrator.add_message(
                    session.id, MessageType.USER, user_msg, user_id,
                    {"operation_index": i, "timestamp": datetime.now().isoformat()}
                )
                
                ai_response = f"Performance AI response {i}"
                filter_result = moderator.filter_response(ai_response)
                
                if filter_result["approved"]:
                    orchestrator.add_message(
                        session.id, MessageType.ASSISTANT, ai_response, None,
                        {"operation_index": i, "backend_processing": True}
                    )
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Verify performance metrics
        stats = orchestrator.get_session_stats(session.id)
        assert stats["total_messages"] == 100  # 50 user + 50 AI messages
        assert total_time < 10.0  # Should complete within 10 seconds
          # Verify conversation integrity
        conversation = orchestrator.get_session_messages(session.id)
        assert len(conversation) == 100

    def test_backend_error_handling_integration(self, backend_chat_system):
        """Test backend error handling in integrated chat system."""
        moderator = backend_chat_system["moderator"]
        orchestrator = backend_chat_system["orchestrator"]
        
        user_id = "error_handling_user"
        session = orchestrator.create_chat_session(user_id, "Error Handling Test")
        
        # Test moderation failure handling
        with patch.object(moderator, 'moderate_message', side_effect=Exception("Moderation service down")):
            try:
                moderator.moderate_message("Test message")
                assert False, "Should have raised exception"
            except Exception as e:
                # Backend should handle moderation service failures gracefully
                error_message = orchestrator.add_message(
                    session.id,
                    MessageType.SYSTEM,
                    "Moderation service temporarily unavailable",
                    None,
                    {"error": str(e), "backend_fallback": True}
                )
                assert error_message.metadata["backend_fallback"] is True
          # Test orchestrator resilience
        conversation = orchestrator.get_session_messages(session.id)
        assert len(conversation) == 1
        assert conversation[0].message_type == MessageType.SYSTEM

    def test_backend_data_export_integration(self, backend_chat_system):
        """Test backend data export for analytics and storage."""
        moderator = backend_chat_system["moderator"]
        orchestrator = backend_chat_system["orchestrator"]
        
        user_id = "export_integration_user"
        session = orchestrator.create_chat_session(user_id, "Data Export Integration")
        
        # Create conversation with moderation metadata
        messages = [
            ("Hello, I need help with AI development", "POSITIVE", 0.9),
            ("What are the best practices for model training?", "NEUTRAL", 0.7),
            ("Thank you for the detailed explanation!", "POSITIVE", 0.95)
        ]
        
        for msg_content, sentiment, confidence in messages:
            # Mock moderation results
            moderator.sentiment_analyzer = MagicMock(
                return_value=[{"label": sentiment, "score": confidence}]
            )
            
            mod_result = moderator.moderate_message(msg_content)
            user_message = orchestrator.add_message(
                session.id, MessageType.USER, msg_content, user_id,
                {"moderation_result": mod_result}
            )
            
            # Add AI response
            ai_response = f"AI response to: {msg_content[:20]}..."
            filter_result = moderator.filter_response(ai_response)
            orchestrator.add_message(
                session.id, MessageType.ASSISTANT, ai_response, None,
                {"filter_result": filter_result, "ai_model": "gemma-3-4b-it"}
            )
          # Export complete session data for backend analytics
        exported_data = orchestrator.export_session(session.id)
          # Verify export contains moderation and filtering metadata
        assert "session" in exported_data
        assert "messages" in exported_data
        assert len(exported_data["messages"]) == 6  # 3 user + 3 AI messages
        
        assert len(exported_data["messages"]) == 6  # 3 user + 3 AI messages
        
        # Verify moderation metadata is preserved
        user_messages = [msg for msg in exported_data["messages"] if msg["message_type"] == "user"]
        assert len(user_messages) == 3
        
        for user_msg in user_messages:
            assert "moderation_result" in user_msg["metadata"]
        
        # Verify AI filtering metadata
        ai_messages = [msg for msg in exported_data["messages"] if msg["message_type"] == "assistant"]
        assert len(ai_messages) == 3
        
        for ai_msg in ai_messages:
            assert "filter_result" in ai_msg["metadata"]
            assert ai_msg["metadata"]["ai_model"] == "gemma-3-4b-it"


if __name__ == "__main__":
    pytest.main([__file__])
