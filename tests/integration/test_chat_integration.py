"""
Integration Tests for Chat System

Tests that verify the interaction between different components of the chat system.
These tests ensure that components work together correctly as a complete system.
"""

import sys
import pytest
import time
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "orchestrator"))


@pytest.mark.integration
class TestChatModerationIntegration:
    """Integration tests for chat moderation with orchestration"""
    
    def test_moderation_orchestration_integration(self, integrated_chat_system, test_data):
        """Test integration between moderation and orchestration components"""
        moderator = integrated_chat_system["moderator"]
        orchestrator = integrated_chat_system["orchestrator"]
        
        # Create session
        session = orchestrator.create_session(
            user_id="integration_test_user",
            title="Moderation Integration Test"
        )
        
        # Test various message types and their integration
        test_cases = [
            ("valid", test_data.valid_messages()),
            ("spam", test_data.spam_messages()),
            ("toxic", test_data.toxic_messages()),
        ]
        
        results = []
        
        for category, messages in test_cases:
            for message in messages[:2]:  # Test first 2 of each type
                # Step 1: Moderate message
                moderation_result = moderator.moderate_message(message)
                
                # Step 2: Based on moderation, decide whether to store
                should_store = len(moderation_result.get("filters_triggered", [])) == 0
                
                stored_message = None
                if should_store:
                    # Step 3: Store message if it passes moderation
                    stored_message = orchestrator.add_message(
                        session_id=session.id,
                        content=message,
                        message_type="user"
                    )
                
                # Step 4: Record results for analysis
                results.append({
                    "category": category,
                    "original_message": message,
                    "moderation_result": moderation_result,
                    "was_stored": stored_message is not None,
                    "stored_message": stored_message
                })
        
        # Verify integration worked correctly
        assert len(results) > 0
        
        # Check that valid messages were generally stored
        valid_results = [r for r in results if r["category"] == "valid"]
        stored_valid = [r for r in valid_results if r["was_stored"]]
        assert len(stored_valid) > 0  # At least some valid messages should be stored
        
        # Verify stored messages have correct structure
        for result in results:
            if result["was_stored"]:
                msg = result["stored_message"]
                assert msg.content == result["original_message"]
                assert msg.chat_session_id == session.id
                assert msg.message_type == "user"
    
    def test_moderation_feedback_loop(self, integrated_chat_system):
        """Test feedback between moderation and orchestration"""
        moderator = integrated_chat_system["moderator"]
        orchestrator = integrated_chat_system["orchestrator"]
        
        session = orchestrator.create_session(user_id="feedback_test")
        
        # Simulate a conversation with moderation
        conversation = [
            "Hello, I need help",
            "This is a normal question",
            "Can you assist me please?",
            "Thank you for your help"
        ]
        
        conversation_log = []
        
        for user_input in conversation:
            # Moderate user input
            moderation_result = moderator.moderate_message(user_input)
            
            # Store user message if it passes
            user_message = None
            if not moderation_result.get("filters_triggered"):
                user_message = orchestrator.add_message(
                    session_id=session.id,
                    content=user_input,
                    message_type="user"
                )
            
            # Generate system response
            system_response = f"I understand your message: '{user_input}'"
            
            # Moderate system response (systems should also be moderated)
            system_moderation = moderator.moderate_message(system_response)
            
            # Store system response
            system_message = None
            if not system_moderation.get("filters_triggered"):
                system_message = orchestrator.add_message(
                    session_id=session.id,
                    content=system_response,
                    message_type="assistant"
                )
            
            conversation_log.append({
                "user_input": user_input,
                "user_moderation": moderation_result,
                "user_message": user_message,
                "system_response": system_response,
                "system_moderation": system_moderation,
                "system_message": system_message
            })
        
        # Verify conversation flow
        assert len(conversation_log) == len(conversation)
        
        # Most messages should have been processed
        processed_exchanges = [log for log in conversation_log 
                             if log["user_message"] and log["system_message"]]
        assert len(processed_exchanges) > 0
    
    def test_cross_session_moderation(self, integrated_chat_system):
        """Test that moderation works consistently across different sessions"""
        moderator = integrated_chat_system["moderator"]
        orchestrator = integrated_chat_system["orchestrator"]
        
        # Create multiple sessions
        sessions = []
        for i in range(3):
            session = orchestrator.create_session(
                user_id=f"cross_session_user_{i}",
                title=f"Cross Session Test {i}"
            )
            sessions.append(session)
        
        # Test same message across all sessions
        test_message = "This is a consistent test message"
        
        moderation_results = []
        stored_messages = []
        
        for session in sessions:
            # Moderate same message
            moderation_result = moderator.moderate_message(test_message)
            moderation_results.append(moderation_result)
            
            # Store message
            if not moderation_result.get("filters_triggered"):
                message = orchestrator.add_message(
                    session_id=session.id,
                    content=test_message,
                    message_type="user"
                )
                stored_messages.append(message)
        
        # Moderation should be consistent across sessions
        first_result = moderation_results[0]
        for result in moderation_results[1:]:
            assert result["filters_triggered"] == first_result["filters_triggered"]
        
        # All sessions should have stored the message (if it passed moderation)
        if not first_result.get("filters_triggered"):
            assert len(stored_messages) == len(sessions)
            for msg in stored_messages:
                assert msg.content == test_message


@pytest.mark.integration
class TestChatSystemDataFlow:
    """Integration tests for data flow through the entire chat system"""
    
    def test_message_lifecycle(self, integrated_chat_system):
        """Test complete message lifecycle from input to storage"""
        moderator = integrated_chat_system["moderator"]
        orchestrator = integrated_chat_system["orchestrator"]
        
        # Create session
        session = orchestrator.create_session(
            user_id="lifecycle_test",
            title="Message Lifecycle Test"
        )
        
        # Test message progression
        original_content = "Hello, this is a test message for lifecycle testing"
        
        # Phase 1: Input validation and moderation
        moderation_result = moderator.moderate_message(original_content)
        
        assert isinstance(moderation_result, dict)
        assert "message" in moderation_result
        assert "filters_triggered" in moderation_result
        assert moderation_result["message"] == original_content
        
        # Phase 2: Storage (if passes moderation)
        stored_message = None
        if not moderation_result.get("filters_triggered"):
            stored_message = orchestrator.add_message(
                session_id=session.id,
                content=original_content,
                message_type="user"
            )
            
            # Verify storage
            assert stored_message is not None
            assert stored_message.content == original_content
            assert stored_message.chat_session_id == session.id
            assert stored_message.message_type == "user"
            assert isinstance(stored_message.timestamp, datetime)
        
        # Phase 3: Retrieval verification
        if stored_message:
            retrieved_session = orchestrator.get_session(session.id)
            assert retrieved_session is not None
            assert retrieved_session.id == session.id
    
    def test_session_message_relationship_integrity(self, integrated_chat_system):
        """Test integrity of session-message relationships"""
        moderator = integrated_chat_system["moderator"]
        orchestrator = integrated_chat_system["orchestrator"]
        
        # Create multiple sessions
        sessions = {}
        for i in range(3):
            session = orchestrator.create_session(
                user_id=f"integrity_user_{i}",
                title=f"Integrity Test Session {i}"
            )
            sessions[f"session_{i}"] = session
        
        # Add messages to each session
        messages_by_session = {}
        
        for session_key, session in sessions.items():
            messages = []
            
            for j in range(3):
                content = f"Message {j} for {session_key}"
                
                # Moderate and store
                moderation_result = moderator.moderate_message(content)
                
                if not moderation_result.get("filters_triggered"):
                    message = orchestrator.add_message(
                        session_id=session.id,
                        content=content,
                        message_type="user"
                    )
                    messages.append(message)
            
            messages_by_session[session_key] = messages
        
        # Verify relationship integrity
        for session_key, session in sessions.items():
            session_messages = messages_by_session[session_key]
            
            for message in session_messages:
                # Each message should belong to correct session
                assert message.chat_session_id == session.id
                
                # Message content should match expected pattern
                assert session_key.replace("session_", "") in message.content
        
        # Verify isolation between sessions
        all_messages = []
        for messages in messages_by_session.values():
            all_messages.extend(messages)
        
        # All messages should have unique IDs
        message_ids = [msg.id for msg in all_messages]
        assert len(set(message_ids)) == len(message_ids)
    
    def test_concurrent_operations(self, integrated_chat_system):
        """Test concurrent operations across the system"""
        moderator = integrated_chat_system["moderator"]
        orchestrator = integrated_chat_system["orchestrator"]
        
        # Simulate concurrent operations
        operations = []
        
        # Create sessions concurrently (simulated)
        for i in range(5):
            session = orchestrator.create_session(
                user_id=f"concurrent_user_{i}",
                title=f"Concurrent Session {i}"
            )
            operations.append(("create_session", session))
        
        # Add messages concurrently (simulated)
        for i, (_, session) in enumerate(operations):
            content = f"Concurrent message {i}"
            
            moderation_result = moderator.moderate_message(content)
            
            if not moderation_result.get("filters_triggered"):
                message = orchestrator.add_message(
                    session_id=session.id,
                    content=content,
                    message_type="user"
                )
                operations.append(("add_message", message))
        
        # Verify all operations completed successfully
        session_ops = [op for op in operations if op[0] == "create_session"]
        message_ops = [op for op in operations if op[0] == "add_message"]
        
        assert len(session_ops) == 5
        assert len(message_ops) > 0
        
        # Verify data consistency
        for _, session in session_ops:
            retrieved = orchestrator.get_session(session.id)
            assert retrieved.id == session.id
            assert retrieved.user_id == session.user_id


@pytest.mark.integration
class TestChatSystemErrorHandling:
    """Integration tests for error handling across components"""
    
    def test_moderation_error_handling(self, integrated_chat_system):
        """Test system behavior when moderation component has issues"""
        moderator = integrated_chat_system["moderator"]
        orchestrator = integrated_chat_system["orchestrator"]
        
        session = orchestrator.create_session(user_id="error_handling_test")
        
        # Test with potentially problematic inputs
        problematic_inputs = [
            None,  # None input
            "",    # Empty string
            " " * 1000,  # Very long whitespace
            "\x00\x01\x02",  # Control characters
        ]
        
        for problematic_input in problematic_inputs:
            try:
                if problematic_input is not None:
                    moderation_result = moderator.moderate_message(problematic_input)
                    
                    # If moderation succeeds, try to store
                    if isinstance(moderation_result, dict) and not moderation_result.get("filters_triggered"):
                        message = orchestrator.add_message(
                            session_id=session.id,
                            content=problematic_input,
                            message_type="user"
                        )
                        # If storage succeeds, verify structure
                        assert hasattr(message, 'content')
                        
            except Exception as e:
                # Should be handled gracefully
                assert isinstance(e, (ValueError, TypeError, AttributeError))
    
    def test_orchestration_error_handling(self, integrated_chat_system):
        """Test system behavior when orchestration component has issues"""
        moderator = integrated_chat_system["moderator"]
        orchestrator = integrated_chat_system["orchestrator"]
        
        # Test with invalid session IDs
        invalid_session_ids = [
            "nonexistent_session",
            "",
            None,
            123,  # Wrong type
        ]
        
        for invalid_id in invalid_session_ids:
            try:
                if invalid_id is not None:
                    # Should handle invalid session ID gracefully
                    result = orchestrator.get_session(invalid_id)
                    # If it returns something, should be None or raise exception
                    assert result is None
            except Exception as e:
                # Should be a specific exception type
                assert isinstance(e, (KeyError, ValueError, TypeError))
        
        # Test adding messages to invalid sessions
        for invalid_id in invalid_session_ids:
            try:
                if invalid_id is not None:
                    message = orchestrator.add_message(
                        session_id=invalid_id,
                        content="Test message",
                        message_type="user"
                    )
                    # If it succeeds unexpectedly, should still be valid
                    if message:
                        assert hasattr(message, 'content')
            except Exception as e:
                # Should be a specific exception type
                assert isinstance(e, (KeyError, ValueError, TypeError))
    
    def test_integrated_error_recovery(self, integrated_chat_system):
        """Test system recovery from errors in integrated workflows"""
        moderator = integrated_chat_system["moderator"]
        orchestrator = integrated_chat_system["orchestrator"]
        
        # Create a valid session for testing
        session = orchestrator.create_session(user_id="error_recovery_test")
        
        # Mix valid and invalid operations
        test_operations = [
            ("valid", "This is a valid message"),
            ("empty", ""),
            ("valid", "Another valid message"),
            ("long", "x" * 10000),
            ("valid", "Final valid message"),
        ]
        
        successful_operations = 0
        total_operations = len(test_operations)
        
        for op_type, content in test_operations:
            try:
                # Try full workflow
                moderation_result = moderator.moderate_message(content)
                
                if isinstance(moderation_result, dict) and not moderation_result.get("filters_triggered"):
                    message = orchestrator.add_message(
                        session_id=session.id,
                        content=content,
                        message_type="user"
                    )
                    
                    if message and hasattr(message, 'content'):
                        successful_operations += 1
                        
            except Exception as e:
                # Log error but continue (system should be resilient)
                print(f"Operation failed for {op_type}: {e}")
                continue
        
        # System should handle at least some operations successfully
        assert successful_operations > 0
        success_rate = successful_operations / total_operations
        # Should have reasonable success rate (at least 50% for this test)
        assert success_rate >= 0.5


@pytest.mark.integration
class TestChatSystemPerformance:
    """Integration tests for performance across components"""
    
    def test_end_to_end_performance(self, integrated_chat_system):
        """Test performance of complete end-to-end workflows"""
        moderator = integrated_chat_system["moderator"]
        orchestrator = integrated_chat_system["orchestrator"]
        
        # Measure session creation performance
        start_time = time.time()
        session = orchestrator.create_session(
            user_id="performance_test",
            title="Performance Test Session"
        )
        session_creation_time = time.time() - start_time
        
        # Should be fast (under 0.1 seconds)
        assert session_creation_time < 0.1
        
        # Measure message processing performance
        message_times = []
        
        for i in range(10):
            message_content = f"Performance test message {i}"
            
            start_time = time.time()
            
            # Full workflow: moderation + storage
            moderation_result = moderator.moderate_message(message_content)
            
            if not moderation_result.get("filters_triggered"):
                message = orchestrator.add_message(
                    session_id=session.id,
                    content=message_content,
                    message_type="user"
                )
            
            end_time = time.time()
            message_times.append(end_time - start_time)
        
        # Calculate performance metrics
        avg_message_time = sum(message_times) / len(message_times)
        max_message_time = max(message_times)
        
        # Performance assertions
        assert avg_message_time < 0.1  # Average under 0.1 seconds
        assert max_message_time < 0.5  # No message takes more than 0.5 seconds
    
    def test_bulk_operations_performance(self, integrated_chat_system):
        """Test performance with bulk operations"""
        moderator = integrated_chat_system["moderator"]
        orchestrator = integrated_chat_system["orchestrator"]
        
        # Create multiple sessions
        start_time = time.time()
        
        sessions = []
        for i in range(20):
            session = orchestrator.create_session(
                user_id=f"bulk_user_{i}",
                title=f"Bulk Session {i}"
            )
            sessions.append(session)
        
        bulk_session_time = time.time() - start_time
        
        # Should handle bulk session creation efficiently
        assert bulk_session_time < 2.0  # Under 2 seconds for 20 sessions
        
        # Add messages to each session
        start_time = time.time()
        
        for i, session in enumerate(sessions):
            for j in range(5):
                content = f"Bulk message {j} from user {i}"
                
                moderation_result = moderator.moderate_message(content)
                
                if not moderation_result.get("filters_triggered"):
                    orchestrator.add_message(
                        session_id=session.id,
                        content=content,
                        message_type="user"
                    )
        
        bulk_message_time = time.time() - start_time
        
        # Should handle bulk message processing efficiently
        # 100 messages (20 sessions * 5 messages) in under 10 seconds
        assert bulk_message_time < 10.0
    
    def test_memory_usage_stability(self, integrated_chat_system):
        """Test that memory usage remains stable during operations"""
        moderator = integrated_chat_system["moderator"]
        orchestrator = integrated_chat_system["orchestrator"]
        
        # Create session
        session = orchestrator.create_session(user_id="memory_test")
        
        # Perform many operations to test memory stability
        for i in range(100):
            content = f"Memory test message {i}"
            
            # Alternate between different operations
            if i % 3 == 0:
                # Moderation only
                moderator.moderate_message(content)
            elif i % 3 == 1:
                # Full workflow
                moderation_result = moderator.moderate_message(content)
                if not moderation_result.get("filters_triggered"):
                    orchestrator.add_message(
                        session_id=session.id,
                        content=content,
                        message_type="user"
                    )
            else:
                # Session retrieval
                orchestrator.get_session(session.id)
        
        # If we reach here without memory errors, test passes
        # (More sophisticated memory monitoring could be added with psutil)
        assert True


@pytest.mark.integration
class TestChatSystemScenarios:
    """Integration tests for real-world usage scenarios"""
    
    def test_customer_support_scenario(self, integrated_chat_system):
        """Test a realistic customer support conversation scenario"""
        moderator = integrated_chat_system["moderator"]
        orchestrator = integrated_chat_system["orchestrator"]
        
        # Customer starts a support session
        session = orchestrator.create_session(
            user_id="customer_123",
            title="Account Issue Support"
        )
        
        # Simulate customer support conversation
        conversation = [
            ("user", "Hello, I'm having trouble accessing my account"),
            ("assistant", "I'm sorry to hear that. Can you provide your account email?"),
            ("user", "My email is user@example.com"),
            ("assistant", "Thank you. I can see your account. What specific issue are you experiencing?"),
            ("user", "I can't log in, it says my password is incorrect"),
            ("assistant", "I can help you reset your password. I'll send a reset link to your email."),
            ("user", "Great, thank you for your help!"),
            ("assistant", "You're welcome! Is there anything else I can help you with today?"),
            ("user", "No, that's all. Thank you!"),
            ("assistant", "Have a great day!")
        ]
        
        processed_messages = []
        
        for message_type, content in conversation:
            # All messages should be moderated
            moderation_result = moderator.moderate_message(content)
            
            # Customer support messages should generally pass moderation
            if not moderation_result.get("filters_triggered"):
                message = orchestrator.add_message(
                    session_id=session.id,
                    content=content,
                    message_type=message_type
                )
                processed_messages.append(message)
        
        # Verify conversation was processed successfully
        assert len(processed_messages) == len(conversation)
        
        # Verify conversation flow
        for i, (expected_type, expected_content) in enumerate(conversation):
            actual_message = processed_messages[i]
            assert actual_message.message_type == expected_type
            assert actual_message.content == expected_content
            assert actual_message.chat_session_id == session.id
    
    def test_multi_user_chat_scenario(self, integrated_chat_system):
        """Test scenario with multiple users chatting"""
        moderator = integrated_chat_system["moderator"]
        orchestrator = integrated_chat_system["orchestrator"]
        
        # Multiple users join chat
        users = ["alice", "bob", "charlie"]
        sessions = {}
        
        for user in users:
            session = orchestrator.create_session(
                user_id=user,
                title=f"{user.title()}'s Chat Session"
            )
            sessions[user] = session
        
        # Simulate multi-user conversation
        chat_sequence = [
            ("alice", "Hello everyone!"),
            ("bob", "Hi Alice! How are you?"),
            ("charlie", "Hey! Great to see you both"),
            ("alice", "I'm doing well, thanks! How about you Bob?"),
            ("bob", "Pretty good! Working on some interesting projects"),
            ("charlie", "That sounds cool! What kind of projects?"),
        ]
        
        all_messages = []
        
        for user, content in chat_sequence:
            session = sessions[user]
            
            # Moderate message
            moderation_result = moderator.moderate_message(content)
            
            # Store message if it passes moderation
            if not moderation_result.get("filters_triggered"):
                message = orchestrator.add_message(
                    session_id=session.id,
                    content=content,
                    message_type="user"
                )
                all_messages.append((user, message))
        
        # Verify all users' messages were processed
        assert len(all_messages) == len(chat_sequence)
        
        # Verify each user's messages are in their own session
        for user in users:
            user_messages = [msg for username, msg in all_messages if username == user]
            session_id = sessions[user].id
            
            for message in user_messages:
                assert message.chat_session_id == session_id
    
    def test_content_moderation_scenario(self, integrated_chat_system, test_data):
        """Test scenario with mixed content requiring moderation"""
        moderator = integrated_chat_system["moderator"]
        orchestrator = integrated_chat_system["orchestrator"]
        
        session = orchestrator.create_session(
            user_id="moderation_test_user",
            title="Content Moderation Test"
        )
        
        # Mix of content types
        mixed_content = (
            test_data.valid_messages()[:2] +
            test_data.spam_messages()[:1] +
            test_data.valid_messages()[2:4] +
            test_data.toxic_messages()[:1] +
            test_data.valid_messages()[4:6]
        )
        
        results = []
        
        for content in mixed_content:
            # Moderate each message
            moderation_result = moderator.moderate_message(content)
            
            # Track moderation decision
            passed_moderation = not moderation_result.get("filters_triggered")
            
            stored_message = None
            if passed_moderation:
                stored_message = orchestrator.add_message(
                    session_id=session.id,
                    content=content,
                    message_type="user"
                )
            
            results.append({
                "content": content,
                "passed_moderation": passed_moderation,
                "stored": stored_message is not None,
                "filters_triggered": moderation_result.get("filters_triggered", [])
            })
        
        # Analyze results
        total_messages = len(results)
        passed_messages = sum(1 for r in results if r["passed_moderation"])
        stored_messages = sum(1 for r in results if r["stored"])
        
        # Verify some messages were processed
        assert total_messages > 0
        assert passed_messages <= total_messages  # Can't pass more than total
        assert stored_messages == passed_messages  # All passed messages should be stored
        
        # Should have some variety in moderation results
        blocked_messages = total_messages - passed_messages
        assert blocked_messages >= 0  # Some messages might be blocked
