"""
Test Script for Chat Moderator and Orchestrator

This script tests the chat moderator and orchestrator functionality
without requiring the full model loading process.
"""

import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_chat_moderator():
    """Test the chat moderator functionality."""
    print("🧪 Testing Chat Moderator...")
    
    try:
        from orchestrator.chatmod import get_chat_moderator, moderate_user_message, filter_ai_response
        
        moderator = get_chat_moderator()
        print("✅ Chat moderator initialized successfully")
        
        # Test message validation
        print("\n📝 Testing message validation...")
        
        test_cases = [
            ("Hello, how are you?", True, "Normal message"),
            ("", False, "Empty message"),
            ("a" * 10000, False, "Message too long"),
            ("\x00Hello", False, "Message with null bytes"),
            ("HELLO " * 10, True, "Caps message (spam pattern)"),
            ("This message contains hate speech", True, "Message with toxic keyword"),
        ]
        
        for message, expected_approval, description in test_cases:
            result = moderate_user_message(message, "test-user")
            approved = result.get("approved", False)
            
            print(f"   {description}: {'✅' if approved else '❌'}")
            if result.get("filters_triggered"):
                print(f"      Filters: {result['filters_triggered']}")
            if result.get("warnings"):
                print(f"      Warnings: {result['warnings']}")
        
        # Test response filtering
        print("\n🔍 Testing response filtering...")
        
        test_response = "Here's a helpful response with a link: https://example.com"
        filter_result = filter_ai_response(test_response)
        
        print(f"   Original: {test_response}")
        print(f"   Filtered: {filter_result.get('filtered_response', 'ERROR')}")
        print(f"   Approved: {'✅' if filter_result.get('approved') else '❌'}")
        
        print("✅ Chat moderator tests completed\n")
        return True
        
    except Exception as e:
        print(f"❌ Chat moderator test failed: {e}")
        return False

def test_chat_orchestrator():
    """Test the chat orchestrator functionality."""
    print("🧪 Testing Chat Orchestrator...")
    
    try:
        from orchestrator.chat import (
            get_chat_orchestrator, create_session, send_message, 
            add_assistant_response, get_conversation_history, MessageType
        )
        
        orchestrator = get_chat_orchestrator()
        print("✅ Chat orchestrator initialized successfully")
        
        # Test session creation
        print("\n📁 Testing session creation...")
        session = create_session("test-user", "Test Session")
        print(f"   Created session: {session.id}")
        print(f"   Session title: {session.title}")
        print(f"   Created at: {session.created_at}")
        
        # Test message handling
        print("\n💬 Testing message handling...")
        
        # Send user message
        user_msg = send_message(session.id, "Hello, this is a test message!", "test-user")
        print(f"   User message added: {user_msg.id}")
        
        # Add assistant response
        ai_msg = add_assistant_response(session.id, "Hello! I'm here to help you.")
        print(f"   Assistant response added: {ai_msg.id}")
        
        # Add another exchange
        user_msg2 = send_message(session.id, "Can you tell me about the weather?", "test-user")
        ai_msg2 = add_assistant_response(session.id, "I don't have access to real-time weather data, but I'd be happy to help with other questions!")
        
        # Test conversation history
        print("\n📜 Testing conversation history...")
        history = get_conversation_history(session.id)
        
        for i, msg in enumerate(history, 1):
            role = "👤 User" if msg.message_type == MessageType.USER else "🤖 Assistant"
            print(f"   {i}. {role}: {msg.content[:50]}{'...' if len(msg.content) > 50 else ''}")
        
        # Test session stats
        print("\n📊 Testing session statistics...")
        stats = orchestrator.get_session_stats(session.id)
        if stats:
            print(f"   Total messages: {stats['total_messages']}")
            print(f"   User messages: {stats['user_messages']}")
            print(f"   Assistant messages: {stats['assistant_messages']}")
            print(f"   Total characters: {stats['total_characters']}")
        
        # Test context generation
        print("\n🔄 Testing conversation context...")
        context = orchestrator.get_conversation_context(session.id)
        print(f"   Context length: {len(context)} characters")
        print(f"   Context preview: {context[:100]}...")
        
        # Test session export
        print("\n📤 Testing session export...")
        export_data = orchestrator.export_session(session.id)
        if export_data:
            print(f"   Exported session with {len(export_data['messages'])} messages")
        
        print("✅ Chat orchestrator tests completed\n")
        return True
        
    except Exception as e:
        print(f"❌ Chat orchestrator test failed: {e}")
        return False

def test_database_integration():
    """Test the database integration."""
    print("🧪 Testing Database Integration...")
    
    try:
        from back.database.chats import (
            get_chat_database, initiate_chat_database, test_chat_database
        )
        
        # Initialize database
        if not initiate_chat_database():
            print("❌ Failed to initialize chat database")
            return False
        
        print("✅ Chat database initialized")
        
        # Test database functionality
        if not test_chat_database():
            print("❌ Chat database test failed")
            return False
        
        print("✅ Chat database test passed")
        
        # Test database stats
        db = get_chat_database()
        stats = db.get_database_stats()
        print(f"   Database stats: {stats}")
        
        print("✅ Database integration tests completed\n")
        return True
        
    except Exception as e:
        print(f"❌ Database integration test failed: {e}")
        return False

def test_integration():
    """Test integration between moderator and orchestrator."""
    print("🧪 Testing Integration...")
    
    try:
        from orchestrator.chat import create_session, get_chat_orchestrator, MessageType
        from orchestrator.chatmod import moderate_user_message, filter_ai_response
        
        # Create a test session
        session = create_session("integration-test-user", "Integration Test")
        orchestrator = get_chat_orchestrator()
        
        print(f"✅ Created integration test session: {session.id}")
        
        # Test full message flow
        test_message = "Hello, I'd like to test the integration!"
        
        # 1. Moderate the message
        moderation_result = moderate_user_message(test_message, "integration-test-user")
        print(f"   Message moderation: {'✅ APPROVED' if moderation_result.get('approved') else '❌ REJECTED'}")
        
        if moderation_result.get("approved"):
            # 2. Add to orchestrator
            user_msg = orchestrator.add_message(
                session.id, MessageType.USER, test_message, "integration-test-user"
            )
            print(f"   Message added to session: {user_msg.id}")
            
            # 3. Simulate AI response
            ai_response = "Hello! I'm happy to help you test the integration. Everything seems to be working well!"
            
            # 4. Filter the response
            filter_result = filter_ai_response(ai_response)
            print(f"   Response filtering: {'✅ APPROVED' if filter_result.get('approved') else '❌ REJECTED'}")
            
            if filter_result.get("approved"):
                final_response = filter_result.get("filtered_response", ai_response)
                
                # 5. Add filtered response to session
                ai_msg = orchestrator.add_message(
                    session.id, MessageType.ASSISTANT, final_response
                )
                print(f"   AI response added to session: {ai_msg.id}")
                
                # 6. Verify conversation
                messages = orchestrator.get_session_messages(session.id)
                print(f"   Total messages in session: {len(messages)}")
                
                print("✅ Integration test completed successfully")
                return True
        
        print("❌ Integration test failed during message flow")
        return False
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Chat System Tests")
    print("=" * 50)
    
    tests = [
        ("Chat Moderator", test_chat_moderator),
        ("Chat Orchestrator", test_chat_orchestrator),
        ("Database Integration", test_database_integration),
        ("System Integration", test_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} tests...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Chat system is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
