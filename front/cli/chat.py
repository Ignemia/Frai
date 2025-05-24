"""
Chat CLI Module

Provides a command-line interface for testing chat functionality.
"""

import logging
from typing import Optional
from orchestrator.chat import get_chat_orchestrator, create_session
from back.ai.chat import chat_completion, get_chat_ai

logger = logging.getLogger(__name__)

def start_chat_session(user_id: str = "cli-user") -> str:
    """Start a new chat session and return the session ID."""
    try:
        session = create_session(user_id, "CLI Chat Session")
        logger.info(f"Started new chat session: {session.id}")
        return session.id
    except Exception as e:
        logger.error(f"Failed to start chat session: {e}")
        return ""

def chat_loop(session_id: str, user_id: str = "cli-user"):
    """Run an interactive chat loop."""
    print("🤖 Frai Chat Interface")
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'history' to see conversation history")
    print("-" * 50)
    
    chat_ai = get_chat_ai()
    if not chat_ai.is_loaded:
        print("⚠️  Chat model not loaded. Please ensure the model is properly initialized.")
        return
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'history':
                show_conversation_history(session_id)
                continue
            
            if not user_input:
                print("Please enter a message.")
                continue
            
            print("\n🤖 Frai: ", end="", flush=True)
            
            # Process the message
            result = chat_completion(session_id, user_input, user_id)
            
            if result["success"]:
                print(result["ai_response"])
                
                # Show any warnings from moderation or filtering
                if result["moderation"].get("warnings"):
                    print("\n⚠️  Moderation warnings:", result["moderation"]["warnings"])
                
                if result["filtering"].get("filters_applied"):
                    print("\n🔧 Filters applied:", result["filtering"]["filters_applied"])
            else:
                print(f"❌ Error: {result['error']}")
                
                # Show moderation details if message was rejected
                if result["moderation"]:
                    mod = result["moderation"]
                    if mod.get("filters_triggered"):
                        print(f"🚫 Filters triggered: {mod['filters_triggered']}")
                    if mod.get("warnings"):
                        print(f"⚠️  Warnings: {mod['warnings']}")
        
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error in chat loop: {e}")
            print(f"\n❌ Unexpected error: {e}")

def show_conversation_history(session_id: str):
    """Show the conversation history for a session."""
    try:
        orchestrator = get_chat_orchestrator()
        messages = orchestrator.get_session_messages(session_id)
        
        print("\n📜 Conversation History:")
        print("-" * 30)
        
        if not messages:
            print("No messages in this session.")
            return
        
        for message in messages:
            timestamp = message.timestamp.strftime("%H:%M:%S")
            role = "👤 You" if message.message_type.value == "user" else "🤖 Frai"
            print(f"[{timestamp}] {role}: {message.content}")
        
        print("-" * 30)
        
    except Exception as e:
        logger.error(f"Error showing history: {e}")
        print(f"❌ Error showing history: {e}")

def run_chat_demo():
    """Run a complete chat demonstration."""
    print("🚀 Starting Frai Chat Demo...")
    
    try:
        # Start a new session
        session_id = start_chat_session()
        if not session_id:
            print("❌ Failed to start chat session")
            return
        
        print(f"✅ Session started: {session_id}")
        
        # Run the chat loop
        chat_loop(session_id)
        
        # Show final session stats
        orchestrator = get_chat_orchestrator()
        stats = orchestrator.get_session_stats(session_id)
        if stats:
            print(f"\n📊 Session Stats:")
            print(f"   Messages: {stats['total_messages']}")
            print(f"   User messages: {stats['user_messages']}")
            print(f"   AI responses: {stats['assistant_messages']}")
        
    except Exception as e:
        logger.error(f"Error in chat demo: {e}")
        print(f"❌ Demo error: {e}")

if __name__ == "__main__":
    run_chat_demo()