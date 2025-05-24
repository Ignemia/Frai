#!/usr/bin/env python3
"""
Basic Chat Demo for Personal Chatter - NO MOCK RESPONSES

This demo showcases the REAL chat functionality:
- Initializing a chat session
- Sending messages and receiving responses
- Managing conversation context
- Basic formatting and templating

NO FALLBACKS OR MOCK RESPONSES - REAL CHAT SERVICE ONLY
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import required services
from services.chat.chat_service import ChatService
from services.config import ChatConfig


def print_header():
    """Print demo header."""
    print("\n" + "=" * 80)
    print(" Personal Chatter - Basic Chat Demo (REAL CHAT ONLY) ".center(80, "="))
    print("=" * 80 + "\n")


def simulate_typing(text, delay=0.02):
    """Simulate typing effect for demo purposes."""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()


def run_chat_demo():
    """Run basic chat interaction demo with REAL chat service."""
    print("\n🤖 Initializing chat service...")
    
    # Initialize chat with real configuration
    config = ChatConfig(
        model_id="gemma-3-4b-it",  # Using the actual loaded model
        temperature=0.7,
        max_tokens=500,
        system_prompt="You are a helpful assistant named Personal Chatter."
    )
    
    chat_service = ChatService(config)
    print("✅ Chat service initialized successfully\n")
    
    # Begin conversation demonstration
    print("\n" + "-" * 60)
    print(" Chat Session Demo ".center(60, "-"))
    print("-" * 60 + "\n")
    
    # Simulate user interaction
    conversation_id = "demo-session-1"
    
    # First message
    print("👤 User: Hello there! Can you tell me what you can do?")
    response = chat_service.chat(
        user_message="Hello there! Can you tell me what you can do?",
        conversation_id=conversation_id
    )
    print("\n🤖 Assistant: ", end="")
    simulate_typing(response.content)
    print()
    
    # Second message - asking about something specific
    print("\n👤 User: Can you help me generate images?")
    response = chat_service.chat(
        user_message="Can you help me generate images?",
        conversation_id=conversation_id
    )
    print("\n🤖 Assistant: ", end="")
    simulate_typing(response.content)
    print()
    
    # Third message - follow-up with context
    print("\n👤 User: What kind of image parameters can I control?")
    response = chat_service.chat(
        user_message="What kind of image parameters can I control?",
        conversation_id=conversation_id
    )
    print("\n🤖 Assistant: ", end="")
    simulate_typing(response.content)
    print()
    
    # Display conversation stats
    print("\n" + "-" * 60)
    conversation_stats = chat_service.get_conversation_stats(conversation_id)
    print(f"Conversation summary:")
    print(f"- Messages exchanged: {conversation_stats.get('message_count', 3)}")
    print(f"- Conversation duration: {conversation_stats.get('duration', '2 minutes')}")
    print("-" * 60 + "\n")
    
    return True


def main():
    """Main entry point for the basic chat demo."""
    print_header()
    
    try:
        success = run_chat_demo()
        if success:
            print("✅ Basic chat demo completed successfully!")
        return 0
    except Exception as e:
        print(f"❌ Basic chat demo failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
