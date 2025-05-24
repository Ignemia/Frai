#!/usr/bin/env python3
"""
Basic Chat Demo for Personal Chatter

This demo showcases the basic chat functionality including:
- Initializing a chat session
- Sending messages and receiving responses
- Managing conversation context
- Basic formatting and templating
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def print_header():
    """Print demo header."""
    print("\n" + "=" * 80)
    print(" Personal Chatter - Basic Chat Demo ".center(80, "="))
    print("=" * 80 + "\n")


def simulate_typing(text, delay=0.02):
    """Simulate typing effect for demo purposes."""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()


def run_chat_demo():
    """Run basic chat interaction demo."""
    from services.chat.chat_service import ChatService
    from services.config import ChatConfig
    
    print("\n🤖 Initializing chat service...")
    
    # Initialize chat with mock/demo configuration
    config = ChatConfig(
        model_id="gpt-3.5-turbo",  # This would be a real model in production
        temperature=0.7,
        max_tokens=500,
        system_prompt="You are a helpful assistant named Personal Chatter."
    )
    
    try:
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
        
    except Exception as e:
        print(f"❌ Error during chat demo: {e}")
        return False
        
    return True


def main():
    """Main entry point for the basic chat demo."""
    print_header()
    
    # Mock implementation for demo purposes
    class MockResponse:
        def __init__(self, content):
            self.content = content
    
    # Mock the chat service if it's not available
    try:
        from services.chat.chat_service import ChatService
        success = run_chat_demo()
    except ImportError:
        print("⚠️ Chat service not available. Running mock demonstration...")
        
        # Define mock responses
        mock_responses = [
            "Hello! I'm Personal Chatter, your AI assistant. I can help you with various tasks including answering questions, providing information, assisting with image generation, and having general conversations.",
            
            "Yes, I can help you generate images! Personal Chatter includes an image generation capability. You can provide a text prompt describing what you want to see, and I'll generate an image based on that description. Would you like me to create an image for you?",
            
            "When generating images with Personal Chatter, you can control several parameters:\n\n1. Prompt: The text description of what you want to see.\n2. Image Size: Width and height dimensions (e.g., 512x512, 768x768, 1024x1024).\n3. Guidance Scale: How closely the image follows your prompt (typically 7-15).\n4. Number of Steps: More steps generally means higher quality (typically 20-50).\n5. Seed: For reproducible results.\n\nWould you like to try generating an image with some of these parameters?"
        ]
        
        # Simulate chat interaction with mock responses
        print("\n👤 User: Hello there! Can you tell me what you can do?")
        print("\n🤖 Assistant: ", end="")
        simulate_typing(mock_responses[0])
        print()
        
        print("\n👤 User: Can you help me generate images?")
        print("\n🤖 Assistant: ", end="")
        simulate_typing(mock_responses[1])
        print()
        
        print("\n👤 User: What kind of image parameters can I control?")
        print("\n🤖 Assistant: ", end="")
        simulate_typing(mock_responses[2])
        print()
        
        print("\n" + "-" * 60)
        print(f"Conversation summary (mock):")
        print(f"- Messages exchanged: 3")
        print(f"- Conversation duration: 2 minutes")
        print("-" * 60 + "\n")
        
        success = True
    
    if success:
        print("✅ Basic chat demo completed successfully!")
    else:
        print("❌ Basic chat demo encountered errors.")
        
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
