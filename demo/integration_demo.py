#!/usr/bin/env python3
"""
Integration Demo for Personal Chatter

This demo shows how the chat and image generation services work together,
mimicking a real user interaction flow where chat leads to image generation.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import demo utils
from demo.utils import print_header, print_section, simulate_typing, display_image


def run_integration_demo():
    """Run the integration demo showing chat-to-image-generation flow."""
    print_header("Personal Chatter - Chat & Image Generation Integration Demo")
    
    print("This demo showcases how a chat conversation can lead to image generation.\n")
    
    # Try to import required services
    try:
        from services.chat.chat_service import ChatService
        from services.image_generation.image_generator import ImageGenerator
        from services.config import ChatConfig, ImageGenerationConfig
        
        has_services = True
        print("✅ Required services are available")
        
    except ImportError:
        has_services = False
        print("⚠️ Required services not found. Running in mock mode.")
    
    # Define conversation flow
    print_section("Starting Conversation")
    
    # User's initial message
    user_message = "I'd like to generate an image of a beautiful mountain landscape."
    print("👤 User: " + user_message)
    
    # Assistant's response
    if has_services:
        # Initialize services
        chat_config = ChatConfig(
            model_id="gpt-3.5-turbo",
            temperature=0.7,
            system_prompt="You are an assistant that can generate images. When a user asks for an image, help refine their request and suggest a detailed prompt."
        )
        
        chat_service = ChatService(chat_config)
        
        # Get response from actual service
        response = chat_service.chat(user_message, "integration-demo-1")
        assistant_message = response.content
    else:
        # Mock response
        assistant_message = "I'd be happy to help you generate a mountain landscape image! To create the best possible image, could you tell me a bit more about what you'd like to see? For example:\n\n- Time of day (sunrise, sunset, midday)\n- Weather conditions\n- Any specific features (lakes, snow, forests)\n- Style preferences (realistic, artistic, dreamy)"
    
    print("\n🤖 Assistant: ", end="")
    simulate_typing(assistant_message, delay=0.01)
    
    # User's follow-up with details
    user_followup = "I'd like a sunset view with snow-capped mountains, a clear lake reflecting the mountains, and some pine trees in the foreground. Make it realistic but vibrant."
    print("\n👤 User: " + user_followup)
    
    # Assistant suggests a detailed prompt
    if has_services:
        # Get response from actual service
        response = chat_service.chat(user_followup, "integration-demo-1")
        assistant_followup = response.content
    else:
        # Mock response
        assistant_followup = "That sounds beautiful! I'll generate that image for you. I'll use this prompt to create your image:\n\n\"A breathtaking realistic landscape of snow-capped mountains at sunset, with vibrant orange and pink sky. A perfectly clear alpine lake in the foreground reflecting the mountains like a mirror. Lush green pine trees framing the scene. Highly detailed, professional photography style.\"\n\nWould you like me to generate this image now?"
    
    print("\n🤖 Assistant: ", end="")
    simulate_typing(assistant_followup, delay=0.01)
    
    # User confirms
    user_confirm = "Yes, please generate that image."
    print("\n👤 User: " + user_confirm)
    
    # Assistant generates the image
    print_section("Generating Image")
    
    # Extract the prompt from the assistant's message
    if "\"" in assistant_followup:
        image_prompt = assistant_followup.split("\"")[1]
    else:
        image_prompt = "A breathtaking realistic landscape of snow-capped mountains at sunset, with vibrant orange and pink sky. A perfectly clear alpine lake in the foreground reflecting the mountains like a mirror. Lush green pine trees framing the scene. Highly detailed, professional photography style."
    
    print("📝 Using prompt: \"" + image_prompt + "\"")
    print("\n🖼️ Generating image...")
    
    # Progress simulation
    for i in range(0, 101, 5):
        progress_bar = f"[{'=' * (i // 5)}>{' ' * (20 - i // 5)}] {i}%"
        print(f"\r{progress_bar}", end="")
        time.sleep(0.1)
    print("\n")
    
    if has_services:
        # Use actual image generation service
        img_config = ImageGenerationConfig(
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            output_dir=str(project_root / "outputs" / "demo_images"),
            width=768,
            height=512,
            num_inference_steps=30,
            guidance_scale=7.5
        )
        
        generator = ImageGenerator(img_config)
        result = generator.generate_image(prompt=image_prompt, seed=12345)
        
        if result.get("success", False):
            print("✅ Image generated successfully!")
            image_path = result.get("image_path")
            display_image(image_path)
        else:
            print(f"❌ Image generation failed: {result.get('error', 'Unknown error')}")
            image_path = None
    else:
        # Mock image generation
        image_path = str(project_root / "outputs" / "demo_images" / "mock_mountain_landscape.png")
        print("✅ Image generated successfully! (mock)")
        display_image(image_path)
    
    # Assistant provides the image
    assistant_final = f"Here's the mountain landscape image I've created for you! It shows snow-capped mountains at sunset with the golden light reflecting off a clear alpine lake, surrounded by pine trees. I hope you like it!"
    
    print("\n🤖 Assistant: ", end="")
    simulate_typing(assistant_final, delay=0.01)
    
    # Show summary
    print_section("Demo Summary")
    print("This demonstration showed how Personal Chatter integrates:")
    print("1. Natural language conversation to understand user needs")
    print("2. Collaborative refinement of image requirements")
    print("3. Converting conversation into a detailed image generation prompt")
    print("4. Generating the requested image")
    print("5. Presenting results back to the user")
    
    return True


def main():
    """Main entry point for the integration demo."""
    success = run_integration_demo()
    
    if success:
        print("\n✅ Integration demo completed successfully!")
    else:
        print("\n❌ Integration demo encountered errors.")
        
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
