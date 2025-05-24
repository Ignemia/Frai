#!/usr/bin/env python3
"""
API Usage Demo for Personal Chatter - NO MOCK RESPONSES

This demo showcases how to use Personal Chatter's API programmatically:
- Making API calls to various endpoints
- Handling requests and responses
- Working with the chat and image generation APIs
- Displaying results

NO FALLBACKS OR MOCK RESPONSES - REAL API CALLS ONLY
"""

import json
import sys
import time
import os
from datetime import datetime
from pathlib import Path
import requests

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import demo utils
from demo.utils import print_header, print_section, display_image

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
    print("✅ Loaded configuration from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed, falling back to default configuration")


def run_api_usage_demo():
    """Run the API usage demo with REAL API calls only."""
    print_header("Personal Chatter - API Usage Demo (REAL API ONLY)")
    
    # Get API configuration from environment variables
    api_host = os.getenv("API_HOST", "localhost")
    api_port = os.getenv("API_PORT", "8000")
    debug_mode = os.getenv("DEBUG", "False").lower() == "true"
    
    if debug_mode:
        print(f"🔧 Debug mode enabled")
        print(f"🔌 API Host: {api_host}:{api_port}")
    
    # API URLs
    SERVER_API_URL = f"http://{api_host}:{api_port}/server"
    USER_API_URL = f"http://{api_host}:{api_port}/user"
    CHAT_API_URL = f"http://{api_host}:{api_port}/chat"
    IMAGE_API_URL = f"http://{api_host}:{api_port}/image"

    auth_token = None    # 0. Authenticate User
    print_section("0. Authenticating User")
    login_credentials = {
        "username": "testuser", 
        "password": "testpassword"
    }
    
    print(f"Attempting login for user: {login_credentials['username']}")
    response = requests.post(f"{USER_API_URL}/login", data=login_credentials, timeout=10)
    response.raise_for_status()
    token_data = response.json()
    auth_token = token_data["access_token"]
    print(f"✅ Successfully authenticated. Token obtained.")
    
    # 1. Check system status
    print_section("1. Checking API Status")
    response = requests.get(f"{SERVER_API_URL}/status", timeout=5)
    response.raise_for_status()
    status_data = response.json()
    print(f"API Status: {status_data['status']}")
    print(f"Version: {status_data['version']}")
    print(f"Uptime: {status_data['uptime']}")
    
    # 2. Making a chat request
    print_section("2. Making a Chat API Request")
    chat_request = {
        "message": "Hello, can you tell me about image generation?",
        "conversation_id": "demo-conversation-1",
        "options": {
            "temperature": 0.7,
            "max_tokens": 300
        }
    }
    
    print("Request data:")
    print(json.dumps(chat_request, indent=2))
    
    headers = {"Authorization": f"Bearer {auth_token}"}
    response = requests.post(f"{CHAT_API_URL}/message", json=chat_request, headers=headers, timeout=30)
    response.raise_for_status()
    chat_response = response.json()
    print("\nResponse data:")
    print(json.dumps(chat_response, indent=2))
    
    # 3. Making an image generation request
    print_section("3. Making an Image Generation API Request")
    
    image_request = {
        "prompt": "A futuristic cityscape at night with neon lights and flying cars",
        "width": 512,
        "height": 512,
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "negative_prompt": "blurry, distorted, low quality"
    }
    
    print("Request data:")
    print(json.dumps(image_request, indent=2))
    
    # Create demo output directory if it doesn't exist
    demo_output_dir = project_root / "outputs" / "demo_images"
    demo_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating image...")
    headers = {"Authorization": f"Bearer {auth_token}"}
    response = requests.post(f"{IMAGE_API_URL}/generate", json=image_request, headers=headers, timeout=120)
    response.raise_for_status()
    image_response = response.json()
    print("Response data:")
    print(json.dumps(image_response, indent=2))
    
    # Display the image if available
    if image_response.get("success", False):
        image_url = image_response.get("image_url")
        if image_url:
            if image_url.startswith("http"):
                # Download the image
                img_response = requests.get(image_url, stream=True)
                img_response.raise_for_status()
                
                local_img_path = demo_output_dir / "api_generated_image.png"
                with open(local_img_path, 'wb') as f:
                    for chunk in img_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
                print(f"\nImage downloaded to: {local_img_path}")
                display_image(local_img_path)
            else:
                # Local file path
                local_path = project_root / image_url.lstrip('/')
                if local_path.exists():
                    print(f"\nImage available at: {local_path}")
                    display_image(local_path)
                else:
                    print(f"\nImage file not found at: {local_path}")
                    print(f"Expected URL: http://localhost:8000{image_url}")
    
    # 4. Making a batch request
    print_section("4. Making a Batch Processing API Request")
    
    batch_request = {
        "requests": [
            {"prompt": "A cat playing with a ball of yarn", "width": 512, "height": 512},
            {"prompt": "A dog running in a park", "width": 512, "height": 512},
            {"prompt": "A fish swimming in a coral reef", "width": 512, "height": 512}
        ],
        "common_parameters": {
            "num_inference_steps": 25,
            "guidance_scale": 7.0
        }
    }
    
    print("Batch request data:")
    print(json.dumps(batch_request, indent=2))
    
    print("\nProcessing batch request...")
    headers = {"Authorization": f"Bearer {auth_token}"}
    response = requests.post(f"{IMAGE_API_URL}/batch_generate", json=batch_request, headers=headers, timeout=300)
    response.raise_for_status()
    batch_response = response.json()
    print("Response data:")
    print(json.dumps(batch_response, indent=2))
    
    # Display the first image from the batch if available
    if batch_response.get("success", False) and batch_response.get("results", []):
        first_result = batch_response["results"][0]
        if first_result.get("success", False) and first_result.get("image_url"):
            image_url = first_result["image_url"]
            local_path = project_root / image_url.lstrip('/')
            if local_path.exists():
                print(f"\nFirst batch image available at: {local_path}")
                display_image(local_path)
            else:
                print(f"\nFirst batch image would be available at: http://localhost:8000{image_url}")
    
    # Summary
    print_section("API Usage Summary")
    print("✅ Completed API demo with REAL responses:")
    print("1. ✅ Authenticated user")
    print("2. ✅ Checked system status")
    print("3. ✅ Sent chat message and received response")
    print("4. ✅ Generated image with parameters")
    print("5. ✅ Processed batch request for multiple images")
    print("\nAPI Reference:")
    print("- Documentation: http://localhost:8000/docs")
    print("- OpenAPI Spec: http://localhost:8000/openapi.json")
    
    return True


def main():
    """Main entry point for the API usage demo."""
    try:
        success = run_api_usage_demo()
        if success:
            print("\n✅ API usage demo completed successfully!")
        return 0
    except Exception as e:
        print(f"\n❌ API usage demo failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
