#!/usr/bin/env python3
"""
API Usage Demo for Personal Chatter

This demo showcases how to use Personal Chatter's API programmatically:
- Making API calls to various endpoints
- Handling requests and responses
- Working with the chat and image generation APIs
- Displaying results
"""

import json
import sys
import time
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import demo utils
from demo.utils import print_header, print_section, display_image

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load from the root directory .env file
    load_dotenv(project_root / ".env")
    print("✅ Loaded configuration from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed, falling back to default configuration")


def run_api_usage_demo():
    """Run the API usage demo."""
    print_header("Personal Chatter - API Usage Demo")
    
    # Get API configuration from environment variables
    api_host = os.getenv("API_HOST", "localhost")
    api_port = os.getenv("API_PORT", "8000")
    api_base_url = os.getenv("API_BASE_URL", f"http://{api_host}:{api_port}/api/v1")
    debug_mode = os.getenv("DEBUG", "False").lower() == "true"
    
    if debug_mode:
        print(f"🔧 Debug mode enabled")
        print(f"🔌 Using API at: {api_base_url}")
    
    # Try to import requests for API calls
    try:
        import requests
        HAS_REQUESTS = True
    except ImportError:
        print("⚠️ 'requests' module not found. Will simulate API calls.")
        HAS_REQUESTS = False
    
    # Check if API is running
    API_RUNNING = False
    if HAS_REQUESTS:
        try:
            status_url = f"http://{api_host}:{api_port}/status"
            response = requests.get(status_url, timeout=2)
            API_RUNNING = response.status_code == 200
            print(f"✅ API is running: {API_RUNNING}")
        except Exception as e:
            print(f"⚠️ API server does not appear to be running: {e}. Will simulate API calls.")
    
    # API base URL - now using value from environment
    API_URL = api_base_url
    
    # 1. Check system status
    print_section("1. Checking API Status")
    
    if API_RUNNING and HAS_REQUESTS:
        try:
            response = requests.get(f"{API_URL}/status")
            status_data = response.json()
            print(f"API Status: {status_data.get('status', 'unknown')}")
            print(f"Version: {status_data.get('version', 'unknown')}")
            print(f"Uptime: {status_data.get('uptime', 'unknown')}")
        except Exception as e:
            print(f"❌ Error checking API status: {e}")
            # Fall back to mock status
            status_data = {
                "status": "operational (simulated)",
                "version": "1.0.0",
                "uptime": "2h 34m",
                "timestamp": datetime.now().isoformat()
            }
            print(f"API Status (mock fallback): {status_data['status']}")
            print(f"Version: {status_data['version']}")
            print(f"Uptime: {status_data['uptime']}")
    else:
        # Mock status response
        mock_status = {
            "status": "operational",
            "version": "1.0.0",
            "uptime": "2h 34m",
            "timestamp": datetime.now().isoformat()
        }
        print(f"API Status (mock): {mock_status['status']}")
        print(f"Version: {mock_status['version']}")
        print(f"Uptime: {mock_status['uptime']}")
    
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
    
    if API_RUNNING and HAS_REQUESTS:
        try:
            response = requests.post(f"{API_URL}/chat", json=chat_request)
            response.raise_for_status()  # Raise exception for HTTP errors
            chat_response = response.json()
            print("\nResponse data:")
            print(json.dumps(chat_response, indent=2))
        except Exception as e:
            print(f"\n❌ Error making chat request: {e}")
            # Fall back to mock response
            chat_response = {
                "id": "msg_1234567890",
                "conversation_id": "demo-conversation-1",
                "content": "Hello! I'd be happy to tell you about image generation. Personal Chatter uses state-of-the-art diffusion models to generate high-quality images from text descriptions (prompts). You can specify parameters like image size, guidance scale, and number of steps to control the generation process. Would you like me to show you how to generate an image?",
                "created_at": datetime.now().isoformat(),
                "processing_time": 0.45
            }
            print("\nResponse data (mock fallback):")
            print(json.dumps(chat_response, indent=2))
    else:
        # Mock chat response
        mock_chat_response = {
            "id": "msg_1234567890",
            "conversation_id": "demo-conversation-1",
            "content": "Hello! I'd be happy to tell you about image generation. Personal Chatter uses state-of-the-art diffusion models to generate high-quality images from text descriptions (prompts). You can specify parameters like image size, guidance_scale, and number of steps to control the generation process. Would you like me to show you how to generate an image?",
            "created_at": datetime.now().isoformat(),
            "processing_time": 0.45
        }
        print("\nResponse data (mock):")
        print(json.dumps(mock_chat_response, indent=2))
    
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
    
    # Simulate request being processed
    print("\nProcessing image generation request...")
    for i in range(0, 101, 10):
        progress_bar = f"[{'=' * (i // 10)}>{' ' * (10 - i // 10)}] {i}%"
        print(f"\r{progress_bar}", end="")
        time.sleep(0.2)
    print("\n")
    
    # Create demo output directory if it doesn't exist
    demo_output_dir = project_root / "outputs" / "demo_images"
    demo_output_dir.mkdir(parents=True, exist_ok=True)
    
    if API_RUNNING and HAS_REQUESTS:
        try:
            response = requests.post(f"{API_URL}/generate_image", json=image_request)
            response.raise_for_status()  # Raise exception for HTTP errors
            image_response = response.json()
            print("Response data:")
            print(json.dumps(image_response, indent=2))
            
            # Display the image if available
            if image_response.get("success", False):
                image_url = image_response.get("image_url")
                if image_url:
                    if image_url.startswith("http"):
                        # For demo, let's download the image so we can display it
                        try:
                            img_response = requests.get(image_url, stream=True)
                            img_response.raise_for_status()
                            
                            local_img_path = demo_output_dir / "api_generated_image.png"
                            with open(local_img_path, 'wb') as f:
                                for chunk in img_response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                                    
                            print(f"\nImage downloaded to: {local_img_path}")
                            display_image(local_img_path)
                        except Exception as e:
                            print(f"❌ Error downloading image: {e}")
                            print(f"Image URL: {image_url}")
                    else:
                        # Local file path
                        local_path = project_root / image_url.lstrip('/')
                        if local_path.exists():
                            print(f"\nImage available at: {local_path}")
                            display_image(local_path)
                        else:
                            print(f"\nImage file not found at: {local_path}")
                            print(f"Would be available at: http://localhost:8000{image_url}")
        except Exception as e:
            print(f"❌ Error making image generation request: {e}")
            
            # Fall back to mock image response
            mock_img_path = demo_output_dir / "futuristic_cityscape.png"
            mock_image_response = {
                "success": True,
                "image_url": f"/outputs/demo_images/{mock_img_path.name}",
                "request_id": "req_" + str(int(time.time())),
                "processing_time": 3.45,
                "metadata": {
                    "prompt": image_request["prompt"],
                    "width": image_request["width"],
                    "height": image_request["height"],
                    "num_inference_steps": image_request["num_inference_steps"],
                    "guidance_scale": image_request["guidance_scale"],
                    "seed": 1234567890
                }
            }
            print("Response data (mock fallback):")
            print(json.dumps(mock_image_response, indent=2))
            print(f"\nImage would be available at: http://localhost:8000{mock_image_response['image_url']}")
            
            # Use mock image for demo if available
            if not mock_img_path.exists():
                print(f"Note: For a complete demo, place a sample image at {mock_img_path}")
            else:
                display_image(mock_img_path)
    else:
        # Mock image generation response
        mock_img_path = demo_output_dir / "futuristic_cityscape.png"
        mock_image_response = {
            "success": True,
            "image_url": f"/outputs/demo_images/{mock_img_path.name}",
            "request_id": "req_" + str(int(time.time())),
            "processing_time": 3.45,
            "metadata": {
                "prompt": image_request["prompt"],
                "width": image_request["width"],
                "height": image_request["height"],
                "num_inference_steps": image_request["num_inference_steps"],
                "guidance_scale": image_request["guidance_scale"],
                "seed": 1234567890
            }
        }
        print("Response data (mock):")
        print(json.dumps(mock_image_response, indent=2))
        print(f"\nImage would be available at: http://localhost:8000{mock_image_response['image_url']}")
        
        # Use mock image for demo if available
        if not mock_img_path.exists():
            print(f"Note: For a complete demo, place a sample image at {mock_img_path}")
        else:
            display_image(mock_img_path)
    
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
    
    # Simulate longer processing time for batch
    print("\nProcessing batch request...")
    for i in range(0, 101, 5):
        progress_bar = f"[{'=' * (i // 5)}>{' ' * (20 - i // 5)}] {i}%"
        print(f"\r{progress_bar}", end="")
        time.sleep(0.15)
    print("\n")
    
    if API_RUNNING and HAS_REQUESTS:
        try:
            response = requests.post(f"{API_URL}/generate_batch", json=batch_request)
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
        except Exception as e:
            print(f"❌ Error making batch request: {e}")
            # Fall back to mock batch response
            mock_batch_response = create_mock_batch_response(batch_request, demo_output_dir)
            print("Response data (mock fallback):")
            print(json.dumps(mock_batch_response, indent=2))
    else:
        # Mock batch response
        mock_batch_response = create_mock_batch_response(batch_request, demo_output_dir)
        print("Response data (mock):")
        print(json.dumps(mock_batch_response, indent=2))
    
    # Summary
    print_section("API Usage Summary")
    print("This demo has shown how to use the Personal Chatter API for:")
    print("1. Checking system status")
    print("2. Sending chat messages and receiving responses")
    print("3. Requesting image generation with parameters")
    print("4. Processing batch requests for multiple images")
    print("\nAPI Reference:")
    print("- Documentation: http://localhost:8000/docs")
    print("- OpenAPI Spec: http://localhost:8000/openapi.json")
    
    return True


def create_mock_batch_response(batch_request, output_dir):
    """Create a mock batch response with appropriate image paths."""
    batch_id = f"batch_{int(time.time())}"
    results = []
    
    for i, req in enumerate(batch_request["requests"]):
        # Create a sensible filename based on the prompt
        prompt_words = req["prompt"].lower().split()[:3]
        filename = f"{prompt_words[0]}_{i+1}.png"
        
        result = {
            "success": True,
            "image_url": f"/outputs/demo_images/{filename}",
            "request_id": f"req_{prompt_words[0]}_{int(time.time())}",
        }
        results.append(result)
        
        # Note if the mock file exists
        mock_path = output_dir / filename
        if not mock_path.exists():
            print(f"Note: For a complete demo, place a sample image at {mock_path}")
        
    return {
        "success": True,
        "batch_id": batch_id,
        "total_processed": len(results),
        "successful": len(results),
        "failed": 0,
        "processing_time": 9.72,
        "results": results
    }


def main():
    """Main entry point for the API usage demo."""
    success = run_api_usage_demo()
    
    if success:
        print("\n✅ API usage demo completed successfully!")
    else:
        print("\n❌ API usage demo encountered errors.")
        
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
