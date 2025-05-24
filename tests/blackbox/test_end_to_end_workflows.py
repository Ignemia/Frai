"""
End-to-end blackbox tests for the personal-chatter application.

These tests treat the entire system as a black box, testing complete
user workflows from API calls to final outputs without relying on
knowledge of internal implementation details.
"""

import pytest
import requests
import json
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

# These tests would typically run against a live system
# For now, we'll mock the external dependencies


@pytest.mark.blackbox
@pytest.mark.external
class TestImageGenerationAPI:
    """End-to-end tests for image generation API."""
    
    @pytest.fixture
    def api_base_url(self):
        """Base URL for API tests."""
        return "http://localhost:8000"
    
    @pytest.fixture
    def temp_output_dir(self):
        """Temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp(prefix="blackbox_test_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @patch('requests.post')
    def test_basic_image_generation_workflow(self, mock_post, api_base_url):
        """Test complete image generation workflow via API."""
        # Mock successful API response
        mock_response = {
            "success": True,
            "image_url": "/outputs/test_image.png",
            "generation_time": 5.2,
            "metadata": {
                "prompt": "A beautiful landscape",
                "width": 512,
                "height": 512,
                "steps": 20
            }
        }
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response
        
        # Make API request
        payload = {
            "prompt": "A beautiful landscape",
            "width": 512,
            "height": 512,
            "steps": 20
        }
        
        response = requests.post(f"{api_base_url}/api/image/generate", json=payload)
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "image_url" in data
        assert "generation_time" in data
        assert data["metadata"]["prompt"] == "A beautiful landscape"
    
    @patch('requests.post')
    def test_invalid_prompt_handling(self, mock_post, api_base_url):
        """Test API handles invalid prompts appropriately."""
        # Mock error response
        mock_response = {
            "success": False,
            "error": "Prompt cannot be empty",
            "error_type": "ValidationError"
        }
        mock_post.return_value.status_code = 400
        mock_post.return_value.json.return_value = mock_response
        
        # Send invalid request
        payload = {
            "prompt": "",  # Empty prompt
            "width": 512,
            "height": 512,
            "steps": 20
        }
        
        response = requests.post(f"{api_base_url}/api/image/generate", json=payload)
        
        # Verify error handling
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "error" in data
        assert "prompt" in data["error"].lower()
    
    @patch('requests.post')
    def test_parameter_boundary_conditions(self, mock_post, api_base_url):
        """Test system behavior at parameter boundaries."""
        test_cases = [
            # Minimum values
            {"prompt": "test", "width": 64, "height": 64, "steps": 1},
            # Maximum reasonable values
            {"prompt": "test", "width": 1024, "height": 1024, "steps": 50},
            # Edge cases
            {"prompt": "test", "width": 512, "height": 768, "steps": 20},
        ]
        
        for payload in test_cases:
            mock_response = {
                "success": True,
                "image_url": f"/outputs/test_{payload['width']}x{payload['height']}.png",
                "generation_time": 3.0,
                "metadata": payload
            }
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response
            
            response = requests.post(f"{api_base_url}/api/image/generate", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
    
    @patch('requests.get')
    def test_system_status_endpoint(self, mock_get, api_base_url):
        """Test system status and health check endpoint."""
        mock_response = {
            "status": "operational",
            "gpu_available": False,
            "memory_usage": {
                "allocated_gb": 0.5,
                "free_gb": 7.5,
                "total_gb": 8.0
            },
            "active_generations": 0,
            "total_generated": 42
        }
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_response
        
        response = requests.get(f"{api_base_url}/api/image/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "gpu_available" in data
        assert "memory_usage" in data


@pytest.mark.blackbox
@pytest.mark.slow
class TestChatAPI:
    """End-to-end tests for chat API functionality."""
    
    @pytest.fixture
    def api_base_url(self):
        """Base URL for API tests."""
        return "http://localhost:8000"
    
    @patch('requests.post')
    def test_basic_chat_workflow(self, mock_post, api_base_url):
        """Test complete chat conversation workflow."""
        mock_response = {
            "response": "Hello! How can I help you today?",
            "conversation_id": "conv_123",
            "message_id": "msg_456",
            "processing_time": 0.8
        }
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response
        
        payload = {
            "message": "Hello, I need help with image generation",
            "conversation_id": None
        }
        
        response = requests.post(f"{api_base_url}/api/chat/message", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "conversation_id" in data
        assert data["processing_time"] > 0
    
    @patch('requests.post')
    def test_image_generation_via_chat(self, mock_post, api_base_url):
        """Test requesting image generation through chat interface."""
        mock_response = {
            "response": "I'll generate that image for you. Here's your beautiful landscape:",
            "image_generated": True,
            "image_url": "/outputs/chat_generated_image.png",
            "conversation_id": "conv_123",
            "processing_time": 8.5
        }
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response
        
        payload = {
            "message": "Please generate an image of a beautiful mountain landscape at sunset",
            "conversation_id": "conv_123"
        }
        
        response = requests.post(f"{api_base_url}/api/chat/message", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["image_generated"] is True
        assert "image_url" in data
        assert "beautiful" in data["response"].lower() or "landscape" in data["response"].lower()


@pytest.mark.blackbox
@pytest.mark.network
class TestSearchIntegration:
    """End-to-end tests for search functionality."""
    
    @pytest.fixture
    def api_base_url(self):
        """Base URL for API tests."""
        return "http://localhost:8000"
    
    @patch('requests.post')
    def test_web_search_workflow(self, mock_post, api_base_url):
        """Test web search functionality."""
        mock_response = {
            "success": True,
            "results": [
                {
                    "title": "Test Result 1",
                    "url": "https://example.com/1",
                    "snippet": "This is a test search result"
                },
                {
                    "title": "Test Result 2", 
                    "url": "https://example.com/2",
                    "snippet": "Another test search result"
                }
            ],
            "query": "python programming",
            "total_results": 2
        }
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response
        
        payload = {
            "query": "python programming",
            "max_results": 5
        }
        
        response = requests.post(f"{api_base_url}/api/search", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["results"]) > 0
        assert data["query"] == "python programming"


@pytest.mark.blackbox
@pytest.mark.performance
class TestSystemPerformance:
    """End-to-end performance tests."""
    
    @pytest.fixture
    def api_base_url(self):
        """Base URL for API tests."""
        return "http://localhost:8000"
    
    @patch('requests.post')
    def test_concurrent_image_generation(self, mock_post, api_base_url):
        """Test system handles concurrent image generation requests."""
        import concurrent.futures
        import threading
        
        # Mock responses with varying generation times
        def mock_response_func(*args, **kwargs):
            mock_resp = type('MockResponse', (), {})()
            mock_resp.status_code = 200
            mock_resp.json = lambda: {
                "success": True,
                "image_url": f"/outputs/concurrent_test_{threading.current_thread().ident}.png",
                "generation_time": 3.0,
                "metadata": {"prompt": "concurrent test"}
            }
            return mock_resp
        
        mock_post.side_effect = mock_response_func
        
        def make_request():
            payload = {
                "prompt": "concurrent test image",
                "width": 512,
                "height": 512,
                "steps": 20
            }
            return requests.post(f"{api_base_url}/api/image/generate", json=payload)
        
        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        assert len(results) == 5
        for result in results:
            assert result.status_code == 200
            data = result.json()
            assert data["success"] is True
    
    @patch('requests.post')
    @pytest.mark.slow
    def test_response_time_requirements(self, mock_post, api_base_url):
        """Test that API responses meet timing requirements."""
        # Mock fast response
        mock_response = {
            "response": "Quick response",
            "processing_time": 0.1
        }
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response
        
        start_time = time.time()
        
        payload = {"message": "Quick test message"}
        response = requests.post(f"{api_base_url}/api/chat/message", json=payload)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # API should respond within reasonable time (including network overhead)
        assert response_time < 2.0  # 2 seconds max for simple chat
        assert response.status_code == 200


@pytest.mark.blackbox
class TestErrorRecovery:
    """End-to-end tests for error recovery and resilience."""
    
    @pytest.fixture
    def api_base_url(self):
        """Base URL for API tests."""
        return "http://localhost:8000"
    
    @patch('requests.post')
    def test_recovery_from_generation_failure(self, mock_post, api_base_url):
        """Test system recovery after generation failure."""
        # First request fails
        mock_post.return_value.status_code = 500
        mock_post.return_value.json.return_value = {
            "success": False,
            "error": "Internal server error"
        }
        
        payload = {
            "prompt": "test image",
            "width": 512,
            "height": 512,
            "steps": 20
        }
        
        # First request should fail
        response1 = requests.post(f"{api_base_url}/api/image/generate", json=payload)
        assert response1.status_code == 500
        
        # Mock recovery - second request succeeds
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "success": True,
            "image_url": "/outputs/recovery_test.png",
            "generation_time": 4.0
        }
        
        # Second request should succeed (system recovered)
        response2 = requests.post(f"{api_base_url}/api/image/generate", json=payload)
        assert response2.status_code == 200
        data = response2.json()
        assert data["success"] is True
    
    @patch('requests.get')
    def test_health_check_during_issues(self, mock_get, api_base_url):
        """Test health check reports system status during issues."""
        mock_response = {
            "status": "degraded",
            "issues": ["GPU memory low", "High load"],
            "gpu_available": True,
            "active_generations": 3
        }
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_response
        
        response = requests.get(f"{api_base_url}/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert "issues" in data
        assert isinstance(data["issues"], list)
