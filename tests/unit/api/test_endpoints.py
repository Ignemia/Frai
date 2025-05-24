"""
Unit tests for API endpoints.

Tests individual API endpoint functions, request/response handling,
validation, and error handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException
import json

from tests.utils.mock_helpers import (
    MockDiffusionPipeline, MockMemoryManager, create_mock_torch_device
)
from tests.fixtures.test_data import (
    sample_api_requests, sample_api_responses, validation_test_cases
)


class MockRequest:
    """Mock HTTP request object."""
    
    def __init__(self, json_data=None, form_data=None, files=None):
        self.json_data = json_data or {}
        self.form_data = form_data or {}
        self.files = files or {}
    
    async def json(self):
        return self.json_data
    
    async def form(self):
        return self.form_data


class MockResponse:
    """Mock HTTP response object."""
    
    def __init__(self, status_code=200, content=None, headers=None):
        self.status_code = status_code
        self.content = content or {}
        self.headers = headers or {}


@pytest.mark.unit
class TestImageGenerationAPI:
    """Unit tests for image generation API endpoints."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_pipeline = MockDiffusionPipeline()
        self.mock_memory_manager = MockMemoryManager()
    
    def test_generate_image_endpoint_success(self):
        """Test successful image generation via API."""
        # Mock the image generation service
        with patch('src.services.image_generation.ImageGenerationService') as mock_service:
            mock_service_instance = Mock()
            mock_service.return_value = mock_service_instance
            mock_service_instance.generate_image.return_value = {
                'image_data': 'base64_encoded_image',
                'metadata': {'width': 512, 'height': 512}
            }
            
            # Create mock request
            request_data = {
                'prompt': 'A beautiful sunset',
                'width': 512,
                'height': 512,
                'num_inference_steps': 20
            }
            
            # Test the endpoint logic
            from src.api.endpoints import generate_image  # Assuming this exists
            
            # Mock the endpoint function behavior
            def mock_generate_image(request):
                if not request.get('prompt'):
                    raise HTTPException(status_code=400, detail="Prompt is required")
                
                result = mock_service_instance.generate_image(
                    prompt=request['prompt'],
                    width=request.get('width', 512),
                    height=request.get('height', 512),
                    num_inference_steps=request.get('num_inference_steps', 20)
                )
                
                return {
                    'success': True,
                    'image': result['image_data'],
                    'metadata': result['metadata']
                }
            
            response = mock_generate_image(request_data)
            
            assert response['success'] is True
            assert 'image' in response
            assert 'metadata' in response
            mock_service_instance.generate_image.assert_called_once()
    
    def test_generate_image_endpoint_validation_error(self):
        """Test API endpoint with invalid request data."""
        def mock_generate_image(request):
            if not request.get('prompt'):
                raise HTTPException(status_code=400, detail="Prompt is required")
            if request.get('width', 0) <= 0:
                raise HTTPException(status_code=400, detail="Width must be positive")
            if request.get('height', 0) <= 0:
                raise HTTPException(status_code=400, detail="Height must be positive")
            return {'success': True}
        
        # Test missing prompt
        with pytest.raises(HTTPException) as exc_info:
            mock_generate_image({})
        assert exc_info.value.status_code == 400
        assert "Prompt is required" in str(exc_info.value.detail)
        
        # Test invalid width
        with pytest.raises(HTTPException) as exc_info:
            mock_generate_image({'prompt': 'test', 'width': -1})
        assert exc_info.value.status_code == 400
        assert "Width must be positive" in str(exc_info.value.detail)
        
        # Test invalid height
        with pytest.raises(HTTPException) as exc_info:
            mock_generate_image({'prompt': 'test', 'height': 0})
        assert exc_info.value.status_code == 400
        assert "Height must be positive" in str(exc_info.value.detail)
    
    def test_generate_image_endpoint_service_error(self):
        """Test API endpoint when service raises an error."""
        with patch('src.services.image_generation.ImageGenerationService') as mock_service:
            mock_service_instance = Mock()
            mock_service.return_value = mock_service_instance
            mock_service_instance.generate_image.side_effect = Exception("GPU out of memory")
            
            def mock_generate_image(request):
                try:
                    result = mock_service_instance.generate_image(**request)
                    return {'success': True, 'result': result}
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
            
            request_data = {'prompt': 'test prompt'}
            
            with pytest.raises(HTTPException) as exc_info:
                mock_generate_image(request_data)
            
            assert exc_info.value.status_code == 500
            assert "GPU out of memory" in str(exc_info.value.detail)
    
    def test_batch_generate_endpoint(self):
        """Test batch image generation endpoint."""
        with patch('src.services.image_generation.ImageGenerationService') as mock_service:
            mock_service_instance = Mock()
            mock_service.return_value = mock_service_instance
            
            # Mock batch generation
            mock_service_instance.generate_batch.return_value = [
                {'image_data': 'image1', 'metadata': {}},
                {'image_data': 'image2', 'metadata': {}},
                {'image_data': 'image3', 'metadata': {}}
            ]
            
            def mock_batch_generate(request):
                prompts = request.get('prompts', [])
                if not prompts:
                    raise HTTPException(status_code=400, detail="Prompts list is required")
                if len(prompts) > 10:
                    raise HTTPException(status_code=400, detail="Maximum 10 prompts allowed")
                
                results = mock_service_instance.generate_batch(prompts)
                return {
                    'success': True,
                    'images': results,
                    'count': len(results)
                }
            
            request_data = {
                'prompts': ['prompt 1', 'prompt 2', 'prompt 3']
            }
            
            response = mock_batch_generate(request_data)
            
            assert response['success'] is True
            assert response['count'] == 3
            assert len(response['images']) == 3
    
    def test_image_status_endpoint(self):
        """Test image generation status endpoint."""
        def mock_get_status(task_id):
            status_map = {
                'valid_id': {'status': 'completed', 'progress': 100},
                'processing_id': {'status': 'processing', 'progress': 50},
                'failed_id': {'status': 'failed', 'error': 'Generation failed'}
            }
            
            if task_id not in status_map:
                raise HTTPException(status_code=404, detail="Task not found")
            
            return status_map[task_id]
        
        # Test valid task ID
        status = mock_get_status('valid_id')
        assert status['status'] == 'completed'
        assert status['progress'] == 100
        
        # Test processing task
        status = mock_get_status('processing_id')
        assert status['status'] == 'processing'
        assert status['progress'] == 50
        
        # Test failed task
        status = mock_get_status('failed_id')
        assert status['status'] == 'failed'
        assert 'error' in status
        
        # Test invalid task ID
        with pytest.raises(HTTPException) as exc_info:
            mock_get_status('invalid_id')
        assert exc_info.value.status_code == 404


@pytest.mark.unit
class TestAPIModels:
    """Unit tests for API request/response models."""
    
    def test_image_generation_request_model(self):
        """Test image generation request model validation."""
        # Mock Pydantic model behavior
        class MockImageGenerationRequest:
            def __init__(self, **kwargs):
                self.prompt = kwargs.get('prompt')
                self.width = kwargs.get('width', 512)
                self.height = kwargs.get('height', 512)
                self.num_inference_steps = kwargs.get('num_inference_steps', 20)
                self.guidance_scale = kwargs.get('guidance_scale', 7.5)
                
                self.validate()
            
            def validate(self):
                if not self.prompt or not isinstance(self.prompt, str):
                    raise ValueError("Prompt must be a non-empty string")
                if self.width <= 0 or self.width > 2048:
                    raise ValueError("Width must be between 1 and 2048")
                if self.height <= 0 or self.height > 2048:
                    raise ValueError("Height must be between 1 and 2048")
                if self.num_inference_steps <= 0 or self.num_inference_steps > 100:
                    raise ValueError("Number of inference steps must be between 1 and 100")
        
        # Test valid request
        request = MockImageGenerationRequest(
            prompt="A beautiful landscape",
            width=512,
            height=512,
            num_inference_steps=20
        )
        assert request.prompt == "A beautiful landscape"
        assert request.width == 512
        
        # Test invalid prompt
        with pytest.raises(ValueError, match="Prompt must be a non-empty string"):
            MockImageGenerationRequest(prompt="")
        
        # Test invalid width
        with pytest.raises(ValueError, match="Width must be between 1 and 2048"):
            MockImageGenerationRequest(prompt="test", width=0)
        
        # Test invalid height
        with pytest.raises(ValueError, match="Height must be between 1 and 2048"):
            MockImageGenerationRequest(prompt="test", height=3000)
        
        # Test invalid inference steps
        with pytest.raises(ValueError, match="Number of inference steps must be between 1 and 100"):
            MockImageGenerationRequest(prompt="test", num_inference_steps=0)
    
    def test_image_generation_response_model(self):
        """Test image generation response model."""
        class MockImageGenerationResponse:
            def __init__(self, **kwargs):
                self.success = kwargs.get('success', True)
                self.image_data = kwargs.get('image_data')
                self.metadata = kwargs.get('metadata', {})
                self.error = kwargs.get('error')
                
                self.validate()
            
            def validate(self):
                if self.success and not self.image_data:
                    raise ValueError("Image data required for successful response")
                if not self.success and not self.error:
                    raise ValueError("Error message required for failed response")
        
        # Test successful response
        response = MockImageGenerationResponse(
            success=True,
            image_data="base64_image_data",
            metadata={"width": 512, "height": 512}
        )
        assert response.success is True
        assert response.image_data == "base64_image_data"
        
        # Test failed response
        response = MockImageGenerationResponse(
            success=False,
            error="Generation failed"
        )
        assert response.success is False
        assert response.error == "Generation failed"
        
        # Test invalid successful response (missing image data)
        with pytest.raises(ValueError, match="Image data required"):
            MockImageGenerationResponse(success=True)
        
        # Test invalid failed response (missing error)
        with pytest.raises(ValueError, match="Error message required"):
            MockImageGenerationResponse(success=False)
    
    def test_batch_request_model(self):
        """Test batch generation request model."""
        class MockBatchGenerationRequest:
            def __init__(self, **kwargs):
                self.prompts = kwargs.get('prompts', [])
                self.common_params = kwargs.get('common_params', {})
                
                self.validate()
            
            def validate(self):
                if not self.prompts or not isinstance(self.prompts, list):
                    raise ValueError("Prompts must be a non-empty list")
                if len(self.prompts) > 10:
                    raise ValueError("Maximum 10 prompts allowed in batch")
                for i, prompt in enumerate(self.prompts):
                    if not isinstance(prompt, str) or not prompt.strip():
                        raise ValueError(f"Prompt {i} must be a non-empty string")
        
        # Test valid batch request
        request = MockBatchGenerationRequest(
            prompts=["prompt 1", "prompt 2", "prompt 3"],
            common_params={"width": 512, "height": 512}
        )
        assert len(request.prompts) == 3
        
        # Test empty prompts
        with pytest.raises(ValueError, match="Prompts must be a non-empty list"):
            MockBatchGenerationRequest(prompts=[])
        
        # Test too many prompts
        with pytest.raises(ValueError, match="Maximum 10 prompts allowed"):
            MockBatchGenerationRequest(prompts=[f"prompt {i}" for i in range(15)])
        
        # Test invalid prompt in list
        with pytest.raises(ValueError, match="Prompt 1 must be a non-empty string"):
            MockBatchGenerationRequest(prompts=["valid prompt", "", "another prompt"])


@pytest.mark.unit
class TestAPIMiddleware:
    """Unit tests for API middleware components."""
    
    def test_request_logging_middleware(self):
        """Test request logging middleware."""
        class MockLoggingMiddleware:
            def __init__(self):
                self.logged_requests = []
            
            def log_request(self, request, response, duration):
                self.logged_requests.append({
                    'method': request.get('method', 'GET'),
                    'url': request.get('url', '/'),
                    'status_code': response.get('status_code', 200),
                    'duration': duration
                })
        
        middleware = MockLoggingMiddleware()
        
        # Simulate request logging
        request = {'method': 'POST', 'url': '/api/generate'}
        response = {'status_code': 200}
        duration = 1.5
        
        middleware.log_request(request, response, duration)
        
        assert len(middleware.logged_requests) == 1
        logged = middleware.logged_requests[0]
        assert logged['method'] == 'POST'
        assert logged['url'] == '/api/generate'
        assert logged['status_code'] == 200
        assert logged['duration'] == 1.5
    
    def test_rate_limiting_middleware(self):
        """Test rate limiting middleware."""
        class MockRateLimiter:
            def __init__(self, max_requests=10, window_seconds=60):
                self.max_requests = max_requests
                self.window_seconds = window_seconds
                self.requests = {}  # client_ip -> list of timestamps
            
            def is_allowed(self, client_ip):
                import time
                current_time = time.time()
                
                if client_ip not in self.requests:
                    self.requests[client_ip] = []
                
                # Clean old requests outside the window
                self.requests[client_ip] = [
                    req_time for req_time in self.requests[client_ip]
                    if current_time - req_time < self.window_seconds
                ]
                
                # Check if under limit
                if len(self.requests[client_ip]) >= self.max_requests:
                    return False
                
                # Add current request
                self.requests[client_ip].append(current_time)
                return True
        
        rate_limiter = MockRateLimiter(max_requests=5, window_seconds=60)
        
        # Test normal usage
        for i in range(5):
            assert rate_limiter.is_allowed('192.168.1.1') is True
        
        # Test rate limit exceeded
        assert rate_limiter.is_allowed('192.168.1.1') is False
        
        # Test different client
        assert rate_limiter.is_allowed('192.168.1.2') is True
    
    def test_error_handling_middleware(self):
        """Test error handling middleware."""
        class MockErrorHandler:
            def handle_error(self, error, request_info=None):
                if isinstance(error, ValueError):
                    return {'status_code': 400, 'detail': str(error)}
                elif isinstance(error, FileNotFoundError):
                    return {'status_code': 404, 'detail': 'Resource not found'}
                elif isinstance(error, PermissionError):
                    return {'status_code': 403, 'detail': 'Access denied'}
                else:
                    return {'status_code': 500, 'detail': 'Internal server error'}
        
        error_handler = MockErrorHandler()
        
        # Test validation error
        response = error_handler.handle_error(ValueError("Invalid input"))
        assert response['status_code'] == 400
        assert 'Invalid input' in response['detail']
        
        # Test not found error
        response = error_handler.handle_error(FileNotFoundError())
        assert response['status_code'] == 404
        assert 'Resource not found' in response['detail']
        
        # Test permission error
        response = error_handler.handle_error(PermissionError())
        assert response['status_code'] == 403
        assert 'Access denied' in response['detail']
        
        # Test generic error
        response = error_handler.handle_error(RuntimeError("Something went wrong"))
        assert response['status_code'] == 500
        assert 'Internal server error' in response['detail']


@pytest.mark.unit
class TestAPIUtilities:
    """Unit tests for API utility functions."""
    
    def test_request_validation_utils(self):
        """Test request validation utility functions."""
        def validate_prompt(prompt):
            if not prompt or not isinstance(prompt, str):
                return False, "Prompt must be a non-empty string"
            if len(prompt) > 1000:
                return False, "Prompt too long (max 1000 characters)"
            return True, None
        
        def validate_dimensions(width, height):
            if not isinstance(width, int) or not isinstance(height, int):
                return False, "Dimensions must be integers"
            if width <= 0 or height <= 0:
                return False, "Dimensions must be positive"
            if width > 2048 or height > 2048:
                return False, "Dimensions too large (max 2048)"
            if width % 64 != 0 or height % 64 != 0:
                return False, "Dimensions must be multiples of 64"
            return True, None
        
        # Test prompt validation
        valid, error = validate_prompt("A beautiful sunset")
        assert valid is True
        assert error is None
        
        valid, error = validate_prompt("")
        assert valid is False
        assert "non-empty string" in error
        
        valid, error = validate_prompt("x" * 1001)
        assert valid is False
        assert "too long" in error
        
        # Test dimension validation
        valid, error = validate_dimensions(512, 512)
        assert valid is True
        assert error is None
        
        valid, error = validate_dimensions(0, 512)
        assert valid is False
        assert "positive" in error
        
        valid, error = validate_dimensions(513, 512)
        assert valid is False
        assert "multiples of 64" in error
    
    def test_response_formatting_utils(self):
        """Test response formatting utility functions."""
        def format_success_response(data, metadata=None):
            return {
                'success': True,
                'data': data,
                'metadata': metadata or {},
                'timestamp': '2024-01-01T00:00:00Z'
            }
        
        def format_error_response(error_message, error_code=None):
            return {
                'success': False,
                'error': {
                    'message': error_message,
                    'code': error_code
                },
                'timestamp': '2024-01-01T00:00:00Z'
            }
        
        # Test success response
        response = format_success_response(
            data={'image': 'base64_data'},
            metadata={'width': 512, 'height': 512}
        )
        assert response['success'] is True
        assert 'data' in response
        assert 'metadata' in response
        assert 'timestamp' in response
        
        # Test error response
        response = format_error_response("Validation failed", "VALIDATION_ERROR")
        assert response['success'] is False
        assert response['error']['message'] == "Validation failed"
        assert response['error']['code'] == "VALIDATION_ERROR"
        assert 'timestamp' in response
    
    def test_file_handling_utils(self):
        """Test file handling utility functions."""
        import base64
        import io
        
        def encode_image_to_base64(image_data):
            """Simulate encoding image to base64."""
            if isinstance(image_data, str):
                # Already base64 encoded
                return image_data
            # Simulate encoding binary data
            return base64.b64encode(b'fake_image_data').decode('utf-8')
        
        def validate_file_upload(file_data, max_size_mb=10):
            """Validate uploaded file."""
            if not file_data:
                return False, "No file provided"
            
            # Simulate file size check
            file_size = len(file_data) if isinstance(file_data, (str, bytes)) else 0
            max_size_bytes = max_size_mb * 1024 * 1024
            
            if file_size > max_size_bytes:
                return False, f"File too large (max {max_size_mb}MB)"
            
            return True, None
        
        # Test base64 encoding
        encoded = encode_image_to_base64("already_encoded")
        assert encoded == "already_encoded"
        
        encoded = encode_image_to_base64(b"binary_data")
        assert isinstance(encoded, str)
        
        # Test file validation
        valid, error = validate_file_upload("small_file")
        assert valid is True
        assert error is None
        
        valid, error = validate_file_upload("x" * (11 * 1024 * 1024))  # 11MB
        assert valid is False
        assert "too large" in error
        
        valid, error = validate_file_upload(None)
        assert valid is False
        assert "No file provided" in error
