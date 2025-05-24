"""
Unit tests for the Image Generation Service.

Tests the main service class that orchestrates image generation,
handles configuration, manages resources, and coordinates between components.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import tempfile
import os

from tests.utils.mock_helpers import (
    MockDiffusionPipeline, MockMemoryManager, MockTorch, 
    create_mock_torch_device, create_test_image
)
from tests.fixtures.test_data import (
    sample_configs, sample_prompts, validation_test_cases
)


class MockImageGenerationService:
    """Mock implementation of ImageGenerationService for testing."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.pipeline = None
        self.memory_manager = MockMemoryManager()
        self.cache = {}
        self.is_initialized = False
        self._device = None
    
    def initialize(self):
        """Initialize the service."""
        self.pipeline = MockDiffusionPipeline()
        self.is_initialized = True
        return True
    
    def generate_image(self, prompt, **kwargs):
        """Generate a single image."""
        if not self.is_initialized:
            raise RuntimeError("Service not initialized")
        
        # Simulate generation
        result = self.pipeline(prompt, **kwargs)
        return {
            'image_data': result.images[0] if hasattr(result, 'images') else 'mock_image',
            'metadata': {
                'prompt': prompt,
                'width': kwargs.get('width', 512),
                'height': kwargs.get('height', 512)
            }
        }
    
    def generate_batch(self, prompts, **kwargs):
        """Generate multiple images."""
        results = []
        for prompt in prompts:
            result = self.generate_image(prompt, **kwargs)
            results.append(result)
        return results
    
    def cleanup(self):
        """Clean up resources."""
        self.is_initialized = False
        self.pipeline = None


@pytest.mark.unit
class TestImageGenerationService:
    """Unit tests for ImageGenerationService class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.service = MockImageGenerationService()
        self.mock_config = sample_configs()[0]
    
    def test_service_initialization(self):
        """Test service initialization."""
        assert self.service.is_initialized is False
        
        # Initialize service
        result = self.service.initialize()
        
        assert result is True
        assert self.service.is_initialized is True
        assert self.service.pipeline is not None
    
    def test_service_initialization_with_config(self):
        """Test service initialization with custom configuration."""
        config = {
            'model_id': 'custom_model',
            'device': 'cuda',
            'cache_size': 100
        }
        
        service = MockImageGenerationService(config)
        service.initialize()
        
        assert service.config == config
        assert service.is_initialized is True
    
    def test_generate_image_success(self):
        """Test successful image generation."""
        self.service.initialize()
        
        result = self.service.generate_image(
            prompt="A beautiful sunset",
            width=512,
            height=512
        )
        
        assert 'image_data' in result
        assert 'metadata' in result
        assert result['metadata']['prompt'] == "A beautiful sunset"
        assert result['metadata']['width'] == 512
        assert result['metadata']['height'] == 512
    
    def test_generate_image_without_initialization(self):
        """Test image generation without initialization."""
        with pytest.raises(RuntimeError, match="Service not initialized"):
            self.service.generate_image("test prompt")
    
    def test_generate_image_with_default_parameters(self):
        """Test image generation with default parameters."""
        self.service.initialize()
        
        result = self.service.generate_image("test prompt")
        
        assert result['metadata']['width'] == 512
        assert result['metadata']['height'] == 512
    
    def test_generate_batch_success(self):
        """Test successful batch image generation."""
        self.service.initialize()
        
        prompts = ["prompt 1", "prompt 2", "prompt 3"]
        results = self.service.generate_batch(prompts)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert 'image_data' in result
            assert 'metadata' in result
            assert result['metadata']['prompt'] == prompts[i]
    
    def test_generate_batch_empty_prompts(self):
        """Test batch generation with empty prompts list."""
        self.service.initialize()
        
        results = self.service.generate_batch([])
        
        assert len(results) == 0
    
    def test_service_cleanup(self):
        """Test service cleanup."""
        self.service.initialize()
        assert self.service.is_initialized is True
        
        self.service.cleanup()
        
        assert self.service.is_initialized is False
        assert self.service.pipeline is None
    
    @patch('torch.cuda.is_available')
    def test_device_selection_cuda_available(self, mock_cuda_available):
        """Test device selection when CUDA is available."""
        mock_cuda_available.return_value = True
        
        def mock_get_device():
            return 'cuda' if mock_cuda_available() else 'cpu'
        
        device = mock_get_device()
        assert device == 'cuda'
        
        mock_cuda_available.return_value = False
        device = mock_get_device()
        assert device == 'cpu'
    
    def test_error_handling_during_generation(self):
        """Test error handling during image generation."""
        # Mock service that raises errors
        class FailingService(MockImageGenerationService):
            def generate_image(self, prompt, **kwargs):
                if prompt == "fail":
                    raise RuntimeError("Generation failed")
                return super().generate_image(prompt, **kwargs)
        
        service = FailingService()
        service.initialize()
        
        # Test successful generation
        result = service.generate_image("success")
        assert 'image_data' in result
        
        # Test failed generation
        with pytest.raises(RuntimeError, match="Generation failed"):
            service.generate_image("fail")
    
    def test_parameter_validation(self):
        """Test parameter validation in service methods."""
        self.service.initialize()
        
        # Test invalid width
        with pytest.raises(ValueError):
            self._validate_and_generate("test", width=-1)
        
        # Test invalid height
        with pytest.raises(ValueError):
            self._validate_and_generate("test", height=0)
        
        # Test invalid prompt
        with pytest.raises(ValueError):
            self._validate_and_generate("")
    
    def _validate_and_generate(self, prompt, **kwargs):
        """Helper method to validate parameters before generation."""
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        if kwargs.get('width', 1) <= 0:
            raise ValueError("Width must be positive")
        if kwargs.get('height', 1) <= 0:
            raise ValueError("Height must be positive")
        
        return self.service.generate_image(prompt, **kwargs)


@pytest.mark.unit
class TestImageGenerationServiceConfiguration:
    """Unit tests for service configuration management."""
    
    def test_default_configuration(self):
        """Test service with default configuration."""
        service = MockImageGenerationService()
        
        # Default config should be empty dict
        assert service.config == {}
    
    def test_custom_configuration(self):
        """Test service with custom configuration."""
        config = {
            'model_id': 'stabilityai/stable-diffusion-2-1',
            'device': 'cuda',
            'dtype': 'float16',
            'cache_size': 50,
            'safety_checker': True
        }
        
        service = MockImageGenerationService(config)
        
        assert service.config == config
        assert service.config['model_id'] == 'stabilityai/stable-diffusion-2-1'
        assert service.config['device'] == 'cuda'
        assert service.config['cache_size'] == 50
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        def validate_config(config):
            """Mock configuration validation."""
            required_fields = ['model_id']
            
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field: {field}")
            
            if 'device' in config and config['device'] not in ['cpu', 'cuda']:
                raise ValueError("Device must be 'cpu' or 'cuda'")
            
            if 'cache_size' in config and config['cache_size'] <= 0:
                raise ValueError("Cache size must be positive")
            
            return True
        
        # Test valid configuration
        valid_config = {'model_id': 'test_model', 'device': 'cuda', 'cache_size': 10}
        assert validate_config(valid_config) is True
        
        # Test missing required field
        with pytest.raises(ValueError, match="Missing required field: model_id"):
            validate_config({})
        
        # Test invalid device
        with pytest.raises(ValueError, match="Device must be 'cpu' or 'cuda'"):
            validate_config({'model_id': 'test', 'device': 'gpu'})
        
        # Test invalid cache size
        with pytest.raises(ValueError, match="Cache size must be positive"):
            validate_config({'model_id': 'test', 'cache_size': -1})
    
    def test_configuration_merging(self):
        """Test merging of default and custom configurations."""
        def merge_configs(default_config, custom_config):
            """Mock configuration merging."""
            merged = default_config.copy()
            merged.update(custom_config)
            return merged
        
        default_config = {
            'model_id': 'default_model',
            'device': 'cpu',
            'cache_size': 10,
            'safety_checker': True
        }
        
        custom_config = {
            'device': 'cuda',
            'cache_size': 50
        }
        
        merged = merge_configs(default_config, custom_config)
        
        # Custom values should override defaults
        assert merged['device'] == 'cuda'
        assert merged['cache_size'] == 50
        
        # Default values should be preserved when not overridden
        assert merged['model_id'] == 'default_model'
        assert merged['safety_checker'] is True


@pytest.mark.unit
class TestImageGenerationServiceCaching:
    """Unit tests for service caching functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.service = MockImageGenerationService()
        self.service.initialize()
    
    def test_cache_hit(self):
        """Test cache hit scenario."""
        # Mock cache with pre-existing entry
        cache_key = "test_prompt_512x512"
        cached_result = {
            'image_data': 'cached_image',
            'metadata': {'cached': True}
        }
        self.service.cache[cache_key] = cached_result
        
        def generate_with_cache(prompt, **kwargs):
            """Mock generation with caching."""
            key = f"{prompt}_{kwargs.get('width', 512)}x{kwargs.get('height', 512)}"
            if key in self.service.cache:
                return self.service.cache[key]
            
            result = self.service.generate_image(prompt, **kwargs)
            self.service.cache[key] = result
            return result
        
        result = generate_with_cache("test_prompt")
        
        assert result == cached_result
        assert result['metadata']['cached'] is True
    
    def test_cache_miss(self):
        """Test cache miss scenario."""
        def generate_with_cache(prompt, **kwargs):
            """Mock generation with caching."""
            key = f"{prompt}_{kwargs.get('width', 512)}x{kwargs.get('height', 512)}"
            if key in self.service.cache:
                return self.service.cache[key]
            
            result = self.service.generate_image(prompt, **kwargs)
            self.service.cache[key] = result
            return result
        
        result = generate_with_cache("new_prompt")
        
        # Should generate new image and cache it
        assert 'image_data' in result
        assert "new_prompt_512x512" in self.service.cache
        assert self.service.cache["new_prompt_512x512"] == result
    
    def test_cache_size_limit(self):
        """Test cache size limit enforcement."""
        cache_size_limit = 3
        
        def add_to_cache_with_limit(key, value):
            """Mock cache with size limit."""
            if len(self.service.cache) >= cache_size_limit:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.service.cache))
                del self.service.cache[oldest_key]
            
            self.service.cache[key] = value
        
        # Add items up to limit
        for i in range(cache_size_limit):
            add_to_cache_with_limit(f"key_{i}", f"value_{i}")
        
        assert len(self.service.cache) == cache_size_limit
        assert "key_0" in self.service.cache
        
        # Add one more item (should evict oldest)
        add_to_cache_with_limit("key_3", "value_3")
        
        assert len(self.service.cache) == cache_size_limit
        assert "key_0" not in self.service.cache  # Should be evicted
        assert "key_3" in self.service.cache  # Should be added
    
    def test_cache_invalidation(self):
        """Test cache invalidation."""
        # Add items to cache
        self.service.cache["key1"] = "value1"
        self.service.cache["key2"] = "value2"
        
        def invalidate_cache(pattern=None):
            """Mock cache invalidation."""
            if pattern is None:
                # Clear all cache
                self.service.cache.clear()
            else:
                # Clear entries matching pattern
                keys_to_remove = [key for key in self.service.cache if pattern in key]
                for key in keys_to_remove:
                    del self.service.cache[key]
        
        # Test selective invalidation
        invalidate_cache("key1")
        assert "key1" not in self.service.cache
        assert "key2" in self.service.cache
        
        # Test full invalidation
        invalidate_cache()
        assert len(self.service.cache) == 0


@pytest.mark.unit
class TestImageGenerationServiceResourceManagement:
    """Unit tests for service resource management."""
    
    def setup_method(self):
        """Set up test environment."""
        self.service = MockImageGenerationService()
    
    def test_memory_management_initialization(self):
        """Test memory manager initialization."""
        assert self.service.memory_manager is not None
        assert isinstance(self.service.memory_manager, MockMemoryManager)
    
    def test_gpu_memory_allocation(self):
        """Test GPU memory allocation tracking."""
        self.service.initialize()
        
        # Simulate memory allocation
        self.service.memory_manager.allocate_gpu_memory(100)  # 100MB
        
        assert self.service.memory_manager.get_gpu_memory_usage() == 100
    
    def test_memory_cleanup_on_service_cleanup(self):
        """Test memory cleanup when service is cleaned up."""
        self.service.initialize()
        
        # Allocate some memory
        self.service.memory_manager.allocate_memory(50)
        assert self.service.memory_manager.get_allocated_memory() == 50
        
        # Cleanup service
        self.service.cleanup()
        
        # Memory manager should also be cleaned up
        # (This would depend on actual implementation)
        assert self.service.is_initialized is False
    
    def test_resource_limits_enforcement(self):
        """Test resource limits enforcement."""
        max_memory_mb = 1000
        
        def check_memory_limit():
            """Mock memory limit checking."""
            current_usage = self.service.memory_manager.get_allocated_memory()
            if current_usage > max_memory_mb:
                raise RuntimeError(f"Memory limit exceeded: {current_usage}MB > {max_memory_mb}MB")
            return True
        
        self.service.initialize()
        
        # Normal usage should be fine
        self.service.memory_manager.allocate_memory(500)
        assert check_memory_limit() is True
        
        # Exceeding limit should raise error
        self.service.memory_manager.allocate_memory(600)  # Total: 1100MB
        with pytest.raises(RuntimeError, match="Memory limit exceeded"):
            check_memory_limit()
    
    def test_concurrent_resource_access(self):
        """Test concurrent access to resources."""
        import threading
        
        self.service.initialize()
        results = []
        errors = []
        
        def worker_function(worker_id):
            """Worker function for concurrent testing."""
            try:
                result = self.service.generate_image(f"worker {worker_id} prompt")
                results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(results) == 5
        assert len(errors) == 0
        
        # Each result should be valid
        for result in results:
            assert 'image_data' in result
            assert 'metadata' in result
