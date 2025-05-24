"""
Mock helpers for testing.

This module provides common mock objects and helper functions
for creating test doubles across different test categories.
"""

from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List, Optional
import torch
from pathlib import Path


class MockDiffusionPipeline:
    """Mock diffusion pipeline for testing."""
    
    def __init__(self, *args, **kwargs):
        self.device = "cpu"
        self.components = {}
        self._call_count = 0
    
    def to(self, device: str):
        """Mock device movement."""
        self.device = device
        return self
    
    def enable_sequential_cpu_offload(self):
        """Mock CPU offload."""
        pass
    
    def enable_attention_slicing(self, slice_size="auto"):
        """Mock attention slicing."""
        pass
    
    def enable_vae_slicing(self):
        """Mock VAE slicing."""
        pass
    
    def enable_vae_tiling(self):
        """Mock VAE tiling."""
        pass
    
    def __call__(self, prompt: str, **kwargs):
        """Mock pipeline call."""
        self._call_count += 1
        
        # Create mock image
        mock_image = Mock()
        mock_image.save = Mock()
        mock_image.size = (kwargs.get('width', 512), kwargs.get('height', 512))
        
        # Create mock result
        result = Mock()
        result.images = [mock_image]
        return result
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Mock from_pretrained class method."""
        return cls()


class MockTorch:
    """Mock torch module for testing without CUDA dependency."""
    
    class cuda:
        @staticmethod
        def is_available():
            return False
        
        @staticmethod
        def empty_cache():
            pass
        
        @staticmethod
        def synchronize():
            pass
        
        @staticmethod
        def mem_get_info():
            return (2000000000, 4000000000)  # 2GB free, 4GB total
        
        @staticmethod
        def memory_allocated():
            return 500000000  # 500MB allocated
        
        @staticmethod
        def memory_reserved():
            return 1000000000  # 1GB reserved
        
        @staticmethod
        def get_device_name(device_id: int = 0):
            return "Mock GPU Device"
        
        class OutOfMemoryError(Exception):
            pass


def create_mock_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a mock configuration dictionary."""
    base_config = {
        "model_path": "test/model/path",
        "output_dir": "/tmp/test_outputs",
        "default_steps": 20,
        "default_width": 512,
        "default_height": 512,
        "memory": {
            "gpu_offload_timeout": 300,
            "enable_attention_slicing": True,
            "enable_memory_efficient_attention": True,
            "enable_cpu_offload": True,
            "auto_gc_after_generation": True
        },
        "api": {
            "host": "127.0.0.1",
            "port": 8000,
            "debug": True
        }
    }
    
    if overrides:
        base_config.update(overrides)
    
    return base_config


def create_mock_generation_result(
    success: bool = True,
    image_path: Optional[str] = None,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """Create a mock generation result."""
    if success:
        return {
            "success": True,
            "image_path": image_path or "/tmp/test_outputs/mock_image.png",
            "generation_time": 3.5,
            "metadata": {
                "prompt": "test prompt",
                "width": 512,
                "height": 512,
                "num_inference_steps": 20,
                "timestamp": "2025-05-24T11:23:45Z",
                "model_info": {
                    "name": "FLUX.1-dev",
                    "version": "test"
                }
            }
        }
    else:
        return {
            "success": False,
            "error": error or "Mock generation error",
            "error_type": "MockError"
        }


class MockLogger:
    """Mock logger for testing."""
    
    def __init__(self):
        self.debug_calls = []
        self.info_calls = []
        self.warning_calls = []
        self.error_calls = []
    
    def debug(self, message: str):
        self.debug_calls.append(message)
    
    def info(self, message: str):
        self.info_calls.append(message)
    
    def warning(self, message: str):
        self.warning_calls.append(message)
    
    def error(self, message: str):
        self.error_calls.append(message)


def patch_torch_cuda(available: bool = False):
    """Context manager to patch torch.cuda for testing."""
    patches = [
        patch('torch.cuda.is_available', return_value=available),
        patch('torch.cuda.empty_cache'),
        patch('torch.cuda.synchronize'),
    ]
    
    if available:
        patches.extend([
            patch('torch.cuda.mem_get_info', return_value=(2000000000, 4000000000)),
            patch('torch.cuda.memory_allocated', return_value=500000000),
            patch('torch.cuda.memory_reserved', return_value=1000000000),
            patch('torch.cuda.get_device_name', return_value="Mock GPU"),
        ])
    
    class CudaPatcher:
        def __enter__(self):
            self.active_patches = [p.__enter__() for p in patches]
            return self
        
        def __exit__(self, *args):
            for p in reversed(patches):
                p.__exit__(*args)
    
    return CudaPatcher()


def create_test_image_file(output_path: Path, width: int = 512, height: int = 512):
    """Create a test image file for testing."""
    try:
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        image_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        image.save(output_path)
        return True
    except ImportError:
        # If PIL not available, create a dummy file
        output_path.write_text("mock image content")
        return False


class MockMemoryManager:
    """Mock memory manager for testing."""
    
    def __init__(self):
        self.pipeline_on_gpu = False
        self.last_access_time = 0
        self.config = {
            "gpu_offload_timeout": 300,
            "enable_attention_slicing": True,
            "enable_cpu_offload": True,
            "auto_gc_after_generation": True
        }
    
    def clear_gpu_memory(self):
        pass
    
    def get_gpu_memory_info(self):
        return {
            "available": True,
            "free_gb": 2.0,
            "total_gb": 4.0,
            "allocated_gb": 0.5,
            "reserved_gb": 1.0,
            "device_name": "Mock GPU"
        }
    
    def update_access_time(self):
        import time
        self.last_access_time = time.time()
    
    def move_pipeline_to_gpu(self, pipeline):
        self.pipeline_on_gpu = True
        return pipeline
    
    def move_pipeline_to_cpu(self, pipeline):
        self.pipeline_on_gpu = False
        return pipeline
    
    def schedule_offload(self, pipeline, delay=None):
        pass
    
    def apply_memory_optimizations(self, pipeline):
        return pipeline
    
    def get_status(self):
        return {
            "pipeline_on_gpu": self.pipeline_on_gpu,
            "last_access_time": self.last_access_time,
            "memory_config": self.config,
            "available": True,
            "free_gb": 2.0,
            "total_gb": 4.0
        }


def assert_valid_image_result(result: Dict[str, Any]):
    """Assert that a result is a valid image generation result."""
    assert isinstance(result, dict)
    assert "success" in result
    
    if result["success"]:
        assert "image_path" in result
        assert "generation_time" in result
        assert "metadata" in result
        assert isinstance(result["generation_time"], (int, float))
        assert result["generation_time"] > 0
        
        metadata = result["metadata"]
        assert "prompt" in metadata
        assert "width" in metadata
        assert "height" in metadata
        assert "num_inference_steps" in metadata
    else:
        assert "error" in result
        assert isinstance(result["error"], str)


def assert_valid_config(config: Dict[str, Any]):
    """Assert that a config dictionary is valid."""
    assert isinstance(config, dict)
    required_keys = ["model_path", "default_steps", "default_width", "default_height"]
    
    for key in required_keys:
        assert key in config, f"Missing required config key: {key}"
    
    assert isinstance(config["default_steps"], int)
    assert config["default_steps"] > 0
    assert isinstance(config["default_width"], int)
    assert config["default_width"] > 0
    assert isinstance(config["default_height"], int)
    assert config["default_height"] > 0


class TestDataGenerator:
    """Generate test data for various scenarios."""
    
    @staticmethod
    def valid_prompts() -> List[str]:
        """Generate list of valid test prompts."""
        return [
            "A beautiful landscape with mountains and a lake",
            "A futuristic city skyline at sunset",
            "A close-up portrait of a smiling person",
            "An abstract painting with vibrant colors",
            "A peaceful forest scene with sunlight filtering through trees"
        ]
    
    @staticmethod
    def invalid_prompts() -> List[str]:
        """Generate list of invalid test prompts."""
        return [
            "",  # Empty
            " ",  # Whitespace only
            "a" * 1000,  # Too long
        ]
    
    @staticmethod
    def valid_dimensions() -> List[tuple]:
        """Generate list of valid dimension pairs."""
        return [
            (512, 512),
            (768, 512),
            (512, 768),
            (1024, 768),
            (768, 1024),
            (256, 256),
            (1024, 1024)
        ]
    
    @staticmethod
    def invalid_dimensions() -> List[tuple]:
        """Generate list of invalid dimension pairs."""
        return [
            (0, 512),      # Zero width
            (512, 0),      # Zero height
            (-100, 512),   # Negative width
            (512, -100),   # Negative height
            (10000, 512),  # Too large width
            (512, 10000),  # Too large height
            (33, 512),     # Not divisible by 8
            (512, 33),     # Not divisible by 8
        ]
    
    @staticmethod
    def valid_steps() -> List[int]:
        """Generate list of valid step counts."""
        return [1, 5, 10, 15, 20, 25, 30, 40, 50]
    
    @staticmethod
    def invalid_steps() -> List[int]:
        """Generate list of invalid step counts."""
        return [0, -1, -10, 101, 1000]
