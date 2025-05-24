"""
Test fixtures for common test data and objects.

This module provides reusable test fixtures and data that can be
shared across different test modules.
"""

import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
import json


class TestFixtures:
    """Collection of test fixtures and data."""
    
    @staticmethod
    def sample_configurations() -> Dict[str, Dict[str, Any]]:
        """Sample configurations for different test scenarios."""
        return {
            "minimal": {
                "model_path": "test/minimal/model",
                "default_steps": 10,
                "default_width": 256,
                "default_height": 256
            },
            "standard": {
                "model_path": "test/standard/model",
                "output_dir": "/tmp/test_outputs",
                "default_steps": 20,
                "default_width": 512,
                "default_height": 512,
                "memory": {
                    "gpu_offload_timeout": 300,
                    "enable_attention_slicing": True,
                    "enable_cpu_offload": True,
                    "auto_gc_after_generation": True
                }
            },
            "high_quality": {
                "model_path": "test/hq/model",
                "output_dir": "/tmp/hq_outputs",
                "default_steps": 50,
                "default_width": 1024,
                "default_height": 1024,
                "memory": {
                    "gpu_offload_timeout": 600,
                    "enable_attention_slicing": False,
                    "enable_cpu_offload": False,
                    "auto_gc_after_generation": True
                }
            },
            "memory_constrained": {
                "model_path": "test/constrained/model",
                "default_steps": 15,
                "default_width": 512,
                "default_height": 512,
                "memory": {
                    "gpu_offload_timeout": 60,
                    "enable_attention_slicing": True,
                    "enable_cpu_offload": True,
                    "auto_gc_after_generation": True
                }
            }
        }
    
    @staticmethod
    def sample_prompts() -> Dict[str, List[str]]:
        """Sample prompts categorized by type."""
        return {
            "simple": [
                "A red apple",
                "Blue sky",
                "Green grass",
                "Yellow flower"
            ],
            "detailed": [
                "A photorealistic portrait of a young woman with curly brown hair, wearing a red dress, standing in a field of sunflowers at golden hour",
                "A majestic mountain landscape with snow-capped peaks reflected in a crystal clear alpine lake, surrounded by pine trees",
                "A futuristic cyberpunk cityscape at night with neon lights, flying cars, and towering skyscrapers"
            ],
            "artistic": [
                "An abstract expressionist painting with bold brushstrokes in primary colors",
                "A watercolor illustration of a peaceful Japanese garden with cherry blossoms",
                "A surreal Salvador Dali-style painting with melting clocks and impossible geometries"
            ],
            "technical": [
                "A detailed technical diagram of a steam engine with labels and cross-sections",
                "An architectural blueprint of a modern house with measurements and annotations",
                "A scientific illustration of DNA double helix structure with molecular details"
            ]
        }
    
    @staticmethod
    def sample_generation_results() -> Dict[str, Dict[str, Any]]:
        """Sample generation results for different scenarios."""
        return {
            "successful": {
                "success": True,
                "image_path": "/test/outputs/success_image.png",
                "generation_time": 4.2,
                "metadata": {
                    "prompt": "A beautiful landscape",
                    "width": 512,
                    "height": 512,
                    "num_inference_steps": 20,
                    "timestamp": "2025-05-24T11:23:45Z",
                    "model_info": {
                        "name": "FLUX.1-dev",
                        "version": "1.0.0"
                    }
                }
            },
            "failed_validation": {
                "success": False,
                "error": "Invalid prompt: cannot be empty",
                "error_type": "ValidationError"
            },
            "failed_memory": {
                "success": False,
                "error": "Insufficient GPU memory for generation",
                "error_type": "MemoryError"
            },
            "failed_model": {
                "success": False,
                "error": "Failed to load model from path",
                "error_type": "ModelLoadError"
            }
        }
    
    @staticmethod
    def sample_api_responses() -> Dict[str, Dict[str, Any]]:
        """Sample API responses for different endpoints."""
        return {
            "image_generation_success": {
                "success": True,
                "image_url": "/api/outputs/generated_image.png",
                "generation_time": 5.8,
                "metadata": {
                    "prompt": "API test prompt",
                    "width": 512,
                    "height": 512,
                    "steps": 20
                }
            },
            "image_generation_error": {
                "success": False,
                "error": "Generation failed due to invalid parameters",
                "error_code": "INVALID_PARAMS"
            },
            "status_operational": {
                "status": "operational",
                "gpu_available": True,
                "memory_usage": {
                    "allocated_gb": 2.1,
                    "free_gb": 5.9,
                    "total_gb": 8.0
                },
                "active_generations": 2,
                "total_generated": 1247
            },
            "status_degraded": {
                "status": "degraded",
                "gpu_available": True,
                "issues": ["High memory usage", "Queue backlog"],
                "memory_usage": {
                    "allocated_gb": 7.2,
                    "free_gb": 0.8,
                    "total_gb": 8.0
                },
                "active_generations": 5,
                "total_generated": 1248
            },
            "chat_response": {
                "response": "I can help you generate an image! Please describe what you'd like to see.",
                "conversation_id": "conv_12345",
                "message_id": "msg_67890",
                "processing_time": 0.45
            }
        }
    
    @staticmethod
    def validation_test_cases() -> List[Dict[str, Any]]:
        """Test cases for parameter validation."""
        return [
            # Valid cases
            {
                "params": {"prompt": "test", "width": 512, "height": 512, "steps": 20},
                "expected_valid": True,
                "description": "Standard valid parameters"
            },
            {
                "params": {"prompt": "test", "width": 1024, "height": 768, "steps": 30},
                "expected_valid": True,
                "description": "High resolution valid parameters"
            },
            # Invalid cases
            {
                "params": {"prompt": "", "width": 512, "height": 512, "steps": 20},
                "expected_valid": False,
                "description": "Empty prompt"
            },
            {
                "params": {"prompt": "test", "width": 0, "height": 512, "steps": 20},
                "expected_valid": False,
                "description": "Zero width"
            },
            {
                "params": {"prompt": "test", "width": 512, "height": 0, "steps": 20},
                "expected_valid": False,
                "description": "Zero height"
            },
            {
                "params": {"prompt": "test", "width": 512, "height": 512, "steps": 0},
                "expected_valid": False,
                "description": "Zero steps"
            },
            {
                "params": {"prompt": "test", "width": -100, "height": 512, "steps": 20},
                "expected_valid": False,
                "description": "Negative width"
            },
            {
                "params": {"prompt": "test", "width": 512, "height": -100, "steps": 20},
                "expected_valid": False,
                "description": "Negative height"
            },
            {
                "params": {"prompt": "test", "width": 513, "height": 512, "steps": 20},
                "expected_valid": False,
                "description": "Width not divisible by 8"
            },
            {
                "params": {"prompt": "test", "width": 10000, "height": 512, "steps": 20},
                "expected_valid": False,
                "description": "Width too large"
            }
        ]


class TempDirectoryManager:
    """Manages temporary directories for tests."""
    
    def __init__(self, prefix: str = "test_"):
        self.prefix = prefix
        self.temp_dirs = []
    
    def create_temp_dir(self) -> Path:
        """Create a new temporary directory."""
        temp_dir = Path(tempfile.mkdtemp(prefix=self.prefix))
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def cleanup_all(self):
        """Clean up all created temporary directories."""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        self.temp_dirs.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_all()


def create_test_config_file(config_data: Dict[str, Any], output_path: Path) -> Path:
    """Create a test configuration file."""
    config_file = output_path / "test_config.json"
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    return config_file


def create_test_output_structure(base_dir: Path) -> Dict[str, Path]:
    """Create a test output directory structure."""
    dirs = {
        "outputs": base_dir / "outputs",
        "models": base_dir / "models", 
        "cache": base_dir / "cache",
        "logs": base_dir / "logs"
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


class MockFileSystem:
    """Mock file system for testing file operations."""
    
    def __init__(self):
        self.files = {}
        self.directories = set()
    
    def create_file(self, path: str, content: str = ""):
        """Create a mock file."""
        self.files[path] = content
        # Add parent directories
        parent = str(Path(path).parent)
        if parent != ".":
            self.directories.add(parent)
    
    def create_directory(self, path: str):
        """Create a mock directory."""
        self.directories.add(path)
    
    def file_exists(self, path: str) -> bool:
        """Check if mock file exists."""
        return path in self.files
    
    def directory_exists(self, path: str) -> bool:
        """Check if mock directory exists."""
        return path in self.directories
    
    def read_file(self, path: str) -> str:
        """Read mock file content."""
        if path not in self.files:
            raise FileNotFoundError(f"Mock file not found: {path}")
        return self.files[path]
    
    def write_file(self, path: str, content: str):
        """Write mock file content."""
        self.files[path] = content
