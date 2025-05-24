"""
Test configuration and shared fixtures.

This module provides common test configuration, fixtures, and utilities
that are shared across all test modules.
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Generator, Dict, Any
import torch


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Test configuration fixture."""
    return {
        "test_mode": True,
        "log_level": "DEBUG",
        "temp_dir": None,
        "mock_gpu": not torch.cuda.is_available(),
        "timeout": 30
    }


@pytest.fixture(scope="session")
def temp_directory() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp(prefix="personal_chatter_test_")
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def mock_gpu():
    """Mock GPU availability for testing."""
    with patch('torch.cuda.is_available', return_value=False):
        with patch('torch.cuda.empty_cache'):
            with patch('torch.cuda.synchronize'):
                yield


@pytest.fixture(scope="function")
def mock_logger():
    """Mock logger for testing."""
    return Mock()


@pytest.fixture(scope="function")
def sample_config() -> Dict[str, Any]:
    """Sample configuration for testing."""
    return {
        "image_generation": {
            "model_path": "test/model/path",
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
        "api": {
            "host": "127.0.0.1",
            "port": 8000,
            "debug": True
        }
    }


@pytest.fixture(scope="function")
def mock_diffusion_pipeline():
    """Mock diffusion pipeline for testing."""
    pipeline = Mock()
    pipeline.to.return_value = pipeline
    pipeline.enable_sequential_cpu_offload.return_value = None
    pipeline.enable_attention_slicing.return_value = None
    pipeline.enable_vae_slicing.return_value = None
    pipeline.enable_vae_tiling.return_value = None
    return pipeline


@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    """Automatically cleanup GPU memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Test categories
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )
    config.addinivalue_line(
        "markers", "implementation: Implementation-specific tests"
    )
    config.addinivalue_line(
        "markers", "blackbox: End-to-end blackbox tests"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and load tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests (skip in quick test runs)"
    )
    config.addinivalue_line(
        "markers", "gpu: Tests requiring GPU hardware"
    )
    config.addinivalue_line(
        "markers", "network: Tests requiring network access"
    )
    config.addinivalue_line(
        "markers", "external: Tests requiring external services"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on location."""
    for item in items:
        # Get the relative path from tests directory
        rel_path = Path(item.fspath).relative_to(Path(config.rootdir) / "tests")
        
        # Add markers based on directory
        if rel_path.parts[0] == "unit":
            item.add_marker(pytest.mark.unit)
        elif rel_path.parts[0] == "integration":
            item.add_marker(pytest.mark.integration)
        elif rel_path.parts[0] == "implementation":
            item.add_marker(pytest.mark.implementation)
        elif rel_path.parts[0] == "blackbox":
            item.add_marker(pytest.mark.blackbox)
        elif rel_path.parts[0] == "performance":
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
