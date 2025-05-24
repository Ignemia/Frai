"""
Unit tests for image generation memory manager.

Tests the MemoryManager class in isolation with mocked dependencies.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from services.ai.image_generation.memory_manager import (
    MemoryManager, 
    get_memory_manager,
    clear_gpu_memory,
    get_memory_status
)


class TestMemoryManager:
    """Test cases for MemoryManager class."""
    
    def test_init(self):
        """Test MemoryManager initialization."""
        manager = MemoryManager()
        
        assert manager.config is not None
        assert manager.config["gpu_offload_timeout"] == 300
        assert manager.config["enable_attention_slicing"] is True
        assert manager._last_access_time == 0
        assert manager._pipeline_on_gpu is False
        assert manager._memory_lock is not None
    
    def test_memory_config_property(self):
        """Test memory_config property returns copy of config."""
        manager = MemoryManager()
        config = manager.memory_config
        
        # Modify returned config
        config["gpu_offload_timeout"] = 600
        
        # Original should be unchanged
        assert manager.config["gpu_offload_timeout"] == 300
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_clear_gpu_memory_no_cuda(self, mock_cuda_available):
        """Test clear_gpu_memory when CUDA is not available."""
        manager = MemoryManager()
        
        # Should not raise an exception
        manager.clear_gpu_memory()
        
        mock_cuda_available.assert_called_once()
    
    @patch('torch.cuda.synchronize')
    @patch('torch.cuda.empty_cache')
    @patch('torch.cuda.is_available', return_value=True)
    @patch('gc.collect')
    def test_clear_gpu_memory_with_cuda(self, mock_gc, mock_cuda_available, 
                                       mock_empty_cache, mock_synchronize):
        """Test clear_gpu_memory when CUDA is available."""
        manager = MemoryManager()
        manager.clear_gpu_memory()
        
        mock_cuda_available.assert_called_once()
        mock_empty_cache.assert_called_once()
        mock_synchronize.assert_called_once()
        mock_gc.assert_called_once()
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_get_gpu_memory_info_no_cuda(self, mock_cuda_available):
        """Test get_gpu_memory_info when CUDA is not available."""
        manager = MemoryManager()
        info = manager.get_gpu_memory_info()
        
        assert info["available"] is False
        mock_cuda_available.assert_called_once()
    
    @patch('torch.cuda.get_device_name', return_value="Test GPU")
    @patch('torch.cuda.memory_reserved', return_value=1000000000)
    @patch('torch.cuda.memory_allocated', return_value=500000000)
    @patch('torch.cuda.mem_get_info', return_value=(2000000000, 4000000000))
    @patch('torch.cuda.is_available', return_value=True)
    def test_get_gpu_memory_info_with_cuda(self, mock_cuda_available, mock_mem_info,
                                          mock_allocated, mock_reserved, mock_device_name):
        """Test get_gpu_memory_info when CUDA is available."""
        manager = MemoryManager()
        info = manager.get_gpu_memory_info()
        
        assert info["available"] is True
        assert info["free_gb"] == 2.0
        assert info["total_gb"] == 4.0
        assert info["allocated_gb"] == 0.5
        assert info["reserved_gb"] == 1.0
        assert info["device_name"] == "Test GPU"
    
    @patch('time.time', return_value=12345)
    def test_update_access_time(self, mock_time):
        """Test update_access_time updates the timestamp."""
        manager = MemoryManager()
        manager.update_access_time()
        
        assert manager._last_access_time == 12345
        mock_time.assert_called_once()
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_move_pipeline_to_gpu_no_cuda(self, mock_cuda_available, mock_diffusion_pipeline):
        """Test move_pipeline_to_gpu when CUDA is not available."""
        manager = MemoryManager()
        result = manager.move_pipeline_to_gpu(mock_diffusion_pipeline)
        
        assert result is mock_diffusion_pipeline
        mock_diffusion_pipeline.to.assert_not_called()
    
    @patch('torch.cuda.is_available', return_value=True)
    def test_move_pipeline_to_gpu_success(self, mock_cuda_available, mock_diffusion_pipeline):
        """Test successful move_pipeline_to_gpu."""
        manager = MemoryManager()
        
        with patch.object(manager, 'update_access_time') as mock_update:
            result = manager.move_pipeline_to_gpu(mock_diffusion_pipeline)
        
        assert result is mock_diffusion_pipeline
        assert manager._pipeline_on_gpu is True
        mock_diffusion_pipeline.to.assert_called_once_with("cuda")
        mock_update.assert_called_once()
    
    def test_move_pipeline_to_cpu(self, mock_diffusion_pipeline):
        """Test move_pipeline_to_cpu."""
        manager = MemoryManager()
        manager._pipeline_on_gpu = True
        
        with patch.object(manager, 'clear_gpu_memory') as mock_clear:
            result = manager.move_pipeline_to_cpu(mock_diffusion_pipeline)
        
        assert result is mock_diffusion_pipeline
        assert manager._pipeline_on_gpu is False
        mock_diffusion_pipeline.to.assert_called_once_with("cpu")
        mock_clear.assert_called_once()
    
    def test_apply_memory_optimizations(self, mock_diffusion_pipeline):
        """Test apply_memory_optimizations applies all optimizations."""
        manager = MemoryManager()
        
        result = manager.apply_memory_optimizations(mock_diffusion_pipeline)
        
        assert result is mock_diffusion_pipeline
        mock_diffusion_pipeline.enable_sequential_cpu_offload.assert_called_once()
        mock_diffusion_pipeline.enable_attention_slicing.assert_called_once_with("max")
        mock_diffusion_pipeline.enable_vae_slicing.assert_called_once()
        mock_diffusion_pipeline.enable_vae_tiling.assert_called_once()
    
    @patch('torch.cuda.is_available', return_value=True)
    def test_get_status(self, mock_cuda_available):
        """Test get_status returns complete status."""
        manager = MemoryManager()
        manager._pipeline_on_gpu = True
        manager._last_access_time = 12345
        
        with patch.object(manager, 'get_gpu_memory_info', return_value={"available": True}):
            status = manager.get_status()
        
        assert status["pipeline_on_gpu"] is True
        assert status["last_access_time"] == 12345
        assert "memory_config" in status
        assert status["available"] is True


class TestMemoryManagerGlobalFunctions:
    """Test global memory manager functions."""
    
    @patch('services.ai.image_generation.memory_manager._memory_manager', None)
    def test_get_memory_manager_creates_instance(self):
        """Test get_memory_manager creates new instance when none exists."""
        manager = get_memory_manager()
        
        assert isinstance(manager, MemoryManager)
        
        # Second call should return same instance
        manager2 = get_memory_manager()
        assert manager is manager2
    
    def test_clear_gpu_memory_function(self):
        """Test global clear_gpu_memory function."""
        with patch('services.ai.image_generation.memory_manager.get_memory_manager') as mock_get:
            mock_manager = Mock()
            mock_get.return_value = mock_manager
            
            clear_gpu_memory()
            
            mock_get.assert_called_once()
            mock_manager.clear_gpu_memory.assert_called_once()
    
    def test_get_memory_status_function(self):
        """Test global get_memory_status function."""
        with patch('services.ai.image_generation.memory_manager.get_memory_manager') as mock_get:
            mock_manager = Mock()
            mock_manager.get_status.return_value = {"test": "status"}
            mock_get.return_value = mock_manager
            
            result = get_memory_status()
            
            assert result == {"test": "status"}
            mock_get.assert_called_once()
            mock_manager.get_status.assert_called_once()
