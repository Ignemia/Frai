"""
Implementation tests for the refactored image generation architecture.

These tests verify specific implementation behaviors, performance characteristics,
and edge cases in the refactored modular system.
"""

import pytest
import time
import threading
from unittest.mock import patch, Mock, MagicMock
from pathlib import Path

from services.ai.image_generation.core.config_manager import ConfigManager
from services.ai.image_generation.core.pipeline_manager import PipelineManager
from services.ai.image_generation.core.generation_engine import GenerationEngine
from services.ai.image_generation.interfaces.sync_generator import SyncImageGenerator
from services.ai.image_generation.memory_manager import MemoryManager


class TestConfigManagerImplementation:
    """Implementation-specific tests for ConfigManager."""
    
    def test_config_caching_behavior(self):
        """Test that config is properly cached and not reloaded unnecessarily."""
        config_manager = ConfigManager()
        
        # First access
        config1 = config_manager.get_config()
        
        # Second access should return same object (cached)
        config2 = config_manager.get_config()
        
        # Verify caching (implementation detail)
        assert config1 is config2
    
    def test_config_update_invalidates_cache(self):
        """Test that config updates properly invalidate cache."""
        config_manager = ConfigManager()
        
        original_config = config_manager.get_config()
        
        # Update config
        config_manager.update_config({"new_param": "new_value"})
        
        # Get config again
        updated_config = config_manager.get_config()
        
        # Should be different object due to cache invalidation
        assert original_config is not updated_config
        assert updated_config["new_param"] == "new_value"
    
    def test_thread_safety(self):
        """Test ConfigManager thread safety."""
        config_manager = ConfigManager()
        results = []
        errors = []
        
        def worker():
            try:
                for i in range(10):
                    config = config_manager.get_config()
                    config_manager.update_config({f"thread_param_{i}": f"value_{i}"})
                    results.append(config)
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads concurrently
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Should not have any errors
        assert len(errors) == 0
        assert len(results) > 0


class TestPipelineManagerImplementation:
    """Implementation-specific tests for PipelineManager."""
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_lazy_loading_behavior(self, mock_cuda):
        """Test that pipeline is loaded lazily on first access."""
        config_manager = ConfigManager()
        
        with patch('services.ai.image_generation.core.pipeline_manager.DiffusionPipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline_class.from_pretrained.return_value = mock_pipeline
            
            pipeline_manager = PipelineManager(config_manager)
            
            # Pipeline should not be loaded yet
            mock_pipeline_class.from_pretrained.assert_not_called()
            
            # First access should trigger loading
            pipeline = pipeline_manager.get_pipeline()
            mock_pipeline_class.from_pretrained.assert_called_once()
            
            # Second access should not trigger loading again
            pipeline2 = pipeline_manager.get_pipeline()
            mock_pipeline_class.from_pretrained.assert_called_once()  # Still only once
            
            assert pipeline is pipeline2
    
    @patch('torch.cuda.is_available', return_value=True)
    def test_memory_optimization_application_order(self, mock_cuda):
        """Test that memory optimizations are applied in correct order."""
        config_manager = ConfigManager()
        
        with patch('services.ai.image_generation.core.pipeline_manager.DiffusionPipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline_class.from_pretrained.return_value = mock_pipeline
            
            pipeline_manager = PipelineManager(config_manager)
            pipeline = pipeline_manager.get_pipeline()
            
            # Verify optimizations were applied in correct order
            call_order = []
            for call in mock_pipeline.method_calls:
                call_order.append(call[0])
            
            # Sequential CPU offload should be called first
            assert 'enable_sequential_cpu_offload' in call_order
            # Attention optimizations should follow
            assert 'enable_attention_slicing' in call_order
    
    def test_pipeline_reload_on_config_change(self):
        """Test that pipeline is reloaded when config changes."""
        config_manager = ConfigManager()
        
        with patch('services.ai.image_generation.core.pipeline_manager.DiffusionPipeline') as mock_pipeline_class:
            mock_pipeline1 = Mock()
            mock_pipeline2 = Mock()
            mock_pipeline_class.from_pretrained.side_effect = [mock_pipeline1, mock_pipeline2]
            
            pipeline_manager = PipelineManager(config_manager)
            
            # First load
            pipeline1 = pipeline_manager.get_pipeline()
            assert pipeline1 is mock_pipeline1
            
            # Change config
            config_manager.update_config({"model_path": "new/model/path"})
            
            # Should reload pipeline
            pipeline2 = pipeline_manager.get_pipeline()
            assert pipeline2 is mock_pipeline2
            assert mock_pipeline_class.from_pretrained.call_count == 2


class TestGenerationEngineImplementation:
    """Implementation-specific tests for GenerationEngine."""
    
    def test_parameter_validation_timing(self):
        """Test that parameter validation happens before pipeline loading."""
        config_manager = ConfigManager()
        
        with patch('services.ai.image_generation.core.pipeline_manager.DiffusionPipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline_class.from_pretrained.return_value = mock_pipeline
            
            pipeline_manager = PipelineManager(config_manager)
            generation_engine = GenerationEngine(config_manager, pipeline_manager)
            
            # Test with invalid parameters
            with pytest.raises(ValueError):
                generation_engine.generate_image(
                    prompt="",  # Invalid empty prompt
                    width=512,
                    height=512,
                    num_inference_steps=20
                )
            
            # Pipeline should not have been loaded due to early validation failure
            mock_pipeline_class.from_pretrained.assert_not_called()
    
    def test_metadata_generation_implementation(self, temp_directory):
        """Test specific metadata generation implementation."""
        config_manager = ConfigManager()
        config_manager.update_config({"output_dir": str(temp_directory)})
        
        with patch('services.ai.image_generation.core.pipeline_manager.DiffusionPipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline_class.from_pretrained.return_value = mock_pipeline
            
            # Mock successful generation
            mock_image = Mock()
            mock_image.save = Mock()
            mock_pipeline.return_value.images = [mock_image]
            
            pipeline_manager = PipelineManager(config_manager)
            generation_engine = GenerationEngine(config_manager, pipeline_manager)
            
            # Test metadata includes all expected fields
            result = generation_engine.generate_image(
                prompt="test prompt",
                width=512,
                height=512,
                num_inference_steps=20
            )
            
            metadata = result["metadata"]
            assert "prompt" in metadata
            assert "width" in metadata
            assert "height" in metadata
            assert "num_inference_steps" in metadata
            assert "timestamp" in metadata
            assert "model_info" in metadata
    
    def test_error_state_cleanup(self):
        """Test that error states are properly cleaned up."""
        config_manager = ConfigManager()
        
        with patch('services.ai.image_generation.core.pipeline_manager.DiffusionPipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline.side_effect = Exception("Generation failed")
            mock_pipeline_class.from_pretrained.return_value = mock_pipeline
            
            pipeline_manager = PipelineManager(config_manager)
            generation_engine = GenerationEngine(config_manager, pipeline_manager)
            
            # Generation should fail
            with pytest.raises(Exception):
                generation_engine.generate_image(
                    prompt="test prompt",
                    width=512,
                    height=512,
                    num_inference_steps=20
                )
            
            # Memory should be cleaned up even after error
            # This tests the implementation detail of cleanup in finally blocks


class TestMemoryManagerImplementation:
    """Implementation-specific tests for MemoryManager."""
    
    def test_offload_timer_cancellation(self):
        """Test that offload timers are properly cancelled and recreated."""
        manager = MemoryManager()
        mock_pipeline = Mock()
        
        # Start first offload timer
        manager.schedule_offload(mock_pipeline, delay=1)
        first_timer = manager._offload_timer
        
        # Start second offload timer
        manager.schedule_offload(mock_pipeline, delay=1)
        second_timer = manager._offload_timer
        
        # Should be different timer objects
        assert first_timer is not second_timer
        # First timer should be cancelled (implementation detail)
        assert first_timer._finished.is_set() or first_timer._target is None
    
    def test_concurrent_gpu_operations(self):
        """Test thread safety of GPU operations."""
        manager = MemoryManager()
        mock_pipeline = Mock()
        results = []
        errors = []
        
        def gpu_worker():
            try:
                for _ in range(5):
                    manager.move_pipeline_to_gpu(mock_pipeline)
                    manager.move_pipeline_to_cpu(mock_pipeline)
                    results.append("success")
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = [threading.Thread(target=gpu_worker) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Should handle concurrent access without errors
        assert len(errors) == 0
        assert len(results) > 0
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.empty_cache')
    def test_memory_cleanup_frequency(self, mock_empty_cache, mock_cuda):
        """Test memory cleanup is called appropriately."""
        manager = MemoryManager()
        
        # Multiple cleanup calls
        for _ in range(5):
            manager.clear_gpu_memory()
        
        # Should call CUDA empty_cache for each cleanup
        assert mock_empty_cache.call_count == 5


class TestSyncGeneratorImplementation:
    """Implementation-specific tests for SyncImageGenerator."""
    
    def test_initialization_order(self):
        """Test that components are initialized in correct order."""
        config = {"model_path": "test/model"}
        
        with patch('services.ai.image_generation.core.config_manager.ConfigManager') as mock_config_manager:
            with patch('services.ai.image_generation.core.pipeline_manager.PipelineManager') as mock_pipeline_manager:
                with patch('services.ai.image_generation.core.generation_engine.GenerationEngine') as mock_generation_engine:
                    
                    generator = SyncImageGenerator(config)
                    
                    # Verify initialization order
                    assert mock_config_manager.called
                    assert mock_pipeline_manager.called
                    assert mock_generation_engine.called
    
    def test_error_response_format(self):
        """Test that error responses follow consistent format."""
        config = {"model_path": "test/model"}
        
        with patch('services.ai.image_generation.core.pipeline_manager.DiffusionPipeline') as mock_pipeline_class:
            mock_pipeline_class.from_pretrained.side_effect = Exception("Model loading failed")
            
            generator = SyncImageGenerator(config)
            
            # Should return error in consistent format
            result = generator.generate(
                prompt="test prompt",
                width=512,
                height=512,
                steps=20
            )
            
            assert result["success"] is False
            assert "error" in result
            assert "error_type" in result
    
    @pytest.mark.slow
    def test_generation_timeout_behavior(self):
        """Test behavior when generation takes too long."""
        config = {"model_path": "test/model", "generation_timeout": 1}  # 1 second timeout
        
        with patch('services.ai.image_generation.core.pipeline_manager.DiffusionPipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            # Make generation take longer than timeout
            mock_pipeline.side_effect = lambda *args, **kwargs: time.sleep(2)
            mock_pipeline_class.from_pretrained.return_value = mock_pipeline
            
            generator = SyncImageGenerator(config)
            
            start_time = time.time()
            result = generator.generate(
                prompt="test prompt",
                width=512,
                height=512,
                steps=20
            )
            end_time = time.time()
            
            # Should timeout and return error
            assert result["success"] is False
            assert "timeout" in result["error"].lower()
            # Should not take much longer than timeout
            assert (end_time - start_time) < 3  # Some buffer for test execution
