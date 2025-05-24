"""
Integration tests for the refactored image generation system.

Tests the interaction between core components, interfaces, and utilities
in the new modular image generation architecture.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock

from services.ai.image_generation.core.config_manager import ConfigManager
from services.ai.image_generation.core.pipeline_manager import PipelineManager
from services.ai.image_generation.core.generation_engine import GenerationEngine
from services.ai.image_generation.interfaces.sync_generator import SyncImageGenerator
from services.ai.image_generation.utils.validation import validate_generation_params
from services.ai.image_generation.utils.file_utils import generate_filename
from services.ai.image_generation.memory_manager import get_memory_manager


class TestImageGenerationIntegration:
    """Integration tests for image generation components."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp(prefix="img_gen_test_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_config_manager_integration(self):
        """Test ConfigManager loads and provides config correctly."""
        config_manager = ConfigManager()
        
        # Test config loading
        config = config_manager.get_config()
        assert isinstance(config, dict)
        assert "model_path" in config
        assert "default_steps" in config
        
        # Test config updates
        config_manager.update_config({"test_param": "test_value"})
        updated_config = config_manager.get_config()
        assert updated_config["test_param"] == "test_value"
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_pipeline_manager_integration(self, mock_cuda):
        """Test PipelineManager loads and manages pipeline."""
        config_manager = ConfigManager()
        
        with patch('services.ai.image_generation.core.pipeline_manager.DiffusionPipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline_class.from_pretrained.return_value = mock_pipeline
            
            pipeline_manager = PipelineManager(config_manager)
            
            # Test pipeline loading
            pipeline = pipeline_manager.get_pipeline()
            assert pipeline is not None
            
            # Test pipeline caching
            pipeline2 = pipeline_manager.get_pipeline()
            assert pipeline is pipeline2
    
    def test_generation_engine_integration(self, temp_output_dir):
        """Test GenerationEngine with real components."""
        config_manager = ConfigManager()
        config_manager.update_config({"output_dir": str(temp_output_dir)})
        
        with patch('services.ai.image_generation.core.pipeline_manager.DiffusionPipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline_class.from_pretrained.return_value = mock_pipeline
            
            # Mock the pipeline call to return a mock image
            mock_image = Mock()
            mock_image.save = Mock()
            mock_pipeline.return_value.images = [mock_image]
            
            pipeline_manager = PipelineManager(config_manager)
            generation_engine = GenerationEngine(config_manager, pipeline_manager)
            
            # Test image generation
            result = generation_engine.generate_image(
                prompt="test prompt",
                width=512,
                height=512,
                num_inference_steps=20
            )
            
            assert result is not None
            assert "image_path" in result
            assert "metadata" in result
    
    def test_sync_generator_integration(self, temp_output_dir):
        """Test SyncImageGenerator end-to-end."""
        config = {
            "model_path": "test/model",
            "output_dir": str(temp_output_dir),
            "default_steps": 20,
            "default_width": 512,
            "default_height": 512
        }
        
        with patch('services.ai.image_generation.core.pipeline_manager.DiffusionPipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline_class.from_pretrained.return_value = mock_pipeline
            
            # Mock successful generation
            mock_image = Mock()
            mock_image.save = Mock()
            mock_pipeline.return_value.images = [mock_image]
            
            generator = SyncImageGenerator(config)
            
            # Test generation
            result = generator.generate(
                prompt="integration test prompt",
                width=512,
                height=512,
                steps=20
            )
            
            assert result["success"] is True
            assert "image_path" in result
            assert "generation_time" in result
            assert result["metadata"]["prompt"] == "integration test prompt"
    
    def test_validation_integration(self):
        """Test validation utilities integration."""
        # Test valid parameters
        result = validate_generation_params(
            prompt="test prompt",
            width=512,
            height=512,
            steps=20
        )
        assert result["valid"] is True
        assert result["errors"] == []
        
        # Test invalid parameters
        result = validate_generation_params(
            prompt="",  # Empty prompt
            width=10000,  # Too large
            height=10000,  # Too large
            steps=0  # Invalid steps
        )
        assert result["valid"] is False
        assert len(result["errors"]) > 0
    
    def test_file_utils_integration(self, temp_output_dir):
        """Test file utilities integration."""
        # Test filename generation
        filename = generate_filename(
            prompt="test prompt for filename",
            width=512,
            height=512,
            steps=20
        )
        assert filename.endswith(".png")
        assert "test_prompt" in filename
        
        # Test with output directory
        full_path = temp_output_dir / filename
        assert str(temp_output_dir) in str(full_path)
    
    def test_memory_manager_integration(self):
        """Test memory manager integration."""
        memory_manager = get_memory_manager()
        
        # Test status retrieval
        status = memory_manager.get_status()
        assert isinstance(status, dict)
        assert "pipeline_on_gpu" in status
        assert "memory_config" in status
        
        # Test memory operations
        memory_manager.clear_gpu_memory()  # Should not raise
        memory_manager.update_access_time()  # Should not raise
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_full_system_integration(self, mock_cuda, temp_output_dir):
        """Test full system integration from config to output."""
        from services.ai.image_generation.flux_generator import FluxGenerator
        
        # Create generator with test config
        config = {
            "model_path": "test/model",
            "output_dir": str(temp_output_dir)
        }
        
        with patch('services.ai.image_generation.core.pipeline_manager.DiffusionPipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline_class.from_pretrained.return_value = mock_pipeline
            
            # Mock successful generation
            mock_image = Mock()
            mock_image.save = Mock()
            mock_pipeline.return_value.images = [mock_image]
            
            generator = FluxGenerator(config)
            
            # Test generation through main interface
            result = generator.generate_image(
                prompt="full system test",
                width=512,
                height=512,
                num_inference_steps=20
            )
            
            assert result["success"] is True
            assert "image_path" in result
            assert "metadata" in result


class TestErrorHandlingIntegration:
    """Integration tests for error handling across components."""
    
    def test_config_error_propagation(self):
        """Test error handling in configuration chain."""
        config_manager = ConfigManager()
        
        # Test with invalid config
        with pytest.raises(Exception):
            config_manager.update_config(None)
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_pipeline_error_handling(self, mock_cuda):
        """Test error handling in pipeline loading."""
        config_manager = ConfigManager()
        
        with patch('services.ai.image_generation.core.pipeline_manager.DiffusionPipeline') as mock_pipeline_class:
            # Mock pipeline loading failure
            mock_pipeline_class.from_pretrained.side_effect = Exception("Model not found")
            
            pipeline_manager = PipelineManager(config_manager)
            
            with pytest.raises(Exception, match="Model not found"):
                pipeline_manager.get_pipeline()
    
    def test_validation_error_integration(self):
        """Test validation error integration."""
        # Test with multiple validation errors
        result = validate_generation_params(
            prompt="",
            width=-100,
            height=0,
            steps=-5
        )
        
        assert result["valid"] is False
        assert len(result["errors"]) >= 3  # Multiple validation errors
        assert any("prompt" in error.lower() for error in result["errors"])
        assert any("width" in error.lower() for error in result["errors"])
        assert any("steps" in error.lower() for error in result["errors"])
