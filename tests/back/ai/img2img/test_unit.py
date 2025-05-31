"""
Unit tests for individual functions and classes in the image-to-image module.

This module tests the core components of the image-to-image generation system,
including initialization, preprocessing, and basic functionality.
"""

import pytest
import logging
from typing import Dict, Any
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from Frai.back.ai.img2img import (
    initialize_img2img_system,
    get_img2img_ai_instance,
    generate_img2img
)

# Set up logging
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def setup_img2img_ai():
    """
    Initialize the image-to-image system once for all tests.
    """
    success = initialize_img2img_system()
    if not success:
        pytest.fail("Failed to initialize image-to-image system")
    
    img2img_ai = get_img2img_ai_instance()
    return img2img_ai


class TestImg2ImgComponents:
    """Test individual components of the image-to-image system."""
    
    def test_system_initialization(self, setup_img2img_ai):
        """Test that the img2img system initializes correctly."""
        assert setup_img2img_ai is not None
        logger.info("Image-to-image system initialized successfully")
    
    def test_model_loading(self, setup_img2img_ai):
        """Test that the img2img model loads properly."""
        assert hasattr(setup_img2img_ai, 'model')
        assert hasattr(setup_img2img_ai, 'processor')
        logger.info("Image-to-image model loaded with required components")
    
    def test_image_preprocessing(self):
        """Test image preprocessing functionality."""
        from Frai.back.ai.img2img.preprocessing import preprocess_image
        
        # Test with image path
        image_path = "test_image.jpg"
        processed = preprocess_image(image_path)
        assert processed is not None
        
        logger.info("Image preprocessing function working correctly")
    
    def test_image_validation(self):
        """Test image input validation."""
        from Frai.back.ai.img2img.utils import validate_image_input
        
        # Test valid image
        assert validate_image_input("test.jpg") == True
        assert validate_image_input("test.png") == True
        
        # Test invalid inputs
        assert validate_image_input(None) == False
        assert validate_image_input("") == False
        assert validate_image_input("test.txt") == False
        
        logger.info("Image validation working correctly")
    
    def test_style_extraction(self):
        """Test style feature extraction."""
        from Frai.back.ai.img2img.style_transfer import extract_style_features
        
        style_image_path = "style_reference.jpg"
        style_features = extract_style_features(style_image_path)
        assert style_features is not None
        
        logger.info("Style extraction working correctly")
    
    def test_transformation_parameters_validation(self):
        """Test transformation parameters validation."""
        from Frai.back.ai.img2img.utils import validate_transformation_params
        
        # Test valid parameters
        valid_params = {
            'strength': 0.8,
            'guidance_scale': 7.5,
            'num_inference_steps': 20,
            'seed': 42
        }
        assert validate_transformation_params(valid_params) == True
        
        # Test invalid parameters
        invalid_params = {
            'strength': 1.5,  # Should be <= 1.0
            'guidance_scale': -1,  # Should be positive
            'num_inference_steps': 0  # Should be positive
        }
        assert validate_transformation_params(invalid_params) == False
        
        logger.info("Transformation parameters validation working correctly")


class TestImg2ImgAPI:
    """Test the image-to-image API functions."""
    
    def test_generate_img2img_basic(self, setup_img2img_ai):
        """Test basic image-to-image generation function."""
        result = generate_img2img(
            source_image="source.jpg",
            reference_image="style.jpg",
            transformation_type="style_transfer"
        )
        
        assert isinstance(result, dict)
        assert 'success' in result
        assert result['success'] == True
        assert 'generated_image' in result
        assert result['generated_image'] is not None
        
        logger.info("Basic img2img generation working correctly")
    
    def test_generate_img2img_style_transfer(self, setup_img2img_ai):
        """Test style transfer transformation."""
        result = generate_img2img(
            source_image="portrait.jpg",
            reference_image="art_style.jpg",
            transformation_type="style_transfer",
            strength=0.8
        )
        
        assert isinstance(result, dict)
        assert result['success'] == True
        assert 'generated_image' in result
        
        logger.info("Style transfer generation working correctly")
    
    def test_generate_img2img_face_swap(self, setup_img2img_ai):
        """Test face swap transformation."""
        result = generate_img2img(
            source_image="face1.jpg",
            reference_image="face2.jpg",
            transformation_type="face_swap"
        )
        
        assert isinstance(result, dict)
        assert result['success'] == True
        assert 'generated_image' in result
        
        logger.info("Face swap generation working correctly")
    
    def test_generate_img2img_invalid_source(self, setup_img2img_ai):
        """Test img2img generation with invalid source image."""
        result = generate_img2img(
            source_image="nonexistent.jpg",
            reference_image="style.jpg",
            transformation_type="style_transfer"
        )
        
        assert isinstance(result, dict)
        assert result['success'] == False
        assert 'error' in result
        
        logger.info("Invalid source image handling working correctly")
    
    def test_generate_img2img_none_inputs(self, setup_img2img_ai):
        """Test img2img generation with None inputs."""
        result = generate_img2img(
            source_image=None,
            reference_image=None,
            transformation_type="style_transfer"
        )
        
        assert isinstance(result, dict)
        assert result['success'] == False
        assert 'error' in result
        
        logger.info("None input handling working correctly")
    
    def test_generate_img2img_invalid_transformation(self, setup_img2img_ai):
        """Test img2img generation with invalid transformation type."""
        result = generate_img2img(
            source_image="source.jpg",
            reference_image="reference.jpg",
            transformation_type="invalid_type"
        )
        
        assert isinstance(result, dict)
        assert result['success'] == False
        assert 'error' in result
        
        logger.info("Invalid transformation type handling working correctly")
    
    def test_generate_img2img_with_parameters(self, setup_img2img_ai):
        """Test img2img generation with custom parameters."""
        result = generate_img2img(
            source_image="source.jpg",
            reference_image="style.jpg",
            transformation_type="style_transfer",
            strength=0.6,
            guidance_scale=10.0,
            seed=42
        )
        
        assert isinstance(result, dict)
        assert result['success'] == True
        assert 'generated_image' in result
        assert 'parameters_used' in result
        
        logger.info("Custom parameters generation working correctly")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])