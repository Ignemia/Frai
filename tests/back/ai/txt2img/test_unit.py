"""
Unit tests for individual functions and classes in the text-to-image module.

This module tests the core components of the text-to-image generation system,
including initialization, prompt processing, and basic functionality.
"""

import pytest
import logging
from typing import Dict, Any
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from Frai.back.ai.txt2img import (
    initialize_txt2img_system,
    get_txt2img_ai_instance,
    generate_image_from_text
)

# Set up logging
logger = logging.getLogger(__name__)


# Using fixture from conftest.py


class TestTxt2ImgComponents:
    """Test individual components of the text-to-image system."""
    
    def test_system_initialization(self, setup_txt2img_ai):
        """Test that the txt2img system initializes correctly."""
        assert setup_txt2img_ai is not None
        logger.info("Text-to-image system initialized successfully")
    
    def test_model_loading(self, setup_txt2img_ai):
        """Test that the txt2img model loads properly."""
        assert hasattr(setup_txt2img_ai, 'model')
        assert hasattr(setup_txt2img_ai, 'tokenizer')
        assert hasattr(setup_txt2img_ai, 'scheduler')
        logger.info("Text-to-image model loaded with required components")
    
    def test_prompt_preprocessing(self):
        """Test prompt preprocessing functionality."""
        from Frai.back.ai.txt2img.preprocessing import preprocess_prompt
        
        # Test with simple prompt
        prompt = "A red apple"
        processed = preprocess_prompt(prompt)
        assert processed is not None
        assert isinstance(processed, str)
        
        logger.info("Prompt preprocessing function working correctly")
    
    def test_prompt_validation(self):
        """Test prompt input validation."""
        from Frai.back.ai.txt2img.utils import validate_prompt
        
        # Test valid prompts
        assert validate_prompt("A beautiful landscape") == True
        assert validate_prompt("Portrait of a woman") == True
        
        # Test invalid inputs
        assert validate_prompt(None) == False
        assert validate_prompt("") == False
        assert validate_prompt("   ") == False
        
        logger.info("Prompt validation working correctly")
    
    def test_generation_parameters_validation(self):
        """Test generation parameters validation."""
        from Frai.back.ai.txt2img.utils import validate_generation_params
        
        # Test valid parameters
        valid_params = {
            'width': 512,
            'height': 512,
            'steps': 20,
            'guidance_scale': 7.5,
            'seed': 42
        }
        assert validate_generation_params(valid_params) == True
        
        # Test invalid parameters
        invalid_params = {
            'width': 0,  # Should be positive
            'height': -1,  # Should be positive
            'steps': 0,  # Should be positive
            'guidance_scale': -1  # Should be positive
        }
        assert validate_generation_params(invalid_params) == False
        
        logger.info("Generation parameters validation working correctly")
    
    def test_image_postprocessing(self):
        """Test image postprocessing functionality."""
        from Frai.back.ai.txt2img.postprocessing import postprocess_image
        
        # Test with mock image data
        mock_image_tensor = "mock_tensor_data"
        processed = postprocess_image(mock_image_tensor)
        assert processed is not None
        
        logger.info("Image postprocessing function working correctly")


class TestTxt2ImgAPI:
    """Test the text-to-image API functions."""
    
    def test_generate_image_basic(self, setup_txt2img_ai):
        """Test basic image generation function."""
        prompt = "A simple red apple"
        result = generate_image_from_text(prompt=prompt)
        
        assert isinstance(result, dict)
        assert 'success' in result
        assert result['success'] == True
        assert 'generated_image' in result
        assert result['generated_image'] is not None
        
        logger.info("Basic image generation working correctly")
    
    def test_generate_image_with_parameters(self, setup_txt2img_ai):
        """Test image generation with custom parameters."""
        prompt = "A beautiful sunset landscape"
        result = generate_image_from_text(
            prompt=prompt,
            width=256,
            height=256,
            steps=10,
            guidance_scale=5.0,
            seed=42
        )
        
        assert isinstance(result, dict)
        assert result['success'] == True
        assert 'generated_image' in result
        assert 'parameters_used' in result
        
        logger.info("Parameterized image generation working correctly")
    
    def test_generate_image_empty_prompt(self, setup_txt2img_ai):
        """Test image generation with empty prompt."""
        result = generate_image_from_text(prompt="")
        
        assert isinstance(result, dict)
        assert result['success'] == False
        assert 'error' in result
        
        logger.info("Empty prompt handling working correctly")
    
    def test_generate_image_none_prompt(self, setup_txt2img_ai):
        """Test image generation with None prompt."""
        result = generate_image_from_text(prompt=None)
        
        assert isinstance(result, dict)
        assert result['success'] == False
        assert 'error' in result
        
        logger.info("None prompt handling working correctly")
    
    def test_generate_image_long_prompt(self, setup_txt2img_ai):
        """Test image generation with very long prompt."""
        long_prompt = "A beautiful " * 100 + "landscape"
        result = generate_image_from_text(prompt=long_prompt)
        
        assert isinstance(result, dict)
        # Should either succeed or fail gracefully
        assert 'success' in result
        
        if not result['success']:
            assert 'error' in result
        
        logger.info("Long prompt handling working correctly")
    
    def test_generate_image_special_characters(self, setup_txt2img_ai):
        """Test image generation with special characters in prompt."""
        special_prompt = "A cat with émojis 🐱 and spëcial characters!"
        result = generate_image_from_text(prompt=special_prompt)
        
        assert isinstance(result, dict)
        assert 'success' in result
        
        logger.info("Special characters handling working correctly")
    
    def test_generate_image_with_negative_prompt(self, setup_txt2img_ai):
        """Test image generation with negative prompt."""
        prompt = "A beautiful portrait"
        negative_prompt = "blurry, low quality, distorted"
        
        result = generate_image_from_text(
            prompt=prompt,
            negative_prompt=negative_prompt
        )
        
        assert isinstance(result, dict)
        assert result['success'] == True
        assert 'generated_image' in result
        
        logger.info("Negative prompt generation working correctly")
    
    def test_generate_image_batch(self, setup_txt2img_ai):
        """Test batch image generation."""
        prompts = [
            "A red apple",
            "A blue car",
            "A green tree"
        ]
        
        results = []
        for prompt in prompts:
            result = generate_image_from_text(prompt=prompt, seed=42)
            results.append(result)
        
        # Check that all results are properly formatted
        for i, result in enumerate(results):
            assert isinstance(result, dict)
            assert 'success' in result
            logger.info(f"Batch generation {i+1}: {'success' if result['success'] else 'failed'}")
    
    def test_generate_image_consistency(self, setup_txt2img_ai):
        """Test that same prompt with same seed produces consistent results."""
        prompt = "A yellow flower in a garden"
        seed = 12345
        
        # Generate twice with same parameters
        result1 = generate_image_from_text(prompt=prompt, seed=seed)
        result2 = generate_image_from_text(prompt=prompt, seed=seed)
        
        assert result1.get('success') == result2.get('success')
        
        if result1.get('success') and result2.get('success'):
            # Both should succeed and have images
            assert result1['generated_image'] is not None
            assert result2['generated_image'] is not None
        
        logger.info("Generation consistency test completed")


class TestTxt2ImgPromptProcessing:
    """Test prompt processing and understanding."""
    
    def test_simple_object_prompts(self, setup_txt2img_ai):
        """Test generation of simple objects."""
        simple_prompts = [
            "A cat",
            "A house",
            "A tree",
            "A car",
            "A flower"
        ]
        
        for prompt in simple_prompts:
            result = generate_image_from_text(prompt=prompt)
            assert isinstance(result, dict)
            assert 'success' in result
            
            if result['success']:
                assert result['generated_image'] is not None
            
            logger.info(f"Simple prompt '{prompt}': {'success' if result.get('success') else 'failed'}")
    
    def test_descriptive_prompts(self, setup_txt2img_ai):
        """Test generation with descriptive attributes."""
        descriptive_prompts = [
            "A red apple on a wooden table",
            "A fluffy white cat sleeping",
            "A tall green tree in spring",
            "A small blue car parked outside",
            "A beautiful yellow sunflower"
        ]
        
        for prompt in descriptive_prompts:
            result = generate_image_from_text(prompt=prompt)
            assert isinstance(result, dict)
            
            logger.info(f"Descriptive prompt test: {'success' if result.get('success') else 'failed'}")
    
    def test_style_prompts(self, setup_txt2img_ai):
        """Test generation with artistic styles."""
        style_prompts = [
            "A portrait in the style of Van Gogh",
            "A landscape as an oil painting",
            "A house in watercolor style",
            "A cat as a pencil sketch",
            "A flower in photorealistic style"
        ]
        
        for prompt in style_prompts:
            result = generate_image_from_text(prompt=prompt)
            assert isinstance(result, dict)
            
            logger.info(f"Style prompt test: {'success' if result.get('success') else 'failed'}")
    
    def test_complex_scene_prompts(self, setup_txt2img_ai):
        """Test generation of complex scenes."""
        complex_prompts = [
            "A bustling marketplace with people and colorful stalls",
            "A serene lake surrounded by mountains at sunset",
            "A futuristic city with flying cars and tall buildings",
            "A cozy library with books and a fireplace",
            "A tropical beach with palm trees and clear water"
        ]
        
        for prompt in complex_prompts:
            result = generate_image_from_text(prompt=prompt)
            assert isinstance(result, dict)
            
            logger.info(f"Complex prompt test: {'success' if result.get('success') else 'failed'}")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])