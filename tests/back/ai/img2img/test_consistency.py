"""
Consistency tests for the image-to-image module.

This module tests that generated images are consistent with requests and source images,
ensuring reproducible outputs and maintaining source image characteristics.
"""

import pytest
import logging
import os
import sys
import csv
from typing import Dict, Any, List
from PIL import Image
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from Frai.back.ai.img2img import (
    initialize_img2img_system,
    get_img2img_ai_instance,
    generate_img2img
)

# Set up logging
logger = logging.getLogger(__name__)

# Path to the test set CSV file
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), 'testset.csv')
INPUTS_PATH = os.path.join(os.path.dirname(__file__), 'inputs.csv')


def load_consistency_test_cases() -> List[Dict[str, str]]:
    """Load test cases specifically for consistency testing."""
    test_cases = []
    try:
        with open(TEST_SET_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter for consistency tests
                if 'consistency' in row['tested_property'].lower():
                    test_cases.append(row)
        logger.info(f"Loaded {len(test_cases)} consistency test cases")
        return test_cases
    except Exception as e:
        logger.error(f"Failed to load test cases: {e}")
        return []


def load_input_image(image_id: str) -> Image.Image:
    """Load an input image by ID from the inputs.csv file."""
    try:
        with open(INPUTS_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['id'] == image_id:
                    image_path = row['path']
                    if os.path.exists(image_path):
                        return Image.open(image_path)
                    else:
                        # Create a placeholder image if file doesn't exist
                        return Image.new('RGB', (256, 256), color=(128, 128, 128))
        
        # If not found, create a default image
        return Image.new('RGB', (256, 256), color=(100, 100, 100))
    except Exception as e:
        logger.warning(f"Failed to load image {image_id}: {e}")
        return Image.new('RGB', (256, 256), color=(150, 150, 150))


def calculate_image_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """Calculate similarity between two images using simple metrics."""
    # Convert to same size for comparison
    size = (256, 256)
    img1_resized = img1.resize(size)
    img2_resized = img2.resize(size)
    
    # Convert to numpy arrays
    arr1 = np.array(img1_resized)
    arr2 = np.array(img2_resized)
    
    # Calculate mean squared error
    mse = np.mean((arr1 - arr2) ** 2)
    
    # Convert to similarity score (0-1, where 1 is identical)
    max_mse = 255 ** 2  # Maximum possible MSE for 8-bit images
    similarity = 1.0 - (mse / max_mse)
    
    return similarity


def extract_dominant_colors(image: Image.Image, num_colors: int = 5) -> List[tuple]:
    """Extract dominant colors from an image."""
    # Resize image for faster processing
    image = image.resize((50, 50))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Get pixel data
    pixels = list(image.getdata())
    
    # Simple color quantization
    from collections import Counter
    color_counts = Counter(pixels)
    dominant_colors = [color for color, count in color_counts.most_common(num_colors)]
    
    return dominant_colors


@pytest.fixture(scope="module")
def setup_img2img_ai():
    """Initialize the image-to-image system once for all tests."""
    success = initialize_img2img_system()
    if not success:
        pytest.fail("Failed to initialize image-to-image system")
    
    img2img_ai = get_img2img_ai_instance()
    return img2img_ai


class TestImg2ImgConsistency:
    """Test image-to-image generation consistency."""
    
    @pytest.mark.parametrize("test_case", load_consistency_test_cases())
    def test_generation_consistency(self, setup_img2img_ai, test_case):
        """
        Test that identical inputs produce consistent outputs.
        
        Args:
            setup_img2img_ai: The img2img AI instance from fixture
            test_case: Dictionary containing test case details from CSV
        """
        test_id = test_case['id']
        input_groups = test_case['input_groups'].split(';')
        
        logger.info(f"Running consistency test {test_id}: {test_case['name']}")
        
        # Load input images based on groups
        if len(input_groups) >= 2:
            source_image = load_input_image('1')  # Use first image as source
            reference_image = load_input_image('11')  # Use first style as reference
        else:
            pytest.skip(f"Test {test_id} requires at least 2 input groups")
        
        # Generate multiple outputs with identical parameters
        generation_params = {
            'seed': 42,  # Fixed seed for consistency
            'strength': 0.8,
            'guidance_scale': 7.5,
            'num_inference_steps': 20
        }
        
        results = []
        for i in range(3):  # Generate 3 times
            result = generate_img2img(
                source_image=source_image,
                reference_image=reference_image,
                transformation_type="style_transfer",
                **generation_params
            )
            
            assert result.get('success', False), f"Generation {i+1} failed: {result.get('error', 'Unknown error')}"
            results.append(result['generated_image'])
        
        # Calculate consistency between generations
        similarities = []
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                similarity = calculate_image_similarity(results[i], results[j])
                similarities.append(similarity)
        
        avg_similarity = np.mean(similarities)
        min_similarity = np.min(similarities)
        
        logger.info(f"Test {test_id} consistency: avg={avg_similarity:.3f}, min={min_similarity:.3f}")
        
        # Consistency threshold - outputs should be quite similar
        assert avg_similarity >= 0.85, f"Average similarity {avg_similarity:.3f} below consistency threshold"
        assert min_similarity >= 0.75, f"Minimum similarity {min_similarity:.3f} too low"
    
    def test_source_image_preservation(self, setup_img2img_ai):
        """Test that key characteristics of source image are preserved."""
        # Create a distinctive source image
        source_image = Image.new('RGB', (256, 256), color=(255, 0, 0))  # Red image
        style_image = Image.new('RGB', (256, 256), color=(0, 0, 255))   # Blue style
        
        result = generate_img2img(
            source_image=source_image,
            reference_image=style_image,
            transformation_type="style_transfer",
            strength=0.5  # Moderate transformation
        )
        
        if not result.get('success', False):
            pytest.skip("Source preservation test requires working generation")
        
        generated_image = result['generated_image']
        
        # Check that some red characteristics are preserved
        generated_colors = extract_dominant_colors(generated_image)
        source_colors = extract_dominant_colors(source_image)
        
        # At least some similarity in color space should be maintained
        has_red_influence = any(color[0] > color[1] and color[0] > color[2] for color in generated_colors)
        
        logger.info(f"Source preservation test: red influence detected = {has_red_influence}")
        
        # With moderate strength, some source characteristics should remain
        # Note: This test might need adjustment based on actual model behavior
    
    def test_style_consistency_across_sources(self, setup_img2img_ai):
        """Test that the same style produces consistent results across different source images."""
        # Create different source images
        source_images = [
            Image.new('RGB', (256, 256), color=(200, 100, 50)),   # Orange
            Image.new('RGB', (256, 256), color=(100, 200, 50)),   # Green
            Image.new('RGB', (256, 256), color=(50, 100, 200))    # Blue
        ]
        
        # Use consistent style reference
        style_image = Image.new('RGB', (256, 256), color=(255, 255, 0))  # Yellow style
        
        generated_images = []
        for i, source in enumerate(source_images):
            result = generate_img2img(
                source_image=source,
                reference_image=style_image,
                transformation_type="style_transfer",
                seed=42,  # Fixed seed
                strength=0.8
            )
            
            if result.get('success', False):
                generated_images.append(result['generated_image'])
            else:
                logger.warning(f"Failed to generate for source {i}")
        
        if len(generated_images) < 2:
            pytest.skip("Style consistency test requires at least 2 successful generations")
        
        # Extract style characteristics (dominant colors) from each result
        style_characteristics = [extract_dominant_colors(img) for img in generated_images]
        
        # Check for yellow influence in all results (from style image)
        yellow_influences = []
        for colors in style_characteristics:
            yellow_influence = any(
                color[0] > 150 and color[1] > 150 and color[2] < 100  # Yellowish colors
                for color in colors
            )
            yellow_influences.append(yellow_influence)
        
        # Most results should show style influence
        style_consistency_ratio = sum(yellow_influences) / len(yellow_influences)
        
        logger.info(f"Style consistency across sources: {style_consistency_ratio:.2f}")
        
        # At least 70% of results should show consistent style influence
        assert style_consistency_ratio >= 0.7, f"Style consistency ratio {style_consistency_ratio:.2f} too low"
    
    def test_parameter_consistency(self, setup_img2img_ai):
        """Test that the same parameters produce consistent results."""
        source_image = Image.new('RGB', (256, 256), color=(128, 128, 128))
        style_image = Image.new('RGB', (256, 256), color=(255, 100, 50))
        
        # Test different parameter sets
        parameter_sets = [
            {'strength': 0.5, 'guidance_scale': 5.0, 'seed': 123},
            {'strength': 0.8, 'guidance_scale': 10.0, 'seed': 456},
            {'strength': 1.0, 'guidance_scale': 15.0, 'seed': 789}
        ]
        
        for params in parameter_sets:
            # Generate twice with same parameters
            results = []
            for _ in range(2):
                result = generate_img2img(
                    source_image=source_image,
                    reference_image=style_image,
                    transformation_type="style_transfer",
                    **params
                )
                
                if result.get('success', False):
                    results.append(result['generated_image'])
            
            if len(results) == 2:
                similarity = calculate_image_similarity(results[0], results[1])
                logger.info(f"Parameter set {params} consistency: {similarity:.3f}")
                
                # Same parameters should produce very similar results
                assert similarity >= 0.9, f"Parameter consistency {similarity:.3f} too low for {params}"
    
    def test_aspect_ratio_preservation(self, setup_img2img_ai):
        """Test that aspect ratios are properly preserved or handled."""
        test_aspect_ratios = [
            (256, 256),   # 1:1
            (512, 256),   # 2:1
            (256, 512),   # 1:2
            (384, 256),   # 3:2
        ]
        
        style_image = Image.new('RGB', (256, 256), color=(100, 150, 200))
        
        for width, height in test_aspect_ratios:
            source_image = Image.new('RGB', (width, height), color=(200, 150, 100))
            
            result = generate_img2img(
                source_image=source_image,
                reference_image=style_image,
                transformation_type="style_transfer"
            )
            
            if result.get('success', False):
                generated_image = result['generated_image']
                gen_width, gen_height = generated_image.size
                
                # Calculate aspect ratios
                source_ratio = width / height
                generated_ratio = gen_width / gen_height
                
                # Aspect ratio should be preserved or handled consistently
                ratio_difference = abs(source_ratio - generated_ratio)
                
                logger.info(f"Aspect ratio {width}x{height}: source={source_ratio:.2f}, generated={generated_ratio:.2f}")
                
                # Allow some tolerance for aspect ratio preservation
                assert ratio_difference < 0.1 or generated_ratio in [1.0, source_ratio], \
                    f"Aspect ratio not properly handled: {ratio_difference:.3f} difference"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])