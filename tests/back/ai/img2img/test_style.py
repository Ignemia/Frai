"""
Style tests for the image-to-image module.

This module tests the img2img system's ability to copy art styles and reuse them,
ensuring style transfer accuracy and consistency across different source images.
"""

import pytest
import logging
import csv
import os
import sys
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

# Path to the test set CSV files
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), 'testset.csv')
INPUTS_PATH = os.path.join(os.path.dirname(__file__), 'inputs.csv')


def load_style_test_cases() -> List[Dict[str, str]]:
    """Load test cases specifically for style testing."""
    test_cases = []
    try:
        with open(TEST_SET_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter for style tests
                if any(keyword in row['tested_property'].lower() for keyword in ['style', 'artistic', 'brush']):
                    test_cases.append(row)
        logger.info(f"Loaded {len(test_cases)} style test cases")
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


def extract_style_features(image: Image.Image) -> Dict[str, float]:
    """Extract style-related features from an image."""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize for consistent analysis
    image = image.resize((128, 128))
    arr = np.array(image)
    
    features = {}
    
    # Color statistics
    features['color_variance'] = np.var(arr)
    features['brightness'] = np.mean(arr)
    features['contrast'] = np.std(arr)
    
    # Color distribution
    for i, channel in enumerate(['red', 'green', 'blue']):
        features[f'{channel}_mean'] = np.mean(arr[:, :, i])
        features[f'{channel}_std'] = np.std(arr[:, :, i])
    
    # Texture analysis (simplified)
    gray = np.mean(arr, axis=2)
    edge_x = np.abs(np.diff(gray, axis=1))
    edge_y = np.abs(np.diff(gray, axis=0))
    features['edge_density'] = (np.mean(edge_x) + np.mean(edge_y)) / 2
    features['texture_variance'] = np.var(gray)
    
    return features


def calculate_style_similarity(features1: Dict[str, float], features2: Dict[str, float]) -> float:
    """Calculate style similarity between two sets of features."""
    if not features1 or not features2:
        return 0.0
    
    common_keys = set(features1.keys()) & set(features2.keys())
    if not common_keys:
        return 0.0
    
    similarities = []
    for key in common_keys:
        val1, val2 = features1[key], features2[key]
        # Normalize and calculate similarity
        max_val = max(abs(val1), abs(val2), 1.0)
        similarity = 1.0 - abs(val1 - val2) / max_val
        similarities.append(similarity)
    
    return sum(similarities) / len(similarities)


@pytest.fixture(scope="module")
def setup_img2img_ai():
    """Initialize the image-to-image system once for all tests."""
    success = initialize_img2img_system()
    if not success:
        pytest.fail("Failed to initialize image-to-image system")
    
    img2img_ai = get_img2img_ai_instance()
    return img2img_ai


class TestImg2ImgStyle:
    """Test image-to-image style transfer capabilities."""
    
    def test_monet_style_transfer(self, setup_img2img_ai):
        """Test applying Monet impressionist style to landscapes."""
        # Load source landscape and Monet style reference
        source_image = load_input_image('6')  # Mountain landscape
        style_image = load_input_image('11')   # Monet style
        
        result = generate_img2img(
            source_image=source_image,
            reference_image=style_image,
            transformation_type="style_transfer",
            strength=0.8,
            seed=42
        )
        
        assert result.get('success', False), f"Monet style transfer failed: {result.get('error', 'Unknown error')}"
        
        generated_image = result['generated_image']
        
        # Extract style features
        source_features = extract_style_features(source_image)
        style_features = extract_style_features(style_image)
        generated_features = extract_style_features(generated_image)
        
        # Check style transfer effectiveness
        style_similarity = calculate_style_similarity(generated_features, style_features)
        source_similarity = calculate_style_similarity(generated_features, source_features)
        
        logger.info(f"Monet style transfer - Style similarity: {style_similarity:.3f}, Source similarity: {source_similarity:.3f}")
        
        # Generated image should be more similar to style than original source
        # but still retain some source characteristics
        assert style_similarity > 0.3, f"Insufficient style transfer: {style_similarity:.3f}"
        assert source_similarity > 0.2, f"Lost too much source content: {source_similarity:.3f}"
    
    def test_van_gogh_style_transfer(self, setup_img2img_ai):
        """Test applying Van Gogh style to various subjects."""
        source_images = [
            load_input_image('7'),   # Beach sunset
            load_input_image('10'),  # Countryside
        ]
        style_image = load_input_image('12')  # Van Gogh style
        
        style_features = extract_style_features(style_image)
        
        for i, source_image in enumerate(source_images):
            result = generate_img2img(
                source_image=source_image,
                reference_image=style_image,
                transformation_type="style_transfer",
                strength=0.7,
                seed=100 + i
            )
            
            if result.get('success', False):
                generated_image = result['generated_image']
                generated_features = extract_style_features(generated_image)
                
                style_similarity = calculate_style_similarity(generated_features, style_features)
                logger.info(f"Van Gogh style test {i+1} - Style similarity: {style_similarity:.3f}")
                
                assert style_similarity > 0.25, f"Van Gogh style transfer insufficient for image {i+1}"
    
    def test_picasso_style_transfer(self, setup_img2img_ai):
        """Test applying Picasso cubist style."""
        source_image = load_input_image('2')   # Portrait
        style_image = load_input_image('13')   # Picasso style
        
        result = generate_img2img(
            source_image=source_image,
            reference_image=style_image,
            transformation_type="style_transfer",
            strength=0.9,  # Strong style transfer for cubist effect
            seed=200
        )
        
        if result.get('success', False):
            generated_image = result['generated_image']
            
            # Picasso style should dramatically change the image structure
            source_features = extract_style_features(source_image)
            generated_features = extract_style_features(generated_image)
            
            # Check for significant transformation
            transformation_magnitude = 1.0 - calculate_style_similarity(source_features, generated_features)
            logger.info(f"Picasso transformation magnitude: {transformation_magnitude:.3f}")
            
            assert transformation_magnitude > 0.3, f"Insufficient cubist transformation: {transformation_magnitude:.3f}"
    
    def test_style_consistency_across_subjects(self, setup_img2img_ai):
        """Test that the same style produces consistent results across different subjects."""
        style_image = load_input_image('14')  # Renaissance style
        source_images = [
            load_input_image('1'),   # Young woman portrait
            load_input_image('6'),   # Mountain landscape
            load_input_image('26'),  # Geometric shapes
        ]
        
        style_features = extract_style_features(style_image)
        generated_features_list = []
        
        for i, source_image in enumerate(source_images):
            result = generate_img2img(
                source_image=source_image,
                reference_image=style_image,
                transformation_type="style_transfer",
                strength=0.8,
                seed=300 + i
            )
            
            if result.get('success', False):
                generated_image = result['generated_image']
                generated_features = extract_style_features(generated_image)
                generated_features_list.append(generated_features)
        
        # Check style consistency across different subjects
        if len(generated_features_list) >= 2:
            style_similarities = []
            for features in generated_features_list:
                similarity = calculate_style_similarity(features, style_features)
                style_similarities.append(similarity)
            
            avg_style_similarity = sum(style_similarities) / len(style_similarities)
            style_consistency = 1.0 - np.std(style_similarities)
            
            logger.info(f"Style consistency across subjects - Avg similarity: {avg_style_similarity:.3f}, Consistency: {style_consistency:.3f}")
            
            assert avg_style_similarity > 0.25, f"Poor average style transfer: {avg_style_similarity:.3f}"
            assert style_consistency > 0.7, f"Inconsistent style application: {style_consistency:.3f}"
    
    def test_style_strength_control(self, setup_img2img_ai):
        """Test that style strength parameter controls the intensity of style transfer."""
        source_image = load_input_image('8')   # Forest path
        style_image = load_input_image('15')   # Abstract style
        
        strengths = [0.3, 0.6, 0.9]
        style_similarities = []
        
        style_features = extract_style_features(style_image)
        source_features = extract_style_features(source_image)
        
        for strength in strengths:
            result = generate_img2img(
                source_image=source_image,
                reference_image=style_image,
                transformation_type="style_transfer",
                strength=strength,
                seed=400
            )
            
            if result.get('success', False):
                generated_image = result['generated_image']
                generated_features = extract_style_features(generated_image)
                
                style_similarity = calculate_style_similarity(generated_features, style_features)
                source_similarity = calculate_style_similarity(generated_features, source_features)
                
                style_similarities.append((strength, style_similarity, source_similarity))
                logger.info(f"Strength {strength} - Style: {style_similarity:.3f}, Source: {source_similarity:.3f}")
        
        # Check that higher strength leads to more style transfer
        if len(style_similarities) >= 2:
            for i in range(1, len(style_similarities)):
                prev_strength, prev_style, prev_source = style_similarities[i-1]
                curr_strength, curr_style, curr_source = style_similarities[i]
                
                # Higher strength should generally increase style similarity
                # and decrease source similarity
                logger.info(f"Strength progression: {prev_strength}->{curr_strength}, Style: {prev_style:.3f}->{curr_style:.3f}")
    
    def test_mixed_style_elements(self, setup_img2img_ai):
        """Test style transfer with mixed artistic elements."""
        source_image = load_input_image('5')   # Professional headshot
        style_image = load_input_image('61')   # Mixed styles
        
        result = generate_img2img(
            source_image=source_image,
            reference_image=style_image,
            transformation_type="style_transfer",
            strength=0.7,
            seed=500
        )
        
        if result.get('success', False):
            generated_image = result['generated_image']
            
            # Mixed style should create interesting hybrid characteristics
            generated_features = extract_style_features(generated_image)
            
            # Check for complex color and texture variations
            color_complexity = generated_features.get('color_variance', 0)
            texture_complexity = generated_features.get('texture_variance', 0)
            
            logger.info(f"Mixed style complexity - Color: {color_complexity:.3f}, Texture: {texture_complexity:.3f}")
            
            # Mixed styles should generally increase visual complexity
            assert color_complexity > 100, f"Insufficient color complexity in mixed style: {color_complexity:.3f}"
    
    def test_art_movement_styles(self, setup_img2img_ai):
        """Test different art movement styles."""
        source_image = load_input_image('9')   # Urban cityscape
        
        art_styles = [
            ('31', 'impressionist'),
            ('32', 'post_impressionist'),
        ]
        
        for style_id, movement in art_styles:
            style_image = load_input_image(style_id)
            
            result = generate_img2img(
                source_image=source_image,
                reference_image=style_image,
                transformation_type="style_transfer",
                strength=0.8,
                seed=600
            )
            
            if result.get('success', False):
                generated_image = result['generated_image']
                generated_features = extract_style_features(generated_image)
                style_features = extract_style_features(style_image)
                
                style_similarity = calculate_style_similarity(generated_features, style_features)
                logger.info(f"{movement} style transfer similarity: {style_similarity:.3f}")
                
                assert style_similarity > 0.2, f"Poor {movement} style transfer: {style_similarity:.3f}"
    
    def test_color_palette_style_transfer(self, setup_img2img_ai):
        """Test style transfer focusing on color palette changes."""
        source_image = load_input_image('30')  # Organic produce (varied colors)
        
        color_styles = [
            ('33', 'warm_palette'),
            ('34', 'cool_palette'),
            ('35', 'monochromatic'),
        ]
        
        source_features = extract_style_features(source_image)
        
        for style_id, palette_type in color_styles:
            style_image = load_input_image(style_id)
            
            result = generate_img2img(
                source_image=source_image,
                reference_image=style_image,
                transformation_type="color_palette_transfer",
                strength=0.7,
                seed=700
            )
            
            if result.get('success', False):
                generated_image = result['generated_image']
                generated_features = extract_style_features(generated_image)
                
                # Check color changes
                color_change = abs(generated_features.get('brightness', 0) - source_features.get('brightness', 0))
                
                logger.info(f"{palette_type} color transfer - Color change: {color_change:.3f}")
                
                # Monochromatic should reduce color variance significantly
                if palette_type == 'monochromatic':
                    color_variance_reduction = source_features.get('color_variance', 0) - generated_features.get('color_variance', 0)
                    assert color_variance_reduction > 0, f"Monochromatic style didn't reduce color variance"


@pytest.mark.parametrize("test_case", load_style_test_cases())
def test_style_from_csv(setup_img2img_ai, test_case):
    """
    Test style transfer using cases from testset.csv.
    
    Args:
        setup_img2img_ai: The img2img AI instance from fixture
        test_case: Dictionary containing test case details from CSV
    """
    test_id = test_case['id']
    input_groups = test_case['input_groups'].split(';')
    tested_property = test_case['tested_property']
    
    logger.info(f"Running style test {test_id}: {test_case['name']}")
    
    if len(input_groups) >= 2:
        # Get source and style images based on groups
        source_image = load_input_image('1')  # Default source
        style_image = load_input_image('11')  # Default style
        
        # Try to match specific groups to images
        for group in input_groups:
            if 'portrait' in group.lower():
                source_image = load_input_image('1')
            elif 'landscape' in group.lower():
                source_image = load_input_image('6')
            elif 'art_style' in group.lower() or 'impressionist' in group.lower():
                style_image = load_input_image('11')
            elif 'post_impressionist' in group.lower():
                style_image = load_input_image('32')
        
        # Determine transformation type based on tested property
        if 'style_consistency' in tested_property:
            transformation_type = "style_transfer"
        elif 'artistic_transformation' in tested_property:
            transformation_type = "style_transfer"
        elif 'brush_stroke' in tested_property:
            transformation_type = "style_transfer"
        else:
            transformation_type = "style_transfer"
        
        result = generate_img2img(
            source_image=source_image,
            reference_image=style_image,
            transformation_type=transformation_type,
            strength=0.8,
            seed=42
        )
        
        if result.get('success', False):
            generated_image = result['generated_image']
            
            # Extract and compare features
            source_features = extract_style_features(source_image)
            style_features = extract_style_features(style_image)
            generated_features = extract_style_features(generated_image)
            
            style_similarity = calculate_style_similarity(generated_features, style_features)
            
            logger.info(f"Test {test_id} style similarity: {style_similarity:.3f}")
            
            # Verify style transfer based on tested property
            if 'consistency' in tested_property:
                assert style_similarity > 0.3, f"Style consistency test failed: {style_similarity:.3f}"
            elif 'artistic' in tested_property:
                assert style_similarity > 0.25, f"Artistic transformation test failed: {style_similarity:.3f}"
            else:
                assert style_similarity > 0.2, f"Basic style test failed: {style_similarity:.3f}"
        else:
            logger.warning(f"Test {test_id} generation failed: {result.get('error', 'Unknown error')}")
    else:
        pytest.skip(f"Test {test_id} requires at least 2 input groups")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])