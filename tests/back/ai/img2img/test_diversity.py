"""
Diversity tests for the image-to-image module.

This module tests that the img2img system can generate diverse images,
ensuring the generator can produce varied outputs for the same inputs.
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


def load_diversity_test_cases() -> List[Dict[str, str]]:
    """Load test cases specifically for diversity testing."""
    test_cases = []
    try:
        with open(TEST_SET_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter for diversity tests
                if 'diversity' in row['tested_property'].lower():
                    test_cases.append(row)
        logger.info(f"Loaded {len(test_cases)} diversity test cases")
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


def calculate_image_diversity(images: List[Image.Image]) -> float:
    """Calculate diversity score across a set of images."""
    if len(images) < 2:
        return 0.0
    
    # Resize all images to same size for comparison
    size = (128, 128)
    resized_images = [img.resize(size) for img in images]
    
    # Convert to numpy arrays
    arrays = [np.array(img) for img in resized_images]
    
    # Calculate pairwise differences
    total_differences = 0
    comparison_count = 0
    
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            # Calculate mean squared difference
            mse = np.mean((arrays[i] - arrays[j]) ** 2)
            total_differences += mse
            comparison_count += 1
    
    # Return average difference normalized to 0-1 scale
    if comparison_count > 0:
        avg_difference = total_differences / comparison_count
        # Normalize by maximum possible difference (255^2)
        diversity_score = min(avg_difference / (255.0 ** 2), 1.0)
        return diversity_score
    
    return 0.0


def extract_visual_features(image: Image.Image) -> Dict[str, float]:
    """Extract visual features for diversity analysis."""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize for consistent feature extraction
    image = image.resize((64, 64))
    arr = np.array(image)
    
    features = {}
    
    # Color statistics
    features['mean_red'] = np.mean(arr[:, :, 0])
    features['mean_green'] = np.mean(arr[:, :, 1])
    features['mean_blue'] = np.mean(arr[:, :, 2])
    features['color_variance'] = np.var(arr)
    
    # Brightness and contrast
    gray = np.mean(arr, axis=2)
    features['brightness'] = np.mean(gray)
    features['contrast'] = np.std(gray)
    
    # Texture features
    edge_x = np.abs(np.diff(gray, axis=1))
    edge_y = np.abs(np.diff(gray, axis=0))
    features['edge_density'] = (np.mean(edge_x) + np.mean(edge_y)) / 2
    features['texture_variance'] = np.var(gray)
    
    # Spatial frequency content
    features['high_freq_content'] = np.mean(edge_x > np.percentile(edge_x, 75)) + np.mean(edge_y > np.percentile(edge_y, 75))
    
    return features


def calculate_feature_diversity(feature_sets: List[Dict[str, float]]) -> float:
    """Calculate diversity based on extracted visual features."""
    if len(feature_sets) < 2:
        return 0.0
    
    feature_keys = list(feature_sets[0].keys())
    total_variance = 0
    
    for key in feature_keys:
        values = [fs[key] for fs in feature_sets]
        variance = np.var(values)
        # Normalize variance (rough approximation)
        normalized_variance = min(variance / 10000.0, 1.0)
        total_variance += normalized_variance
    
    return total_variance / len(feature_keys)


@pytest.fixture(scope="module")
def setup_img2img_ai():
    """Initialize the image-to-image system once for all tests."""
    success = initialize_img2img_system()
    if not success:
        pytest.fail("Failed to initialize image-to-image system")
    
    img2img_ai = get_img2img_ai_instance()
    return img2img_ai


class TestImg2ImgDiversity:
    """Test image-to-image generation diversity."""
    
    def test_random_seed_diversity(self, setup_img2img_ai):
        """Test that different random seeds produce diverse outputs."""
        source_image = load_input_image('1')  # Young woman portrait
        style_image = load_input_image('11')  # Monet style
        
        images = []
        seeds = [42, 123, 456, 789, 1000]
        
        for seed in seeds:
            result = generate_img2img(
                source_image=source_image,
                reference_image=style_image,
                transformation_type="style_transfer",
                strength=0.7,
                seed=seed
            )
            
            if result.get('success', False):
                images.append(result['generated_image'])
        
        if len(images) >= 3:
            diversity_score = calculate_image_diversity(images)
            logger.info(f"Random seed diversity score: {diversity_score:.3f}")
            
            # Different seeds should produce diverse results
            assert diversity_score >= 0.1, f"Insufficient seed diversity: {diversity_score:.3f}"
        else:
            pytest.skip("Insufficient successful generations for diversity test")
    
    def test_style_diversity_single_source(self, setup_img2img_ai):
        """Test diversity when applying different styles to the same source."""
        source_image = load_input_image('6')  # Mountain landscape
        
        style_images = [
            load_input_image('11'),  # Monet
            load_input_image('12'),  # Van Gogh
            load_input_image('13'),  # Picasso
            load_input_image('14'),  # Renaissance
        ]
        
        generated_images = []
        
        for i, style_image in enumerate(style_images):
            result = generate_img2img(
                source_image=source_image,
                reference_image=style_image,
                transformation_type="style_transfer",
                strength=0.8,
                seed=42  # Same seed to focus on style diversity
            )
            
            if result.get('success', False):
                generated_images.append(result['generated_image'])
        
        if len(generated_images) >= 3:
            diversity_score = calculate_image_diversity(generated_images)
            logger.info(f"Style diversity score: {diversity_score:.3f}")
            
            # Different styles should create high diversity
            assert diversity_score >= 0.2, f"Insufficient style diversity: {diversity_score:.3f}"
    
    def test_transformation_type_diversity(self, setup_img2img_ai):
        """Test diversity across different transformation types."""
        source_image = load_input_image('2')  # Middle-aged man portrait
        reference_image = load_input_image('11')  # Art style reference
        
        transformation_types = [
            "style_transfer",
            "face_swap",
            "background_replacement",
            "color_palette_transfer",
            "texture_transfer"
        ]
        
        generated_images = []
        
        for transform_type in transformation_types:
            result = generate_img2img(
                source_image=source_image,
                reference_image=reference_image,
                transformation_type=transform_type,
                strength=0.7,
                seed=100
            )
            
            if result.get('success', False):
                generated_images.append(result['generated_image'])
        
        if len(generated_images) >= 3:
            diversity_score = calculate_image_diversity(generated_images)
            logger.info(f"Transformation type diversity score: {diversity_score:.3f}")
            
            # Different transformation types should be very diverse
            assert diversity_score >= 0.15, f"Insufficient transformation diversity: {diversity_score:.3f}"
    
    def test_strength_parameter_diversity(self, setup_img2img_ai):
        """Test diversity across different strength parameters."""
        source_image = load_input_image('8')  # Forest path
        style_image = load_input_image('15')  # Abstract style
        
        strengths = [0.2, 0.5, 0.8]
        generated_images = []
        
        for strength in strengths:
            result = generate_img2img(
                source_image=source_image,
                reference_image=style_image,
                transformation_type="style_transfer",
                strength=strength,
                seed=200
            )
            
            if result.get('success', False):
                generated_images.append(result['generated_image'])
        
        if len(generated_images) >= 2:
            diversity_score = calculate_image_diversity(generated_images)
            logger.info(f"Strength parameter diversity score: {diversity_score:.3f}")
            
            # Different strengths should create some diversity
            assert diversity_score >= 0.05, f"Insufficient strength diversity: {diversity_score:.3f}"
    
    def test_source_image_diversity(self, setup_img2img_ai):
        """Test diversity when using different source images with same style."""
        source_images = [
            load_input_image('1'),   # Young woman
            load_input_image('2'),   # Middle-aged man
            load_input_image('3'),   # Child
            load_input_image('4'),   # Elderly woman
        ]
        style_image = load_input_image('12')  # Van Gogh style
        
        generated_images = []
        
        for source_image in source_images:
            result = generate_img2img(
                source_image=source_image,
                reference_image=style_image,
                transformation_type="style_transfer",
                strength=0.7,
                seed=300
            )
            
            if result.get('success', False):
                generated_images.append(result['generated_image'])
        
        if len(generated_images) >= 3:
            diversity_score = calculate_image_diversity(generated_images)
            logger.info(f"Source image diversity score: {diversity_score:.3f}")
            
            # Different source images should maintain diversity even with same style
            assert diversity_score >= 0.15, f"Insufficient source diversity: {diversity_score:.3f}"
    
    def test_background_replacement_diversity(self, setup_img2img_ai):
        """Test diversity in background replacement transformations."""
        source_image = load_input_image('5')  # Professional headshot
        
        background_images = [
            load_input_image('16'),  # White studio
            load_input_image('17'),  # Outdoor nature
            load_input_image('18'),  # Office interior
            load_input_image('19'),  # Gradient background
        ]
        
        generated_images = []
        
        for bg_image in background_images:
            result = generate_img2img(
                source_image=source_image,
                reference_image=bg_image,
                transformation_type="background_replacement",
                strength=0.8,
                seed=400
            )
            
            if result.get('success', False):
                generated_images.append(result['generated_image'])
        
        if len(generated_images) >= 3:
            diversity_score = calculate_image_diversity(generated_images)
            logger.info(f"Background diversity score: {diversity_score:.3f}")
            
            # Different backgrounds should create diverse results
            assert diversity_score >= 0.12, f"Insufficient background diversity: {diversity_score:.3f}"
    
    def test_texture_transfer_diversity(self, setup_img2img_ai):
        """Test diversity in texture transfer applications."""
        source_image = load_input_image('27')  # Household items
        
        texture_images = [
            load_input_image('21'),  # Wood grain
            load_input_image('22'),  # Marble surface
            load_input_image('23'),  # Fabric weave
            load_input_image('24'),  # Brushed metal
        ]
        
        generated_images = []
        feature_sets = []
        
        for texture_image in texture_images:
            result = generate_img2img(
                source_image=source_image,
                reference_image=texture_image,
                transformation_type="texture_transfer",
                strength=0.6,
                seed=500
            )
            
            if result.get('success', False):
                img = result['generated_image']
                generated_images.append(img)
                feature_sets.append(extract_visual_features(img))
        
        if len(generated_images) >= 3:
            diversity_score = calculate_image_diversity(generated_images)
            feature_diversity = calculate_feature_diversity(feature_sets)
            
            logger.info(f"Texture diversity score: {diversity_score:.3f}")
            logger.info(f"Texture feature diversity: {feature_diversity:.3f}")
            
            # Different textures should create diverse surface appearances
            assert diversity_score >= 0.1, f"Insufficient texture diversity: {diversity_score:.3f}"
            assert feature_diversity >= 0.08, f"Insufficient texture feature diversity: {feature_diversity:.3f}"
    
    def test_creative_interpretation_diversity(self, setup_img2img_ai):
        """Test diversity in creative interpretation of ambiguous references."""
        source_image = load_input_image('26')  # Geometric shapes
        reference_image = load_input_image('61')  # Mixed styles
        
        # Generate multiple variations with different seeds
        generated_images = []
        seeds = [600, 700, 800, 900, 1000]
        
        for seed in seeds:
            result = generate_img2img(
                source_image=source_image,
                reference_image=reference_image,
                transformation_type="style_transfer",
                strength=0.9,  # High strength for creative interpretation
                seed=seed
            )
            
            if result.get('success', False):
                generated_images.append(result['generated_image'])
        
        if len(generated_images) >= 4:
            diversity_score = calculate_image_diversity(generated_images)
            logger.info(f"Creative interpretation diversity score: {diversity_score:.3f}")
            
            # Creative interpretations should be highly diverse
            assert diversity_score >= 0.15, f"Insufficient creative diversity: {diversity_score:.3f}"
    
    def test_batch_generation_diversity(self, setup_img2img_ai):
        """Test diversity when generating multiple images in sequence."""
        source_image = load_input_image('75')  # Natural landscape
        style_image = load_input_image('31')   # Impressionist landscape
        
        # Generate a batch of images with random seeds
        generated_images = []
        num_generations = 6
        
        for i in range(num_generations):
            result = generate_img2img(
                source_image=source_image,
                reference_image=style_image,
                transformation_type="style_transfer",
                strength=0.7,
                seed=None  # Random seed
            )
            
            if result.get('success', False):
                generated_images.append(result['generated_image'])
        
        if len(generated_images) >= 4:
            diversity_score = calculate_image_diversity(generated_images)
            logger.info(f"Batch generation diversity score: {diversity_score:.3f}")
            
            # Batch generations should maintain good diversity
            assert diversity_score >= 0.1, f"Insufficient batch diversity: {diversity_score:.3f}"


@pytest.mark.parametrize("test_case", load_diversity_test_cases())
def test_diversity_from_csv(setup_img2img_ai, test_case):
    """
    Test diversity using cases from testset.csv.
    
    Args:
        setup_img2img_ai: The img2img AI instance from fixture
        test_case: Dictionary containing test case details from CSV
    """
    test_id = test_case['id']
    input_groups = test_case['input_groups'].split(';')
    tested_property = test_case['tested_property']
    
    logger.info(f"Running diversity test {test_id}: {test_case['name']}")
    
    if len(input_groups) >= 2:
        # Load appropriate images based on groups
        source_image = load_input_image('1')  # Default source
        reference_image = load_input_image('11')  # Default reference
        
        # Match groups to specific images
        for group in input_groups:
            if 'portrait' in group.lower():
                source_image = load_input_image('1')
            elif 'landscape' in group.lower():
                source_image = load_input_image('6')
            elif 'art_style' in group.lower():
                reference_image = load_input_image('11')
        
        # Generate multiple variations for diversity testing
        generated_images = []
        num_variations = 4
        
        for i in range(num_variations):
            result = generate_img2img(
                source_image=source_image,
                reference_image=reference_image,
                transformation_type="style_transfer",
                strength=0.7,
                seed=1000 + i * 100  # Different seeds for diversity
            )
            
            if result.get('success', False):
                generated_images.append(result['generated_image'])
        
        if len(generated_images) >= 3:
            diversity_score = calculate_image_diversity(generated_images)
            logger.info(f"Test {test_id} diversity score: {diversity_score:.3f}")
            
            # Diversity threshold based on tested property
            if 'output_diversity' in tested_property:
                assert diversity_score >= 0.12, f"High diversity test failed: {diversity_score:.3f}"
            else:
                assert diversity_score >= 0.08, f"Basic diversity test failed: {diversity_score:.3f}"
        else:
            logger.warning(f"Test {test_id} could not generate sufficient results for diversity analysis")
    else:
        pytest.skip(f"Test {test_id} requires at least 2 input groups")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])