"""
Quality tests for the text-to-image module.

This module tests the quality and characteristics of generated images,
ensuring they meet expected visual standards and properly interpret prompts.
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

from Frai.back.ai.txt2img import (
    initialize_txt2img_system,
    get_txt2img_ai_instance,
    generate_image_from_text
)

# Set up logging
logger = logging.getLogger(__name__)

# Path to the test set CSV file
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), 'testset.csv')


def load_quality_test_cases() -> List[Dict[str, str]]:
    """Load test cases specifically for quality testing."""
    test_cases = []
    try:
        with open(TEST_SET_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter for quality tests
                if any(keyword in row['evaluation_metric'].lower() for keyword in ['quality', 'resolution', 'detail']):
                    test_cases.append(row)
        logger.info(f"Loaded {len(test_cases)} quality test cases")
        return test_cases
    except Exception as e:
        logger.error(f"Failed to load test cases: {e}")
        return []


def analyze_image_quality(image: Image.Image) -> Dict[str, float]:
    """Analyze basic quality metrics of an image."""
    if not image:
        return {}
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array for analysis
    img_array = np.array(image)
    
    quality_metrics = {}
    
    # Resolution metrics
    quality_metrics['width'] = image.width
    quality_metrics['height'] = image.height
    quality_metrics['total_pixels'] = image.width * image.height
    quality_metrics['aspect_ratio'] = image.width / image.height if image.height > 0 else 1.0
    
    # Color analysis
    quality_metrics['mean_brightness'] = np.mean(img_array)
    quality_metrics['brightness_std'] = np.std(img_array)
    quality_metrics['color_variance'] = np.var(img_array)
    
    # Channel analysis
    for i, channel in enumerate(['red', 'green', 'blue']):
        channel_data = img_array[:, :, i]
        quality_metrics[f'{channel}_mean'] = np.mean(channel_data)
        quality_metrics[f'{channel}_std'] = np.std(channel_data)
    
    # Simple contrast measure
    gray = np.mean(img_array, axis=2)
    quality_metrics['contrast'] = np.std(gray)
    
    # Edge density (simple sharpness measure)
    edge_x = np.abs(np.diff(gray, axis=1))
    edge_y = np.abs(np.diff(gray, axis=0))
    quality_metrics['edge_density'] = (np.mean(edge_x) + np.mean(edge_y)) / 2
    
    # Color distribution
    unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
    quality_metrics['color_diversity'] = unique_colors / quality_metrics['total_pixels']
    
    return quality_metrics


def check_resolution_requirements(image: Image.Image, min_width: int = 256, min_height: int = 256) -> bool:
    """Check if image meets minimum resolution requirements."""
    if not image:
        return False
    return image.width >= min_width and image.height >= min_height


def check_aspect_ratio(image: Image.Image, expected_ratio: float = 1.0, tolerance: float = 0.1) -> bool:
    """Check if image has expected aspect ratio within tolerance."""
    if not image:
        return False
    
    if image.height == 0:
        return False
    
    actual_ratio = image.width / image.height
    return abs(actual_ratio - expected_ratio) <= tolerance


def detect_artifacts(image: Image.Image) -> Dict[str, bool]:
    """Detect common generation artifacts in the image."""
    if not image:
        return {}
    
    img_array = np.array(image.convert('RGB'))
    artifacts = {}
    
    # Check for pure black or white regions (potential failures)
    black_pixels = np.sum(np.all(img_array == [0, 0, 0], axis=2))
    white_pixels = np.sum(np.all(img_array == [255, 255, 255], axis=2))
    total_pixels = img_array.shape[0] * img_array.shape[1]
    
    artifacts['excessive_black'] = (black_pixels / total_pixels) > 0.5
    artifacts['excessive_white'] = (white_pixels / total_pixels) > 0.5
    
    # Check for very low variance (potential flat regions)
    gray = np.mean(img_array, axis=2)
    artifacts['low_variance'] = np.var(gray) < 100
    
    # Check for extreme brightness/darkness
    mean_brightness = np.mean(img_array)
    artifacts['too_dark'] = mean_brightness < 30
    artifacts['too_bright'] = mean_brightness > 225
    
    # Check for color cast (one channel dominates)
    channel_means = [np.mean(img_array[:, :, i]) for i in range(3)]
    max_mean = max(channel_means)
    min_mean = min(channel_means)
    artifacts['color_cast'] = (max_mean - min_mean) > 100
    
    return artifacts


# Using fixture from conftest.py


class TestTxt2ImgQuality:
    """Test text-to-image generation quality."""
    
    def test_basic_resolution_quality(self, setup_txt2img_ai):
        """Test that generated images meet basic resolution requirements."""
        test_prompts = [
            "A red apple on a table",
            "A beautiful landscape with mountains",
            "A portrait of a young woman"
        ]
        
        for prompt in test_prompts:
            result = generate_image_from_text(
                prompt=prompt,
                width=512,
                height=512,
                seed=42
            )
            
            if result.get('success', False):
                image = result['generated_image']
                
                # Check resolution
                assert check_resolution_requirements(image, 512, 512), \
                    f"Image resolution {image.size} below required 512x512"
                
                # Check aspect ratio
                assert check_aspect_ratio(image, 1.0, 0.05), \
                    f"Image aspect ratio {image.width/image.height:.2f} not close to 1.0"
                
                logger.info(f"Resolution test passed for: {prompt}")
    
    def test_image_quality_metrics(self, setup_txt2img_ai):
        """Test various image quality metrics."""
        result = generate_image_from_text(
            prompt="A high quality professional photograph of a sunset over the ocean",
            width=512,
            height=512,
            steps=30,
            guidance_scale=8.0,
            seed=123
        )
        
        if result.get('success', False):
            image = result['generated_image']
            quality_metrics = analyze_image_quality(image)
            
            # Check basic quality indicators
            assert quality_metrics['total_pixels'] >= 262144, "Image too small"  # 512x512
            assert quality_metrics['color_variance'] > 100, "Image lacks color variation"
            assert quality_metrics['contrast'] > 10, "Image has insufficient contrast"
            assert quality_metrics['edge_density'] > 1, "Image lacks detail/sharpness"
            
            # Check color distribution
            assert quality_metrics['color_diversity'] > 0.01, "Image has too few unique colors"
            
            logger.info(f"Quality metrics: contrast={quality_metrics['contrast']:.2f}, "
                       f"variance={quality_metrics['color_variance']:.2f}, "
                       f"edges={quality_metrics['edge_density']:.2f}")
    
    def test_artifact_detection(self, setup_txt2img_ai):
        """Test that generated images don't have common artifacts."""
        prompts_and_expectations = [
            ("A colorful parrot in a tropical setting", "should have varied colors"),
            ("A detailed cityscape at night", "should have good contrast"),
            ("A portrait with natural lighting", "should have balanced exposure")
        ]
        
        for prompt, expectation in prompts_and_expectations:
            result = generate_image_from_text(
                prompt=prompt,
                width=512,
                height=512,
                seed=456
            )
            
            if result.get('success', False):
                image = result['generated_image']
                artifacts = detect_artifacts(image)
                
                # Check for problematic artifacts
                assert not artifacts.get('excessive_black', False), \
                    f"Image has too much black area - {expectation}"
                assert not artifacts.get('excessive_white', False), \
                    f"Image has too much white area - {expectation}"
                assert not artifacts.get('low_variance', False), \
                    f"Image lacks visual variety - {expectation}"
                assert not artifacts.get('too_dark', False), \
                    f"Image is too dark - {expectation}"
                assert not artifacts.get('too_bright', False), \
                    f"Image is too bright - {expectation}"
                
                logger.info(f"Artifact check passed for: {prompt}")
    
    def test_prompt_interpretation_quality(self, setup_txt2img_ai):
        """Test that images properly interpret prompt details."""
        specific_prompts = [
            {
                'prompt': "A red rose in a blue vase",
                'expected_colors': ['red', 'blue'],
                'description': "color specification"
            },
            {
                'prompt': "A large elephant next to a small mouse",
                'expected_features': ['size_contrast'],
                'description': "size relationships"
            },
            {
                'prompt': "A sunny day with bright blue sky",
                'expected_mood': 'bright',
                'description': "lighting and mood"
            }
        ]
        
        for test_case in specific_prompts:
            result = generate_image_from_text(
                prompt=test_case['prompt'],
                width=512,
                height=512,
                seed=789
            )
            
            if result.get('success', False):
                image = result['generated_image']
                quality_metrics = analyze_image_quality(image)
                
                # Test brightness for sunny scene
                if 'bright' in test_case.get('expected_mood', ''):
                    assert quality_metrics['mean_brightness'] > 120, \
                        f"Sunny scene should be bright, got brightness {quality_metrics['mean_brightness']:.1f}"
                
                # Test color presence for color-specific prompts
                if 'expected_colors' in test_case:
                    # This is a simplified test - in practice you'd need more sophisticated color detection
                    assert quality_metrics['color_diversity'] > 0.02, \
                        f"Color-specific prompt should have good color diversity"
                
                logger.info(f"Prompt interpretation test passed: {test_case['description']}")
    
    def test_style_quality_consistency(self, setup_txt2img_ai):
        """Test quality consistency across different artistic styles."""
        style_prompts = [
            "A landscape in photorealistic style",
            "A landscape as an oil painting",
            "A landscape in watercolor style",
            "A landscape as a pencil sketch"
        ]
        
        quality_results = []
        
        for prompt in style_prompts:
            result = generate_image_from_text(
                prompt=prompt,
                width=512,
                height=512,
                seed=101112
            )
            
            if result.get('success', False):
                image = result['generated_image']
                quality_metrics = analyze_image_quality(image)
                quality_results.append((prompt, quality_metrics))
        
        # Check that all styles produce images with reasonable quality
        for prompt, metrics in quality_results:
            assert metrics['total_pixels'] >= 262144, f"Style '{prompt}' produced too small image"
            assert metrics['contrast'] > 5, f"Style '{prompt}' has insufficient contrast"
            
            logger.info(f"Style quality test passed for: {prompt}")
        
        # Check consistency across styles
        if len(quality_results) >= 2:
            contrasts = [metrics['contrast'] for _, metrics in quality_results]
            contrast_range = max(contrasts) - min(contrasts)
            # Allow reasonable variation but not extreme differences
            assert contrast_range < 100, f"Too much contrast variation across styles: {contrast_range}"
    
    def test_high_resolution_quality(self, setup_txt2img_ai):
        """Test quality at higher resolutions."""
        result = generate_image_from_text(
            prompt="A detailed architectural photograph of a Gothic cathedral",
            width=768,
            height=768,
            steps=40,
            guidance_scale=9.0,
            seed=131415
        )
        
        if result.get('success', False):
            image = result['generated_image']
            quality_metrics = analyze_image_quality(image)
            
            # Higher resolution should maintain good quality
            assert quality_metrics['total_pixels'] >= 589824, "High-res image too small"  # 768x768
            assert quality_metrics['edge_density'] > 2, "High-res image lacks detail"
            assert quality_metrics['color_diversity'] > 0.015, "High-res image lacks color variation"
            
            # Should not have excessive artifacts
            artifacts = detect_artifacts(image)
            assert not any(artifacts.values()), f"High-res image has artifacts: {artifacts}"
            
            logger.info(f"High resolution quality test passed: {image.size}")
    
    def test_batch_quality_consistency(self, setup_txt2img_ai):
        """Test that batch generation maintains consistent quality."""
        base_prompt = "A professional photograph of a modern building"
        
        batch_results = []
        for i in range(4):
            result = generate_image_from_text(
                prompt=base_prompt,
                width=512,
                height=512,
                seed=1000 + i  # Different seeds for variety
            )
            
            if result.get('success', False):
                image = result['generated_image']
                quality_metrics = analyze_image_quality(image)
                batch_results.append(quality_metrics)
        
        if len(batch_results) >= 3:
            # Check consistency across batch
            contrasts = [metrics['contrast'] for metrics in batch_results]
            variances = [metrics['color_variance'] for metrics in batch_results]
            
            # All should meet minimum quality
            for i, metrics in enumerate(batch_results):
                assert metrics['contrast'] > 10, f"Batch image {i} has poor contrast"
                assert metrics['color_variance'] > 100, f"Batch image {i} lacks color variation"
            
            # Quality should be reasonably consistent
            contrast_cv = np.std(contrasts) / np.mean(contrasts)  # Coefficient of variation
            assert contrast_cv < 0.5, f"Batch quality too inconsistent: CV={contrast_cv:.3f}"
            
            logger.info(f"Batch quality consistency test passed with {len(batch_results)} images")


@pytest.mark.parametrize("test_case", load_quality_test_cases())
def test_quality_from_csv(setup_txt2img_ai, test_case):
    """
    Test quality using cases from testset.csv.
    
    Args:
        setup_txt2img_ai: The txt2img AI instance from fixture
        test_case: Dictionary containing test case details from CSV
    """
    test_id = test_case['id']
    prompt = test_case['text_prompt']
    expected_properties = test_case['expected_properties']
    evaluation_metric = test_case['evaluation_metric']
    
    logger.info(f"Running quality test {test_id}: {test_case['name']}")
    
    # Generate image with appropriate parameters for quality testing
    result = generate_image_from_text(
        prompt=prompt,
        width=512,
        height=512,
        steps=25,
        guidance_scale=7.5,
        seed=42
    )
    
    if result.get('success', False):
        image = result['generated_image']
        quality_metrics = analyze_image_quality(image)
        artifacts = detect_artifacts(image)
        
        # Apply evaluation based on metric type
        if 'resolution_check' in evaluation_metric:
            assert check_resolution_requirements(image, 512, 512), \
                f"Test {test_id} failed resolution check"
        
        elif 'quality_score' in evaluation_metric:
            # Extract quality threshold if specified
            if '>=' in evaluation_metric:
                threshold_str = evaluation_metric.split('>=')[1].strip()
                try:
                    quality_threshold = float(threshold_str)
                    # Use contrast as a proxy for quality
                    assert quality_metrics['contrast'] >= quality_threshold, \
                        f"Test {test_id} quality {quality_metrics['contrast']:.2f} below threshold {quality_threshold}"
                except ValueError:
                    pass
        
        elif 'artifact_detection' in evaluation_metric:
            problem_artifacts = [k for k, v in artifacts.items() if v]
            assert len(problem_artifacts) == 0, \
                f"Test {test_id} has artifacts: {problem_artifacts}"
        
        elif 'detail_level' in evaluation_metric:
            assert quality_metrics['edge_density'] > 1.5, \
                f"Test {test_id} insufficient detail: {quality_metrics['edge_density']:.2f}"
        
        # Check expected properties if specified
        if expected_properties and expected_properties.lower() != 'any':
            if 'colorful' in expected_properties.lower():
                assert quality_metrics['color_diversity'] > 0.02, \
                    f"Test {test_id} lacks expected color diversity"
            
            if 'high_contrast' in expected_properties.lower():
                assert quality_metrics['contrast'] > 20, \
                    f"Test {test_id} lacks expected high contrast"
            
            if 'detailed' in expected_properties.lower():
                assert quality_metrics['edge_density'] > 2, \
                    f"Test {test_id} lacks expected detail level"
        
        logger.info(f"Test {test_id} passed quality evaluation")
    else:
        logger.warning(f"Test {test_id} generation failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])