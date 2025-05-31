"""
Prompt interpretation tests for the text-to-image module.

This module tests the txt2img system's ability to understand and interpret
various types of text prompts, from simple objects to complex scenes.
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


def load_prompt_test_cases() -> List[Dict[str, str]]:
    """Load test cases specifically for prompt interpretation testing."""
    test_cases = []
    try:
        with open(TEST_SET_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter for prompt interpretation tests
                if any(keyword in row['evaluation_metric'].lower() for keyword in ['prompt', 'interpretation', 'understanding']):
                    test_cases.append(row)
        logger.info(f"Loaded {len(test_cases)} prompt interpretation test cases")
        return test_cases
    except Exception as e:
        logger.error(f"Failed to load test cases: {e}")
        return []


def analyze_color_presence(image: Image.Image, expected_colors: List[str]) -> Dict[str, bool]:
    """Analyze if expected colors are present in the image."""
    if not image:
        return {}
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    
    # Define color ranges (simplified)
    color_ranges = {
        'red': {'min': [150, 0, 0], 'max': [255, 100, 100]},
        'green': {'min': [0, 150, 0], 'max': [100, 255, 100]},
        'blue': {'min': [0, 0, 150], 'max': [100, 100, 255]},
        'yellow': {'min': [200, 200, 0], 'max': [255, 255, 100]},
        'orange': {'min': [200, 100, 0], 'max': [255, 200, 100]},
        'purple': {'min': [100, 0, 150], 'max': [200, 100, 255]},
        'pink': {'min': [200, 100, 150], 'max': [255, 200, 255]},
        'brown': {'min': [100, 50, 0], 'max': [150, 100, 50]},
        'black': {'min': [0, 0, 0], 'max': [50, 50, 50]},
        'white': {'min': [200, 200, 200], 'max': [255, 255, 255]}
    }
    
    color_presence = {}
    
    for color in expected_colors:
        if color.lower() in color_ranges:
            color_range = color_ranges[color.lower()]
            min_vals = np.array(color_range['min'])
            max_vals = np.array(color_range['max'])
            
            # Check if any pixels fall within the color range
            in_range = np.all((img_array >= min_vals) & (img_array <= max_vals), axis=2)
            pixel_count = np.sum(in_range)
            total_pixels = img_array.shape[0] * img_array.shape[1]
            
            # Consider color present if it covers at least 1% of the image
            color_presence[color] = (pixel_count / total_pixels) > 0.01
        else:
            color_presence[color] = False
    
    return color_presence


def analyze_composition_complexity(image: Image.Image) -> Dict[str, float]:
    """Analyze the compositional complexity of the image."""
    if not image:
        return {}
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    gray = np.mean(img_array, axis=2)
    
    complexity_metrics = {}
    
    # Edge density (indicates detail level)
    edge_x = np.abs(np.diff(gray, axis=1))
    edge_y = np.abs(np.diff(gray, axis=0))
    complexity_metrics['edge_density'] = (np.mean(edge_x) + np.mean(edge_y)) / 2
    
    # Local variance (indicates texture complexity)
    from scipy import ndimage
    local_var = ndimage.generic_filter(gray, np.var, size=3)
    complexity_metrics['texture_complexity'] = np.mean(local_var)
    
    # Color complexity
    unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
    total_pixels = img_array.shape[0] * img_array.shape[1]
    complexity_metrics['color_complexity'] = unique_colors / total_pixels
    
    # Spatial frequency content
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)
    magnitude_spectrum = np.abs(fft_shifted)
    complexity_metrics['frequency_content'] = np.mean(magnitude_spectrum)
    
    return complexity_metrics


# Using fixture from conftest.py


class TestTxt2ImgPromptInterpretation:
    """Test text-to-image prompt interpretation capabilities."""
    
    def test_simple_object_prompts(self, setup_txt2img_ai):
        """Test interpretation of simple object descriptions."""
        simple_prompts = [
            {
                'prompt': "A red apple",
                'expected_colors': ['red'],
                'expected_objects': ['apple']
            },
            {
                'prompt': "A blue car",
                'expected_colors': ['blue'],
                'expected_objects': ['car']
            },
            {
                'prompt': "A yellow flower",
                'expected_colors': ['yellow'],
                'expected_objects': ['flower']
            },
            {
                'prompt': "A green tree",
                'expected_colors': ['green'],
                'expected_objects': ['tree']
            },
            {
                'prompt': "A white house",
                'expected_colors': ['white'],
                'expected_objects': ['house']
            }
        ]
        
        for test_case in simple_prompts:
            result = generate_image_from_text(
                prompt=test_case['prompt'],
                width=512,
                height=512,
                seed=42
            )
            
            if result.get('success', False):
                image = result['generated_image']
                
                # Check color presence
                color_presence = analyze_color_presence(image, test_case['expected_colors'])
                for color in test_case['expected_colors']:
                    assert color_presence.get(color, False), \
                        f"Expected color '{color}' not found in image for prompt: {test_case['prompt']}"
                
                logger.info(f"Simple object test passed: {test_case['prompt']}")
    
    def test_color_attribute_prompts(self, setup_txt2img_ai):
        """Test interpretation of specific color attributes."""
        color_prompts = [
            {
                'prompt': "A bright red sports car",
                'expected_colors': ['red'],
                'color_intensity': 'bright'
            },
            {
                'prompt': "A deep blue ocean",
                'expected_colors': ['blue'],
                'color_intensity': 'deep'
            },
            {
                'prompt': "A pale yellow sunrise",
                'expected_colors': ['yellow'],
                'color_intensity': 'pale'
            },
            {
                'prompt': "A vibrant green forest",
                'expected_colors': ['green'],
                'color_intensity': 'vibrant'
            }
        ]
        
        for test_case in color_prompts:
            result = generate_image_from_text(
                prompt=test_case['prompt'],
                width=512,
                height=512,
                seed=123
            )
            
            if result.get('success', False):
                image = result['generated_image']
                color_presence = analyze_color_presence(image, test_case['expected_colors'])
                
                for color in test_case['expected_colors']:
                    assert color_presence.get(color, False), \
                        f"Expected color '{color}' with intensity '{test_case['color_intensity']}' not found"
                
                logger.info(f"Color attribute test passed: {test_case['prompt']}")
    
    def test_size_and_quantity_prompts(self, setup_txt2img_ai):
        """Test interpretation of size and quantity descriptors."""
        size_quantity_prompts = [
            "A large elephant",
            "A small mouse",
            "A tall building",
            "A wide river",
            "Three red balloons",
            "Many colorful flowers",
            "A single white dove",
            "A huge mountain"
        ]
        
        for prompt in size_quantity_prompts:
            result = generate_image_from_text(
                prompt=prompt,
                width=512,
                height=512,
                seed=456
            )
            
            if result.get('success', False):
                image = result['generated_image']
                complexity = analyze_composition_complexity(image)
                
                # Basic check that image was generated successfully
                assert image is not None, f"Failed to generate image for prompt: {prompt}"
                assert complexity['edge_density'] > 0, f"Image lacks detail for prompt: {prompt}"
                
                logger.info(f"Size/quantity test passed: {prompt}")
    
    def test_spatial_relationship_prompts(self, setup_txt2img_ai):
        """Test interpretation of spatial relationships."""
        spatial_prompts = [
            "A cat on a table",
            "A bird above a tree",
            "A car next to a house",
            "A flower in a vase",
            "A book under a lamp",
            "Mountains behind a lake",
            "A bridge over water",
            "A person inside a room"
        ]
        
        for prompt in spatial_prompts:
            result = generate_image_from_text(
                prompt=prompt,
                width=512,
                height=512,
                seed=789
            )
            
            if result.get('success', False):
                image = result['generated_image']
                complexity = analyze_composition_complexity(image)
                
                # Spatial relationships should create more complex compositions
                assert complexity['edge_density'] > 5, \
                    f"Spatial relationship prompt should create detailed image: {prompt}"
                assert complexity['color_complexity'] > 0.01, \
                    f"Spatial prompt should have varied colors: {prompt}"
                
                logger.info(f"Spatial relationship test passed: {prompt}")
    
    def test_action_and_movement_prompts(self, setup_txt2img_ai):
        """Test interpretation of actions and movement."""
        action_prompts = [
            "A dog running in a park",
            "A person jumping over a fence",
            "A bird flying in the sky",
            "A car driving on a road",
            "A dancer performing on stage",
            "Water flowing down a waterfall",
            "Leaves falling from a tree",
            "A horse galloping in a field"
        ]
        
        for prompt in action_prompts:
            result = generate_image_from_text(
                prompt=prompt,
                width=512,
                height=512,
                seed=101112
            )
            
            if result.get('success', False):
                image = result['generated_image']
                complexity = analyze_composition_complexity(image)
                
                # Action scenes should have good detail and composition
                assert complexity['edge_density'] > 3, \
                    f"Action prompt should create detailed scene: {prompt}"
                
                logger.info(f"Action/movement test passed: {prompt}")
    
    def test_emotion_and_mood_prompts(self, setup_txt2img_ai):
        """Test interpretation of emotional and mood descriptors."""
        mood_prompts = [
            {
                'prompt': "A happy child playing",
                'expected_mood': 'positive'
            },
            {
                'prompt': "A peaceful garden at sunset",
                'expected_mood': 'serene'
            },
            {
                'prompt': "A dramatic stormy sky",
                'expected_mood': 'intense'
            },
            {
                'prompt': "A cozy fireplace scene",
                'expected_mood': 'warm'
            },
            {
                'prompt': "A mysterious forest at night",
                'expected_mood': 'dark'
            }
        ]
        
        for test_case in mood_prompts:
            result = generate_image_from_text(
                prompt=test_case['prompt'],
                width=512,
                height=512,
                seed=131415
            )
            
            if result.get('success', False):
                image = result['generated_image']
                img_array = np.array(image.convert('RGB'))
                mean_brightness = np.mean(img_array)
                
                # Check mood interpretation through brightness
                if test_case['expected_mood'] == 'positive':
                    assert mean_brightness > 100, f"Positive mood should be bright: {test_case['prompt']}"
                elif test_case['expected_mood'] == 'dark':
                    assert mean_brightness < 150, f"Dark mood should be less bright: {test_case['prompt']}"
                elif test_case['expected_mood'] == 'warm':
                    # Check for warm colors (simplified)
                    red_channel = np.mean(img_array[:, :, 0])
                    assert red_channel > 100, f"Warm mood should have warm tones: {test_case['prompt']}"
                
                logger.info(f"Mood test passed: {test_case['prompt']}")
    
    def test_artistic_style_prompts(self, setup_txt2img_ai):
        """Test interpretation of artistic style descriptors."""
        style_prompts = [
            "A portrait in the style of Van Gogh",
            "A landscape as an oil painting",
            "A city scene in watercolor style",
            "A flower in photorealistic style",
            "A mountain in impressionist style",
            "A building in architectural blueprint style",
            "A person in cartoon animation style",
            "A forest in pencil sketch style"
        ]
        
        for prompt in style_prompts:
            result = generate_image_from_text(
                prompt=prompt,
                width=512,
                height=512,
                seed=161718
            )
            
            if result.get('success', False):
                image = result['generated_image']
                complexity = analyze_composition_complexity(image)
                
                # Artistic styles should produce distinctive visual characteristics
                assert image is not None, f"Style prompt failed to generate: {prompt}"
                assert complexity['color_complexity'] > 0.005, \
                    f"Artistic style should have color variation: {prompt}"
                
                logger.info(f"Artistic style test passed: {prompt}")
    
    def test_environmental_context_prompts(self, setup_txt2img_ai):
        """Test interpretation of environmental context."""
        environment_prompts = [
            "A polar bear in the Arctic",
            "A camel in the desert",
            "A fish in coral reef",
            "A skier on snowy mountains",
            "A surfer on ocean waves",
            "A hiker in a rainforest",
            "A cyclist in city streets",
            "A farmer in wheat fields"
        ]
        
        for prompt in environment_prompts:
            result = generate_image_from_text(
                prompt=prompt,
                width=512,
                height=512,
                seed=192021
            )
            
            if result.get('success', False):
                image = result['generated_image']
                complexity = analyze_composition_complexity(image)
                
                # Environmental contexts should create rich, detailed scenes
                assert complexity['edge_density'] > 4, \
                    f"Environmental prompt should create detailed scene: {prompt}"
                assert complexity['texture_complexity'] > 50, \
                    f"Environmental prompt should have textural variety: {prompt}"
                
                logger.info(f"Environmental context test passed: {prompt}")
    
    def test_time_and_lighting_prompts(self, setup_txt2img_ai):
        """Test interpretation of time and lighting conditions."""
        lighting_prompts = [
            {
                'prompt': "A cityscape at dawn",
                'expected_lighting': 'soft'
            },
            {
                'prompt': "A portrait in bright sunlight",
                'expected_lighting': 'bright'
            },
            {
                'prompt': "A landscape at sunset",
                'expected_lighting': 'warm'
            },
            {
                'prompt': "A street scene at night",
                'expected_lighting': 'dark'
            },
            {
                'prompt': "A room with candlelight",
                'expected_lighting': 'dim'
            }
        ]
        
        for test_case in lighting_prompts:
            result = generate_image_from_text(
                prompt=test_case['prompt'],
                width=512,
                height=512,
                seed=222324
            )
            
            if result.get('success', False):
                image = result['generated_image']
                img_array = np.array(image.convert('RGB'))
                mean_brightness = np.mean(img_array)
                contrast = np.std(img_array)
                
                # Check lighting interpretation
                if test_case['expected_lighting'] == 'bright':
                    assert mean_brightness > 140, f"Bright lighting should be bright: {test_case['prompt']}"
                elif test_case['expected_lighting'] == 'dark':
                    assert mean_brightness < 100, f"Night/dark lighting should be dark: {test_case['prompt']}"
                elif test_case['expected_lighting'] == 'dim':
                    assert mean_brightness < 120, f"Dim lighting should be subdued: {test_case['prompt']}"
                
                # All lighting scenarios should have some contrast
                assert contrast > 10, f"Image should have contrast for lighting: {test_case['prompt']}"
                
                logger.info(f"Lighting test passed: {test_case['prompt']}")
    
    def test_complex_multi_element_prompts(self, setup_txt2img_ai):
        """Test interpretation of complex prompts with multiple elements."""
        complex_prompts = [
            "A red sports car parked next to a blue house under a clear sky",
            "A young woman with brown hair reading a book in a green garden",
            "Three white cats sleeping on a wooden table near a window",
            "A tall lighthouse on a rocky cliff overlooking a stormy ocean",
            "An elderly man feeding pigeons in a busy city square at sunset"
        ]
        
        for prompt in complex_prompts:
            result = generate_image_from_text(
                prompt=prompt,
                width=512,
                height=512,
                steps=30,  # More steps for complex prompts
                seed=252627
            )
            
            if result.get('success', False):
                image = result['generated_image']
                complexity = analyze_composition_complexity(image)
                
                # Complex prompts should create rich, detailed compositions
                assert complexity['edge_density'] > 6, \
                    f"Complex prompt should create very detailed image: {prompt}"
                assert complexity['color_complexity'] > 0.02, \
                    f"Complex prompt should have high color variety: {prompt}"
                assert complexity['texture_complexity'] > 100, \
                    f"Complex prompt should have rich textures: {prompt}"
                
                logger.info(f"Complex multi-element test passed: {prompt}")


@pytest.mark.parametrize("test_case", load_prompt_test_cases())
def test_prompt_interpretation_from_csv(setup_txt2img_ai, test_case):
    """
    Test prompt interpretation using cases from testset.csv.
    
    Args:
        setup_txt2img_ai: The txt2img AI instance from fixture
        test_case: Dictionary containing test case details from CSV
    """
    test_id = test_case['id']
    prompt = test_case['text_prompt']
    expected_properties = test_case['expected_properties']
    evaluation_metric = test_case['evaluation_metric']
    
    logger.info(f"Running prompt interpretation test {test_id}: {test_case['name']}")
    
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
        
        # Apply evaluation based on metric type
        if 'prompt_understanding' in evaluation_metric:
            complexity = analyze_composition_complexity(image)
            assert complexity['edge_density'] > 2, \
                f"Test {test_id} prompt not properly understood - insufficient detail"
            assert complexity['color_complexity'] > 0.01, \
                f"Test {test_id} prompt not properly understood - insufficient color variety"
        
        elif 'color_accuracy' in evaluation_metric:
            # Extract expected colors from properties
            if expected_properties and 'color:' in expected_properties:
                color_spec = expected_properties.split('color:')[1].split(',')[0].strip()
                expected_colors = [color_spec]
                color_presence = analyze_color_presence(image, expected_colors)
                
                for color in expected_colors:
                    assert color_presence.get(color, False), \
                        f"Test {test_id} missing expected color: {color}"
        
        elif 'composition_quality' in evaluation_metric:
            complexity = analyze_composition_complexity(image)
            assert complexity['edge_density'] > 4, \
                f"Test {test_id} poor composition quality - edge density {complexity['edge_density']:.2f}"
            assert complexity['texture_complexity'] > 50, \
                f"Test {test_id} poor composition quality - texture complexity {complexity['texture_complexity']:.2f}"
        
        elif 'object_presence' in evaluation_metric:
            # Basic check that something was generated
            complexity = analyze_composition_complexity(image)
            assert complexity['edge_density'] > 1, \
                f"Test {test_id} no clear objects detected"
        
        logger.info(f"Test {test_id} passed prompt interpretation evaluation")
    else:
        logger.warning(f"Test {test_id} generation failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])