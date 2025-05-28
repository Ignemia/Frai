import pytest
import logging
import os
import sys

from ..test_helpers import (
    safe_import_ai_function,
    MockAIInstance,
    expect_implementation_error
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Safe imports of txt2img functions
initialize_txt2img_system = safe_import_ai_function('Frai.back.ai.txt2img', 'initialize_txt2img_system')
get_txt2img_ai_instance = safe_import_ai_function('Frai.back.ai.txt2img', 'get_txt2img_ai_instance')
generate_image_from_text = safe_import_ai_function('Frai.back.ai.txt2img', 'generate_image_from_text')


@pytest.fixture(scope="session")
def setup_txt2img_ai():
    """
    Initialize the text-to-image system once for all tests.
    This fixture is shared across all txt2img test modules.
    """
    try:
        logger.info("Initializing text-to-image system for tests...")
        success = initialize_txt2img_system()
        
        if not success:
            pytest.fail("Failed to initialize text-to-image system")
            return None
        
        # Get the txt2img AI instance
        txt2img_ai = get_txt2img_ai_instance()
        logger.info("Text-to-image system initialized successfully")
        
        yield txt2img_ai
        
        logger.info("Test session complete. Text-to-image AI instance cleanup.")
        
    except Exception as e:
        logger.warning(f"Could not initialize txt2img system: {e}")
        yield MockAIInstance("txt2img")





@pytest.fixture
def sample_prompts():
    """Provide sample prompts for testing different categories."""
    return {
        'simple_objects': [
            "A red apple on a white table",
            "A blue car in a parking lot",
            "A yellow flower in a garden",
            "A brown dog sitting on grass",
            "A white cat sleeping on a sofa"
        ],
        'complex_scenes': [
            "A bustling marketplace in Morocco with colorful spices and textiles",
            "A serene mountain lake at sunset with reflections of pine trees",
            "A cyberpunk cityscape at night with neon lights and flying cars",
            "A medieval castle on a hill surrounded by a moat and forest",
            "A underwater coral reef teeming with tropical fish and sea life"
        ],
        'artistic_styles': [
            "A portrait of a woman in the style of Leonardo da Vinci",
            "A landscape painting in Van Gogh's impressionist style",
            "A cubist interpretation of a city street by Picasso",
            "A surreal dreamscape in the style of Salvador Dali",
            "A Japanese woodblock print of Mount Fuji"
        ],
        'portraits': [
            "A professional headshot of a smiling businessman",
            "An elderly woman with kind eyes and wrinkled hands",
            "A young child laughing while playing with toys",
            "A warrior in medieval armor holding a sword",
            "A scientist in a laboratory wearing safety goggles"
        ],
        'landscapes': [
            "A vast desert with sand dunes under a starry sky",
            "A tropical beach with palm trees and crystal clear water",
            "A snowy mountain peak with clouds swirling around it",
            "A rolling green countryside with sheep and farmhouses",
            "A dense rainforest with exotic birds and waterfalls"
        ],
        'abstract': [
            "Swirling colors representing the emotion of joy",
            "Geometric shapes floating in zero gravity",
            "A explosion of light and energy in deep space",
            "Flowing water transformed into musical notes",
            "The concept of time visualized as interconnected spirals"
        ],
        'fantasy': [
            "A majestic dragon breathing fire over a medieval village",
            "An enchanted forest with glowing mushrooms and fairy lights",
            "A wizard casting a spell in an ancient library",
            "A phoenix rising from ashes against a stormy sky",
            "A crystal cave with magical glowing gems"
        ],
        'technical': [
            "A detailed architectural blueprint of a modern skyscraper",
            "A cross-section diagram of a jet engine",
            "A molecular structure of DNA with accurate scientific detail",
            "A circuit board with electronic components and traces",
            "A anatomical illustration of the human heart"
        ]
    }


@pytest.fixture
def generation_parameters():
    """Provide standard generation parameters for testing."""
    return {
        'basic': {
            'width': 512,
            'height': 512,
            'steps': 20,
            'guidance_scale': 7.5
        },
        'high_quality': {
            'width': 1024,
            'height': 1024,
            'steps': 50,
            'guidance_scale': 10.0
        },
        'fast': {
            'width': 256,
            'height': 256,
            'steps': 10,
            'guidance_scale': 5.0
        },
        'artistic': {
            'width': 768,
            'height': 768,
            'steps': 30,
            'guidance_scale': 12.0
        },
        'experimental': {
            'width': 512,
            'height': 768,
            'steps': 25,
            'guidance_scale': 8.5
        }
    }


@pytest.fixture
def negative_prompts():
    """Provide negative prompts for better generation control."""
    return {
        'general': "blurry, low quality, distorted, deformed, ugly, bad anatomy",
        'portrait': "extra limbs, missing limbs, elongated body, distorted face, asymmetrical features",
        'landscape': "oversaturated, unrealistic colors, floating objects, impossible perspective",
        'artistic': "modern elements in historical scenes, anachronistic objects, inconsistent style",
        'technical': "inaccurate proportions, impossible mechanics, fictional technology"
    }


@pytest.fixture
def style_modifiers():
    """Provide style modifiers for prompt enhancement."""
    return {
        'photography': [
            "professional photography",
            "high resolution",
            "sharp focus",
            "realistic lighting",
            "detailed textures"
        ],
        'painting': [
            "oil painting",
            "watercolor",
            "acrylic paint",
            "canvas texture",
            "artistic brushstrokes"
        ],
        'digital_art': [
            "digital art",
            "concept art",
            "matte painting",
            "photorealistic rendering",
            "CGI illustration"
        ],
        'vintage': [
            "vintage style",
            "retro aesthetic",
            "film grain",
            "aged paper texture",
            "nostalgic mood"
        ],
        'cinematic': [
            "cinematic lighting",
            "dramatic composition",
            "movie still",
            "epic scale",
            "atmospheric mood"
        ]
    }


@pytest.fixture
def quality_metrics():
    """Provide quality assessment metrics."""
    return {
        'resolution': lambda img: img.size if hasattr(img, 'size') else (0, 0),
        'aspect_ratio': lambda img: img.size[0] / img.size[1] if hasattr(img, 'size') and img.size[1] > 0 else 1.0,
        'file_size': lambda img: len(img.fp.read()) if hasattr(img, 'fp') else 0,
        'color_depth': lambda img: len(img.getbands()) if hasattr(img, 'getbands') else 0,
        'has_transparency': lambda img: 'transparency' in img.info if hasattr(img, 'info') else False
    }


@pytest.fixture
def prompt_complexity_levels():
    """Provide prompts organized by complexity level."""
    return {
        'simple': [
            "A cat",
            "Red rose",
            "Blue sky",
            "Green tree",
            "White house"
        ],
        'medium': [
            "A fluffy orange cat sitting on a windowsill",
            "A red rose in a crystal vase on a wooden table",
            "A clear blue sky with white puffy clouds",
            "A large oak tree in a peaceful meadow",
            "A charming white cottage with a red roof"
        ],
        'complex': [
            "A majestic orange tabby cat with bright green eyes sitting regally on an ornate Victorian windowsill, golden sunlight streaming through lace curtains",
            "A single perfect red rose with dewdrops on its petals, arranged in an elegant cut crystal vase, sitting on a polished mahogany table with intricate carvings",
            "A vast azure sky filled with billowing cumulus clouds, painted in the golden light of late afternoon, stretching endlessly over a peaceful landscape",
            "An ancient oak tree with gnarled branches and rich green foliage, standing sentinel in a flower-filled meadow beside a babbling brook",
            "A picturesque white cottage with climbing roses, a thatched roof, and smoke curling from the chimney, nestled in an English countryside setting"
        ]
    }