import pytest
import logging
import os
import sys

# Add project root to path


from ..test_helpers import (
    safe_import_ai_function,
    MockAIInstance,
    expect_implementation_error
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Safe imports of img2img functions
initialize_img2img_system = safe_import_ai_function('Frai.back.ai.img2img', 'initialize_img2img_system')
get_img2img_ai_instance = safe_import_ai_function('Frai.back.ai.img2img', 'get_img2img_ai_instance')
generate_img2img = safe_import_ai_function('Frai.back.ai.img2img', 'generate_img2img')


@pytest.fixture(scope="session")
def setup_img2img_ai():
    """
    Initialize the image-to-image system once for all tests.
    This fixture is shared across all img2img test modules.
    """
    try:
        logger.info("Initializing image-to-image system for tests...")
        success = initialize_img2img_system()
        
        if not success:
            pytest.fail("Failed to initialize image-to-image system")
            return None
        
        # Get the img2img AI instance
        img2img_ai = get_img2img_ai_instance()
        logger.info("Image-to-image system initialized successfully")
        
        yield img2img_ai
        
        logger.info("Test session complete. Image-to-image AI instance cleanup.")
        
    except Exception as e:
        logger.warning(f"Could not initialize img2img system: {e}")
        yield MockAIInstance("img2img")


@pytest.fixture
def img2img_response():
    """
    Fixture to generate responses from the image-to-image system.
    Provides a function that can be called with image inputs.
    """
    def _generate_img2img(source_image, reference_image, transformation_type, **kwargs):
        """
        Generate image-to-image transformation.
        
        Args:
            source_image: Source image path or data
            reference_image: Reference image path or data
            transformation_type: Type of transformation
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary
        """
        try:
            response = generate_img2img(
                source_image=source_image,
                reference_image=reference_image,
                transformation_type=transformation_type,
                **kwargs
            )
            return response
        except Exception as e:
            return {
                'success': False,
                'error': f'Generation failed: {str(e)}'
            }
    
    return _generate_img2img


@pytest.fixture
def sample_image_paths():
    """Provide sample image paths for testing."""
    return {
        'portraits': [
            "Frai/tests/back/ai/img2img/inputs/portraits/young_woman.jpg",
            "Frai/tests/back/ai/img2img/inputs/portraits/middle_aged_man.jpg",
            "Frai/tests/back/ai/img2img/inputs/portraits/child_smiling.jpg"
        ],
        'landscapes': [
            "Frai/tests/back/ai/img2img/inputs/landscapes/mountain_landscape.jpg",
            "Frai/tests/back/ai/img2img/inputs/landscapes/beach_sunset.jpg",
            "Frai/tests/back/ai/img2img/inputs/landscapes/forest_path.jpg"
        ],
        'art_styles': [
            "Frai/tests/back/ai/img2img/inputs/styles/monet_style.jpg",
            "Frai/tests/back/ai/img2img/inputs/styles/van_gogh_style.jpg",
            "Frai/tests/back/ai/img2img/inputs/styles/picasso_style.jpg"
        ],
        'backgrounds': [
            "Frai/tests/back/ai/img2img/inputs/backgrounds/white_studio.jpg",
            "Frai/tests/back/ai/img2img/inputs/backgrounds/outdoor_nature.jpg",
            "Frai/tests/back/ai/img2img/inputs/backgrounds/office_interior.jpg"
        ]
    }


@pytest.fixture
def transformation_types():
    """Provide available transformation types for testing."""
    return [
        "style_transfer",
        "face_swap",
        "background_replacement",
        "color_palette_transfer",
        "texture_transfer",
        "lighting_adaptation",
        "seasonal_transformation",
        "age_progression",
        "cartoon_to_realistic",
        "realistic_to_cartoon"
    ]