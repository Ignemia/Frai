#!/usr/bin/env python3
"""
Minimal image generation test with the FLUX.1 model.
"""
import logging
import time
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_actual_generation():
    """Test actual image generation with conservative settings."""
    logger.info("Testing actual image generation...")
    
    try:
        from services.image_generator import generate_image
        from services.ai.image_generation.style_presets import StylePreset
        
        # Test with very conservative settings to avoid memory issues
        prompt = "A simple red apple on a white background"
        
        logger.info(f"Generating image with prompt: '{prompt}'")
        logger.info("Using conservative settings: 256x256, 10 steps")
        
        start_time = time.time()
        
        result = generate_image(
            prompt=prompt,
            height=256,  # Small size to avoid memory issues
            width=256,   # Small size to avoid memory issues
            steps=10,    # Few steps for faster generation
            guidance_scale=7.5,
            style=StylePreset.REALISTIC,  # Use realistic style
            session_id=f"test_{int(time.time())}",
            progress_callback=None  # No progress callback for simplicity
        )
        
        generation_time = time.time() - start_time
        
        if result and result[0]:
            logger.info(f"✓ Image generated successfully in {generation_time:.2f} seconds!")
            logger.info(f"  Saved to: {result[0]}")
            logger.info(f"  URL: {result[1]}")
            
            # Check if file actually exists
            if os.path.exists(result[0]):
                file_size = os.path.getsize(result[0])
                logger.info(f"  File size: {file_size:,} bytes")
                return True
            else:
                logger.error("Image file was not created")
                return False
        else:
            logger.error("No image was generated")
            return False
            
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        return False

def main():
    """Run the minimal image generation test."""
    logger.info("=" * 60)
    logger.info("Minimal Image Generation Test")
    logger.info("=" * 60)
    
    # Check if model exists
    model_path = "./models/FLUX.1-dev"
    if not os.path.exists(model_path):
        logger.error(f"FLUX.1 model not found at {model_path}")
        logger.error("Please ensure the model is downloaded and available")
        return False
    
    logger.info(f"✓ Model directory found: {model_path}")
    
    # Run the test
    success = test_actual_generation()
    
    if success:
        logger.info("✅ Image generation test PASSED!")
        logger.info("The system is working correctly with FLUX.1")
    else:
        logger.error("❌ Image generation test FAILED!")
        logger.error("Check the logs above for error details")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
