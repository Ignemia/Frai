#!/usr/bin/env python3
"""
Basic functionality test for the image generation system.
"""
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        from services.ai.image_generation.flux_generator import check_gpu_memory_for_generation
        logger.info("✓ flux_generator imports OK")
        
        from services.ai.image_generation.memory_manager import get_memory_status, clear_gpu_memory
        logger.info("✓ memory_manager imports OK")
        
        from services.ai.image_generation.style_presets import StylePreset
        logger.info("✓ style_presets imports OK")
        
        from services.image_generator import generate_image, validate_prompt
        logger.info("✓ image_generator imports OK")
        
        return True
    except Exception as e:
        logger.error(f"Import failed: {e}")
        return False

def test_memory_check():
    """Test GPU memory checking."""
    logger.info("Testing GPU memory check...")
    
    try:
        from services.ai.image_generation.flux_generator import check_gpu_memory_for_generation
        
        # Test with small image
        can_gen, msg = check_gpu_memory_for_generation(256, 256, 10)
        logger.info(f"256x256, 10 steps: {can_gen} - {msg}")
        
        # Test with large image
        can_gen, msg = check_gpu_memory_for_generation(1024, 1024, 50)
        logger.info(f"1024x1024, 50 steps: {can_gen} - {msg}")
        
        return True
    except Exception as e:
        logger.error(f"Memory check test failed: {e}")
        return False

def test_memory_status():
    """Test getting memory status."""
    logger.info("Testing memory status...")
    
    try:
        from services.ai.image_generation.memory_manager import get_memory_status
        
        status = get_memory_status()
        logger.info(f"CUDA available: {status.get('cuda_available', False)}")
        
        if status.get('cuda_available'):
            logger.info(f"GPU: {status.get('gpu_name', 'Unknown')}")
            logger.info(f"Total memory: {status.get('memory_total_gb', 0):.2f} GB")
            logger.info(f"Free memory: {status.get('memory_free_gb', 0):.2f} GB")
        
        return True
    except Exception as e:
        logger.error(f"Memory status test failed: {e}")
        return False

def test_style_presets():
    """Test style preset functionality."""
    logger.info("Testing style presets...")
    
    try:
        from services.ai.image_generation.style_presets import StylePreset, enhance_prompt_with_style
        
        # Test all style presets
        test_prompt = "A beautiful landscape"
        
        for style in StylePreset:
            enhanced, negative, params = enhance_prompt_with_style(test_prompt, style)
            logger.info(f"Style {style.value}: Enhanced prompt length = {len(enhanced)}")
        
        return True
    except Exception as e:
        logger.error(f"Style preset test failed: {e}")
        return False

def main():
    """Run all basic functionality tests."""
    logger.info("=" * 50)
    logger.info("Basic Functionality Test Suite")
    logger.info("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Memory Check Test", test_memory_check),
        ("Memory Status Test", test_memory_status),
        ("Style Presets Test", test_style_presets)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if test_func():
                logger.info(f"✓ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info(f"\n{'=' * 50}")
    logger.info(f"Test Results: {passed}/{total} tests passed")
    logger.info(f"{'=' * 50}")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("All tests passed! System is ready for image generation.")
        exit(0)
    else:
        logger.error("Some tests failed. Please check the errors above.")
        exit(1)
