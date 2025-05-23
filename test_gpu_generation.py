#!/usr/bin/env python3
"""
Test GPU-prioritized image generation with FLUX.1.
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

def test_gpu_generation():
    """Test GPU-prioritized image generation."""
    logger.info("Testing GPU-prioritized image generation...")
    
    try:
        from services.image_generator import generate_image
        from services.ai.image_generation.style_presets import StylePreset
        from services.ai.image_generation.memory_manager import get_memory_status
        
        # Check initial memory status
        logger.info("=== Initial Memory Status ===")
        memory_status = get_memory_status()
        for key, value in memory_status.items():
            logger.info(f"{key}: {value}")
        
        # Test with small image first (should work on GPU)
        prompt = "A beautiful sunset over mountains, digital art"
        
        logger.info(f"Generating small image (256x256) with prompt: '{prompt}'")
        
        start_time = time.time()
        
        result = generate_image(
            prompt=prompt,
            height=256,  # Small size for GPU
            width=256,   # Small size for GPU
            steps=15,    # Moderate steps
            guidance_scale=7.5,
            style=StylePreset.REALISTIC,
            session_id=f"gpu_test_{int(time.time())}",
            progress_callback=None
        )
        
        generation_time = time.time() - start_time
        
        if result and result[0]:
            logger.info(f"✅ Small image generated successfully in {generation_time:.2f} seconds!")
            logger.info(f"  Saved to: {result[0]}")
            
            # Check if file exists and get size
            if os.path.exists(result[0]):
                file_size = os.path.getsize(result[0])
                logger.info(f"  File size: {file_size:,} bytes")
            
            # Check memory status after generation
            logger.info("=== Memory Status After Small Image ===")
            memory_status = get_memory_status()
            for key, value in memory_status.items():
                logger.info(f"{key}: {value}")
            
            # Test with medium image (might work on GPU with 12GB+)
            logger.info(f"Generating medium image (512x512) with prompt: '{prompt}'")
            
            start_time = time.time()
            
            result2 = generate_image(
                prompt=prompt,
                height=512,  # Medium size
                width=512,   # Medium size
                steps=15,    # Moderate steps
                guidance_scale=7.5,
                style=StylePreset.REALISTIC,
                session_id=f"gpu_test_medium_{int(time.time())}",
                progress_callback=None
            )
            
            generation_time = time.time() - start_time
            
            if result2 and result2[0]:
                logger.info(f"✅ Medium image also generated successfully in {generation_time:.2f} seconds!")
                logger.info(f"  Saved to: {result2[0]}")
            else:
                logger.warning("❌ Medium image generation failed (possibly due to memory constraints)")
            
            return True
        else:
            logger.error("❌ Small image generation failed")
            return False
            
    except Exception as e:
        logger.error(f"GPU generation test failed: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the GPU generation test."""
    logger.info("=" * 60)
    logger.info("GPU-Prioritized Image Generation Test")
    logger.info("=" * 60)
    
    # Check if model exists
    model_path = "./models/FLUX.1-dev"
    if not os.path.exists(model_path):
        logger.error(f"FLUX.1 model not found at {model_path}")
        return False
    
    logger.info(f"✓ Model directory found: {model_path}")
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            total_memory = props.total_memory / 1e9
            available_memory = torch.cuda.mem_get_info()[0] / 1e9
            logger.info(f"✓ GPU Memory: {available_memory:.2f}GB available / {total_memory:.2f}GB total")
            
            if available_memory < 6.0:
                logger.warning("⚠️  Low GPU memory detected. Generation may fail or be slow.")
            elif available_memory >= 8.0:
                logger.info("✅ Excellent! Plenty of GPU memory for high-quality generation.")
            else:
                logger.info("✅ Good GPU memory for standard generation.")
        else:
            logger.error("❌ CUDA not available")
            return False
    except Exception as e:
        logger.error(f"Error checking CUDA: {e}")
        return False
    
    # Run the test
    success = test_gpu_generation()
    
    if success:
        logger.info("🎉 GPU-prioritized generation test PASSED!")
        logger.info("The system is working correctly with GPU prioritization")
    else:
        logger.error("💥 GPU-prioritized generation test FAILED!")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
