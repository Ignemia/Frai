#!/usr/bin/env python3
from typing import List
"""
Simplified debug version of the image generator to isolate the hanging issue.
"""
import logging
import time
import os
import torch
from typing import Optional, Tuple

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_generate_image(prompt: str = "A simple test image") -> Tuple[Optional[str], Optional[str]]:
    """
    Simplified image generation for debugging the hanging issue.
    """
    logger.info("=== DEBUG IMAGE GENERATION START ===")
    
    # Check CUDA availability
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Model path
    model_path = "./models/FLUX.1-dev"
    logger.info(f"Model path: {model_path}")
    logger.info(f"Model exists: {os.path.exists(model_path)}")
    
    if not os.path.exists(model_path):
        logger.error("Model path does not exist!")
        return None, None
    
    # List model files
    try:
        model_files = os.listdir(model_path)
        logger.info(f"Model directory contains {len(model_files)} files")
        for f in model_files[:10]:  # Show first 10 files
            logger.info(f"  - {f}")
    except Exception as e:
        logger.error(f"Error listing model files: {e}")
        return None, None
    
    # Try importing diffusers
    logger.info("Importing diffusers...")
    try:
        from diffusers import FluxPipeline
        logger.info("✓ Diffusers imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import diffusers: {e}")
        return None, None
    
    # Try loading the pipeline
    logger.info("Loading Flux pipeline...")
    load_start = time.time()
    
    try:        # Load with minimal configuration
        pipeline = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        
        load_time = time.time() - load_start
        logger.info(f"✓ Pipeline loaded in {load_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"✗ Failed to load pipeline: {e}")
        return None, None
    
    # Try a simple generation with minimal parameters
    logger.info("Starting simple generation...")
    gen_start = time.time()
    
    try:
        # Use very simple parameters
        result = pipeline(
            prompt=prompt,
            height=512,  # Smaller size for faster generation
            width=512,
            num_inference_steps=10,  # Fewer steps for faster generation
            guidance_scale=3.5  # Lower guidance for faster generation
        )
        
        gen_time = time.time() - gen_start
        logger.info(f"✓ Generation completed in {gen_time:.2f} seconds")
        
        # Save the image
        if hasattr(result, 'images') and len(result.images) > 0:
            image = result.images[0]
            filename = f"debug_image_{int(time.time())}.png"
            filepath = os.path.join("./outputs", filename)
            
            # Ensure output directory exists
            os.makedirs("./outputs", exist_ok=True)
            
            image.save(filepath)
            logger.info(f"✓ Image saved to {filepath}")
            return filepath, filename
        else:
            logger.error("✗ No image in result")
            return None, None
            
    except Exception as e:
        logger.error(f"✗ Generation failed: {e}")
        return None, None
    
    finally:
        # Clean up
        try:
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("✓ Cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

if __name__ == "__main__":
    logger.info("Starting debug image generation...")
    result = debug_generate_image("A beautiful mountain landscape")
    
    if result[0]:
        logger.info(f"🎉 Success! Image generated: {result[0]}")
    else:
        logger.error("❌ Failed to generate image")
