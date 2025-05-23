"""
Image-to-image generation capabilities for style transfer and recreation.

This module provides functionality to generate new images based on existing images,
including style transfer, recreation with different styles, and image enhancement.
"""
import logging
import os
import time
import threading
from typing import Dict, Optional, Any, Tuple, Callable, Union
from pathlib import Path
import torch
from PIL import Image
import numpy as np

from services.config import get_config
from .memory_manager import (
    clear_gpu_memory, move_pipeline_to_gpu, move_pipeline_to_cpu,
    update_access_time, schedule_offload, apply_memory_optimizations,
    get_memory_status
)
from .progress_tracker import create_progress_callback, unregister_progress_callback
from .style_presets import StylePreset, enhance_prompt_with_style

logger = logging.getLogger(__name__)

# Load configuration
config = get_config()

# Constants for image-to-image generation
MODEL_PATH = config.get("models", {}).get("flux_image_path", "./models/FLUX.1-dev")
OUTPUT_DIR = "./outputs"

# Global pipeline objects
_img2img_pipeline = None
_pipeline_lock = threading.Lock()

def _get_img2img_pipeline():
    """
    Get or initialize the Flux image-to-image pipeline.
    
    Returns:
        The Flux image-to-image pipeline object or None if initialization failed
    """
    global _img2img_pipeline
    
    # If we already have a pipeline, update access time and ensure it's ready
    if _img2img_pipeline is not None:
        update_access_time()
        return _img2img_pipeline
    
    # Use a lock to ensure only one thread initializes the pipeline
    with _pipeline_lock:
        # Check again in case another thread initialized it while we were waiting
        if _img2img_pipeline is not None:
            update_access_time()
            return _img2img_pipeline
        
        # Initialize the pipeline
        try:
            logger.info("Initializing Flux.1 image-to-image pipeline")
            start_time = time.time()
            
            # Import here to avoid loading dependencies unless needed
            from diffusers import FluxImg2ImgPipeline
            
            # Check if model path exists
            if not os.path.exists(MODEL_PATH):
                logger.error(f"Flux.1 model directory not found at {MODEL_PATH}")
                return None
            
            # Log GPU information if available
            if torch.cuda.is_available():
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                props = torch.cuda.get_device_properties(0)
                logger.info(f"Total GPU memory: {props.total_memory / 1e9:.2f} GB")
                logger.info(f"Available GPU memory: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")
                
                # Clear CUDA cache before loading
                clear_gpu_memory()
            else:
                logger.warning("CUDA not available, using CPU (will be slow)")
            
            # Load the image-to-image pipeline with optimizations
            logger.info("Loading Flux.1 image-to-image pipeline with memory optimizations")
            
            _img2img_pipeline = FluxImg2ImgPipeline.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=None,  # Load on CPU first
                low_cpu_mem_usage=True,
                use_safetensors=True,
                variant="fp16" if torch.cuda.is_available() else None,
                use_auth_token=False,
                local_files_only=True
            )
            
            # Move to CPU explicitly and apply optimizations
            _img2img_pipeline = _img2img_pipeline.to("cpu")
            _img2img_pipeline = apply_memory_optimizations(_img2img_pipeline)
            
            logger.info(f"Flux.1 image-to-image model loaded in {time.time() - start_time:.2f} seconds")
            update_access_time()
            return _img2img_pipeline
            
        except Exception as e:
            logger.error(f"Error initializing Flux.1 image-to-image pipeline: {e}", exc_info=True)
            return None

def load_and_prepare_image(image_path: str, target_size: Tuple[int, int] = None) -> Optional[Image.Image]:
    """
    Load and prepare an image for image-to-image generation.
    
    Args:
        image_path: Path to the source image
        target_size: Optional target size (width, height) for resizing
        
    Returns:
        PIL Image object or None if loading failed
    """
    try:
        # Load the image
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if target size is specified
        if target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        logger.info(f"Loaded image from {image_path}, size: {image.size}")
        return image
        
    except Exception as e:
        logger.error(f"Error loading image from {image_path}: {e}")
        return None

def generate_image_to_image(
    prompt: str,
    source_image_path: str,
    strength: float = 0.7,
    height: int = 1024,
    width: int = 1024,
    steps: int = 35,
    guidance_scale: float = 7.0,
    negative_prompt: str = None,
    style: StylePreset = None,
    session_id: str = None,
    progress_callback: Optional[Callable] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate an image based on an existing image using Flux.1 image-to-image.
    
    Args:
        prompt: The text prompt for image generation
        source_image_path: Path to the source image
        strength: How much to transform the image (0.0 = no change, 1.0 = completely new)
        height: Target image height in pixels
        width: Target image width in pixels
        steps: Number of inference steps
        guidance_scale: Guidance scale for generation
        negative_prompt: Negative prompt to guide what to avoid
        style: Optional style preset to apply
        session_id: Optional session ID for progress tracking
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Tuple containing:
            - Path to the generated image if successful, None otherwise
            - URL or relative path to the image for displaying
    """
    logger.info(f"Starting image-to-image generation: '{prompt[:50]}...'")
    start_time = time.time()
    
    # Load and prepare the source image
    source_image = load_and_prepare_image(source_image_path, (width, height))
    if source_image is None:
        logger.error(f"Failed to load source image: {source_image_path}")
        return None, None
    
    # Apply style preset if specified
    if style:
        enhanced_prompt, style_negative, style_params = enhance_prompt_with_style(prompt, style)
        prompt = enhanced_prompt
        if style_negative and not negative_prompt:
            negative_prompt = style_negative
        elif style_negative and negative_prompt:
            negative_prompt = f"{negative_prompt}, {style_negative}"
        
        # Use style-specific parameters
        guidance_scale = style_params.get("guidance_scale", guidance_scale)
        steps = style_params.get("steps", steps)
    
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logger.info(f"Created output directory at {OUTPUT_DIR}")
    
    # Get the pipeline
    pipeline = _get_img2img_pipeline()
    if pipeline is None:
        logger.error("Failed to initialize Flux.1 image-to-image pipeline")
        return None, None
    
    # Set up progress tracking
    progress_cb = None
    if session_id:
        progress_cb = create_progress_callback(
            session_id=session_id,
            total_steps=steps,
            external_callback=progress_callback,
            save_checkpoints=False
        )
    
    try:
        # Move pipeline to GPU for inference
        pipeline = move_pipeline_to_gpu(pipeline)
        
        # Generate the image
        logger.info(f"Generating image-to-image with {steps} steps, strength {strength}")
        
        generation_kwargs = {
            "prompt": prompt,
            "image": source_image,
            "strength": strength,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
        }
        
        if negative_prompt:
            generation_kwargs["negative_prompt"] = negative_prompt
        
        # Add progress callback if available
        if progress_cb:
            try:
                def simple_callback(step, timestep, latents):
                    progress_cb(step, steps, step / steps, time.time() - start_time)
                
                generation_kwargs["callback"] = simple_callback
                generation_kwargs["callback_steps"] = 1
                logger.debug("Progress callback configured for image-to-image")
            except Exception as e:
                logger.warning(f"Progress callback not supported: {e}")
        
        logger.info("Starting Flux.1 image-to-image generation...")
        generation_start = time.time()
        
        # Add timeout protection for the generation
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Image-to-image generation timed out after 5 minutes")
        
        # Set 5-minute timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(300)  # 5 minutes
        
        try:
            with torch.inference_mode():
                result = pipeline(**generation_kwargs)
        finally:
            signal.alarm(0)  # Cancel the alarm
            signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
        
        generation_time = time.time() - generation_start
        logger.info(f"Image-to-image generation completed in {generation_time:.2f} seconds")
        
        # Process the result
        if hasattr(result, 'images') and len(result.images) > 0:
            image = result.images[0]
            
            # Generate filename
            safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '_')).strip()
            safe_prompt = safe_prompt.replace(' ', '_')
            timestamp = int(time.time())
            filename = f"img2img_{safe_prompt}_{timestamp}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            # Save the image
            image.save(filepath)
            logger.info(f"Image-to-image result saved to {filepath}")
            
            # Schedule pipeline offload after a delay
            schedule_offload(pipeline)
            
            # Clean up progress tracking
            if session_id and progress_cb:
                unregister_progress_callback(session_id)
            
            total_time = time.time() - start_time
            logger.info(f"Image-to-image generation completed in {total_time:.2f} seconds")
            
            return filepath, filename
        else:
            logger.error("No image generated in image-to-image result")
            return None, None
            
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"GPU out of memory during image-to-image generation: {e}")
        logger.info("Try reducing image size, steps, or strength")
        
        # Move pipeline back to CPU and clear memory
        move_pipeline_to_cpu(pipeline)
        
        return None, None
        
    except Exception as e:
        logger.error(f"Error during image-to-image generation: {e}", exc_info=True)
        
        # Clean up progress tracking
        if session_id:
            unregister_progress_callback(session_id)
        
        return None, None

def style_transfer_image(
    source_image_path: str,
    target_style: StylePreset,
    strength: float = 0.8,
    session_id: str = None,
    progress_callback: Optional[Callable] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Apply a style transfer to an existing image.
    
    Args:
        source_image_path: Path to the source image
        target_style: Style preset to apply
        strength: How strongly to apply the style (0.0 = no change, 1.0 = full style)
        session_id: Optional session ID for progress tracking
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Tuple containing:
            - Path to the generated image if successful, None otherwise
            - URL or relative path to the image for displaying
    """
    # Create a style-based prompt
    style_prompts = {
        StylePreset.RIOT_GAMES: "Transform this into League of Legends game art style",
        StylePreset.REALISTIC: "Transform this into a photorealistic image",
        StylePreset.ANIME: "Transform this into high-quality anime artwork"
    }
    
    prompt = style_prompts.get(target_style, "Transform this image")
    
    return generate_image_to_image(
        prompt=prompt,
        source_image_path=source_image_path,
        strength=strength,
        style=target_style,
        session_id=session_id,
        progress_callback=progress_callback
    )

def recreate_with_variations(
    source_image_path: str,
    variation_prompt: str,
    num_variations: int = 3,
    strength_range: Tuple[float, float] = (0.5, 0.9),
    session_id: str = None,
    progress_callback: Optional[Callable] = None
) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Create multiple variations of an image with different prompts and strengths.
    
    Args:
        source_image_path: Path to the source image
        variation_prompt: Base prompt for variations
        num_variations: Number of variations to create
        strength_range: Range of strength values to use (min, max)
        session_id: Optional session ID for progress tracking
        progress_callback: Optional callback function for progress updates
        
    Returns:
        List of (image_path, filename) tuples for each variation
    """
    variations = []
    strength_values = np.linspace(strength_range[0], strength_range[1], num_variations)
    
    for i, strength in enumerate(strength_values):
        logger.info(f"Creating variation {i+1}/{num_variations} with strength {strength:.2f}")
        
        # Create variation-specific session ID
        var_session_id = f"{session_id}_var_{i}" if session_id else None
        
        result = generate_image_to_image(
            prompt=f"{variation_prompt}, variation {i+1}",
            source_image_path=source_image_path,
            strength=strength,
            session_id=var_session_id,
            progress_callback=progress_callback
        )
        
        variations.append(result)
    
    return variations
