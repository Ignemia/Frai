"""
from typing import List
Main Image generation service for Personal Chatter.

This module serves as the primary interface for image generation,
delegating to the optimized modular Flux.1 system with integrated
memory management, progress tracking, and style presets.
"""
import os
import logging
import time
import asyncio
from typing import Dict, Optional, Any, Tuple, Callable, Union, List 
from pathlib import Path

from services.config import get_config

logger = logging.getLogger(__name__)

# Load configuration
config = get_config()

# Constants for backward compatibility
IMG_DEFAULT_HEIGHT = 1024
IMG_DEFAULT_WIDTH = 1024
NUM_INFERENCE_STEPS = 35
GUIDANCE_SCALE = 7.0
OUTPUT_DIR = "./outputs"

# Import the modular components
try:
    from .ai.image_generation.flux_generator import (
        generate_image as _generate_image,
        generate_image_with_feedback as _generate_image_with_feedback,
        generate_image_async as _generate_image_async,
        get_model_status as _get_model_status,
        unload_model as _unload_model,
        reload_model as _reload_model,
        validate_prompt as _validate_prompt
    )
    from .ai.image_generation.style_presets import StylePreset
    from .ai.image_generation.image_to_image import (
        generate_image_to_image,
        generate_image_to_image_async
    )
    MODULAR_SYSTEM_AVAILABLE = True
    logger.info("Modular image generation system loaded successfully")
except ImportError as e:
    logger.warning(f"Modular system not available, falling back to basic implementation: {e}")
    MODULAR_SYSTEM_AVAILABLE = False
    StylePreset = None

def generate_image(
    prompt: str, 
    height: int = IMG_DEFAULT_HEIGHT, 
    width: int = IMG_DEFAULT_WIDTH,
    steps: int = NUM_INFERENCE_STEPS, 
    guidance_scale: float = GUIDANCE_SCALE,
    negative_prompt: str = None,
    style: str = None,  # Accept string for backward compatibility
    session_id: str = None,
    progress_callback: Optional[Callable] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate an image using the Flux.1 model with style presets and progress tracking.
    
    Args:
        prompt: The text prompt for image generation
        height: Image height in pixels
        width: Image width in pixels
        steps: Number of inference steps
        guidance_scale: Guidance scale for generation
        negative_prompt: Negative prompt to guide what to avoid
        style: Style preset name ("riot_games", "realistic", "anime") or StylePreset enum
        session_id: Optional session ID for progress tracking
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Tuple containing:
            - Path to the generated image if successful, None otherwise
            - URL or relative path to the image for displaying
    """
    if not MODULAR_SYSTEM_AVAILABLE:
        logger.error("Modular image generation system is not available")
        return None, None
    
    # Convert string style to StylePreset enum if needed
    style_preset = None
    if style and StylePreset:
        try:
            if isinstance(style, str):
                style_preset = StylePreset(style.lower())
            else:
                style_preset = style
        except ValueError:
            logger.warning(f"Unknown style preset: {style}")
    
    try:
        return _generate_image(
            prompt=prompt,
            height=height,
            width=width,
            steps=steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            style=style_preset,
            session_id=session_id,
            progress_callback=progress_callback
        )
    except Exception as e:
        logger.error(f"Error in image generation: {e}", exc_info=True)
        return None, None

def generate_image_with_feedback(
    prompt: str,
    height: int = IMG_DEFAULT_HEIGHT,
    width: int = IMG_DEFAULT_WIDTH,
    steps: int = NUM_INFERENCE_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
    negative_prompt: str = None,
    style: str = None,
    session_id: str = None,
    progress_callback: Optional[Callable] = None,
    user_feedback: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """
    Generate an image with sentiment analysis and feedback integration.
    
    Args:
        prompt: The text prompt for image generation
        height: Image height in pixels
        width: Image width in pixels
        steps: Number of inference steps
        guidance_scale: Guidance scale for generation
        negative_prompt: Negative prompt to guide what to avoid
        style: Style preset name or StylePreset enum
        session_id: Optional session ID for progress tracking
        progress_callback: Optional callback function for progress updates
        user_feedback: Optional feedback from previous generations
        
    Returns:
        Tuple containing:
            - Path to the generated image if successful, None otherwise
            - URL or relative path to the image for displaying
            - Analysis results and suggestions for improvement
    """
    if not MODULAR_SYSTEM_AVAILABLE:
        logger.error("Modular image generation system is not available")
        return None, None, None
    
    # Convert string style to StylePreset enum if needed
    style_preset = None
    if style and StylePreset:
        try:
            if isinstance(style, str):
                style_preset = StylePreset(style.lower())
            else:
                style_preset = style
        except ValueError:
            logger.warning(f"Unknown style preset: {style}")
    
    try:
        return _generate_image_with_feedback(
            prompt=prompt,
            height=height,
            width=width,
            steps=steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            style=style_preset,
            session_id=session_id,
            progress_callback=progress_callback,
            user_feedback=user_feedback
        )
    except Exception as e:
        logger.error(f"Error in image generation with feedback: {e}", exc_info=True)
        return None, None, None

async def generate_image_async(
    prompt: str,
    height: int = IMG_DEFAULT_HEIGHT,
    width: int = IMG_DEFAULT_WIDTH,
    steps: int = NUM_INFERENCE_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
    negative_prompt: str = None,
    style: str = None,
    session_id: str = None,
    progress_callback: Optional[Callable] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Asynchronously generate an image using the Flux.1 model.
    
    Args:
        prompt: The text prompt for image generation
        height: Image height in pixels
        width: Image width in pixels
        steps: Number of inference steps
        guidance_scale: Guidance scale for generation
        negative_prompt: Negative prompt to guide what to avoid
        style: Style preset name or StylePreset enum
        session_id: Optional session ID for progress tracking
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Tuple containing:
            - Path to the generated image if successful, None otherwise
            - URL or relative path to the image for displaying
    """
    if not MODULAR_SYSTEM_AVAILABLE:
        logger.error("Modular image generation system is not available")
        return None, None
    
    # Convert string style to StylePreset enum if needed
    style_preset = None
    if style and StylePreset:
        try:
            if isinstance(style, str):
                style_preset = StylePreset(style.lower())
            else:
                style_preset = style
        except ValueError:
            logger.warning(f"Unknown style preset: {style}")
    
    try:
        return await _generate_image_async(
            prompt=prompt,
            height=height,
            width=width,
            steps=steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            style=style_preset,
            session_id=session_id,
            progress_callback=progress_callback
        )
    except Exception as e:
        logger.error(f"Error in async image generation: {e}", exc_info=True)
        return None, None

def get_model_status() -> Dict[str, Any]:
    """
    Get the current status of the image generation models.
    
    Returns:
        Dictionary containing model status information
    """
    if not MODULAR_SYSTEM_AVAILABLE:
        return {
            "error": "Modular image generation system is not available",
            "modular_system_available": False
        }
    
    try:
        status = _get_model_status()
        status["modular_system_available"] = True
        return status
    except Exception as e:
        logger.error(f"Error getting model status: {e}", exc_info=True)
        return {
            "error": str(e),
            "modular_system_available": MODULAR_SYSTEM_AVAILABLE
        }

def unload_model() -> bool:
    """
    Unload the image generation models from memory.
    
    Returns:
        True if successfully unloaded, False otherwise
    """
    if not MODULAR_SYSTEM_AVAILABLE:
        logger.warning("Modular image generation system is not available")
        return False
    
    try:
        return _unload_model()
    except Exception as e:
        logger.error(f"Error unloading model: {e}", exc_info=True)
        return False

def reload_model() -> bool:
    """
    Reload the image generation models.
    
    Returns:
        True if successfully reloaded, False otherwise
    """
    if not MODULAR_SYSTEM_AVAILABLE:
        logger.warning("Modular image generation system is not available")
        return False
    
    try:
        return _reload_model()
    except Exception as e:
        logger.error(f"Error reloading model: {e}", exc_info=True)
        return False

def validate_prompt(prompt: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validate the input prompt and provide suggestions if needed.
    
    Args:
        prompt: The text prompt to validate
        
    Returns:
        Tuple containing:
            - Boolean indicating if prompt is valid
            - Validation message
            - Optional suggested improved prompt
    """
    if not MODULAR_SYSTEM_AVAILABLE:
        # Basic validation if modular system is not available
        if not prompt or not prompt.strip():
            return False, "Prompt cannot be empty", None
        if len(prompt) > 1000:
            return False, "Prompt is too long (max 1000 characters)", prompt[:1000]
        return True, "Prompt is valid (basic validation)", None
    
    try:
        return _validate_prompt(prompt)
    except Exception as e:
        logger.error(f"Error validating prompt: {e}", exc_info=True)
        return False, f"Error validating prompt: {e}", None

# Backward compatibility functions
def generate_flux_image(prompt: str, **kwargs) -> Tuple[Optional[str], Optional[str]]:
    """Backward compatibility wrapper for generate_image."""
    return generate_image(prompt, **kwargs)

async def generate_flux_image_async(prompt: str, **kwargs) -> Tuple[Optional[str], Optional[str]]:
    """Backward compatibility wrapper for generate_image_async."""
    return await generate_image_async(prompt, **kwargs)

def get_flux_model_status() -> Dict[str, Any]:
    """Backward compatibility wrapper for get_model_status."""
    return get_model_status()

# Export main functions
__all__ = [
    "generate_image",
    "generate_image_with_feedback",
    "generate_image_async", 
    "generate_image_to_image",
    "generate_image_to_image_async",
    "get_model_status",
    "unload_model",
    "reload_model",
    "validate_prompt",
    "StylePreset",
    # Backward compatibility
    "generate_flux_image",
    "generate_flux_image_async",
    "get_flux_model_status"
]
