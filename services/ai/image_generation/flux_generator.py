# filepath: c:\Users\andyo\projects\personal-chatter\services\ai\image_generation\flux_generator.py
"""
Main Flux.1 image generator - Refactored Clean Interface.

This module provides the main interface for image generation using
the Flux.1 model with a clean, modular architecture.
"""
import logging
import asyncio
from typing import Dict, Optional, Any, Tuple, Callable

from .interfaces import SyncImageGenerator, AsyncImageGenerator, FeedbackImageGenerator
from .style_presets import StylePreset
from .utils.validation import validate_prompt

logger = logging.getLogger(__name__)

# Global instances (lazy initialization)
_sync_generator = None
_async_generator = None
_feedback_generator = None


def _get_sync_generator() -> SyncImageGenerator:
    """Get or create the synchronous generator instance."""
    global _sync_generator
    if _sync_generator is None:
        _sync_generator = SyncImageGenerator()
    return _sync_generator


def _get_async_generator() -> AsyncImageGenerator:
    """Get or create the asynchronous generator instance.""" 
    global _async_generator
    if _async_generator is None:
        _async_generator = AsyncImageGenerator()
    return _async_generator


def _get_feedback_generator() -> FeedbackImageGenerator:
    """Get or create the feedback generator instance."""
    global _feedback_generator
    if _feedback_generator is None:
        _feedback_generator = FeedbackImageGenerator()
    return _feedback_generator


def generate_image(
    prompt: str, 
    height: Optional[int] = None, 
    width: Optional[int] = None,
    steps: Optional[int] = None, 
    guidance_scale: Optional[float] = None,
    negative_prompt: Optional[str] = None,
    style: Optional[StylePreset] = None,
    session_id: Optional[str] = None,
    progress_callback: Optional[Callable] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate an image using the Flux.1 model with progress tracking and style presets.
    
    Args:
        prompt: The text prompt for image generation
        height: Image height in pixels
        width: Image width in pixels
        steps: Number of inference steps
        guidance_scale: Guidance scale for generation
        negative_prompt: Negative prompt to guide what to avoid
        style: Optional style preset to apply (Riot Games, Realistic, Anime)
        session_id: Optional session ID for progress tracking
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Tuple containing:
            - Path to the generated image if successful, None otherwise
            - URL or relative path to the image for displaying
    """
    generator = _get_sync_generator()
    return generator.generate(
        prompt=prompt,
        height=height,
        width=width,
        steps=steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        style=style,
        session_id=session_id,
        progress_callback=progress_callback
    )


def generate_image_with_feedback(
    prompt: str,
    height: Optional[int] = None,
    width: Optional[int] = None,
    steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    negative_prompt: Optional[str] = None,
    style: Optional[StylePreset] = None,
    session_id: Optional[str] = None,
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
        style: Optional style preset to apply
        session_id: Optional session ID for progress tracking
        progress_callback: Optional callback function for progress updates
        user_feedback: Optional feedback from previous generations
        
    Returns:
        Tuple containing:
            - Path to the generated image if successful, None otherwise
            - URL or relative path to the image for displaying
            - Analysis results and suggestions for improvement
    """
    generator = _get_feedback_generator()
    return generator.generate_with_feedback(
        prompt=prompt,
        height=height,
        width=width,
        steps=steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        style=style,
        session_id=session_id,
        progress_callback=progress_callback,
        user_feedback=user_feedback
    )


async def generate_image_async(
    prompt: str,
    height: Optional[int] = None,
    width: Optional[int] = None,
    steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    negative_prompt: Optional[str] = None,
    style: Optional[StylePreset] = None,
    session_id: Optional[str] = None,
    progress_callback: Optional[Callable] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Asynchronously generate an image using the Flux.1 model.
    
    This function runs the image generation in a thread pool to avoid blocking
    the async event loop.
    
    Args:
        prompt: The text prompt for image generation
        height: Image height in pixels
        width: Image width in pixels
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
    generator = _get_async_generator()
    return await generator.generate(
        prompt=prompt,
        height=height,
        width=width,
        steps=steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        style=style,
        session_id=session_id,
        progress_callback=progress_callback
    )


def get_model_status() -> Dict[str, Any]:
    """
    Get the current status of the Flux.1 model and memory usage.
    
    Returns:
        Dictionary containing model status information
    """
    generator = _get_sync_generator()
    return generator.get_status()


def unload_model() -> bool:
    """
    Unload the Flux.1 model from memory to free up resources.
    
    Returns:
        True if successfully unloaded, False otherwise
    """
    generator = _get_sync_generator()
    return generator.unload()


def reload_model() -> bool:
    """
    Reload the Flux.1 model, useful for recovering from errors or applying updates.
    
    Returns:
        True if successfully reloaded, False otherwise
    """
    generator = _get_sync_generator()
    return generator.reload()


# Export the main functions for use by other modules
__all__ = [
    "generate_image",
    "generate_image_with_feedback", 
    "generate_image_async",
    "get_model_status",
    "unload_model",
    "reload_model",
    "validate_prompt"
]
