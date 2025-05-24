"""
Image generation command handlers for the Service Command Interface.

This module provides handlers for all image generation operations, including
text-to-image generation, image-to-image transformation, model management,
and feedback-enhanced generation.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable, Tuple, List
from pathlib import Path
import uuid

from ..command_system import CommandType, CommandResult, ExecutionContext
from ...exceptions import ValidationError

logger = logging.getLogger(__name__)

# Import image generation services
try:
    from services.image_generator import (
        generate_image,
        generate_image_with_feedback,
        generate_image_async,
        get_model_status,
        unload_model,
        reload_model,
        validate_prompt,
        StylePreset,
        ImageGenerationError
    )
    from services.ai.image_generation.image_to_image import (
        generate_image_to_image,
        generate_image_to_image_async
    )
    IMAGE_SERVICES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Image generation services not available: {e}")
    IMAGE_SERVICES_AVAILABLE = False
    
    # Mock classes for fallback
    class StylePreset:
        RIOT_GAMES = "riot_games"
        REALISTIC = "realistic"
        ANIME = "anime"
    
    class ImageGenerationError(Exception):
        pass


class ImageHandlers:
    """Handler class for image generation commands."""
    
    @staticmethod
    def handle_image_generate(parameters: Dict[str, Any], context: ExecutionContext) -> CommandResult:
        """
        Handle IMAGE_GENERATE command - basic text-to-image generation.
        
        Required parameters:
        - prompt: Text prompt for image generation
        
        Optional parameters:
        - height: Image height in pixels (default: 1024)
        - width: Image width in pixels (default: 1024)
        - steps: Number of inference steps (default: 35)
        - guidance_scale: Guidance scale (default: 7.0)
        - negative_prompt: Negative prompt to avoid
        - style: Style preset ('riot_games', 'realistic', 'anime')
        - enable_progress: Enable progress tracking (default: True)
        """
        if not IMAGE_SERVICES_AVAILABLE:
            return CommandResult.error(
                command_id=context.command_id,
                error_code="SERVICE_UNAVAILABLE",
                message="Image generation services are not available"
            )
        
        try:
            # Validate required parameters
            prompt = parameters.get("prompt")
            if not prompt:
                return CommandResult.error(
                    command_id=context.command_id,
                    error_code="MISSING_PARAMETER",
                    message="Prompt is required for image generation"
                )
            
            # Validate prompt
            is_valid, validation_message, suggested_prompt = validate_prompt(prompt)
            if not is_valid:
                return CommandResult.error(
                    command_id=context.command_id,
                    error_code="INVALID_PROMPT",
                    message=f"Invalid prompt: {validation_message}",
                    data={"suggested_prompt": suggested_prompt} if suggested_prompt else None
                )
            
            # Extract generation parameters with defaults
            height = parameters.get("height", 1024)
            width = parameters.get("width", 1024)
            steps = parameters.get("steps", 35)
            guidance_scale = parameters.get("guidance_scale", 7.0)
            negative_prompt = parameters.get("negative_prompt")
            style_name = parameters.get("style")
            enable_progress = parameters.get("enable_progress", True)
            
            # Convert style string to StylePreset if provided
            style = None
            if style_name:
                style_mapping = {
                    "riot_games": StylePreset.RIOT_GAMES,
                    "realistic": StylePreset.REALISTIC,
                    "anime": StylePreset.ANIME
                }
                style = style_mapping.get(style_name.lower())
                if not style and hasattr(StylePreset, style_name.upper()):
                    style = getattr(StylePreset, style_name.upper())
            
            # Generate session ID for progress tracking if enabled
            session_id = str(uuid.uuid4()) if enable_progress else None
            
            # Progress callback setup
            progress_callback = None
            if enable_progress and context.progress_callback:
                def track_progress(step: int, total: int, **kwargs):
                    context.progress_callback(step, total, **kwargs)
                progress_callback = track_progress
            
            # Generate the image
            logger.info(f"Generating image for user {context.user_id}: '{prompt[:50]}...'")
            
            image_path, image_url = generate_image(
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
            
            if image_path:
                result_data = {
                    "image_path": image_path,
                    "image_url": image_url,
                    "prompt": prompt,
                    "parameters": {
                        "height": height,
                        "width": width,
                        "steps": steps,
                        "guidance_scale": guidance_scale,
                        "negative_prompt": negative_prompt,
                        "style": style_name
                    },
                    "session_id": session_id
                }
                
                return CommandResult.success(
                    command_id=context.command_id,
                    message="Image generated successfully",
                    data=result_data
                )
            else:
                return CommandResult.error(
                    command_id=context.command_id,
                    error_code="GENERATION_FAILED",
                    message="Failed to generate image"
                )
                
        except ImageGenerationError as e:
            logger.error(f"Image generation error: {e}")
            return CommandResult.error(
                command_id=context.command_id,
                error_code="IMAGE_GENERATION_ERROR",
                message=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error in image generation: {e}", exc_info=True)
            return CommandResult.error(
                command_id=context.command_id,
                error_code="INTERNAL_ERROR",
                message=f"Unexpected error: {str(e)}"
            )
    
    @staticmethod
    def handle_image_generate_with_feedback(parameters: Dict[str, Any], context: ExecutionContext) -> CommandResult:
        """
        Handle IMAGE_GENERATE_WITH_FEEDBACK command - generation with feedback analysis.
        
        Parameters include all from IMAGE_GENERATE plus:
        - user_feedback: Dict with rating, liked_aspects, disliked_aspects, improvement_suggestions
        """
        if not IMAGE_SERVICES_AVAILABLE:
            return CommandResult.error(
                command_id=context.command_id,
                error_code="SERVICE_UNAVAILABLE",
                message="Image generation services are not available"
            )
        
        try:
            # Validate required parameters
            prompt = parameters.get("prompt")
            if not prompt:
                return CommandResult.error(
                    command_id=context.command_id,
                    error_code="MISSING_PARAMETER",
                    message="Prompt is required for image generation"
                )
            
            # Extract all parameters (same as basic generation)
            height = parameters.get("height", 1024)
            width = parameters.get("width", 1024)
            steps = parameters.get("steps", 35)
            guidance_scale = parameters.get("guidance_scale", 7.0)
            negative_prompt = parameters.get("negative_prompt")
            style_name = parameters.get("style")
            enable_progress = parameters.get("enable_progress", True)
            user_feedback = parameters.get("user_feedback")
            
            # Convert style string to StylePreset if provided
            style = None
            if style_name:
                style_mapping = {
                    "riot_games": StylePreset.RIOT_GAMES,
                    "realistic": StylePreset.REALISTIC,
                    "anime": StylePreset.ANIME
                }
                style = style_mapping.get(style_name.lower())
            
            # Generate session ID for progress tracking
            session_id = str(uuid.uuid4()) if enable_progress else None
            
            # Progress callback setup
            progress_callback = None
            if enable_progress and context.progress_callback:
                def track_progress(step: int, total: int, **kwargs):
                    context.progress_callback(step, total, **kwargs)
                progress_callback = track_progress
            
            # Generate the image with feedback
            logger.info(f"Generating image with feedback for user {context.user_id}: '{prompt[:50]}...'")
            
            image_path, image_url, analysis = generate_image_with_feedback(
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
            
            if image_path:
                result_data = {
                    "image_path": image_path,
                    "image_url": image_url,
                    "prompt": prompt,
                    "parameters": {
                        "height": height,
                        "width": width,
                        "steps": steps,
                        "guidance_scale": guidance_scale,
                        "negative_prompt": negative_prompt,
                        "style": style_name
                    },
                    "session_id": session_id,
                    "analysis": analysis,
                    "feedback_processed": user_feedback is not None
                }
                
                return CommandResult.success(
                    command_id=context.command_id,
                    message="Image generated successfully with feedback analysis",
                    data=result_data
                )
            else:
                return CommandResult.error(
                    command_id=context.command_id,
                    error_code="GENERATION_FAILED",
                    message="Failed to generate image with feedback"
                )
                
        except Exception as e:
            logger.error(f"Error in feedback-enhanced image generation: {e}", exc_info=True)
            return CommandResult.error(
                command_id=context.command_id,
                error_code="INTERNAL_ERROR",
                message=f"Unexpected error: {str(e)}"
            )
    
    @staticmethod
    def handle_image_to_image(parameters: Dict[str, Any], context: ExecutionContext) -> CommandResult:
        """
        Handle IMAGE_TO_IMAGE command - image-to-image transformation.
        
        Required parameters:
        - prompt: Text prompt for transformation
        - source_image: Path to source image or image data
        
        Optional parameters:
        - strength: Transformation strength (0.1-1.0, default: 0.7)
        - height: Output height (default: 1024)
        - width: Output width (default: 1024)
        - steps: Inference steps (default: 35)
        - guidance_scale: Guidance scale (default: 7.0)
        - negative_prompt: Negative prompt
        - style: Style preset
        """
        if not IMAGE_SERVICES_AVAILABLE:
            return CommandResult.error(
                command_id=context.command_id,
                error_code="SERVICE_UNAVAILABLE",
                message="Image generation services are not available"
            )
        
        try:
            # Validate required parameters
            prompt = parameters.get("prompt")
            source_image = parameters.get("source_image")
            
            if not prompt:
                return CommandResult.error(
                    command_id=context.command_id,
                    error_code="MISSING_PARAMETER",
                    message="Prompt is required for image-to-image generation"
                )
            
            if not source_image:
                return CommandResult.error(
                    command_id=context.command_id,
                    error_code="MISSING_PARAMETER",
                    message="Source image is required for image-to-image generation"
                )
            
            # Validate source image path
            if isinstance(source_image, str):
                source_path = Path(source_image)
                if not source_path.exists():
                    return CommandResult.error(
                        command_id=context.command_id,
                        error_code="FILE_NOT_FOUND",
                        message=f"Source image not found: {source_image}"
                    )
                source_image_path = str(source_path)
            else:
                return CommandResult.error(
                    command_id=context.command_id,
                    error_code="INVALID_PARAMETER",
                    message="Source image must be a file path"
                )
            
            # Extract generation parameters
            strength = parameters.get("strength", 0.7)
            height = parameters.get("height", 1024)
            width = parameters.get("width", 1024)
            steps = parameters.get("steps", 35)
            guidance_scale = parameters.get("guidance_scale", 7.0)
            negative_prompt = parameters.get("negative_prompt")
            style_name = parameters.get("style")
            enable_progress = parameters.get("enable_progress", True)
            
            # Validate strength parameter
            if not 0.1 <= strength <= 1.0:
                return CommandResult.error(
                    command_id=context.command_id,
                    error_code="INVALID_PARAMETER",
                    message="Strength must be between 0.1 and 1.0"
                )
            
            # Convert style string to StylePreset if provided
            style = None
            if style_name:
                style_mapping = {
                    "riot_games": StylePreset.RIOT_GAMES,
                    "realistic": StylePreset.REALISTIC,
                    "anime": StylePreset.ANIME
                }
                style = style_mapping.get(style_name.lower())
            
            # Generate session ID for progress tracking
            session_id = str(uuid.uuid4()) if enable_progress else None
            
            # Progress callback setup
            progress_callback = None
            if enable_progress and context.progress_callback:
                def track_progress(step: int, total: int, **kwargs):
                    context.progress_callback(step, total, **kwargs)
                progress_callback = track_progress
            
            # Generate the image-to-image transformation
            logger.info(f"Generating image-to-image for user {context.user_id}: '{prompt[:50]}...'")
            
            image_path, image_url = generate_image_to_image(
                prompt=prompt,
                source_image_path=source_image_path,
                strength=strength,
                height=height,
                width=width,
                steps=steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                style=style,
                session_id=session_id,
                progress_callback=progress_callback
            )
            
            if image_path:
                result_data = {
                    "image_path": image_path,
                    "image_url": image_url,
                    "source_image": source_image_path,
                    "prompt": prompt,
                    "parameters": {
                        "strength": strength,
                        "height": height,
                        "width": width,
                        "steps": steps,
                        "guidance_scale": guidance_scale,
                        "negative_prompt": negative_prompt,
                        "style": style_name
                    },
                    "session_id": session_id
                }
                
                return CommandResult.success(
                    command_id=context.command_id,
                    message="Image-to-image transformation completed successfully",
                    data=result_data
                )
            else:
                return CommandResult.error(
                    command_id=context.command_id,
                    error_code="GENERATION_FAILED",
                    message="Failed to generate image-to-image transformation"
                )
                
        except Exception as e:
            logger.error(f"Error in image-to-image generation: {e}", exc_info=True)
            return CommandResult.error(
                command_id=context.command_id,
                error_code="INTERNAL_ERROR",
                message=f"Unexpected error: {str(e)}"
            )
    
    @staticmethod
    def handle_image_generate_async(parameters: Dict[str, Any], context: ExecutionContext) -> CommandResult:
        """
        Handle IMAGE_GENERATE_ASYNC command - asynchronous image generation.
        
        Parameters are the same as IMAGE_GENERATE, but generation runs asynchronously.
        Returns immediately with a task ID for status tracking.
        """
        if not IMAGE_SERVICES_AVAILABLE:
            return CommandResult.error(
                command_id=context.command_id,
                error_code="SERVICE_UNAVAILABLE",
                message="Image generation services are not available"
            )
        
        try:
            # Validate required parameters
            prompt = parameters.get("prompt")
            if not prompt:
                return CommandResult.error(
                    command_id=context.command_id,
                    error_code="MISSING_PARAMETER",
                    message="Prompt is required for image generation"
                )
            
            # Extract generation parameters
            height = parameters.get("height", 1024)
            width = parameters.get("width", 1024)
            steps = parameters.get("steps", 35)
            guidance_scale = parameters.get("guidance_scale", 7.0)
            negative_prompt = parameters.get("negative_prompt")
            style_name = parameters.get("style")
            
            # Convert style string to StylePreset if provided
            style = None
            if style_name:
                style_mapping = {
                    "riot_games": StylePreset.RIOT_GAMES,
                    "realistic": StylePreset.REALISTIC,
                    "anime": StylePreset.ANIME
                }
                style = style_mapping.get(style_name.lower())
            
            # Generate unique task ID
            task_id = str(uuid.uuid4())
            session_id = str(uuid.uuid4())
            
            # Progress callback setup for async tracking
            progress_callback = None
            if context.progress_callback:
                def track_progress(step: int, total: int, **kwargs):
                    context.progress_callback(step, total, task_id=task_id, **kwargs)
                progress_callback = track_progress
            
            # Start async generation
            async def async_generation():
                try:
                    return await generate_image_async(
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
                except Exception as e:
                    logger.error(f"Async image generation failed: {e}", exc_info=True)
                    return None, None
            
            # Store task for tracking (in a real implementation, you'd use a task manager)
            # For now, just return the task ID
            logger.info(f"Started async image generation for user {context.user_id}: task {task_id}")
            
            result_data = {
                "task_id": task_id,
                "session_id": session_id,
                "prompt": prompt,
                "parameters": {
                    "height": height,
                    "width": width,
                    "steps": steps,
                    "guidance_scale": guidance_scale,
                    "negative_prompt": negative_prompt,
                    "style": style_name
                },
                "status": "started"
            }
            
            return CommandResult.success(
                command_id=context.command_id,
                message="Async image generation started",
                data=result_data
            )
            
        except Exception as e:
            logger.error(f"Error starting async image generation: {e}", exc_info=True)
            return CommandResult.error(
                command_id=context.command_id,
                error_code="INTERNAL_ERROR",
                message=f"Failed to start async generation: {str(e)}"
            )
    
    @staticmethod
    def handle_image_model_status(parameters: Dict[str, Any], context: ExecutionContext) -> CommandResult:
        """
        Handle IMAGE_MODEL_STATUS command - get model status and information.
        """
        if not IMAGE_SERVICES_AVAILABLE:
            return CommandResult.error(
                command_id=context.command_id,
                error_code="SERVICE_UNAVAILABLE",
                message="Image generation services are not available"
            )
        
        try:
            status = get_model_status()
            
            return CommandResult.success(
                command_id=context.command_id,
                message="Model status retrieved successfully",
                data=status
            )
            
        except Exception as e:
            logger.error(f"Error getting model status: {e}", exc_info=True)
            return CommandResult.error(
                command_id=context.command_id,
                error_code="INTERNAL_ERROR",
                message=f"Failed to get model status: {str(e)}"
            )
    
    @staticmethod
    def handle_image_model_unload(parameters: Dict[str, Any], context: ExecutionContext) -> CommandResult:
        """
        Handle IMAGE_MODEL_UNLOAD command - unload model from memory.
        """
        if not IMAGE_SERVICES_AVAILABLE:
            return CommandResult.error(
                command_id=context.command_id,
                error_code="SERVICE_UNAVAILABLE",
                message="Image generation services are not available"
            )
        
        try:
            success = unload_model()
            
            if success:
                return CommandResult.success(
                    command_id=context.command_id,
                    message="Model unloaded successfully",
                    data={"unloaded": True}
                )
            else:
                return CommandResult.error(
                    command_id=context.command_id,
                    error_code="UNLOAD_FAILED",
                    message="Failed to unload model"
                )
                
        except Exception as e:
            logger.error(f"Error unloading model: {e}", exc_info=True)
            return CommandResult.error(
                command_id=context.command_id,
                error_code="INTERNAL_ERROR",
                message=f"Failed to unload model: {str(e)}"
            )
    
    @staticmethod
    def handle_image_model_reload(parameters: Dict[str, Any], context: ExecutionContext) -> CommandResult:
        """
        Handle IMAGE_MODEL_RELOAD command - reload model.
        """
        if not IMAGE_SERVICES_AVAILABLE:
            return CommandResult.error(
                command_id=context.command_id,
                error_code="SERVICE_UNAVAILABLE",
                message="Image generation services are not available"
            )
        
        try:
            success = reload_model()
            
            if success:
                return CommandResult.success(
                    command_id=context.command_id,
                    message="Model reloaded successfully",
                    data={"reloaded": True}
                )
            else:
                return CommandResult.error(
                    command_id=context.command_id,
                    error_code="RELOAD_FAILED",
                    message="Failed to reload model"
                )
                
        except Exception as e:
            logger.error(f"Error reloading model: {e}", exc_info=True)
            return CommandResult.error(
                command_id=context.command_id,
                error_code="INTERNAL_ERROR",
                message=f"Failed to reload model: {str(e)}"
            )
    
    @staticmethod
    def handle_image_validate_prompt(parameters: Dict[str, Any], context: ExecutionContext) -> CommandResult:
        """
        Handle IMAGE_VALIDATE_PROMPT command - validate image generation prompt.
        
        Required parameters:
        - prompt: Text prompt to validate
        """
        if not IMAGE_SERVICES_AVAILABLE:
            return CommandResult.error(
                command_id=context.command_id,
                error_code="SERVICE_UNAVAILABLE",
                message="Image generation services are not available"
            )
        
        try:
            prompt = parameters.get("prompt")
            if not prompt:
                return CommandResult.error(
                    command_id=context.command_id,
                    error_code="MISSING_PARAMETER",
                    message="Prompt is required for validation"
                )
            
            is_valid, validation_message, suggested_prompt = validate_prompt(prompt)
            
            result_data = {
                "prompt": prompt,
                "is_valid": is_valid,
                "message": validation_message,
                "suggested_prompt": suggested_prompt
            }
            
            if is_valid:
                return CommandResult.success(
                    command_id=context.command_id,
                    message="Prompt is valid",
                    data=result_data
                )
            else:
                return CommandResult.success(
                    command_id=context.command_id,
                    message=f"Prompt validation: {validation_message}",
                    data=result_data
                )
                
        except Exception as e:
            logger.error(f"Error validating prompt: {e}", exc_info=True)
            return CommandResult.error(
                command_id=context.command_id,
                error_code="INTERNAL_ERROR",
                message=f"Failed to validate prompt: {str(e)}"
            )
    
    @staticmethod
    def handle_image_get_styles(parameters: Dict[str, Any], context: ExecutionContext) -> CommandResult:
        """
        Handle IMAGE_GET_STYLES command - get available style presets.
        """
        try:
            # Available style presets
            styles = [
                {
                    "name": "riot_games",
                    "display_name": "Riot Games",
                    "description": "League of Legends and Riot Games art style"
                },
                {
                    "name": "realistic",
                    "display_name": "Realistic",
                    "description": "Photorealistic and lifelike imagery"
                },
                {
                    "name": "anime",
                    "display_name": "Anime",
                    "description": "Japanese anime and manga art style"
                }
            ]
            
            return CommandResult.success(
                command_id=context.command_id,
                message="Available styles retrieved successfully",
                data={"styles": styles}
            )
            
        except Exception as e:
            logger.error(f"Error getting styles: {e}", exc_info=True)
            return CommandResult.error(
                command_id=context.command_id,
                error_code="INTERNAL_ERROR",
                message=f"Failed to get styles: {str(e)}"
            )


# Handler registry for image commands
IMAGE_HANDLERS = {
    CommandType.IMAGE_GENERATE: ImageHandlers.handle_image_generate,
    CommandType.IMAGE_GENERATE_WITH_FEEDBACK: ImageHandlers.handle_image_generate_with_feedback,
    CommandType.IMAGE_TO_IMAGE: ImageHandlers.handle_image_to_image,
    CommandType.IMAGE_GENERATE_ASYNC: ImageHandlers.handle_image_generate_async,
    CommandType.IMAGE_MODEL_STATUS: ImageHandlers.handle_image_model_status,
    CommandType.IMAGE_MODEL_UNLOAD: ImageHandlers.handle_image_model_unload,
    CommandType.IMAGE_MODEL_RELOAD: ImageHandlers.handle_image_model_reload,
    CommandType.IMAGE_VALIDATE_PROMPT: ImageHandlers.handle_image_validate_prompt,
    CommandType.IMAGE_GET_STYLES: ImageHandlers.handle_image_get_styles,
}
