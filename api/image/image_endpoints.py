"""
from typing import List
Image Generation API endpoints for Frai application.

This module provides REST API endpoints for image generation operations including:
- Basic image generation with style presets
- Image-to-image generation
- Progress tracking via WebSocket
- Model status and management
"""
import logging
import asyncio
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query, File, UploadFile, Form
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketState
import json
import uuid
import os
import time

from services.image_generator import (
    generate_image,
    generate_image_with_feedback,
    generate_image_async,
    generate_image_to_image,
    generate_image_to_image_async,
    get_model_status,
    unload_model,
    reload_model,
    validate_prompt,
    StylePreset
)
from api.auth import get_current_user, verify_token
from services.communication.websocket import get_websocket_progress_manager

logger = logging.getLogger(__name__)

# Token validation adapter
async def get_session_token(current_user = Depends(get_current_user)) -> str:
    """Convert JWT token to session token for backward compatibility."""
    return current_user.get("username", "anonymous")

# Request/Response Models
class ImageGenerationRequest(BaseModel):
    """Request model for basic image generation."""
    prompt: str = Field(..., description="Text prompt for image generation", min_length=1, max_length=1000)
    height: int = Field(1024, description="Image height in pixels", ge=256, le=2048)
    width: int = Field(1024, description="Image width in pixels", ge=256, le=2048)
    steps: int = Field(35, description="Number of inference steps", ge=10, le=100)
    guidance_scale: float = Field(7.0, description="Guidance scale for generation", ge=1.0, le=20.0)
    negative_prompt: Optional[str] = Field(None, description="Negative prompt to guide what to avoid", max_length=500)
    style: Optional[str] = Field(None, description="Style preset: 'riot_games', 'realistic', or 'anime'")
    enable_progress: bool = Field(True, description="Enable real-time progress tracking")

class ImageGenerationResponse(BaseModel):
    """Response model for image generation."""
    success: bool
    session_id: Optional[str] = None
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    message: str
    generation_time: Optional[float] = None
    analysis: Optional[Dict[str, Any]] = None

class ImageToImageRequest(BaseModel):
    """Request model for image-to-image generation."""
    prompt: str = Field(..., description="Text prompt for image generation", min_length=1, max_length=1000)
    strength: float = Field(0.7, description="Transformation strength", ge=0.1, le=1.0)
    height: int = Field(1024, description="Image height in pixels", ge=256, le=2048)
    width: int = Field(1024, description="Image width in pixels", ge=256, le=2048)
    steps: int = Field(35, description="Number of inference steps", ge=10, le=100)
    guidance_scale: float = Field(7.0, description="Guidance scale for generation", ge=1.0, le=20.0)
    negative_prompt: Optional[str] = Field(None, description="Negative prompt to guide what to avoid", max_length=500)
    style: Optional[str] = Field(None, description="Style preset: 'riot_games', 'realistic', or 'anime'")
    enable_progress: bool = Field(True, description="Enable real-time progress tracking")

class FeedbackData(BaseModel):
    """Model for user feedback on generated images."""
    rating: int = Field(..., description="User rating from 1-5", ge=1, le=5)
    liked_aspects: List[str] = Field(default_factory=list, description="Aspects the user liked")
    disliked_aspects: List[str] = Field(default_factory=list, description="Aspects the user disliked")
    improvement_suggestions: Optional[str] = Field(None, description="Suggestions for improvement", max_length=500)

class GenerationWithFeedbackRequest(BaseModel):
    """Request model for image generation with feedback integration."""
    prompt: str = Field(..., description="Text prompt for image generation", min_length=1, max_length=1000)
    height: int = Field(1024, description="Image height in pixels", ge=256, le=2048)
    width: int = Field(1024, description="Image width in pixels", ge=256, le=2048)
    steps: int = Field(35, description="Number of inference steps", ge=10, le=100)
    guidance_scale: float = Field(7.0, description="Guidance scale for generation", ge=1.0, le=20.0)
    negative_prompt: Optional[str] = Field(None, description="Negative prompt to guide what to avoid", max_length=500)
    style: Optional[str] = Field(None, description="Style preset: 'riot_games', 'realistic', or 'anime'")
    user_feedback: Optional[FeedbackData] = Field(None, description="Previous generation feedback")
    enable_progress: bool = Field(True, description="Enable real-time progress tracking")

class ModelStatusResponse(BaseModel):
    """Response model for model status."""
    model_loaded: bool
    model_path: str
    model_exists: bool
    gpu_available: bool
    memory_info: Dict[str, Any]
    gpu_memory: Optional[Dict[str, float]] = None
    model_device: Optional[str] = None
    modular_system_available: bool

class PromptValidationRequest(BaseModel):
    """Request model for prompt validation."""
    prompt: str = Field(..., description="Text prompt to validate", max_length=1000)

class PromptValidationResponse(BaseModel):
    """Response model for prompt validation."""
    is_valid: bool
    message: str
    suggested_prompt: Optional[str] = None

# Create router
image_router = APIRouter(prefix="/image", tags=["image"])

@image_router.post(
    "/generate",
    response_model=ImageGenerationResponse,
    summary="Generate an image from text prompt",
    status_code=200)
async def generate_image_endpoint(
    request: ImageGenerationRequest,
    session_token: str = Depends(get_session_token)
):
    """
    Generate an image using the Flux.1 model with optional style presets.
    
    Args:
        request: Image generation parameters
        session_token: User's session token
        
    Returns:
        Generated image information and metadata
    """
    try:
        start_time = time.time()
        session_id = str(uuid.uuid4()) if request.enable_progress else None
        
        logger.info(f"Starting image generation for user {session_token}")
        logger.info(f"Prompt: {request.prompt[:100]}...")
        
        # Validate prompt first
        is_valid, validation_message, suggested_prompt = validate_prompt(request.prompt)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid prompt: {validation_message}")
        
        # Generate the image
        if request.enable_progress:
            image_path, image_url = await generate_image_async(
                prompt=request.prompt,
                height=request.height,
                width=request.width,
                steps=request.steps,
                guidance_scale=request.guidance_scale,
                negative_prompt=request.negative_prompt,
                style=request.style,
                session_id=session_id
            )
        else:
            image_path, image_url = generate_image(
                prompt=request.prompt,
                height=request.height,
                width=request.width,
                steps=request.steps,
                guidance_scale=request.guidance_scale,
                negative_prompt=request.negative_prompt,
                style=request.style
            )
        
        generation_time = time.time() - start_time
        
        if image_path:
            logger.info(f"Image generation successful in {generation_time:.2f}s")
            return ImageGenerationResponse(
                success=True,
                session_id=session_id,
                image_path=image_path,
                image_url=image_url,
                message="Image generated successfully",
                generation_time=generation_time
            )
        else:
            logger.error("Image generation failed")
            return ImageGenerationResponse(
                success=False,
                message="Image generation failed. Check server logs for details."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in image generation endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image generation error: {str(e)}")

@image_router.post(
    "/generate-with-feedback",
    response_model=ImageGenerationResponse,
    summary="Generate an image with feedback integration",
    status_code=200)
async def generate_image_with_feedback_endpoint(
    request: GenerationWithFeedbackRequest,
    session_token: str = Depends(get_session_token)
):
    """
    Generate an image with sentiment analysis and feedback integration.
    
    Args:
        request: Image generation parameters with optional user feedback
        session_token: User's session token
        
    Returns:
        Generated image information with analysis results
    """
    try:
        start_time = time.time()
        session_id = str(uuid.uuid4()) if request.enable_progress else None
        
        logger.info(f"Starting feedback-enhanced image generation for user {session_token}")
        
        # Validate prompt first
        is_valid, validation_message, suggested_prompt = validate_prompt(request.prompt)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid prompt: {validation_message}")
        
        # Convert feedback to the expected format
        user_feedback = None
        if request.user_feedback:
            user_feedback = {
                "rating": request.user_feedback.rating,
                "liked_aspects": request.user_feedback.liked_aspects,
                "disliked_aspects": request.user_feedback.disliked_aspects,
                "improvement_suggestions": request.user_feedback.improvement_suggestions
            }
        
        # Generate the image with feedback
        image_path, image_url, analysis = generate_image_with_feedback(
            prompt=request.prompt,
            height=request.height,
            width=request.width,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
            negative_prompt=request.negative_prompt,
            style=request.style,
            session_id=session_id,
            user_feedback=user_feedback
        )
        
        generation_time = time.time() - start_time
        
        if image_path:
            logger.info(f"Feedback-enhanced image generation successful in {generation_time:.2f}s")
            return ImageGenerationResponse(
                success=True,
                session_id=session_id,
                image_path=image_path,
                image_url=image_url,
                message="Image generated successfully with feedback integration",
                generation_time=generation_time,
                analysis=analysis
            )
        else:
            logger.error("Feedback-enhanced image generation failed")
            return ImageGenerationResponse(
                success=False,
                message="Image generation failed. Check server logs for details."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in feedback-enhanced image generation endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image generation error: {str(e)}")

@image_router.post(
    "/image-to-image",
    response_model=ImageGenerationResponse,
    summary="Generate an image from an input image and text prompt",
    status_code=200)
async def image_to_image_endpoint(
    prompt: str = Form(..., description="Text prompt for image transformation"),
    strength: float = Form(0.7, description="Transformation strength"),
    height: int = Form(1024, description="Output image height"),
    width: int = Form(1024, description="Output image width"),
    steps: int = Form(35, description="Number of inference steps"),
    guidance_scale: float = Form(7.0, description="Guidance scale"),
    negative_prompt: Optional[str] = Form(None, description="Negative prompt"),
    style: Optional[str] = Form(None, description="Style preset"),
    enable_progress: bool = Form(True, description="Enable progress tracking"),
    input_image: UploadFile = File(..., description="Input image file"),
    session_token: str = Depends(get_session_token)
):
    """
    Generate an image from an input image and text prompt using image-to-image generation.
    
    Args:
        prompt: Text prompt for image transformation
        strength: How much to transform the input image (0.1-1.0)
        height: Output image height in pixels
        width: Output image width in pixels
        steps: Number of inference steps
        guidance_scale: Guidance scale for generation
        negative_prompt: Negative prompt to guide what to avoid
        style: Style preset to apply
        enable_progress: Enable real-time progress tracking
        input_image: Input image file (PNG, JPG, JPEG)
        session_token: User's session token
        
    Returns:
        Generated image information and metadata
    """
    try:
        start_time = time.time()
        session_id = str(uuid.uuid4()) if enable_progress else None
        
        logger.info(f"Starting image-to-image generation for user {session_token}")
        logger.info(f"Input image: {input_image.filename}, Prompt: {prompt[:100]}...")
        
        # Validate file type
        allowed_types = ["image/png", "image/jpeg", "image/jpg"]
        if input_image.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Allowed types: {', '.join(allowed_types)}"
            )
        
        # Validate prompt
        is_valid, validation_message, suggested_prompt = validate_prompt(prompt)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid prompt: {validation_message}")
        
        # Save uploaded file temporarily
        temp_dir = "./temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_filename = f"{uuid.uuid4()}_{input_image.filename}"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        try:
            # Read and save the uploaded file
            content = await input_image.read()
            with open(temp_path, "wb") as f:
                f.write(content)
            
            # Generate the image
            if enable_progress:
                image_path, image_url = await generate_image_to_image_async(
                    prompt=prompt,
                    input_image_path=temp_path,
                    strength=strength,
                    height=height,
                    width=width,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    style=style,
                    session_id=session_id
                )
            else:
                image_path, image_url = generate_image_to_image(
                    prompt=prompt,
                    input_image_path=temp_path,
                    strength=strength,
                    height=height,
                    width=width,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    style=style
                )
            
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Could not remove temporary file {temp_path}: {e}")
        
        generation_time = time.time() - start_time
        
        if image_path:
            logger.info(f"Image-to-image generation successful in {generation_time:.2f}s")
            return ImageGenerationResponse(
                success=True,
                session_id=session_id,
                image_path=image_path,
                image_url=image_url,
                message="Image-to-image generation completed successfully",
                generation_time=generation_time
            )
        else:
            logger.error("Image-to-image generation failed")
            return ImageGenerationResponse(
                success=False,
                message="Image-to-image generation failed. Check server logs for details."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in image-to-image generation endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image-to-image generation error: {str(e)}")

@image_router.get(
    "/status",
    response_model=ModelStatusResponse,
    summary="Get image generation model status",
    status_code=200)
async def get_model_status_endpoint(session_token: str = Depends(get_session_token)):
    """
    Get the current status of the image generation models and system.
    
    Args:
        session_token: User's session token
        
    Returns:
        Model status and system information
    """
    try:
        status = get_model_status()
        
        return ModelStatusResponse(
            model_loaded=status.get("model_loaded", False),
            model_path=status.get("model_path", ""),
            model_exists=status.get("model_exists", False),
            gpu_available=status.get("gpu_available", False),
            memory_info=status.get("memory_info", {}),
            gpu_memory=status.get("gpu_memory"),
            model_device=status.get("model_device"),
            modular_system_available=status.get("modular_system_available", False)
        )
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting model status: {str(e)}")

@image_router.post(
    "/model/unload",
    response_model=dict,
    summary="Unload the image generation model from memory",
    status_code=200)
async def unload_model_endpoint(session_token: str = Depends(get_session_token)):
    """
    Unload the image generation model from memory to free up resources.
    
    Args:
        session_token: User's session token
        
    Returns:
        Success status
    """
    try:
        success = unload_model()
        
        if success:
            logger.info("Model unloaded successfully")
            return {"success": True, "message": "Model unloaded successfully"}
        else:
            logger.warning("Failed to unload model")
            return {"success": False, "message": "Failed to unload model"}
            
    except Exception as e:
        logger.error(f"Error unloading model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error unloading model: {str(e)}")

@image_router.post(
    "/model/reload",
    response_model=dict,
    summary="Reload the image generation model",
    status_code=200)
async def reload_model_endpoint(session_token: str = Depends(get_session_token)):
    """
    Reload the image generation model, useful for recovering from errors.
    
    Args:
        session_token: User's session token
        
    Returns:
        Success status
    """
    try:
        success = reload_model()
        
        if success:
            logger.info("Model reloaded successfully")
            return {"success": True, "message": "Model reloaded successfully"}
        else:
            logger.warning("Failed to reload model")
            return {"success": False, "message": "Failed to reload model"}
            
    except Exception as e:
        logger.error(f"Error reloading model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error reloading model: {str(e)}")

@image_router.post(
    "/validate-prompt",
    response_model=PromptValidationResponse,
    summary="Validate a text prompt for image generation",
    status_code=200)
async def validate_prompt_endpoint(
    request: PromptValidationRequest,
    session_token: str = Depends(get_session_token)
):
    """
    Validate a text prompt and provide suggestions for improvement.
    
    Args:
        request: Prompt validation request
        session_token: User's session token
        
    Returns:
        Validation results and suggestions
    """
    try:
        is_valid, message, suggested_prompt = validate_prompt(request.prompt)
        
        return PromptValidationResponse(
            is_valid=is_valid,
            message=message,
            suggested_prompt=suggested_prompt
        )
        
    except Exception as e:
        logger.error(f"Error validating prompt: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error validating prompt: {str(e)}")

@image_router.get(
    "/styles",
    response_model=List[Dict[str, str]],
    summary="Get available style presets",
    status_code=200)
async def get_available_styles(session_token: str = Depends(get_session_token)):
    """
    Get a list of available style presets for image generation.
    
    Args:
        session_token: User's session token
        
    Returns:
        List of available style presets with descriptions
    """
    try:
        styles = []
        
        if StylePreset:
            styles = [
                {
                    "name": "riot_games",
                    "display_name": "Riot Games",
                    "description": "Character-focused art style with vibrant colors and dynamic poses"
                },
                {
                    "name": "realistic",
                    "display_name": "Realistic", 
                    "description": "Photorealistic style with natural lighting and detailed textures"
                },
                {
                    "name": "anime",
                    "display_name": "Anime",
                    "description": "Japanese anime and manga inspired art style"
                }
            ]
        
        return styles
        
    except Exception as e:
        logger.error(f"Error getting available styles: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting styles: {str(e)}")

# Export the router
__all__ = ["image_router"]
