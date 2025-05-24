"""
Synchronous image generation interface.

Provides a clean synchronous API for image generation with proper
error handling and validation.
"""
import logging
from typing import Dict, Optional, Any, Tuple, Callable

from ..core import PipelineManager, GenerationEngine, ConfigManager
from ..style_presets import StylePreset, enhance_prompt_with_style
from ..utils.validation import validate_prompt

logger = logging.getLogger(__name__)


class SyncImageGenerator:
    """Synchronous image generation interface."""
    
    def __init__(self):
        """Initialize the synchronous image generator."""
        self.config = ConfigManager()
        self.pipeline_manager = PipelineManager(self.config)
        self.engine = GenerationEngine(self.pipeline_manager, self.config)
    
    def generate(
        self,
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
        Generate an image using the Flux.1 model.
        
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
        # Validate prompt
        is_valid, validation_message, suggested_prompt = validate_prompt(prompt)
        if not is_valid:
            logger.error(f"Prompt validation failed: {validation_message}")
            return None, None
        
        if suggested_prompt and suggested_prompt != prompt:
            logger.info(f"Using suggested prompt enhancement")
            prompt = suggested_prompt
        
        # Apply style preset if specified
        original_prompt = prompt
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
            
            logger.info(f"Applied {style.value} style preset")
            logger.debug(f"Enhanced prompt: {prompt[:100]}...")
        
        # Set up progress tracking if requested
        progress_cb = None
        if session_id:
            try:
                from ..progress_tracker import create_progress_callback
                progress_cb = create_progress_callback(
                    session_id=session_id,
                    total_steps=steps or self.config.default_steps,
                    external_callback=progress_callback,
                    save_checkpoints=False
                )
            except ImportError:
                logger.warning("Progress tracking not available")
        
        try:
            # Generate the image
            result = self.engine.generate(
                prompt=prompt,
                height=height,
                width=width,
                steps=steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                progress_callback=progress_cb
            )
            
            return result
            
        finally:
            # Clean up progress tracking
            if session_id and progress_cb:
                try:
                    from ..progress_tracker import unregister_progress_callback
                    unregister_progress_callback(session_id)
                except ImportError:
                    pass
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the image generation system.
        
        Returns:
            Dictionary containing status information
        """
        status = {
            "pipeline_loaded": self.pipeline_manager.is_loaded,
            "model_path": self.config.model_path,
            "model_exists": self._model_exists(),
            "gpu_available": self._gpu_available(),
        }
        
        # Add memory information
        try:
            from ..memory_manager import get_memory_status
            status["memory_info"] = get_memory_status()
        except ImportError:
            status["memory_info"] = {"error": "Memory manager not available"}
        
        # Add GPU memory info
        if self._gpu_available():
            try:
                import torch
                gpu_memory = torch.cuda.mem_get_info()
                status["gpu_memory"] = {
                    "free_gb": gpu_memory[0] / 1e9,
                    "total_gb": gpu_memory[1] / 1e9,
                    "used_gb": (gpu_memory[1] - gpu_memory[0]) / 1e9
                }
            except Exception as e:
                logger.warning(f"Could not get GPU memory info: {e}")
                status["gpu_memory"] = {"error": str(e)}
        
        # Add pipeline device info
        if self.pipeline_manager.is_loaded:
            try:
                pipeline = self.pipeline_manager.pipeline
                first_param = next(pipeline.parameters(), None)
                if first_param is not None:
                    status["model_device"] = str(first_param.device)
                else:
                    status["model_device"] = "unknown"
            except Exception as e:
                logger.warning(f"Could not determine model device: {e}")
                status["model_device"] = "error"
        
        return status
    
    def unload(self) -> bool:
        """
        Unload the model from memory.
        
        Returns:
            True if successfully unloaded, False otherwise
        """
        return self.pipeline_manager.unload()
    
    def reload(self) -> bool:
        """
        Reload the model.
        
        Returns:
            True if successfully reloaded, False otherwise
        """
        return self.pipeline_manager.reload()
    
    def _model_exists(self) -> bool:
        """Check if the model directory exists."""
        import os
        return os.path.exists(self.config.model_path)
    
    def _gpu_available(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
