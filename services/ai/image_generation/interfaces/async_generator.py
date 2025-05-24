"""
Asynchronous image generation interface.

Provides a clean asynchronous API for image generation using thread pools
to avoid blocking the event loop.
"""
import asyncio
import logging
from typing import Dict, Optional, Any, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor

from .sync_generator import SyncImageGenerator
from ..style_presets import StylePreset

logger = logging.getLogger(__name__)


class AsyncImageGenerator:
    """Asynchronous image generation interface."""
    
    def __init__(self):
        """Initialize the asynchronous image generator."""
        self._sync_generator = SyncImageGenerator()
        self._executor = None
    
    @property
    def executor(self):
        """Get or create the thread pool executor."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1)
        return self._executor
    
    async def generate(
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
        loop = asyncio.get_event_loop()
        
        try:
            result = await loop.run_in_executor(
                self.executor,
                lambda: self._sync_generator.generate(
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
            )
            return result
        except Exception as e:
            logger.error(f"Error in async image generation: {e}", exc_info=True)
            return None, None
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Asynchronously get the current status of the image generation system.
        
        Returns:
            Dictionary containing status information
        """
        loop = asyncio.get_event_loop()
        
        try:
            result = await loop.run_in_executor(
                self.executor,
                self._sync_generator.get_status
            )
            return result
        except Exception as e:
            logger.error(f"Error getting async status: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def unload(self) -> bool:
        """
        Asynchronously unload the model from memory.
        
        Returns:
            True if successfully unloaded, False otherwise
        """
        loop = asyncio.get_event_loop()
        
        try:
            result = await loop.run_in_executor(
                self.executor,
                self._sync_generator.unload
            )
            return result
        except Exception as e:
            logger.error(f"Error in async unload: {e}", exc_info=True)
            return False
    
    async def reload(self) -> bool:
        """
        Asynchronously reload the model.
        
        Returns:
            True if successfully reloaded, False otherwise
        """
        loop = asyncio.get_event_loop()
        
        try:
            result = await loop.run_in_executor(
                self.executor,
                self._sync_generator.reload
            )
            return result
        except Exception as e:
            logger.error(f"Error in async reload: {e}", exc_info=True)
            return False
    
    def shutdown(self):
        """Shutdown the thread pool executor."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
    
    def __del__(self):
        """Cleanup when the object is destroyed."""
        self.shutdown()
