"""
Core image generation engine.

Handles the actual image generation process with memory management,
progress tracking, and error handling.
"""
import os
import logging
import time
import concurrent.futures
from typing import Dict, Optional, Any, Tuple, Callable
import torch

logger = logging.getLogger(__name__)


class GenerationEngine:
    """Core engine for image generation using Flux.1 model."""
    
    def __init__(self, pipeline_manager, config_manager):
        """Initialize the generation engine.
        
        Args:
            pipeline_manager: Pipeline manager instance
            config_manager: Configuration manager instance
        """
        self.pipeline_manager = pipeline_manager
        self.config = config_manager
        
    def generate(
        self,
        prompt: str,
        height: Optional[int] = None,
        width: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
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
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Tuple containing:
                - Path to the generated image if successful, None otherwise
                - URL or relative path to the image for displaying
        """
        logger.info(f"Starting image generation: '{prompt[:50]}...'")
        start_time = time.time()
        
        # Use defaults from config if not provided
        height = height or self.config.default_height
        width = width or self.config.default_width
        steps = steps or self.config.default_steps
        guidance_scale = guidance_scale or self.config.default_guidance_scale
        
        # Ensure output directory exists
        self.config.ensure_output_dir()
        
        # Get the pipeline
        pipeline = self.pipeline_manager.get_pipeline()
        if pipeline is None:
            logger.error("Failed to initialize Flux.1 pipeline")
            return None, None
        
        # Check GPU memory before attempting generation
        can_generate, memory_message = self._check_gpu_memory(height, width, steps)
        if not can_generate:
            logger.error(f"Memory check failed: {memory_message}")
            logger.info("Consider reducing image size (e.g., 256x256) or inference steps (e.g., 10-15)")
            return None, None
        
        try:
            return self._perform_generation(
                pipeline, prompt, height, width, steps, 
                guidance_scale, negative_prompt, progress_callback, start_time
            )
            
        except torch.cuda.OutOfMemoryError as e:
            return self._handle_oom_error(e, pipeline, width, height, steps)
        except Exception as e:
            logger.error(f"Error during image generation: {e}", exc_info=True)
            return None, None
    
    def _perform_generation(
        self, pipeline, prompt, height, width, steps, 
        guidance_scale, negative_prompt, progress_callback, start_time
    ):
        """Perform the actual image generation."""
        # Aggressive memory cleanup before generation
        import gc
        gc.collect()
        
        try:
            from ..memory_manager import clear_gpu_memory
            clear_gpu_memory()
        except ImportError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Log memory status right before generation
        self._log_memory_status()
        
        # Generate the image
        logger.info(f"Generating image with {steps} steps")
        
        generation_kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "max_sequence_length": self.config.max_sequence_length,
        }
        
        if negative_prompt:
            generation_kwargs["negative_prompt"] = negative_prompt
        
        logger.info("Starting Flux.1 image generation...")
        generation_start = time.time()
        
        # Use threading with timeout for cross-platform compatibility
        result = self._run_generation_with_timeout(pipeline, generation_kwargs)
        
        generation_time = time.time() - generation_start
        logger.info(f"Image generation completed in {generation_time:.2f} seconds")
        
        # Process the result
        return self._process_generation_result(result, prompt, start_time)
    
    def _run_generation_with_timeout(self, pipeline, generation_kwargs):
        """Run generation with timeout handling."""
        def run_generation():
            with torch.inference_mode():
                return pipeline(**generation_kwargs)
        
        # Use ThreadPoolExecutor for timeout handling that works on all platforms
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_generation)
            try:
                # 5-minute timeout (300 seconds)
                return future.result(timeout=300)
            except concurrent.futures.TimeoutError:
                logger.error("Image generation timed out after 5 minutes")
                # Cancel the future if possible and clean up
                future.cancel()
                # Force GPU memory cleanup
                try:
                    from ..memory_manager import clear_gpu_memory
                    clear_gpu_memory()
                except ImportError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                raise TimeoutError("Image generation timed out after 5 minutes")
    
    def _process_generation_result(self, result, prompt, start_time):
        """Process the generation result and save the image."""
        if hasattr(result, 'images') and len(result.images) > 0:
            image = result.images[0]
            
            # Generate filename
            safe_prompt = "".join(c for c in prompt[:50] if c.isalnum() or c in (' ', '_')).strip()
            safe_prompt = safe_prompt.replace(' ', '_')
            timestamp = int(time.time())
            filename = f"image_{safe_prompt}_{timestamp}.png"
            filepath = os.path.join(self.config.output_dir, filename)
            
            # Save the image
            image.save(filepath)
            logger.info(f"Image saved to {filepath}")
            
            # Schedule pipeline offload after a delay
            try:
                from ..memory_manager import schedule_offload
                schedule_offload(self.pipeline_manager.pipeline)
            except ImportError:
                pass
            
            total_time = time.time() - start_time
            logger.info(f"Total image generation completed in {total_time:.2f} seconds")
            
            return filepath, filename
        else:
            logger.error("No image generated in result")
            return None, None
    
    def _handle_oom_error(self, error, pipeline, width, height, steps):
        """Handle out-of-memory errors."""
        logger.error(f"GPU out of memory during generation: {error}")
        logger.info("Moving pipeline to CPU and clearing memory...")
        
        # Move pipeline back to CPU and clear memory
        try:
            from ..memory_manager import move_pipeline_to_cpu, clear_gpu_memory
            pipeline = move_pipeline_to_cpu(pipeline)
            clear_gpu_memory()
        except ImportError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Suggest lower settings
        logger.warning(f"Current settings: {width}x{height}, {steps} steps")
        logger.warning("Try using: 256x256 resolution with 10-15 inference steps")
        
        return None, None
    
    def _log_memory_status(self):
        """Log current GPU memory status."""
        if torch.cuda.is_available():
            free_memory, total_memory = torch.cuda.mem_get_info()
            free_memory_gb = free_memory / 1e9
            total_memory_gb = total_memory / 1e9
            logger.info(f"GPU memory before generation: {free_memory_gb:.2f}GB free / {total_memory_gb:.2f}GB total")
    
    def _check_gpu_memory(self, height: int, width: int, steps: int) -> Tuple[bool, str]:
        """
        Check if there's sufficient GPU memory for the requested generation parameters.
        
        Args:
            height: Image height in pixels
            width: Image width in pixels
            steps: Number of inference steps
            
        Returns:
            Tuple of (can_generate: bool, message: str)
        """
        # Small images should always be allowed to attempt generation
        if height <= 256 and width <= 256 and steps <= 15:
            message = f"Small image requested ({height}x{width}, {steps} steps) - bypassing memory check"
            logger.info(message)
            return True, message
            
        if not torch.cuda.is_available():
            return True, "CUDA not available, will use CPU"
        
        try:
            # Get current GPU memory status
            free_memory, total_device_memory = torch.cuda.mem_get_info()
            free_memory_gb = free_memory / 1e9
            reserved_memory_gb = torch.cuda.memory_reserved() / 1e9
            allocated_memory_gb = torch.cuda.memory_allocated() / 1e9
            
            # Calculate effective free memory
            free_reserved_memory_gb = reserved_memory_gb - allocated_memory_gb
            effective_free_memory = free_memory_gb + max(0, free_reserved_memory_gb)
            
            logger.debug(f"Memory status: Free={free_memory_gb:.2f}GB, Reserved={reserved_memory_gb:.2f}GB, Allocated={allocated_memory_gb:.2f}GB")
            logger.debug(f"Effective free memory: {effective_free_memory:.2f}GB")
            
            # For small images, we can proceed even with limited memory
            if height <= 384 and width <= 384 and steps <= 20:
                if effective_free_memory > 0.2:  # Need at least 200MB free
                    message = f"Small image generation should fit in available memory: {effective_free_memory:.2f}GB available"
                    logger.info(message)
                    return True, message
            
            # Estimate memory requirements
            resolution_factor = (height * width) / (512 * 512)
            base_memory = 1.0  # Base memory for generation (1GB for 512x512)
            estimated_memory = base_memory * resolution_factor
            
            # Add step factor
            step_factor = steps / 20  # Normalize to 20 steps
            estimated_memory *= step_factor
            
            logger.debug(f"Memory estimate: {estimated_memory:.2f}GB for {height}x{width}, {steps} steps")
            
            # Check if we can fit the generation
            if effective_free_memory > estimated_memory:
                message = f"Memory check passed: {effective_free_memory:.2f}GB available, need ~{estimated_memory:.2f}GB"
                logger.info(message)
                return True, message
            else:
                # If close enough, let it try anyway
                if effective_free_memory > (estimated_memory * 0.8):
                    message = f"Memory might be tight, but we'll try: {effective_free_memory:.2f}GB available, need ~{estimated_memory:.2f}GB"
                    logger.warning(message)
                    return True, message
                else:
                    message = f"Insufficient GPU memory: {effective_free_memory:.2f}GB available, need ~{estimated_memory:.2f}GB"
                    logger.warning(message)
                    return False, message
                
        except Exception as e:
            logger.warning(f"Could not check GPU memory: {e}")
            # If memory check fails, proceed cautiously for small images
            if height <= 256 and width <= 256 and steps <= 15:
                return True, "Memory check failed, proceeding with small image"
            else:
                return False, f"Memory check failed: {e}"
