"""
from typing import List
Optimized Image generation service for Personal Chatter.

This module handles the generation of images using the Flux.1 model
with optimized memory management and step-by-step progress updates.
Includes LangChain integration and efficient RAM/VRAM allocation.
"""
import os
import logging
import time
import threading
import asyncio
import weakref
import gc
from typing import Dict, List, Optional, Any, Tuple, Callable, Union 
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

# Constants for image generation
MODEL_PATH = "./models/FLUX.1-dev"
IMG_DEFAULT_HEIGHT = 1024  # Default height (can be adjusted based on prompt/device capabilities)
IMG_DEFAULT_WIDTH = 1024   # Default width
NUM_INFERENCE_STEPS = 35   # Default number of inference steps
GUIDANCE_SCALE = 7.0       # Default guidance scale
MAX_SEQUENCE_LENGTH = 512  # Default max sequence length
OUTPUT_DIR = "./outputs"   # Output directory for generated images

# Memory management constants
MODEL_OFFLOAD_TIMEOUT = 300  # Seconds before offloading model from VRAM
CHECKPOINT_INTERVAL = 5      # Save intermediate results every N steps

# Global pipeline objects with memory optimization
_flux_pipeline = None
_pipeline_lock = threading.Lock()
_last_access_time = 0
_offload_timer = None
_pipeline_on_gpu = False

# Progress tracking
_progress_callbacks = weakref.WeakValueDictionary()  # session_id -> callback function

def _ensure_output_dir():
    """Ensure the output directory exists."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logger.info(f"Created output directory at {OUTPUT_DIR}")

def _clear_gpu_memory():
    """Clear GPU memory and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    logger.debug("GPU memory cleared and garbage collection performed")

def _move_pipeline_to_gpu():
    """Move pipeline components to GPU for inference."""
    global _pipeline_on_gpu, _flux_pipeline
    
    if not _flux_pipeline or _pipeline_on_gpu:
        return
        
    if torch.cuda.is_available():
        try:
            logger.info("Moving pipeline to GPU for inference")
            _flux_pipeline.to("cuda")
            _pipeline_on_gpu = True
            logger.debug("Pipeline successfully moved to GPU")
        except Exception as e:
            logger.warning(f"Failed to move pipeline to GPU: {e}")

def _move_pipeline_to_cpu():
    """Move pipeline components to CPU to free VRAM."""
    global _pipeline_on_gpu, _flux_pipeline
    
    if not _flux_pipeline or not _pipeline_on_gpu:
        return
        
    try:
        logger.info("Moving pipeline to CPU to free VRAM")
        _flux_pipeline.to("cpu")
        _pipeline_on_gpu = False
        _clear_gpu_memory()
        logger.debug("Pipeline successfully moved to CPU")
    except Exception as e:
        logger.warning(f"Failed to move pipeline to CPU: {e}")

def _schedule_offload():
    """Schedule pipeline offload from GPU after timeout."""
    global _offload_timer
    
    # Cancel existing timer
    if _offload_timer:
        _offload_timer.cancel()
    
    # Schedule new offload
    _offload_timer = threading.Timer(MODEL_OFFLOAD_TIMEOUT, _move_pipeline_to_cpu)
    _offload_timer.start()
    logger.debug(f"Scheduled pipeline offload in {MODEL_OFFLOAD_TIMEOUT} seconds")

def _update_access_time():
    """Update the last access time and reschedule offload."""
    global _last_access_time
    _last_access_time = time.time()
    _schedule_offload()

class ProgressCallback:
    """Callback class for tracking image generation progress."""
    
    def __init__(self, session_id: str, callback_func: Optional[Callable] = None):
        self.session_id = session_id
        self.callback_func = callback_func
        self.current_step = 0
        self.total_steps = 0
        self.start_time = time.time()
        
    def __call__(self, step: int, timestep: int, latents: torch.Tensor):
        """Called during each denoising step."""
        self.current_step = step
        elapsed_time = time.time() - self.start_time
        progress = (step / self.total_steps) * 100 if self.total_steps > 0 else 0
        
        # Log progress
        logger.debug(f"Generation progress: {progress:.1f}% (step {step}/{self.total_steps})")
        
        # Call external callback if provided
        if self.callback_func:
            try:
                self.callback_func(step, self.total_steps, progress, elapsed_time)
            except Exception as e:
                logger.warning(f"Error in progress callback: {e}")
        
        # Save checkpoint at intervals
        if step > 0 and step % CHECKPOINT_INTERVAL == 0:
            self._save_checkpoint(latents, step)
    
    def _save_checkpoint(self, latents: torch.Tensor, step: int):
        """Save intermediate generation checkpoint."""
        try:
            # This would require access to the pipeline's VAE to decode latents
            # For now, just log the checkpoint save
            logger.debug(f"Checkpoint saved at step {step}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint at step {step}: {e}")

def register_progress_callback(session_id: str, callback_func: Optional[Callable] = None):
    """Register a progress callback for a generation session."""
    global _progress_callbacks
    _progress_callbacks[session_id] = ProgressCallback(session_id, callback_func)
    logger.debug(f"Registered progress callback for session {session_id}")

def unregister_progress_callback(session_id: str):
    """Unregister a progress callback."""
    global _progress_callbacks
    if session_id in _progress_callbacks:
        del _progress_callbacks[session_id]
        logger.debug(f"Unregistered progress callback for session {session_id}")

def _get_flux_pipeline():
    """
    Get or initialize the Flux pipeline with optimized memory management.
    
    Returns:
        The Flux pipeline object or None if initialization failed
    """
    global _flux_pipeline, _pipeline_on_gpu
    
    # If we already have a pipeline, update access time and ensure it's ready
    if _flux_pipeline is not None:
        _update_access_time()
        return _flux_pipeline
    
    # Use a lock to ensure only one thread initializes the pipeline
    with _pipeline_lock:
        # Check again in case another thread initialized it while we were waiting
        if _flux_pipeline is not None:
            _update_access_time()
            return _flux_pipeline
        
        # Initialize the pipeline
        try:
            logger.info("Initializing Flux.1 image generation pipeline")
            start_time = time.time()
            
            # Import here to avoid loading dependencies unless needed
            from diffusers import FluxPipeline
            
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
                _clear_gpu_memory()
            else:
                logger.warning("CUDA not available, using CPU (will be slow)")
            
            # Load the pipeline on CPU first for memory efficiency
            logger.info("Loading pipeline on CPU for optimal memory management")
            _flux_pipeline = FluxPipeline.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=None  # Load on CPU first
            )
            
            # Move to CPU explicitly and apply optimizations
            _flux_pipeline = _flux_pipeline.to("cpu")
            _pipeline_on_gpu = False
            
            # Apply memory optimizations
            _flux_pipeline.enable_sequential_cpu_offload()
            logger.info("Enabled sequential CPU offloading for maximum compatibility")
            
            if hasattr(_flux_pipeline, "enable_attention_slicing"):
                _flux_pipeline.enable_attention_slicing("max")
                logger.info("Enabled attention slicing")
            
            if hasattr(_flux_pipeline, "enable_memory_efficient_attention"):
                _flux_pipeline.enable_memory_efficient_attention()
                logger.info("Enabled memory efficient attention")
            
            # Enable model CPU offloading for even better memory management
            if hasattr(_flux_pipeline, "enable_model_cpu_offload"):
                _flux_pipeline.enable_model_cpu_offload()
                logger.info("Enabled model CPU offloading")
            
            logger.info(f"Flux.1 model loaded in {time.time() - start_time:.2f} seconds")
            _update_access_time()
            return _flux_pipeline
            
        except Exception as e:
            logger.error(f"Error initializing Flux.1 pipeline: {e}", exc_info=True)
            return None

def generate_image(
    prompt: str, 
    height: int = IMG_DEFAULT_HEIGHT, 
    width: int = IMG_DEFAULT_WIDTH,
    steps: int = NUM_INFERENCE_STEPS, 
    guidance_scale: float = GUIDANCE_SCALE,
    negative_prompt: str = None,
    session_id: str = None,
    progress_callback: Optional[Callable] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate an image using the Flux.1 model with progress tracking.
    
    Args:
        prompt: The text prompt for image generation
        height: Image height in pixels
        width: Image width in pixels
        steps: Number of inference steps
        guidance_scale: Guidance scale for generation
        negative_prompt: Negative prompt to guide what to avoid
        session_id: Optional session ID for progress tracking
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Tuple containing:
            - Path to the generated image if successful, None otherwise
            - URL or relative path to the image for displaying
    """
    logger.info(f"Generating image with prompt: {prompt[:100]}...")
    _ensure_output_dir()
    
    # Default negative prompt if not provided
    if not negative_prompt:
        negative_prompt = """deformed, ugly, amateur, bad art, blurry, pixelated, 
        grainy, low resolution, poorly drawn, distorted proportions, disfigured, 
        oversaturated, undersaturated, bad anatomy, inconsistent style, style change, 
        character morphing, photorealistic, 3D render, 
        text errors, illegible text, garbled text, misspelled text, jumbled text, unreadable text, text artifacts,
        mutated hands, extra limbs, fused fingers, too many fingers, watermark, signature, artist name,
        clashing colors, messy lines, inconsistent lighting."""
    
    # Get the pipeline
    pipeline = _get_flux_pipeline()
    if pipeline is None:
        logger.error("Failed to initialize image generation pipeline")
        return None, None
    
    # Register progress callback if session_id provided
    if session_id:
        register_progress_callback(session_id, progress_callback)
    
    try:
        # Ensure pipeline is on GPU for inference
        if torch.cuda.is_available():
            _move_pipeline_to_gpu()
            
            # Log available memory before generation
            free_memory, total_memory = torch.cuda.mem_get_info()
            logger.info(f"GPU memory before generation: {free_memory / 1e9:.2f} GB free, {total_memory / 1e9:.2f} GB total")
        
        # Prepare progress callback
        callback = None
        if session_id and session_id in _progress_callbacks:
            progress_tracker = _progress_callbacks[session_id]
            progress_tracker.total_steps = steps
            progress_tracker.start_time = time.time()
            callback = progress_tracker
        
        # Generate the image with progress tracking
        start_time = time.time()
        logger.info("Starting image generation with progress tracking...")
        
        with torch.no_grad():
            image = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                callback=callback,
                callback_steps=1  # Call callback every step
            ).images[0]
        
        generation_time = time.time() - start_time
        logger.info(f"Image generated in {generation_time:.2f} seconds")
        
        # Save the image
        timestamp = int(time.time())
        safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)
        safe_prompt = safe_prompt[:50]  # Limit length
        
        filename = f"image_{safe_prompt}_{timestamp}.png"
        image_path = os.path.join(OUTPUT_DIR, filename)
        
        image.save(image_path)
        logger.info(f"Image saved to {image_path}")
        
        # Return the path and URL
        relative_url = f"/outputs/{filename}"  # Relative URL for web serving
        return image_path, relative_url
        
    except Exception as e:
        logger.error(f"Error generating image: {e}", exc_info=True)
        return None, None
    finally:
        # Clean up progress callback
        if session_id:
            unregister_progress_callback(session_id)
        
        # Schedule pipeline offload from GPU to save VRAM
        if torch.cuda.is_available():
            _schedule_offload()

def release_resources():
    """
    Release resources used by the image generation pipeline.
    Call this when the pipeline won't be needed for a while.
    """
    global _flux_pipeline, _offload_timer
    
    with _pipeline_lock:
        # Cancel any pending offload timer
        if _offload_timer:
            _offload_timer.cancel()
            _offload_timer = None
        
        if _flux_pipeline is not None:
            # Move to CPU first if on GPU
            _move_pipeline_to_cpu()
            
            # Delete the pipeline to free resources
            del _flux_pipeline
            _flux_pipeline = None
            
            # Clear all GPU memory
            _clear_gpu_memory()
                
            logger.info("Image generation resources released")

def get_model_status() -> Dict[str, Any]:
    """
    Get the current status of the image generation model.
    
    Returns:
        Dictionary containing model status information
    """
    global _flux_pipeline, _pipeline_on_gpu, _last_access_time
    
    status = {
        "model_loaded": _flux_pipeline is not None,
        "on_gpu": _pipeline_on_gpu,
        "last_access": _last_access_time,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "cuda_available": torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        try:
            free_memory, total_memory = torch.cuda.mem_get_info()
            status.update({
                "gpu_memory_free": free_memory,
                "gpu_memory_total": total_memory,
                "gpu_memory_used": total_memory - free_memory,
                "gpu_name": torch.cuda.get_device_name(0)
            })
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
    
    return status

# LangChain Integration
try:
    from langchain.llms.base import LLM
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain.schema import Generation, LLMResult
    
    class FluxImageGenerationLLM(LLM):
        """
        LangChain-compatible LLM wrapper for Flux.1 image generation.
        
        This allows Flux.1 to be used within LangChain pipelines and chains
        for image generation tasks.
        """
        
        height: int = IMG_DEFAULT_HEIGHT
        width: int = IMG_DEFAULT_WIDTH
        steps: int = NUM_INFERENCE_STEPS
        guidance_scale: float = GUIDANCE_SCALE
        
        @property
        def _llm_type(self) -> str:
            return "flux_image_generator"
        
        def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs
        ) -> str:
            """
            Generate an image and return the path/URL.
            
            Args:
                prompt: The image generation prompt
                stop: Not used for image generation
                run_manager: LangChain callback manager
                **kwargs: Additional generation parameters
                
            Returns:
                String containing the image path and URL
            """
            # Extract parameters from kwargs
            height = kwargs.get("height", self.height)
            width = kwargs.get("width", self.width)
            steps = kwargs.get("steps", self.steps)
            guidance_scale = kwargs.get("guidance_scale", self.guidance_scale)
            negative_prompt = kwargs.get("negative_prompt", None)
            session_id = kwargs.get("session_id", None)
            
            # Progress callback integration with LangChain
            progress_callback = None
            if run_manager:
                def langchain_progress_callback(step, total_steps, progress, elapsed_time):
                    run_manager.on_text(
                        f"Image generation progress: {progress:.1f}% (step {step}/{total_steps})",
                        verbose=True
                    )
                progress_callback = langchain_progress_callback
            
            # Generate the image
            image_path, image_url = generate_image(
                prompt=prompt,
                height=height,
                width=width,
                steps=steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                session_id=session_id,
                progress_callback=progress_callback
            )
            
            if image_path:
                return f"Generated image: {image_url} (saved to: {image_path})"
            else:
                return "Failed to generate image"
        
        async def _acall(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs
        ) -> str:
            """Async version of _call (currently just calls sync version)."""
            return self._call(prompt, stop, run_manager, **kwargs)
    
    # Export the LangChain-compatible class
    __all__ = [
        "generate_image", 
        "release_resources", 
        "get_model_status", 
        "register_progress_callback",
        "unregister_progress_callback",
        "FluxImageGenerationLLM"
    ]
    
    logger.info("LangChain integration available")
    
except ImportError:
    # LangChain not available, just export main functions
    __all__ = [
        "generate_image", 
        "release_resources", 
        "get_model_status", 
        "register_progress_callback",
        "unregister_progress_callback"
    ]
    logger.debug("LangChain not available, image generation will work without chain integration")

# Async variants for better web integration
async def generate_image_async(
    prompt: str,
    height: int = IMG_DEFAULT_HEIGHT,
    width: int = IMG_DEFAULT_WIDTH,
    steps: int = NUM_INFERENCE_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
    negative_prompt: str = None,
    session_id: str = None,
    progress_callback: Optional[Callable] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Async wrapper for image generation.
    
    This runs the generation in a thread pool to avoid blocking the event loop.
    """
    import concurrent.futures
    
    # Run in thread pool to avoid blocking
    with concurrent.futures.ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            generate_image,
            prompt, height, width, steps, guidance_scale, negative_prompt, session_id, progress_callback
        )
    
    return result
