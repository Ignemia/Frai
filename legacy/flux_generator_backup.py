"""
Main Flux.1 image generator - Refactored Clean Interface.

This module provides the main interface for image generation using
the Flux.1 model with a clean, modular architecture.
"""
import logging
from typing import Dict, Optional, Any, Tuple, Callable

from .interfaces import SyncImageGenerator, AsyncImageGenerator, FeedbackImageGenerator
from .style_presets import StylePreset
from .utils.validation import validate_prompt

logger = logging.getLogger(__name__)

# Load configuration
config = get_config()

# Constants for image generation (optimized for GPU memory constraints)
MODEL_PATH = config.get("models", {}).get("flux_image_path", "./models/FLUX.1-dev")
IMG_DEFAULT_HEIGHT = 512  # Reduced from 1024 for memory efficiency
IMG_DEFAULT_WIDTH = 512   # Reduced from 1024 for memory efficiency
NUM_INFERENCE_STEPS = 20  # Reduced from 35 for faster generation and less memory usage
GUIDANCE_SCALE = 7.0
MAX_SEQUENCE_LENGTH = 256  # Reduced from 512 for memory efficiency
OUTPUT_DIR = "./outputs"

# Global pipeline objects
_flux_pipeline = None
_pipeline_lock = threading.Lock()

def _ensure_output_dir():
    """Ensure the output directory exists."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logger.info(f"Created output directory at {OUTPUT_DIR}")

def _get_flux_pipeline():
    """
    Get or initialize the Flux pipeline with optimized memory management.
    
    Returns:
        The Flux pipeline object or None if initialization failed
    """
    global _flux_pipeline
    
    # If we already have a pipeline, update access time and ensure it's ready
    if _flux_pipeline is not None:
        update_access_time()
        return _flux_pipeline
    
    # Use a lock to ensure only one thread initializes the pipeline
    with _pipeline_lock:
        # Check again in case another thread initialized it while we were waiting
        if _flux_pipeline is not None:
            update_access_time()
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
            
            # Log model files for debugging
            model_files = os.listdir(MODEL_PATH)
            logger.info(f"Found {len(model_files)} files in model directory")
              # Log GPU information if available
            if torch.cuda.is_available():
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                props = torch.cuda.get_device_properties(0)
                logger.info(f"Total GPU memory: {props.total_memory / 1e9:.2f} GB")
                available_memory = torch.cuda.mem_get_info()[0] / 1e9
                logger.info(f"Available GPU memory: {available_memory:.2f} GB")
                
                # Clear CUDA cache before loading
                clear_gpu_memory()
                
                # Set environment variable for better memory management
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
                
                # Load directly to GPU if we have enough memory (>8GB available)
                target_device = "cuda" if available_memory > 8.0 else "cpu"
                logger.info(f"Loading model to: {target_device}")
            else:
                logger.warning("CUDA not available, using CPU (will be slow)")
                target_device = "cpu"
              # Load the pipeline with compatibility optimizations (like working prototype)
            logger.info("Loading Flux.1 pipeline with compatibility settings")
            
            # Load to CPU first to avoid initial GPU memory issues
            _flux_pipeline = FluxPipeline.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                variant=None,
                local_files_only=True
            )
            
            # Apply memory optimizations with sequential CPU offload (like working prototype)
            logger.info("Enabling sequential CPU offloading for maximum compatibility")
            if hasattr(_flux_pipeline, "enable_sequential_cpu_offload"):
                _flux_pipeline.enable_sequential_cpu_offload()
                logger.info("Sequential CPU offload enabled")
            
            # Apply additional memory optimizations
            logger.info("Applying compatible memory optimizations") 
            if hasattr(_flux_pipeline, "enable_attention_slicing"):
                _flux_pipeline.enable_attention_slicing("max")
                logger.info("Attention slicing enabled")
            
            if hasattr(_flux_pipeline, "enable_vae_slicing"):
                _flux_pipeline.enable_vae_slicing()
                logger.info("VAE slicing enabled")
                
            if hasattr(_flux_pipeline, "enable_vae_tiling"):
                _flux_pipeline.enable_vae_tiling()
                logger.info("VAE tiling enabled")
            
            logger.info(f"Flux.1 model loaded in {time.time() - start_time:.2f} seconds")
            update_access_time()
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
    style: StylePreset = None,
    session_id: str = None,
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
    logger.info(f"Starting image generation: '{prompt[:50]}...'")
    start_time = time.time()
    
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
    
    # Ensure output directory exists
    _ensure_output_dir()
    
    # Get the pipeline
    pipeline = _get_flux_pipeline()
    if pipeline is None:
        logger.error("Failed to initialize Flux.1 pipeline")
        return None, None
      # Set up progress tracking
    progress_cb = None
    if session_id:
        progress_cb = create_progress_callback(
            session_id=session_id,
            total_steps=steps,
            external_callback=progress_callback,
            save_checkpoints=False  # Can be made configurable
        )
      # Check GPU memory before attempting generation
    can_generate, memory_message = check_gpu_memory_for_generation(height, width, steps)
    if not can_generate:
        logger.error(f"Memory check failed: {memory_message}")
        logger.info("Consider reducing image size (e.g., 256x256) or inference steps (e.g., 10-15)")
        return None, None
    try:
        # Aggressive memory cleanup before generation
        import gc
        gc.collect()
        clear_gpu_memory()
        
        # Sequential CPU offload handles device management automatically, so no manual GPU move needed
        logger.debug("Using sequential CPU offload - device management is automatic")
        
        # Log memory status right before generation
        if torch.cuda.is_available():
            free_memory, total_memory = torch.cuda.mem_get_info()
            free_memory_gb = free_memory / 1e9
            total_memory_gb = total_memory / 1e9
            logger.info(f"GPU memory before generation: {free_memory_gb:.2f}GB free / {total_memory_gb:.2f}GB total")
        
        # Generate the image
        logger.info(f"Generating image with {steps} steps")
        
        # For small images, use plms scheduler if available for better memory efficiency
        if height <= 256 and width <= 256 and hasattr(pipeline, 'scheduler') and hasattr(pipeline.scheduler, 'config'):
            logger.debug("Using optimized memory settings for small images")
            # Try to optimize scheduler for memory efficiency
            if hasattr(pipeline.scheduler, 'config'):
                try:
                    pipeline.scheduler.config.use_karras_sigmas = True
                    logger.debug("Enabled Karras sigmas for more efficient sampling")
                except:
                    pass
        
        generation_kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
        }
        
        if negative_prompt:
            generation_kwargs["negative_prompt"] = negative_prompt
          # For progress tracking, we need to be more careful with the callback
        # Some diffusers versions have issues with callbacks, so wrap in try-catch
        if progress_cb:
            try:
                # Check if this version of FluxPipeline supports callbacks
                if hasattr(pipeline, "register_scheduler_callback"):
                    # Use the register_scheduler_callback method if available (newer diffusers)
                    def simple_callback(step, timestep, latents):
                        progress_cb(step, steps, step / steps, time.time() - generation_start)
                    
                    pipeline.register_scheduler_callback(simple_callback)
                    logger.debug("Progress callback configured with scheduler callback registry")
                    
                # Do not add callback to generation_kwargs as FLUX.1 might not support it
                # Some versions of diffusers don't support direct callback parameter
            except Exception as e:
                logger.warning(f"Progress callback not supported: {e}")
                # Continue without progress callback
        
        logger.info("Starting Flux.1 image generation...")
        generation_start = time.time()        # Use threading with timeout for cross-platform compatibility
        import threading
        import concurrent.futures
        
        # Define a function to run the generation with timeout
        def run_generation_with_timeout():
            with torch.inference_mode():
                return pipeline(**generation_kwargs)
        
        # Use ThreadPoolExecutor for timeout handling that works on all platforms
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_generation_with_timeout)
            try:
                # 5-minute timeout (300 seconds)
                result = future.result(timeout=300)
            except concurrent.futures.TimeoutError:
                logger.error("Image generation timed out after 5 minutes")
                # Cancel the future if possible and clean up
                future.cancel()
                # Force GPU memory cleanup
                clear_gpu_memory()
                raise TimeoutError("Image generation timed out after 5 minutes")
        
        generation_time = time.time() - generation_start
        logger.info(f"Image generation completed in {generation_time:.2f} seconds")
        
        # Process the result
        if hasattr(result, 'images') and len(result.images) > 0:
            image = result.images[0]
            
            # Generate filename
            safe_prompt = "".join(c for c in prompt[:50] if c.isalnum() or c in (' ', '_')).strip()
            safe_prompt = safe_prompt.replace(' ', '_')
            timestamp = int(time.time())
            filename = f"image_{safe_prompt}_{timestamp}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            # Save the image
            image.save(filepath)
            logger.info(f"Image saved to {filepath}")
            
            # Schedule pipeline offload after a delay
            schedule_offload(pipeline)
              # Clean up progress tracking
            if session_id and progress_cb:
                unregister_progress_callback(session_id)
            
            total_time = time.time() - start_time
            logger.info(f"Total image generation completed in {total_time:.2f} seconds")
            
            return filepath, filename
        else:
            logger.error("No image generated in result")
            return None, None
            
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"GPU out of memory during generation: {e}")
        logger.info("Moving pipeline to CPU and clearing memory...")
        
        # Move pipeline back to CPU and clear memory
        pipeline = move_pipeline_to_cpu(pipeline)
        clear_gpu_memory()
        
        # Clean up progress tracking
        if session_id:
            unregister_progress_callback(session_id)
        
        # Suggest lower settings
        logger.warning(f"Current settings: {width}x{height}, {steps} steps")
        logger.warning("Try using: 256x256 resolution with 10-15 inference steps")
        
        return None, None
        
        return None, None
        
        return None, None
        
    except Exception as e:
        logger.error(f"Error during image generation: {e}", exc_info=True)
        
        # Clean up progress tracking
        if session_id:
            unregister_progress_callback(session_id)
        
        return None, None

def generate_image_with_feedback(
    prompt: str,
    height: int = IMG_DEFAULT_HEIGHT,
    width: int = IMG_DEFAULT_WIDTH,
    steps: int = NUM_INFERENCE_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
    negative_prompt: str = None,
    style: StylePreset = None,
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
    # Analyze feedback and enhance prompt if provided
    enhanced_prompt = prompt
    analysis_results = None
    
    if user_feedback:
        try:
            feedback_analysis = analyze_feedback(user_feedback)
            analysis_results = feedback_analysis
            
            # Get prompt suggestions based on feedback
            suggestions = suggest_prompt_improvements(prompt, feedback_analysis)
            if suggestions.get("enhanced_prompt"):
                enhanced_prompt = suggestions["enhanced_prompt"]
                logger.info("Applied feedback-based prompt enhancement")
                
        except Exception as e:
            logger.warning(f"Error processing user feedback: {e}")
    
    # Generate the image with the enhanced prompt
    image_path, image_url = generate_image(
        enhanced_prompt,
        height=height,
        width=width,
        steps=steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        style=style,
        session_id=session_id,
        progress_callback=progress_callback
    )
    
    return image_path, image_url, analysis_results

async def generate_image_async(
    prompt: str,
    height: int = IMG_DEFAULT_HEIGHT,
    width: int = IMG_DEFAULT_WIDTH,
    steps: int = NUM_INFERENCE_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
    negative_prompt: str = None,
    style: StylePreset = None,
    session_id: str = None,
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
    
    # Use ThreadPoolExecutor to run the synchronous function
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            result = await loop.run_in_executor(
                executor,
                lambda: generate_image(
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

def get_model_status() -> Dict[str, Any]:
    """
    Get the current status of the Flux.1 model and memory usage.
    
    Returns:
        Dictionary containing model status information
    """
    global _flux_pipeline
    
    status = {
        "model_loaded": _flux_pipeline is not None,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "gpu_available": torch.cuda.is_available(),
        "memory_info": get_memory_status()
    }
    
    if torch.cuda.is_available():
        try:
            gpu_memory = torch.cuda.mem_get_info()
            status["gpu_memory"] = {
                "free_gb": gpu_memory[0] / 1e9,
                "total_gb": gpu_memory[1] / 1e9,
                "used_gb": (gpu_memory[1] - gpu_memory[0]) / 1e9
            }
        except Exception as e:
            logger.warning(f"Could not get GPU memory info: {e}")
            status["gpu_memory"] = {"error": str(e)}
    
    if _flux_pipeline is not None:
        try:
            # Check if pipeline is on GPU or CPU
            first_param = next(_flux_pipeline.parameters(), None)
            if first_param is not None:
                status["model_device"] = str(first_param.device)
            else:
                status["model_device"] = "unknown"
        except Exception as e:
            logger.warning(f"Could not determine model device: {e}")
            status["model_device"] = "error"
    
    return status

def unload_model() -> bool:
    """
    Unload the Flux.1 model from memory to free up resources.
    
    Returns:
        True if successfully unloaded, False otherwise
    """
    global _flux_pipeline
    
    try:
        with _pipeline_lock:
            if _flux_pipeline is not None:
                logger.info("Unloading Flux.1 model from memory")
                
                # Move to CPU first
                _flux_pipeline = move_pipeline_to_cpu(_flux_pipeline)
                
                # Delete the pipeline
                del _flux_pipeline
                _flux_pipeline = None
                
                # Clear GPU memory
                clear_gpu_memory()
                
                logger.info("Flux.1 model successfully unloaded")
                return True
            else:
                logger.info("Flux.1 model was not loaded")
                return True
                
    except Exception as e:
        logger.error(f"Error unloading Flux.1 model: {e}", exc_info=True)
        return False

def reload_model() -> bool:
    """
    Reload the Flux.1 model, useful for recovering from errors or applying updates.
    
    Returns:
        True if successfully reloaded, False otherwise
    """
    logger.info("Reloading Flux.1 model")
    
    # First unload the current model
    if not unload_model():
        logger.warning("Failed to unload current model, attempting to proceed anyway")
    
    # Wait a moment for cleanup
    time.sleep(2)
    
    # Try to load the model again
    try:
        pipeline = _get_flux_pipeline()
        if pipeline is not None:
            logger.info("Flux.1 model successfully reloaded")
            return True
        else:
            logger.error("Failed to reload Flux.1 model")
            return False
    except Exception as e:
        logger.error(f"Error reloading Flux.1 model: {e}", exc_info=True)
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
    if not prompt or not prompt.strip():
        return False, "Prompt cannot be empty", None
    
    prompt = prompt.strip()
    
    # Check length
    if len(prompt) > 1000:
        return False, "Prompt is too long (max 1000 characters)", prompt[:1000]
    
    # Check for potentially problematic content
    problematic_words = [
        "nude", "naked", "nsfw", "explicit", "sexual", "porn", "xxx",
        "violence", "blood", "gore", "death", "kill", "murder",
        "hate", "racist", "nazi", "terrorist"
    ]
    
    prompt_lower = prompt.lower()
    found_problematic = [word for word in problematic_words if word in prompt_lower]
    
    if found_problematic:
        return False, f"Prompt contains potentially inappropriate content: {', '.join(found_problematic)}", None
    
    # Basic suggestions for improvement
    suggestions = []
    if len(prompt) < 10:
        suggestions.append("Consider adding more descriptive details")
    
    if "," not in prompt and len(prompt.split()) > 3:
        suggestions.append("Consider using commas to separate concepts")
    
    # Check if it's a good artistic prompt
    artistic_keywords = ["style", "art", "painting", "digital", "illustration", "concept", "detailed", "beautiful"]
    has_artistic_terms = any(keyword in prompt_lower for keyword in artistic_keywords)
    
    if not has_artistic_terms:
        suggestions.append("Consider adding artistic style descriptors (e.g., 'digital art', 'detailed illustration')")
    
    suggested_prompt = None
    if suggestions:
        suggested_prompt = prompt
        if not has_artistic_terms:
            suggested_prompt = f"{prompt}, digital art, detailed"
    
    message = "Prompt is valid"
    if suggestions:
        message += f". Suggestions: {'; '.join(suggestions)}"
    
    return True, message, suggested_prompt

def check_gpu_memory_for_generation(height: int, width: int, steps: int) -> Tuple[bool, str]:
    """
    Check if there's sufficient GPU memory for the requested generation parameters.
    
    Args:
        height: Image height in pixels
        width: Image width in pixels
        steps: Number of inference steps
        
    Returns:
        Tuple of (can_generate: bool, message: str)
    """
    # Small images should always be allowed to attempt generation regardless of memory checks
    if height <= 256 and width <= 256 and steps <= 15:
        message = f"Small image requested ({height}x{width}, {steps} steps) - bypassing memory check"
        logger.info(message)
        return True, message
        
    if not torch.cuda.is_available():
        return True, "CUDA not available, will use CPU"
    
    try:
        # Get current GPU memory status
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        free_memory, total_device_memory = torch.cuda.mem_get_info()
        free_memory_gb = free_memory / 1e9
        reserved_memory_gb = torch.cuda.memory_reserved() / 1e9
        allocated_memory_gb = torch.cuda.memory_allocated() / 1e9
        
        # Calculate free memory within reserved blocks
        free_reserved_memory_gb = reserved_memory_gb - allocated_memory_gb
        
        # Actual free memory = completely free + free reserved
        effective_free_memory = free_memory_gb + max(0, free_reserved_memory_gb)
        
        logger.debug(f"Memory status: Free={free_memory_gb:.2f}GB, Reserved={reserved_memory_gb:.2f}GB, Allocated={allocated_memory_gb:.2f}GB")
        logger.debug(f"Effective free memory: {effective_free_memory:.2f}GB")
        
        # For small images, we can proceed even with limited memory
        if height <= 384 and width <= 384 and steps <= 20:
            if effective_free_memory > 0.2:  # Need at least 200MB free
                message = f"Small image generation should fit in available memory: {effective_free_memory:.2f}GB available"
                logger.info(message)
                return True, message
        
        # For larger images or if model loaded, estimate realistically
        resolution_factor = (height * width) / (512 * 512)
        base_memory = 1.0  # Base memory for generation (1GB for 512x512)
        estimated_memory = base_memory * resolution_factor
        
        # Add step factor (more steps = more intermediate tensors)
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
