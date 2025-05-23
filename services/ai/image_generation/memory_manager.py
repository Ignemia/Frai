"""
GPU/VRAM memory management for image generation.

This module handles efficient memory allocation and cleanup
for the Flux.1 image generation pipeline.
"""
import logging
import time
import threading
import gc
import torch
from services.config import get_config

logger = logging.getLogger(__name__)

# Memory management constants from configuration
config = get_config()
memory_config = config.get("memory_management", {})
MODEL_OFFLOAD_TIMEOUT = memory_config.get("gpu_offload_timeout", 300)
ENABLE_ATTENTION_SLICING = memory_config.get("enable_attention_slicing", True)
ENABLE_MEMORY_EFFICIENT_ATTENTION = memory_config.get("enable_memory_efficient_attention", True)
ENABLE_CPU_OFFLOAD = memory_config.get("enable_cpu_offload", True)
AUTO_GC_AFTER_GENERATION = memory_config.get("auto_gc_after_generation", True)

# Global state for memory management
_last_access_time = 0
_offload_timer = None
_pipeline_on_gpu = False
_memory_lock = threading.Lock()

def clear_gpu_memory():
    """Clear GPU memory and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.debug("GPU memory cleared")
    
    if AUTO_GC_AFTER_GENERATION:
        gc.collect()
        logger.debug("Garbage collection completed")

def move_pipeline_to_gpu(pipeline):
    """
    Move pipeline to GPU for inference with smart memory management.
    
    Args:
        pipeline: The diffusion pipeline to move
        
    Returns:
        The pipeline moved to GPU, or original pipeline if no GPU available
    """
    global _pipeline_on_gpu
    
    if not torch.cuda.is_available():
        logger.debug("CUDA not available, keeping pipeline on CPU")
        return pipeline
    
    with _memory_lock:
        if not _pipeline_on_gpu:
            try:
                logger.debug("Moving pipeline to GPU for inference")
                
                # Check available memory before moving
                available_memory = torch.cuda.mem_get_info()[0] / 1e9
                logger.debug(f"Available GPU memory before move: {available_memory:.2f}GB")
                
                # If we have sequential CPU offload enabled, we don't need to manually move
                # The pipeline will automatically manage GPU/CPU placement
                if hasattr(pipeline, 'enable_sequential_cpu_offload') and hasattr(pipeline, '_is_offloaded'):
                    logger.debug("Pipeline has sequential CPU offload, letting it manage device placement")
                    # Just ensure we're ready for inference
                    if hasattr(pipeline, 'to'):
                        # Only move the main components that need to be on GPU during inference
                        pass  # Let sequential offload handle this
                else:
                    # Manual move to GPU
                    pipeline = pipeline.to("cuda")
                
                _pipeline_on_gpu = True
                
                # Log GPU memory usage after move
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1e9
                    memory_reserved = torch.cuda.memory_reserved() / 1e9
                    logger.debug(f"GPU memory after loading: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
                    
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"GPU out of memory when moving pipeline: {e}")
                clear_gpu_memory()
                # Keep pipeline on CPU
                logger.info("Pipeline will remain on CPU due to memory constraints")
        
        return pipeline

def move_pipeline_to_cpu(pipeline):
    """
    Move pipeline to CPU to free VRAM.
    
    Args:
        pipeline: The diffusion pipeline to move
        
    Returns:
        The pipeline moved to CPU
    """
    global _pipeline_on_gpu
    
    with _memory_lock:
        if _pipeline_on_gpu:
            logger.debug("Moving pipeline to CPU to free VRAM")
            pipeline = pipeline.to("cpu")
            _pipeline_on_gpu = False
            clear_gpu_memory()
        
        return pipeline

def update_access_time():
    """Update the last access time for memory management."""
    global _last_access_time
    _last_access_time = time.time()

def schedule_offload(pipeline):
    """
    Schedule pipeline offload to CPU after timeout.
    
    Args:
        pipeline: The pipeline to potentially offload
    """
    global _offload_timer
    
    def offload_if_timeout():
        if time.time() - _last_access_time >= MODEL_OFFLOAD_TIMEOUT:
            logger.info(f"Pipeline unused for {MODEL_OFFLOAD_TIMEOUT}s, offloading to CPU")
            move_pipeline_to_cpu(pipeline)
        else:
            # Reschedule if still being used
            schedule_offload(pipeline)
    
    # Cancel existing timer if any
    if _offload_timer is not None:
        _offload_timer.cancel()
    
    # Schedule new timer
    _offload_timer = threading.Timer(MODEL_OFFLOAD_TIMEOUT, offload_if_timeout)
    _offload_timer.daemon = True
    _offload_timer.start()

def apply_memory_optimizations(pipeline):
    """
    Apply memory optimizations to the pipeline based on configuration.
    
    Args:
        pipeline: The diffusion pipeline to optimize
        
    Returns:
        The optimized pipeline
    """
    # Apply memory optimizations based on configuration
    if ENABLE_CPU_OFFLOAD:
        pipeline.enable_sequential_cpu_offload()
        logger.info("Enabled sequential CPU offloading for maximum compatibility")
    
    if ENABLE_ATTENTION_SLICING and hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing("max")
        logger.info("Enabled attention slicing")
    
    if ENABLE_MEMORY_EFFICIENT_ATTENTION and hasattr(pipeline, "enable_memory_efficient_attention"):
        pipeline.enable_memory_efficient_attention()
        logger.info("Enabled memory efficient attention")
    
    # Enable model CPU offloading for even better memory management
    if ENABLE_CPU_OFFLOAD and hasattr(pipeline, "enable_model_cpu_offload"):
        pipeline.enable_model_cpu_offload()
        logger.info("Enabled model CPU offloading")
    
    return pipeline

def apply_memory_optimizations_manual(pipeline):
    """
    Apply GPU-prioritized memory optimizations.
    Uses partial offloading only when necessary to keep GPU as primary processor.
    
    Args:
        pipeline: The diffusion pipeline to optimize
        
    Returns:
        The optimized pipeline
    """
    # Enable attention slicing for better memory efficiency on GPU
    if ENABLE_ATTENTION_SLICING and hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing("auto")  # Use auto instead of max for better GPU performance
        logger.info("Enabled attention slicing (auto) for GPU efficiency")
    
    # Enable memory efficient attention to reduce GPU memory usage
    if ENABLE_MEMORY_EFFICIENT_ATTENTION and hasattr(pipeline, "enable_memory_efficient_attention"):
        pipeline.enable_memory_efficient_attention()
        logger.info("Enabled memory efficient attention for GPU")
    
    # Check if device mapping is already active
    has_device_map = hasattr(pipeline, 'hf_device_map') and pipeline.hf_device_map is not None
    
    if has_device_map:
        logger.info("Device mapping detected - skipping sequential CPU offload (already handled by device mapping)")
    else:
        # Use sequential CPU offload only if no device mapping is active
        if hasattr(pipeline, "enable_sequential_cpu_offload"):
            pipeline.enable_sequential_cpu_offload()
            logger.info("Enabled sequential CPU offload (keeps main generation on GPU)")
    
    # Enable VAE slicing to reduce memory spikes during VAE operations
    if hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
        logger.info("Enabled VAE slicing for GPU memory efficiency")
    
    # Enable VAE tiling for large images while keeping processing on GPU
    if hasattr(pipeline, "enable_vae_tiling"):
        pipeline.enable_vae_tiling()
        logger.info("Enabled VAE tiling for large image support on GPU")
    
    logger.info("Applied GPU-prioritized memory optimizations")
    
    return pipeline

def get_memory_status():
    """
    Get current memory status information.
    
    Returns:
        Dictionary containing memory status information
    """
    status = {
        "pipeline_on_gpu": _pipeline_on_gpu,
        "last_access_time": _last_access_time,
        "auto_gc_enabled": AUTO_GC_AFTER_GENERATION,
        "offload_timeout": MODEL_OFFLOAD_TIMEOUT
    }
    
    if torch.cuda.is_available():
        try:
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_reserved = torch.cuda.memory_reserved() / 1e9
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            free_memory = total_memory - memory_reserved
            
            status.update({
                "cuda_available": True,
                "gpu_name": torch.cuda.get_device_name(0),
                "memory_allocated_gb": memory_allocated,
                "memory_reserved_gb": memory_reserved,
                "memory_total_gb": total_memory,
                "memory_free_gb": free_memory
            })
        except Exception as e:
            logger.error(f"Error getting GPU memory status: {e}")
            status["cuda_available"] = False
    else:
        status["cuda_available"] = False
    
    return status
