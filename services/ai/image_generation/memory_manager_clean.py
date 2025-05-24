"""
GPU/VRAM memory management for image generation.

This module handles efficient memory allocation and cleanup
for the Flux.1 image generation pipeline.
"""
import logging
import time
import threading
import gc
from typing import Dict, Any, Optional, TYPE_CHECKING
import torch
from services.config import get_config

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages GPU memory and pipeline offloading for image generation."""
    
    def __init__(self):
        self.config = get_config().get("image_generation", {}).get("memory", {})
        self._last_access_time = 0
        self._offload_timer = None
        self._pipeline_on_gpu = False
        self._memory_lock = threading.Lock()
    
    @property
    def memory_config(self) -> Dict[str, Any]:
        """Get memory configuration."""
        return {
            "gpu_offload_timeout": self.config.get("gpu_offload_timeout", 300),
            "enable_attention_slicing": self.config.get("enable_attention_slicing", True),
            "enable_memory_efficient_attention": self.config.get("enable_memory_efficient_attention", True),
            "enable_cpu_offload": self.config.get("enable_cpu_offload", True),
            "auto_gc_after_generation": self.config.get("auto_gc_after_generation", True)
        }
    
    def clear_gpu_memory(self) -> None:
        """Clear GPU memory and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("GPU memory cleared")
        
        if self.memory_config["auto_gc_after_generation"]:
            gc.collect()
            logger.debug("Garbage collection completed")
    
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get GPU memory information in GB."""
        if not torch.cuda.is_available():
            return {"available": False}
        
        try:
            free_memory, total_memory = torch.cuda.mem_get_info()
            allocated_memory = torch.cuda.memory_allocated()
            reserved_memory = torch.cuda.memory_reserved()
            
            return {
                "available": True,
                "free_gb": free_memory / 1e9,
                "total_gb": total_memory / 1e9,
                "allocated_gb": allocated_memory / 1e9,
                "reserved_gb": reserved_memory / 1e9,
                "device_name": torch.cuda.get_device_name(0)
            }
        except Exception as e:
            logger.error(f"Error getting GPU memory info: {e}")
            return {"available": False, "error": str(e)}
    
    def update_access_time(self) -> None:
        """Update the last access time for the pipeline."""
        self._last_access_time = time.time()
    
    def move_pipeline_to_gpu(self, pipeline: 'DiffusionPipeline') -> 'DiffusionPipeline':
        """Move pipeline to GPU for generation."""
        with self._memory_lock:
            if not self._pipeline_on_gpu and torch.cuda.is_available():
                try:
                    logger.debug("Moving pipeline to GPU for generation")
                    pipeline = pipeline.to("cuda")
                    self._pipeline_on_gpu = True
                    self.update_access_time()
                    
                except torch.cuda.OutOfMemoryError as e:
                    logger.error(f"GPU out of memory when moving pipeline: {e}")
                    self.clear_gpu_memory()
                    raise RuntimeError(f"Not enough GPU memory: {e}")
        
        return pipeline
    
    def move_pipeline_to_cpu(self, pipeline: 'DiffusionPipeline') -> 'DiffusionPipeline':
        """Move pipeline to CPU to free VRAM."""
        with self._memory_lock:
            if self._pipeline_on_gpu:
                logger.debug("Moving pipeline to CPU to free VRAM")
                pipeline = pipeline.to("cpu")
                self._pipeline_on_gpu = False
                self.clear_gpu_memory()
        
        return pipeline
    
    def schedule_offload(self, pipeline: 'DiffusionPipeline', delay: Optional[int] = None) -> None:
        """Schedule pipeline offload to CPU after a delay."""
        if delay is None:
            delay = self.memory_config["gpu_offload_timeout"]
        
        if self._offload_timer:
            self._offload_timer.cancel()
        
        def offload():
            time.sleep(delay)
            if time.time() - self._last_access_time >= delay:
                logger.debug("Auto-offloading pipeline to CPU")
                self.move_pipeline_to_cpu(pipeline)
        
        self._offload_timer = threading.Thread(target=offload, daemon=True)
        self._offload_timer.start()
    
    def apply_memory_optimizations(self, pipeline: 'DiffusionPipeline') -> 'DiffusionPipeline':
        """Apply memory optimizations to the pipeline."""
        config = self.memory_config
        
        # Apply sequential CPU offload
        if config.get("enable_cpu_offload", True):
            if hasattr(pipeline, "enable_sequential_cpu_offload"):
                pipeline.enable_sequential_cpu_offload()
                logger.debug("Sequential CPU offload enabled")
        
        # Apply attention slicing
        if config.get("enable_attention_slicing", True):
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing("max")
                logger.debug("Attention slicing enabled")
        
        # Apply VAE slicing if available
        if hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
            logger.debug("VAE slicing enabled")
        
        # Apply VAE tiling if available  
        if hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()
            logger.debug("VAE tiling enabled")
        
        return pipeline
    
    def get_status(self) -> Dict[str, Any]:
        """Get memory manager status."""
        status = {
            "pipeline_on_gpu": self._pipeline_on_gpu,
            "last_access_time": self._last_access_time,
            "memory_config": self.memory_config
        }
        
        # Add GPU memory info
        status.update(self.get_gpu_memory_info())
        
        return status


# Global memory manager instance
_memory_manager = None


def get_memory_manager() -> MemoryManager:
    """Get or create the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


# Convenience functions for backward compatibility
def clear_gpu_memory() -> None:
    """Clear GPU memory."""
    get_memory_manager().clear_gpu_memory()


def move_pipeline_to_gpu(pipeline: 'DiffusionPipeline') -> 'DiffusionPipeline':
    """Move pipeline to GPU."""
    return get_memory_manager().move_pipeline_to_gpu(pipeline)


def move_pipeline_to_cpu(pipeline: 'DiffusionPipeline') -> 'DiffusionPipeline':
    """Move pipeline to CPU."""
    return get_memory_manager().move_pipeline_to_cpu(pipeline)


def update_access_time() -> None:
    """Update pipeline access time."""
    get_memory_manager().update_access_time()


def schedule_offload(pipeline: 'DiffusionPipeline', delay: Optional[int] = None) -> None:
    """Schedule pipeline offload."""
    get_memory_manager().schedule_offload(pipeline, delay)


def apply_memory_optimizations(pipeline: 'DiffusionPipeline') -> 'DiffusionPipeline':
    """Apply memory optimizations."""
    return get_memory_manager().apply_memory_optimizations(pipeline)


def get_memory_status() -> Dict[str, Any]:
    """Get memory status."""
    return get_memory_manager().get_status()
