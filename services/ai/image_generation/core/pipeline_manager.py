"""
Pipeline manager for Flux.1 model.

Handles initialization, lifecycle management, and memory optimization
of the Flux.1 image generation pipeline.
"""
import os
import logging
import time
import threading
from typing import Optional
import torch

logger = logging.getLogger(__name__)


class PipelineManager:
    """Manages the Flux.1 pipeline lifecycle and memory optimization."""
    
    def __init__(self, config_manager):
        """Initialize the pipeline manager.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        self._pipeline = None
        self._pipeline_lock = threading.Lock()
        self._last_access_time = None
        
    @property
    def pipeline(self):
        """Get the current pipeline instance."""
        return self._pipeline
        
    @property
    def is_loaded(self) -> bool:
        """Check if pipeline is loaded."""
        return self._pipeline is not None
        
    def get_pipeline(self):
        """
        Get or initialize the Flux pipeline with optimized memory management.
        
        Returns:
            The Flux pipeline object or None if initialization failed
        """
        # If we already have a pipeline, update access time and return it
        if self._pipeline is not None:
            self._update_access_time()
            return self._pipeline
        
        # Use a lock to ensure only one thread initializes the pipeline
        with self._pipeline_lock:
            # Check again in case another thread initialized it while we were waiting
            if self._pipeline is not None:
                self._update_access_time()
                return self._pipeline
            
            return self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the Flux pipeline."""
        try:
            logger.info("Initializing Flux.1 image generation pipeline")
            start_time = time.time()
            
            # Import here to avoid loading dependencies unless needed
            from diffusers import FluxPipeline
            
            # Check if model path exists
            model_path = self.config.model_path
            if not os.path.exists(model_path):
                logger.error(f"Flux.1 model directory not found at {model_path}")
                return None
            
            # Log model files for debugging
            model_files = os.listdir(model_path)
            logger.info(f"Found {len(model_files)} files in model directory")
            
            # Log GPU information and prepare memory settings
            self._log_gpu_info()
            self._prepare_memory_environment()
            
            # Load the pipeline with compatibility optimizations
            logger.info("Loading Flux.1 pipeline with compatibility settings")
            
            # Load to CPU first to avoid initial GPU memory issues
            self._pipeline = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                variant=None,
                local_files_only=True
            )
            
            # Apply memory optimizations
            self._apply_memory_optimizations()
            
            logger.info(f"Flux.1 model loaded in {time.time() - start_time:.2f} seconds")
            self._update_access_time()
            return self._pipeline
            
        except Exception as e:
            logger.error(f"Error initializing Flux.1 pipeline: {e}", exc_info=True)
            return None
    
    def _log_gpu_info(self):
        """Log GPU information if available."""
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            logger.info(f"Total GPU memory: {props.total_memory / 1e9:.2f} GB")
            available_memory = torch.cuda.mem_get_info()[0] / 1e9
            logger.info(f"Available GPU memory: {available_memory:.2f} GB")
        else:
            logger.warning("CUDA not available, using CPU (will be slow)")
    
    def _prepare_memory_environment(self):
        """Prepare memory environment settings."""
        if torch.cuda.is_available():
            # Clear CUDA cache before loading
            from ..memory_manager import clear_gpu_memory
            clear_gpu_memory()
            
            # Set environment variable for better memory management
            pytorch_conf = self.config.get_memory_setting("pytorch_cuda_alloc_conf")
            if pytorch_conf:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = pytorch_conf
    
    def _apply_memory_optimizations(self):
        """Apply memory optimizations to the pipeline."""
        if not self._pipeline:
            return
            
        memory_settings = self.config.memory_settings
        
        # Apply sequential CPU offload
        if memory_settings.get("enable_sequential_cpu_offload", True):
            if hasattr(self._pipeline, "enable_sequential_cpu_offload"):
                self._pipeline.enable_sequential_cpu_offload()
                logger.info("Sequential CPU offload enabled")
        
        # Apply attention slicing
        if memory_settings.get("enable_attention_slicing", True):
            if hasattr(self._pipeline, "enable_attention_slicing"):
                self._pipeline.enable_attention_slicing("max")
                logger.info("Attention slicing enabled")
        
        # Apply VAE slicing
        if memory_settings.get("enable_vae_slicing", True):
            if hasattr(self._pipeline, "enable_vae_slicing"):
                self._pipeline.enable_vae_slicing()
                logger.info("VAE slicing enabled")
                
        # Apply VAE tiling
        if memory_settings.get("enable_vae_tiling", True):
            if hasattr(self._pipeline, "enable_vae_tiling"):
                self._pipeline.enable_vae_tiling()
                logger.info("VAE tiling enabled")
    
    def _update_access_time(self):
        """Update the last access time."""
        self._last_access_time = time.time()
        # Update access time in memory manager
        try:
            from ..memory_manager import update_access_time
            update_access_time()
        except ImportError:
            pass
    
    def unload(self) -> bool:
        """
        Unload the pipeline from memory.
        
        Returns:
            True if successfully unloaded, False otherwise
        """
        try:
            with self._pipeline_lock:
                if self._pipeline is not None:
                    logger.info("Unloading Flux.1 model from memory")
                    
                    # Move to CPU first
                    try:
                        from ..memory_manager import move_pipeline_to_cpu, clear_gpu_memory
                        self._pipeline = move_pipeline_to_cpu(self._pipeline)
                    except ImportError:
                        pass
                    
                    # Delete the pipeline
                    del self._pipeline
                    self._pipeline = None
                    
                    # Clear GPU memory
                    try:
                        from ..memory_manager import clear_gpu_memory
                        clear_gpu_memory()
                    except ImportError:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    logger.info("Flux.1 model successfully unloaded")
                    return True
                else:
                    logger.info("Flux.1 model was not loaded")
                    return True
                    
        except Exception as e:
            logger.error(f"Error unloading Flux.1 model: {e}", exc_info=True)
            return False
    
    def reload(self) -> bool:
        """
        Reload the pipeline.
        
        Returns:
            True if successfully reloaded, False otherwise
        """
        logger.info("Reloading Flux.1 model")
        
        # First unload the current model
        if not self.unload():
            logger.warning("Failed to unload current model, attempting to proceed anyway")
        
        # Wait a moment for cleanup
        time.sleep(2)
        
        # Try to load the model again
        try:
            pipeline = self.get_pipeline()
            if pipeline is not None:
                logger.info("Flux.1 model successfully reloaded")
                return True
            else:
                logger.error("Failed to reload Flux.1 model")
                return False
        except Exception as e:
            logger.error(f"Error reloading Flux.1 model: {e}", exc_info=True)
            return False
