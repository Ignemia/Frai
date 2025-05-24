"""
Configuration manager for image generation system.

Handles loading and managing configuration settings for the image generation
pipeline, including model paths, default parameters, and memory settings.
"""
import os
from typing import Dict, Any
from services.config import get_config


class ConfigManager:
    """Manages configuration for image generation system."""
    
    def __init__(self):
        """Initialize configuration manager."""
        self._config = get_config()
        self._image_config = self._config.get("image_generation", {})
        
    @property
    def model_path(self) -> str:
        """Get the path to the Flux model."""
        return self._config.get("models", {}).get("flux_image_path", "./models/FLUX.1-dev")
    
    @property
    def output_dir(self) -> str:
        """Get the output directory for generated images."""
        return self._image_config.get("output_dir", "./outputs")
        
    @property
    def default_height(self) -> int:
        """Get default image height."""
        return self._image_config.get("default_height", 512)
        
    @property
    def default_width(self) -> int:
        """Get default image width.""" 
        return self._image_config.get("default_width", 512)
        
    @property
    def default_steps(self) -> int:
        """Get default number of inference steps."""
        return self._image_config.get("default_steps", 20)
        
    @property
    def default_guidance_scale(self) -> float:
        """Get default guidance scale."""
        return self._image_config.get("default_guidance_scale", 7.0)
        
    @property
    def max_sequence_length(self) -> int:
        """Get maximum sequence length."""
        return self._image_config.get("max_sequence_length", 256)
        
    @property
    def memory_settings(self) -> Dict[str, Any]:
        """Get memory management settings."""
        return self._image_config.get("memory", {
            "enable_sequential_cpu_offload": True,
            "enable_attention_slicing": True,
            "enable_vae_slicing": True,
            "enable_vae_tiling": True,
            "pytorch_cuda_alloc_conf": "max_split_size_mb:512"
        })
        
    def ensure_output_dir(self) -> None:
        """Ensure the output directory exists."""
        output_dir = self.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def get_memory_setting(self, key: str, default: Any = None) -> Any:
        """Get a specific memory setting."""
        return self.memory_settings.get(key, default)
