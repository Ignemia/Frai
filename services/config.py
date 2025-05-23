"""
Configuration module for Personal Chatter.

This module provides functions to load, get and update application
configuration settings.
"""
import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default configuration 
DEFAULT_CONFIG = {
    "command_preprocessing": {
        "enabled": True,
        "use_main_model": True,
        "store_memory_enabled": True,
        "store_user_info_enabled": True,
        "image_generation_enabled": True,
        "online_search_enabled": True,
        "local_search_enabled": True
    },
    "models": {
        "flux_image_path": "./models/FLUX.1-dev",
        "main_llm_path": "./models/gemma-3-4b-it",
        "embedding_model_path": "./models/Gemma-Embeddings-v1.0"
    },
    "memory_management": {
        "gpu_offload_timeout": 300,
        "enable_attention_slicing": True,
        "enable_memory_efficient_attention": True,
        "enable_cpu_offload": True,
        "checkpoint_save_interval": 5,
        "auto_gc_after_generation": True
    },
    "api": {
        "brave_search_api_key_env": "BRAVE_SEARCH_API_KEY"
    }
}

# Global configuration object
_config = None

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file or use default.
    
    Args:
        config_path: Path to configuration file, if None uses default path
        
    Returns:
        Dictionary containing configuration settings
    """
    global _config
    
    # Default path is config.json in the root directory
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
    
    # If the config file exists, load it
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                loaded_config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
                
                # Merge with default config to ensure all keys exist
                merged_config = DEFAULT_CONFIG.copy()
                for section, values in loaded_config.items():
                    if section in merged_config and isinstance(merged_config[section], dict):
                        merged_config[section].update(values)
                    else:
                        merged_config[section] = values
                        
                _config = merged_config
                return _config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.warning("Using default configuration")
            _config = DEFAULT_CONFIG.copy()
            return _config
    else:
        logger.warning(f"Configuration file {config_path} not found, using default config")
        _config = DEFAULT_CONFIG.copy()
        return _config

def save_config(config_path: Optional[str] = None) -> bool:
    """
    Save current configuration to file.
    
    Args:
        config_path: Path to save configuration file, if None uses default path
        
    Returns:
        True if successful, False otherwise
    """
    global _config
    
    # Default path is config.json in the root directory
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
    
    if _config is None:
        _config = DEFAULT_CONFIG.copy()
    
    try:
        with open(config_path, "w") as f:
            json.dump(_config, f, indent=4)
            logger.info(f"Saved configuration to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return False

def get_config() -> Dict[str, Any]:
    """
    Get the current configuration.
    
    Returns:
        Dictionary containing configuration settings
    """
    global _config
    
    if _config is None:
        return load_config()
    
    return _config

def update_config(new_values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values.
    
    Args:
        new_values: Dictionary containing new configuration values
        
    Returns:
        Updated configuration dictionary
    """
    global _config
    
    if _config is None:
        _config = load_config()
    
    # Update the configuration recursively
    def update_dict_recursive(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                update_dict_recursive(d[k], v)
            else:
                d[k] = v
    
    update_dict_recursive(_config, new_values)
    return _config
