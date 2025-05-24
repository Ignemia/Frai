"""
Core configuration management for Personal Chatter.

This module provides centralized configuration management with
support for environment-specific settings and dynamic updates.
"""

# Re-export main configuration functions for backwards compatibility
from .config.config_manager import load_config, save_config, get_config, update_config

__all__ = ['load_config', 'save_config', 'get_config', 'update_config']
