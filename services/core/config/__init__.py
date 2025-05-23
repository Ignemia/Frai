"""
Configuration management module for Personal Chatter.

This module provides centralized configuration loading, saving,
and management functionality.
"""

# Re-export configuration functions for backwards compatibility
from .config_manager import load_config, save_config, get_config, update_config

__all__ = ['load_config', 'save_config', 'get_config', 'update_config']
