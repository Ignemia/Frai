"""
API initialization for the Image module.

This module provides the default exports and initialization for the image generation API.
"""
from .image_endpoints import image_router

__all__ = ['image_router']