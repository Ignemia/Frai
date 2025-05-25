"""
Backend Package

This package provides all backend functionality including AI models,
database operations, and preprocessing utilities.

The models submodule provides comprehensive Pydantic models for
type safety and data validation across all backend operations.
"""

# Import key backend functionality
from .ai import (
    load_chat_model,
    load_image_model, 
    load_voice_activity_detection_model,
    load_speech_recognition_model,
    load_text_to_speech_model
)

from .database import (
    initiate_database_connection,
    test_database_connection
)

# Import all models for easy access
from .models import *

__all__ = [
    # AI Functions
    "load_chat_model",
    "load_image_model",
    "load_voice_activity_detection_model", 
    "load_speech_recognition_model",
    "load_text_to_speech_model",
    
    # Database Functions
    "initiate_database_connection",
    "test_database_connection",
    
    # All models are re-exported from models module
]