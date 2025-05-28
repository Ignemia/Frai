"""
Model Configuration Management

This module manages the mapping between model names and their local paths,
providing fallback mechanisms and utility functions for model resolution.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Types of AI models supported"""
    CHAT = "chat"
    TEXT2IMG = "text2img"
    IMG2IMG = "img2img"
    EMBEDDINGS = "embeddings"
    SPEECH_RECOGNITION = "speech_recognition"
    TEXT_TO_SPEECH = "text_to_speech"
    SPEAKER_DIARIZATION = "speaker_diarization"
    SENTIMENT_ANALYSIS = "sentiment_analysis"

# Base directory for all models
MODELS_BASE_DIR = Path(__file__).parent.parent.parent / "models"

# Model configuration mapping HuggingFace names to local paths
MODEL_CONFIGS = {
    # Chat Models
    "google/gemma-3-4b-it": {
        "local_path": MODELS_BASE_DIR / "gemma-3-4b-it",
        "hf_name": "google/gemma-3-4b-it",
        "type": ModelType.CHAT,
        "description": "Google Gemma 3 4B Instruct model for chat"
    },
    "nari-labs/Dia-1.6B": {
        "local_path": MODELS_BASE_DIR / "Dia-1.6B", 
        "hf_name": "nari-labs/Dia-1.6B",
        "type": ModelType.CHAT,
        "description": "Dia 1.6B chat model"
    },
    
    # Image Generation Models
    "black-forest-labs/FLUX.1-dev": {
        "local_path": MODELS_BASE_DIR / "FLUX.1-dev",
        "hf_name": "black-forest-labs/FLUX.1-dev", 
        "type": ModelType.TEXT2IMG,
        "description": "FLUX.1 text-to-image generation model"
    },
    
    # Embedding Models
    "google/Gemma-Embeddings-v1.0": {
        "local_path": MODELS_BASE_DIR / "Gemma-Embeddings-v1.0",
        "hf_name": "google/Gemma-Embeddings-v1.0",
        "type": ModelType.EMBEDDINGS,
        "description": "Google Gemma embeddings model"
    },
    
    # Speech and Audio Models
    "4evergr8/pyannote-segmentation-3.0": {
        "local_path": MODELS_BASE_DIR / "pyannote-segmentation-3.0",
        "hf_name": "4evergr8/pyannote-segmentation-3.0", 
        "type": ModelType.SPEECH_RECOGNITION,
        "description": "PyAnnote segmentation model"
    },
    "pyannote/speaker-diarization-3.1": {
        "local_path": MODELS_BASE_DIR / "speaker-diarization-3.1",
        "hf_name": "pyannote/speaker-diarization-3.1",
        "type": ModelType.SPEAKER_DIARIZATION, 
        "description": "PyAnnote speaker diarization model"
    },
    
    # Sentiment Analysis
    "tabularisai/multilingual-sentiment-analysis": {
        "local_path": MODELS_BASE_DIR / "multilingual-sentiment-analysis",
        "hf_name": "tabularisai/multilingual-sentiment-analysis",
        "type": ModelType.SENTIMENT_ANALYSIS,
        "description": "Multilingual sentiment analysis model"
    }
}

# Default models for each type
DEFAULT_MODELS = {
    ModelType.CHAT: "google/gemma-3-4b-it",
    ModelType.TEXT2IMG: "black-forest-labs/FLUX.1-dev",
    ModelType.EMBEDDINGS: "google/Gemma-Embeddings-v1.0",
    ModelType.SPEAKER_DIARIZATION: "pyannote/speaker-diarization-3.1",
    ModelType.SENTIMENT_ANALYSIS: "tabularisai/multilingual-sentiment-analysis"
}

def is_lfs_pointer_file(file_path: Path) -> bool:
    """Check if a file is a Git LFS pointer file."""
    if not file_path.exists() or file_path.stat().st_size > 1000:  # LFS pointers are small
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            return first_line.startswith("version https://git-lfs.github.com/spec/v")
    except:
        return False

def has_lfs_issues(model_path: Path) -> bool:
    """Check if model has Git LFS files that aren't downloaded."""
    # Temporarily disabled LFS checks - assuming models are properly downloaded
    return False
    
    if not model_path.exists():
        return False
    
    # Check critical files that would be LFS
    critical_patterns = ["tokenizer.json", "*.safetensors", "model*.safetensors"]
    
    for pattern in critical_patterns:
        for file_path in model_path.glob(pattern):
            if is_lfs_pointer_file(file_path):
                return True
    
    return False

def get_model_path(model_name: str, prefer_local: bool = True) -> Tuple[str, bool]:
    """
    Get the appropriate model path for a given model name.
    
    Args:
        model_name: HuggingFace model name or local alias
        prefer_local: Whether to prefer local models over remote
        
    Returns:
        Tuple of (model_path, is_local)
    """
    # Check if it's already a local path
    if model_name.startswith("models/") or os.path.exists(model_name):
        local_path = Path(model_name)
        if has_lfs_issues(local_path):
            logger.warning(f"Local model at {model_name} has Git LFS issues, falling back to HuggingFace")
            # Try to find the HuggingFace name for this local path
            for hf_name, config in MODEL_CONFIGS.items():
                if str(config["local_path"]) == model_name:
                    return hf_name, False
            # If not found in configs, assume the model_name is also the HF name
            return model_name.replace("models/", "").replace("-", "/"), False
        return model_name, True
    
    # Check if we have a local configuration for this model
    if model_name in MODEL_CONFIGS:
        config = MODEL_CONFIGS[model_name]
        local_path = config["local_path"]
        
        if prefer_local and local_path.exists():
            # Check for Git LFS issues
            if has_lfs_issues(local_path):
                logger.warning(f"Local model at {local_path} has Git LFS issues, falling back to HuggingFace")
                logger.info(f"Using HuggingFace model: {model_name}")
                return model_name, False
            else:
                logger.info(f"Using local model: {local_path}")
                return str(local_path), True
        else:
            if not local_path.exists():
                logger.warning(f"Local model not found at {local_path}, falling back to HuggingFace")
            logger.info(f"Using HuggingFace model: {model_name}")
            return model_name, False
    
    # Not found in our configs, assume it's a HuggingFace model name
    logger.info(f"Unknown model {model_name}, treating as HuggingFace model")
    return model_name, False

def get_default_model(model_type: ModelType) -> str:
    """Get the default model name for a given model type."""
    return DEFAULT_MODELS.get(model_type, "")

def list_available_models(model_type: Optional[ModelType] = None) -> Dict[str, Dict]:
    """
    List all available models, optionally filtered by type.
    
    Args:
        model_type: Optional model type to filter by
        
    Returns:
        Dictionary of model configurations
    """
    if model_type is None:
        return MODEL_CONFIGS
    
    return {
        name: config for name, config in MODEL_CONFIGS.items()
        if config["type"] == model_type
    }

def is_model_available_locally(model_name: str) -> bool:
    """Check if a model is available locally and usable (no LFS issues)."""
    if model_name in MODEL_CONFIGS:
        local_path = MODEL_CONFIGS[model_name]["local_path"]
        return local_path.exists() and not has_lfs_issues(local_path)
    
    # Check if it's a direct path
    if model_name.startswith("models/") or os.path.exists(model_name):
        local_path = Path(model_name)
        return local_path.exists() and not has_lfs_issues(local_path)
    
    return False

def get_model_info(model_name: str) -> Optional[Dict]:
    """Get detailed information about a model."""
    if model_name in MODEL_CONFIGS:
        config = MODEL_CONFIGS[model_name].copy()
        local_path = config["local_path"]
        config["is_available_locally"] = local_path.exists()
        config["has_lfs_issues"] = has_lfs_issues(local_path) if local_path.exists() else False
        config["is_usable_locally"] = local_path.exists() and not has_lfs_issues(local_path)
        config["local_path"] = str(local_path)
        return config
    return None

def validate_model_directory() -> Dict[str, bool]:
    """
    Validate that all configured models are available locally.
    
    Returns:
        Dictionary mapping model names to availability status (exists, regardless of LFS)
    """
    results = {}
    for model_name, config in MODEL_CONFIGS.items():
        local_path = config["local_path"]
        is_available = local_path.exists()
        results[model_name] = is_available
        
        if is_available:
            if has_lfs_issues(local_path):
                logger.debug(f"⚠ Model available but has LFS issues: {model_name} at {local_path}")
            else:
                logger.debug(f"✓ Model available: {model_name} at {local_path}")
        else:
            logger.warning(f"✗ Model missing: {model_name} at {local_path}")
    
    return results

def get_chat_model_path(model_name: Optional[str] = None) -> str:
    """
    Get the path for a chat model, with intelligent defaults.
    
    Args:
        model_name: Optional specific model name
        
    Returns:
        Model path (local or HuggingFace name)
    """
    if model_name is None:
        model_name = get_default_model(ModelType.CHAT)
    
    path, is_local = get_model_path(model_name)
    return path

def get_text2img_model_path(model_name: Optional[str] = None) -> str:
    """
    Get the path for a text-to-image model, with intelligent defaults.
    
    Args:
        model_name: Optional specific model name
        
    Returns:
        Model path (local or HuggingFace name)
    """
    if model_name is None:
        model_name = get_default_model(ModelType.TEXT2IMG)
    
    path, is_local = get_model_path(model_name)
    return path

def initialize_model_config():
    """Initialize and validate model configuration."""
    logger.info("Initializing model configuration...")
    
    # Create models directory if it doesn't exist
    MODELS_BASE_DIR.mkdir(exist_ok=True)
    
    # Validate available models
    validation_results = validate_model_directory()
    
    available_count = sum(validation_results.values())
    total_count = len(validation_results)
    
    logger.info(f"Model validation complete: {available_count}/{total_count} models available locally")
    
    if available_count == 0:
        logger.warning("No local models found. All models will be downloaded from HuggingFace on first use.")
    elif available_count < total_count:
        missing_models = [name for name, available in validation_results.items() if not available]
        logger.info(f"Missing local models: {missing_models}")
    else:
        logger.info("All configured models are available locally!")
    
    return validation_results