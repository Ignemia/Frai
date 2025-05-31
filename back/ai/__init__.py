import logging
from .model_config import initialize_model_config

logger = logging.getLogger(__name__)

# Initialize model configuration on module import
try:
    initialize_model_config()
    logger.info("Model configuration initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize model configuration: {e}")

# Try to import chat model functions
try:
    from back.ai.chat import load_chat_model as _load_chat_model
    _chat_model_available = True
except ImportError as e:
    logger.warning(f"Chat model not available: {e}")
    _chat_model_available = False

# Try to import image model functions
try:
    from back.ai.text2img import initialize_text2img_system as _load_text2img_model
    _text2img_model_available = True
except ImportError as e:
    logger.warning(f"Text2img model not available: {e}")
    _text2img_model_available = False

# Try to import other AI model functions
try:
    from back.ai.img2img import initialize_img2img_system as _load_img2img_model
    _img2img_model_available = True
except ImportError as e:
    logger.warning(f"Img2img model not available: {e}")
    _img2img_model_available = False

try:
    from back.ai.voicein import initialize_speech_recognition_system as _load_speech_recognition_model
    _speech_recognition_available = True
except ImportError as e:
    logger.warning(f"Speech recognition model not available: {e}")
    _speech_recognition_available = False

try:
    from back.ai.voiceout import initialize_text_to_speech_system as _load_text_to_speech_model
    _text_to_speech_available = True
except ImportError as e:
    logger.warning(f"Text-to-speech model not available: {e}")
    _text_to_speech_available = False

def load_chat_model():
    """Load the chat model using the chat module."""
    if _chat_model_available:
        try:
            return _load_chat_model()
        except Exception as e:
            logger.error(f"Failed to load chat model: {e}")
            return False
    else:
        logger.warning("Chat model not available - using placeholder")
        print("Chat model loaded (placeholder).")
        return True
    
def load_image_model():
    """Load image generation models (text2img and img2img)."""
    success = True
    
    if _text2img_model_available:
        try:
            if not _load_text2img_model():
                logger.warning("Text2img model failed to load")
                success = False
        except Exception as e:
            logger.error(f"Failed to load text2img model: {e}")
            success = False
    else:
        logger.warning("Text2img model not available - using placeholder")
    
    if _img2img_model_available:
        try:
            if not _load_img2img_model():
                logger.warning("Img2img model failed to load")
                success = False
        except Exception as e:
            logger.error(f"Failed to load img2img model: {e}")
            success = False
    else:
        logger.warning("Img2img model not available - using placeholder")
    
    print("Image models loaded (with available components).")
    return success
    
def load_voice_activity_detection_model():
    """Load voice activity detection model."""
    # This is typically part of speech recognition
    logger.info("Voice activity detection model loaded (placeholder).")
    print("Voice activity detection model loaded.")
    return True
    
def load_speech_recognition_model():
    """Load speech recognition model."""
    if _speech_recognition_available:
        try:
            return _load_speech_recognition_model()
        except Exception as e:
            logger.error(f"Failed to load speech recognition model: {e}")
            print("Speech recognition model loaded (fallback).")
            return False
    else:
        logger.warning("Speech recognition model not available - using placeholder")
        print("Speech recognition model loaded (placeholder).")
        return True
    
def load_text_to_speech_model():
    """Load text-to-speech model."""
    if _text_to_speech_available:
        try:
            return _load_text_to_speech_model()
        except Exception as e:
            logger.error(f"Failed to load text-to-speech model: {e}")
            print("Text-to-speech model loaded (fallback).")
            return False
    else:
        logger.warning("Text-to-speech model not available - using placeholder")
        print("Text-to-speech model loaded (placeholder).")
        return True