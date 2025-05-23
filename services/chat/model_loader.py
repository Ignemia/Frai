"""
Manages the loading of the Hugging Face language model and processor.

This module provides functions to initialize and access the global model and
processor instances. It ensures that the model is loaded from the configured
path and is ready for use by other parts of the chat service.
"""
import logging
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from .pipeline_config import MODEL_PATH

logger = logging.getLogger(__name__)

model: Gemma3ForConditionalGeneration | None = None
processor: AutoProcessor | None = None

def load_model():
    """
    Load the language model and processor from the specified MODEL_PATH.

    Initializes the global `model` and `processor` variables.
    Logs information about the loading process and any errors encountered.
    The model is loaded onto the available device (e.g., GPU or CPU) 
    and set to evaluation mode. Uses bfloat16 for torch_dtype for 
    potentially better numerical stability and performance on compatible hardware.

    Global Args:
        model (Gemma3ForConditionalGeneration | None): The loaded language model.
        processor (AutoProcessor | None): The loaded model processor.

    Raises:
        FileNotFoundError: If model or processor files are not found.
        ImportError: If a required library is missing.
        Exception: For any other unexpected errors during model loading.
    """
    global model, processor

    if model is not None and processor is not None:
        logger.info("Model and processor are already loaded.")
        return

    logger.info(f"Attempting to load model and processor from: {MODEL_PATH}")
    try:
        logger.debug("Loading processor...")
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        logger.info("Processor loaded successfully.")
        
        logger.debug("Loading model...")
        model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()
        logger.info("Model loaded successfully and set to evaluation mode.")

    except FileNotFoundError as e:
        logger.error(f"Model or processor files not found at {MODEL_PATH}: {e}", exc_info=True)
        raise
    except ImportError as e:
        logger.error(f"A required library is missing for model loading: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during model loading: {e}", exc_info=True)
        raise

    logger.info("Model and processor successfully loaded and ready to use.")
