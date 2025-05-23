"""
Configuration constants for the chat pipeline.

This module stores various settings like model paths, token limits,
default prompts, and other parameters used throughout the chat pipeline.
"""
import pathlib
import os
import logging

logger = logging.getLogger(__name__)

# --- Prompts ---
# Attempt to get prompts from environment variables, with fallbacks
try:
    POSITIVE_PROMPT = os.environ.get("POSITIVE_SYSTEM_PROMPT_CHAT", "You are a helpful AI assistant.")
    NEGATIVE_PROMPT = os.environ.get("NEGATIVE_SYSTEM_PROMPT_CHAT", "Avoid inappropriate content.")
except Exception as e:
    logger.error(f"Error reading system prompts from environment variables: {e}", exc_info=True)
    POSITIVE_PROMPT = "You are a helpful AI assistant."
    NEGATIVE_PROMPT = "Avoid inappropriate content."

SYSTEM_PROMPT = f"{POSITIVE_PROMPT}\n{NEGATIVE_PROMPT}"
if not POSITIVE_PROMPT and not NEGATIVE_PROMPT: # If both env vars were empty or not set
    logger.warning("Both POSITIVE_SYSTEM_PROMPT_CHAT and NEGATIVE_SYSTEM_PROMPT_CHAT are empty or not set. Using a default system prompt.")
    SYSTEM_PROMPT = "You are a helpful AI assistant. Be polite and concise."


# --- Model Paths ---
try:
    # Get model path from environment or use default
    MODEL_PATH_STR = os.environ.get("LLM_MODEL_PATH", "models/gemma-3-4b-it")
    MODEL_PATH = pathlib.Path(MODEL_PATH_STR).resolve()
    
    # Validate the model path
    if not MODEL_PATH.parent.exists():
        logger.warning(f"Model directory '{MODEL_PATH.parent}' does not exist. Model loading will likely fail.")
    elif not (MODEL_PATH.exists() or list(MODEL_PATH.parent.glob(f"{MODEL_PATH.name}*"))):
        # Check if either the exact path exists or there are files matching the pattern
        logger.warning(f"No files matching the model path pattern '{MODEL_PATH}' were found. Model loading may fail.")
    else:
        logger.info(f"Model path validated: '{MODEL_PATH}'")
except Exception as e:
    logger.error(f"Error resolving model path '{MODEL_PATH_STR}': {e}", exc_info=True)
    # Use default fallback path
    MODEL_PATH = pathlib.Path("models/gemma-3-4b-it")

# --- Tokenizer and Model Settings ---
MAX_NEW_TOKENS = 1024  # Max tokens for the model to generate in a response
CONTEXT_WINDOW_SIZE = 256 * 1024 # Total context window size for the model
RESPONSE_BUFFER = 1024 # Number of tokens to reserve for the model's response when trimming history

# --- Default Generation Parameters ---
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS_TITLE = 50 # Max tokens for title generation

logger.info("Pipeline configuration loaded.")
logger.debug(f"SYSTEM_PROMPT set to: {SYSTEM_PROMPT[:100]}...")
logger.debug(f"MODEL_PATH: {MODEL_PATH}")
logger.debug(f"MAX_NEW_TOKENS: {MAX_NEW_TOKENS}")
logger.debug(f"CONTEXT_WINDOW_SIZE: {CONTEXT_WINDOW_SIZE}")
