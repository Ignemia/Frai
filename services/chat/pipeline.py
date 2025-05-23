"""
Main pipeline for the chat service.

This module is deprecated and its functionalities have been moved to:
- model_loader.py: For loading the model and processor.
- token_utils.py: For token counting and history trimming.
- llm_interface.py: For sending queries and generating responses.
- pipeline_config.py: For configuration constants.

This file will be removed in a future version.
For new integrations, please use the specific modules directly.
"""

import logging

# For backwards compatibility, re-export key functions/variables if needed,
# though ideally, dependent modules should update their imports.
from .pipeline_config import (
    MODEL_PATH,
    MAX_NEW_TOKENS,
    CONTEXT_WINDOW_SIZE,
    RESPONSE_BUFFER,
    SYSTEM_PROMPT,
    POSITIVE_PROMPT,
    NEGATIVE_PROMPT
)
from .model_loader import model, processor, load_model
from .token_utils import count_tokens, trim_chat_history
from .llm_interface import send_query

logger = logging.getLogger(__name__)

logger.warning(
    "The 'pipeline.py' module is deprecated and will be removed in a future version. "
    "Please update imports to use 'model_loader.py', 'token_utils.py', "
    "'llm_interface.py', and 'pipeline_config.py' directly."
)

# Example of how this file might have looked before, now largely empty or just re-exporting.
# Global variables for model and processor, now managed in model_loader.py
# model = None # Now in model_loader
# processor = None # Now in model_loader

# Configuration constants, now in pipeline_config.py
# MODEL_PATH = "models/gemma-3-4b-it"
# ... and so on for other constants

# Functions, now in their respective new modules
# def load_model_and_processor(): # Now load_model in model_loader.py
#     pass

# def count_tokens(text: str) -> int: # Now in token_utils.py
#     pass

# def trim_chat_history(chat_history: list[dict[str, str]]) -> list[dict[str, str]]: # Now in token_utils.py
#     pass

# def send_query(query: str, chat_history: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]: # Now in llm_interface.py
#     pass