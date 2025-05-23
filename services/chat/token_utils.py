"""
Utility functions for token counting and chat history management.

This module provides functions to count tokens in a given text using the
model's processor and to trim chat history to fit within a specified
context window size, considering a response buffer.
"""
import logging
from .model_loader import processor, load_model
from .pipeline_config import CONTEXT_WINDOW_SIZE, RESPONSE_BUFFER

logger = logging.getLogger(__name__)

def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a given text string.

    Uses the globally loaded processor. If the processor is not loaded,
    it attempts to load it first.

    Args:
        text (str): The text to count tokens for.

    Returns:
        int: The number of tokens in the text.

    Raises:
        ValueError: If the processor is not available or fails to load.
        Exception: For any other unexpected errors during tokenization.
    """
    global processor
    if processor is None:
        logger.warning("Processor not loaded. Attempting to load model and processor.")
        try:
            load_model() # This will load both model and processor
            if processor is None: # Check again after attempting to load
                logger.error("Failed to load processor for token counting.")
                raise ValueError("Processor is not available for token counting.")
        except Exception as e:
            logger.error(f"Error loading model/processor in count_tokens: {e}", exc_info=True)
            raise

    logger.debug(f"Counting tokens for text: '{text[:50]}...' if len > 50 else text")
    try:
        tokens = processor.tokenizer(text, return_tensors="pt")["input_ids"]
        num_tokens = tokens.shape[1]
        logger.debug(f"Number of tokens: {num_tokens}")
        return num_tokens
    except Exception as e:
        logger.error(f"Error during tokenization: {e}", exc_info=True)
        raise

def trim_chat_history(chat_history: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Trim the chat history to fit within the context window size.

    The trimming process respects the CONTEXT_WINDOW_SIZE and RESPONSE_BUFFER.
    It iteratively removes older messages until the total token count is within limits.
    This ensures that new responses have enough token budget to generate properly.

    Args:
        chat_history (list[dict[str, str]]): A list of chat messages, where each
            message is a dictionary with 'role' and 'content' keys.

    Returns:
        list[dict[str, str]]: The trimmed chat history with oldest messages removed
            if necessary to fit within token limits.
    """
    logger.debug(f"Trimming chat history. Initial length: {len(chat_history)}")
    
    # Create a copy to avoid modifying the original
    history_copy = list(chat_history)
      # Calculate total tokens in current history
    current_tokens = sum(count_tokens(msg["content"]) for msg in history_copy)
    logger.debug(f"Initial token count: {current_tokens}")
    
    max_tokens_allowed = CONTEXT_WINDOW_SIZE - RESPONSE_BUFFER
    logger.debug(f"Max tokens allowed (context_window - response_buffer): {max_tokens_allowed}")

    # Remove oldest messages first until we're under the token limit
    while current_tokens > max_tokens_allowed and history_copy:
        logger.debug(f"Current tokens ({current_tokens}) exceed max allowed ({max_tokens_allowed}). Trimming.")
        removed_message = history_copy.pop(0)  # Remove the oldest message
        removed_tokens = count_tokens(removed_message["content"])
        current_tokens -= removed_tokens
        logger.info(f"Removed message ({removed_message['role']}, {len(removed_message['content'])} chars, "
                    f"{removed_tokens} tokens). New token count: {current_tokens}")

    if len(history_copy) < len(chat_history):
        logger.warning(f"Removed {len(chat_history) - len(history_copy)} messages to fit token limit.")
    
    logger.info(f"Chat history trimmed. Final length: {len(history_copy)}, Token count: {current_tokens}")
    return history_copy
