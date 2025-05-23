"""
Provides the core interface for interacting with the LLM.

This module handles sending queries to the language model and generating responses.
It utilizes the loaded model and processor, and incorporates system prompts and
chat history management.
"""
import logging
import torch
from .model_loader import model, processor, load_model as global_load_model # Renamed to avoid conflict
from .pipeline_config import SYSTEM_PROMPT, MAX_NEW_TOKENS, POSITIVE_PROMPT, NEGATIVE_PROMPT
from .token_utils import trim_chat_history

logger = logging.getLogger(__name__)

def _ensure_model_loaded():
    """Ensures the model and processor are loaded."""
    global model, processor
    if model is None or processor is None:
        logger.warning("Model or processor not loaded. Attempting to load.")
        try:
            global_load_model() # Call the renamed global loader
            if model is None or processor is None:
                logger.error("Failed to load model/processor.")
                raise ValueError("Model or processor not available after attempting load.")
            logger.info("Model and processor loaded successfully by _ensure_model_loaded.")
        except Exception as e:
            logger.error(f"Error loading model/processor: {e}", exc_info=True)
            raise

def _prepare_prompt_for_model(current_chat_history: list[dict[str, str]]) -> str:
    """
    Prepares the final prompt string for the model.
    This includes trimming history, adding a system prompt, and applying the chat template.
    """
    logger.debug(f"Preparing prompt. Initial history length: {len(current_chat_history)}")
    # Trim a copy of the history to avoid modifying the original list prematurely
    history_for_prompt = trim_chat_history(list(current_chat_history))
    logger.debug(f"History trimmed for prompt. Length: {len(history_for_prompt)}")

    messages_for_model = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ] + history_for_prompt
    logger.debug("Prepared messages for model (system prompt + trimmed history):")
    # for msg in messages_for_model: # Too verbose for default logging
    #     logger.debug(msg)

    logger.debug("Applying chat template...")
    prompt = processor.tokenizer.apply_chat_template(
        messages_for_model,
        tokenize=False,
        add_generation_prompt=True
    )
    logger.debug(f"Generated prompt for model (first 100 chars): {prompt[:100]}...")
    return prompt

def _tokenize_prompt(prompt_text: str) -> dict:
    """Tokenizes the prompt string for model input."""
    global model # Ensure model is accessible for model.device
    logger.debug("Tokenizing inputs for model...")
    inputs = processor.tokenizer(prompt_text, return_tensors="pt").to(model.device)
    logger.debug(f"Input tensor shape: {inputs['input_ids'].shape}")
    return inputs

def _invoke_model_generation(inputs: dict) -> str:
    """Invokes the model to generate a response and decodes it."""
    global model, processor # Ensure model and processor are accessible
    logger.info("Generating response from model...")
    generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95
    }
    if POSITIVE_PROMPT:
        generation_kwargs["positive_prompt"] = POSITIVE_PROMPT
    if NEGATIVE_PROMPT:
        generation_kwargs["negative_prompt"] = NEGATIVE_PROMPT
    
    logger.debug(f"Generation kwargs: {generation_kwargs}")
    outputs = model.generate(**inputs, **generation_kwargs)
    response_ids = outputs[0][inputs["input_ids"].shape[1]:]
    logger.debug("Model generation complete.")

    logger.debug("Decoding response...")
    response_text = processor.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    logger.info(f"Generated response (first 100 chars): {response_text[:100]}...")
    return response_text

def send_query(query: str, chat_history: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    """
    Send a query to the language model and get a response.

    Manages chat history, system prompts, and model inference.
    Ensures the model and processor are loaded before processing.

    Args:
        query (str): The user's query.
        chat_history (list[dict[str, str]]): The current chat history.

    Returns:
        tuple[str, list[dict[str, str]]]: A tuple containing the model's response
            and the updated chat history.

    Raises:
        ValueError: If the model or processor is not available or fails to load.
        Exception: For any other unexpected errors during query processing or model inference.
    """
    logger.info(f"Received query: '{query[:100]}...' if len(query) > 100 else query")
    
    try:
        _ensure_model_loaded()

        # Add the new user query to the history that will be returned
        # The history passed to _prepare_prompt_for_model will be a processed copy
        current_chat_history = list(chat_history) # Work with a copy
        current_chat_history.append({"role": "user", "content": query})
        logger.debug(f"Appended user query. Current history length: {len(current_chat_history)}")

        prompt_text = _prepare_prompt_for_model(current_chat_history)
        tokenized_inputs = _tokenize_prompt(prompt_text)
        response_text = _invoke_model_generation(tokenized_inputs)

        # Add assistant's response to the current chat_history
        current_chat_history.append({"role": "assistant", "content": response_text})
        logger.debug(f"Appended assistant response. Final history length: {len(current_chat_history)}")

        return response_text, current_chat_history

    except Exception as e:
        logger.error(f"Error during send_query: {e}", exc_info=True)
        # Do not modify original chat_history here if error occurs; 
        # current_chat_history is a copy. The caller manages the original.
        raise
