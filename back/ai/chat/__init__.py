"""
Backend AI Chat Module

This module defines the main ChatAI class that orchestrates model loading,
text generation, and device management (RAM/VRAM transfer).
It exports the ChatAI class and utility functions for accessing it.
"""

import logging
import os
from typing import Dict, Optional, Any, List
import torch

# Local imports from the new modules
from .model_loader import (
    load_model_and_tokenizer,
    move_model_to_device,
    get_generation_pipeline
)
from .text_generator import generate_text
from ..model_config import get_chat_model_path, get_model_path
# Removed: from .orchestration import process_chat_message as process_chat_message_orchestrated, format_conversation_for_model

logger = logging.getLogger(__name__)

class ChatAI:
    """
    AI chat handler that coordinates model loading, generation, and device management.
    The model is loaded into RAM on initialization.
    For generation, it's moved to VRAM (if available) and then back to RAM.
    """

    def __init__(self, model_name: Optional[str] = None, model_path: Optional[str] = None):
        # Resolve model path using configuration
        if model_name is None:
            self.model_path = get_chat_model_path()
            self.model_name = "google/gemma-3-4b-it"  # Keep original name for identification
        else:
            self.model_path, _ = get_model_path(model_name)
            self.model_name = model_name
        
        # Override with explicit model_path if provided
        if model_path is not None:
            self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.vram_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ram_device = "cpu"

        self.generation_params = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "max_new_tokens": 8192
        }

        self.positive_system_prompt_template = os.getenv("POSITIVE_SYSTEM_PROMPT_CHAT", "Be helpful and answer concisely.")
        self.negative_system_prompt_template = os.getenv("NEGATIVE_SYSTEM_PROMPT_CHAT", "Do not use offensive language.")

        self._load_model_to_ram()

    def _load_model_to_ram(self):
        logger.info(f"Attempting to load model {self.model_name} to RAM.")
        self.model, self.tokenizer = load_model_and_tokenizer(self.model_name, self.model_path)
        if self.model and self.tokenizer:
            self.is_loaded = True
            logger.info(f"Model {self.model_name} and tokenizer loaded to RAM successfully.")
        else:
            self.is_loaded = False
            logger.error(f"Failed to load model {self.model_name} to RAM.")
            raise RuntimeError(f"Could not load model {self.model_name} to RAM.")

    def _ensure_model_on_device(self, target_device: str) -> bool:
        if not self.is_loaded or self.model is None:
            logger.error("Model not loaded, cannot move device.")
            return False
        return move_model_to_device(self.model, target_device)

    @staticmethod
    def format_conversation_for_model(
        messages: List[Dict[str, str]],
        positive_system_prompt: str,
        negative_system_prompt: str
    ) -> str:
        """
        Format conversation messages for the model, including system prompts.
        """
        formatted_messages_list = []
        system_prompt_content = f"YOU SHOULD FOLLOW THESE INSTRUCTIONS:`{positive_system_prompt}\n\nYOU SOULD NEVER DO THESE THINGS:{negative_system_prompt}`"
        formatted_messages_list.append(f"System: {system_prompt_content}")

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "user":
                formatted_messages_list.append(f"User: {content}")
            elif role == "assistant":
                formatted_messages_list.append(f"Assistant: {content}")
        formatted_messages_list.append("Assistant:")
        return "\n".join(formatted_messages_list)

    def generate_response(
        self,
        conversation_history: List[Dict[str, str]],
        positive_system_prompt_override: Optional[str] = None,
        negative_system_prompt_override: Optional[str] = None,
        max_new_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generates an AI response based on conversation history and system prompts.
        Manages device VRAM/RAM transfers for the model.

        Args:
            conversation_history: List of message dicts [{"role": "user/assistant", "content": "..."}].
            positive_system_prompt_override: Optional override for the positive system prompt.
            negative_system_prompt_override: Optional override for the negative system prompt.
            max_new_tokens: Optional override for max_new_tokens for this specific generation.

        Returns:
            Dictionary with generation results, including "response", "success", "error", "metadata".
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            return {"success": False, "error": "Model not loaded", "response": "", "metadata": {}}

        current_pipeline = None
        target_device_for_generation = self.vram_device

        final_positive_prompt = positive_system_prompt_override if positive_system_prompt_override else self.positive_system_prompt_template
        final_negative_prompt = negative_system_prompt_override if negative_system_prompt_override else self.negative_system_prompt_template

        conversation_context = self.format_conversation_for_model(
            messages=conversation_history,
            positive_system_prompt=final_positive_prompt,
            negative_system_prompt=final_negative_prompt
        )

        try:
            if not self._ensure_model_on_device(target_device_for_generation):
                raise RuntimeError(f"Failed to move model to {target_device_for_generation}.")

            logger.info(f"Creating/getting pipeline on {target_device_for_generation}")
            current_pipeline = get_generation_pipeline(self.model, self.tokenizer, target_device_for_generation)
            if not current_pipeline:
                raise RuntimeError(f"Failed to create generation pipeline on {target_device_for_generation}.")

            logger.info(f"Generating response on {target_device_for_generation}...")

            generation_config = self.generation_params.copy()
            current_max_tokens = max_new_tokens if max_new_tokens is not None else generation_config["max_new_tokens"]
            generation_config["max_new_tokens"] = current_max_tokens

            gen_result = generate_text(
                generation_pipeline=current_pipeline,
                tokenizer=self.tokenizer,
                conversation_context=conversation_context,
                generation_params=generation_config,
                max_new_tokens=current_max_tokens
            )

            if gen_result["success"]:
                 gen_result["metadata"]["model_name"] = self.model_name
                 gen_result["metadata"]["device_used"] = target_device_for_generation
            return gen_result

        except Exception as e:
            logger.error(f"Error during generation with device handling: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e), "response": "", "metadata": {}}
        finally:
            logger.info(f"Attempting to move model back to {self.ram_device}.")
            if self.model is not None:
                if not self._ensure_model_on_device(self.ram_device):
                    logger.error(f"CRITICAL: Failed to move model back to {self.ram_device} after generation.")
                else:
                    logger.info(f"Model successfully moved back to {self.ram_device}.")
            if target_device_for_generation == "cuda" and current_pipeline is not None:
                logger.info("Clearing VRAM generation pipeline and emptying cache.")
                del current_pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

# --- Global Instance and Accessor Functions ---
_chat_ai_instance: Optional[ChatAI] = None

def get_chat_ai_instance(model_name: Optional[str] = None, model_path: Optional[str] = None) -> ChatAI:
    global _chat_ai_instance
    if _chat_ai_instance is None:
        logger.info("Initializing global ChatAI instance.")
        _chat_ai_instance = ChatAI(
            model_name=model_name,
            model_path=model_path
        )
    elif model_name and (_chat_ai_instance.model_name != model_name or
                        (model_path and _chat_ai_instance.model_path != model_path)):
        logger.warning(
            f"Requesting ChatAI with new model/path {model_name}/{model_path}, but instance with "
            f"{_chat_ai_instance.model_name}/{_chat_ai_instance.model_path} exists. Re-initializing."
        )
        _chat_ai_instance = ChatAI(model_name=model_name, model_path=model_path)
    return _chat_ai_instance

def load_chat_model(model_name: Optional[str] = None, model_path: Optional[str] = None) -> bool:
    """
    Load the chat model. This is an alias for initialize_chat_system for compatibility.
    """
    return initialize_chat_system(model_name, model_path)

def initialize_chat_system(model_name: Optional[str] = None, model_path: Optional[str] = None) -> bool:
    try:
        logger.info("Initializing chat system...")
        chat_ai = get_chat_ai_instance(model_name, model_path)
        if chat_ai.is_loaded:
            logger.info("Chat system initialized successfully. Model is in RAM.")
            return True
        else:
            logger.error("Chat system initialization failed. Model could not be loaded.")
            return False
    except Exception as e:
        logger.error(f"Error during chat system initialization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# These endpoint functions are how other layers (like an orchestrator or API layer) would interact with the AI.
def generate_ai_text(
    conversation_history: List[Dict[str, str]],
    positive_system_prompt: Optional[str] = None,
    negative_system_prompt: Optional[str] = None,
    max_new_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    High-level function to generate AI text based on conversation history and optional system prompts.
    This is intended to be called by the layer above (e.g., Orchestrator).
    """
    chat_ai = get_chat_ai_instance()
    if not chat_ai.is_loaded:
         return {"success": False, "error": "Chat model not ready or failed to load.", "response": ""}
    return chat_ai.generate_response(
        conversation_history=conversation_history,
        positive_system_prompt_override=positive_system_prompt,
        negative_system_prompt_override=negative_system_prompt,
        max_new_tokens=max_new_tokens
    )
