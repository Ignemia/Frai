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
        
        # Device configuration with explicit GPU detection
        self.cuda_available = torch.cuda.is_available()
        self.uses_distributed_model = False  # Will be set after model loading
        
        # Log device configuration
        if self.cuda_available:
            logger.info(f"CUDA available: Partial GPU offloading will be used. Device count: {torch.cuda.device_count()}")
            logger.info(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'Unknown'}")
        else:
            logger.warning("CUDA not available: All operations will run on CPU")

        self.generation_params = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "max_new_tokens": 50
        }
        
        # Add flag

        self.positive_system_prompt_template = os.getenv("POSITIVE_SYSTEM_PROMPT_CHAT", "Be helpful and answer concisely.")
        self.negative_system_prompt_template = os.getenv("NEGATIVE_SYSTEM_PROMPT_CHAT", "Do not use offensive language.")

        self._load_model_to_ram()

    def _load_model_to_ram(self):
        logger.info(f"Attempting to load model {self.model_name} with optimal device mapping.")
        self.model, self.tokenizer = load_model_and_tokenizer(self.model_name, self.model_path)
        if self.model and self.tokenizer:
            self.is_loaded = True
            # Check if model is distributed across devices
            if hasattr(self.model, 'hf_device_map'):
                self.uses_distributed_model = True
                logger.info(f"Model {self.model_name} loaded with distributed device mapping.")
                logger.info(f"Device distribution: {dict(self.model.hf_device_map)}")
            else:
                self.uses_distributed_model = False
                device = next(self.model.parameters()).device
                logger.info(f"Model {self.model_name} loaded to single device: {device}")
        else:
            self.is_loaded = False
            logger.error(f"Failed to load model {self.model_name}.")
            raise RuntimeError(f"Could not load model {self.model_name}.")

    def _ensure_model_ready_for_generation(self) -> bool:
        """Ensure model is ready for generation (no device moves needed for distributed models)."""
        if not self.is_loaded or self.model is None:
            logger.error("Model not loaded, cannot prepare for generation.")
            return False
        
        if self.uses_distributed_model:
            logger.info("Model is distributed across devices - already optimally placed")
            return True
        else:
            # For single-device models, we can still move them if needed
            current_device = next(self.model.parameters()).device
            target_device = "cuda" if self.cuda_available else "cpu"
            
            if str(current_device) != target_device and not str(current_device).startswith(target_device):
                logger.info(f"Moving single-device model from {current_device} to {target_device}")
                return move_model_to_device(self.model, target_device)
            else:
                logger.info(f"Model already on appropriate device: {current_device}")
                return True

    def format_conversation_for_model(
        self,
        messages: List[Dict[str, str]],
        positive_system_prompt: str,
        negative_system_prompt: str
    ) -> str:
        """
        Format conversation messages for the model using tokenizer's chat template.
        """
        # Build chat messages in the format expected by the tokenizer
        chat_messages = []
        
        # Add conversation messages
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if content.strip():  # Only add non-empty messages
                chat_messages.append({"role": role, "content": content})
        
        # Use tokenizer's chat template if available
        if (self.tokenizer is not None and 
            hasattr(self.tokenizer, 'apply_chat_template') and 
            hasattr(self.tokenizer, 'chat_template') and 
            self.tokenizer.chat_template is not None):
            try:
                # For Gemma3, don't include system messages in chat template
                formatted_text = self.tokenizer.apply_chat_template(
                    chat_messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                return formatted_text
            except Exception as e:
                logger.warning(f"Failed to use chat template: {e}, falling back to manual formatting")
        
        # Fallback to manual formatting
        formatted_messages_list = []
        
        # Add system prompt as instruction prefix
        system_prompt_content = f"You should follow these instructions: {positive_system_prompt}\n\nYou should never do these things: {negative_system_prompt}"
        formatted_messages_list.append(f"Instructions: {system_prompt_content}")
        
        for message in chat_messages:
            role = message["role"]
            content = message["content"]
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

        # Validate conversation history
        if not conversation_history:
            return {"success": False, "error": "Empty conversation history", "response": "", "metadata": {}}
        
        # Check if conversation has any valid content
        has_content = any(msg.get("content", "").strip() for msg in conversation_history)
        if not has_content:
            return {"success": False, "error": "No content in conversation history", "response": "", "metadata": {}}

        current_pipeline = None
        target_device_for_generation = "cuda" if self.cuda_available else "cpu"

        final_positive_prompt = positive_system_prompt_override if positive_system_prompt_override else self.positive_system_prompt_template
        final_negative_prompt = negative_system_prompt_override if negative_system_prompt_override else self.negative_system_prompt_template

        conversation_context = self.format_conversation_for_model(
            messages=conversation_history,
            positive_system_prompt=final_positive_prompt,
            negative_system_prompt=final_negative_prompt
        )
        logger.info(f"Formatted conversation context: {repr(conversation_context)}")

        # Prepare generation configuration outside try block to make variables available in except block
        generation_config = self.generation_params.copy()
        current_max_tokens = max_new_tokens if max_new_tokens is not None else generation_config["max_new_tokens"]
        generation_config["max_new_tokens"] = current_max_tokens

        try:
            # Prepare model for generation
            logger.info("Preparing model for generation")
            if not self._ensure_model_ready_for_generation():
                raise RuntimeError("Failed to prepare model for generation.")

            # Determine actual device usage for logging
            if self.uses_distributed_model:
                device_info = "distributed (GPU+CPU)"
                target_device_for_generation = "distributed"
            else:
                actual_device = next(self.model.parameters()).device
                device_info = str(actual_device)
                target_device_for_generation = str(actual_device)
            
            logger.info(f"Model ready on: {device_info}")

            logger.info("Creating generation pipeline")
            current_pipeline = get_generation_pipeline(self.model, self.tokenizer, target_device_for_generation)
            if not current_pipeline:
                raise RuntimeError("Failed to create generation pipeline.")

            logger.info("Starting text generation...")

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
                 gen_result["metadata"]["cuda_available"] = self.cuda_available
                 gen_result["metadata"]["distributed_model"] = self.uses_distributed_model
                 if self.uses_distributed_model:
                     gen_result["metadata"]["device_map"] = dict(self.model.hf_device_map)
                 logger.info("Generation completed successfully")
            return gen_result

        except RuntimeError as e:
            # Handle CUDA errors specifically - for distributed models, try to continue
            error_str = str(e)
            if "CUDA" in error_str:
                logger.warning(f"CUDA error encountered: {error_str}")
                
                # Clear CUDA cache first
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                if not self.uses_distributed_model:
                    logger.info("Attempting fallback to CPU generation...")
                    try:
                        # Move model back to CPU and retry
                        if move_model_to_device(self.model, "cpu"):
                            cpu_pipeline = get_generation_pipeline(self.model, self.tokenizer, "cpu")
                            if cpu_pipeline:
                                logger.info("Retrying generation on CPU...")
                                gen_result = generate_text(
                                    generation_pipeline=cpu_pipeline,
                                    tokenizer=self.tokenizer,
                                    conversation_context=conversation_context,
                                    generation_params=generation_config,
                                    max_new_tokens=current_max_tokens
                                )
                                if gen_result["success"]:
                                    gen_result["metadata"]["model_name"] = self.model_name
                                    gen_result["metadata"]["device_used"] = "cpu"
                                    gen_result["metadata"]["cuda_available"] = self.cuda_available
                                    gen_result["metadata"]["distributed_model"] = False
                                    gen_result["metadata"]["fallback_from_cuda"] = True
                                    logger.info("CPU fallback generation completed successfully")
                                # Clean up CPU pipeline
                                del cpu_pipeline
                                return gen_result
                    except Exception as fallback_e:
                        logger.error(f"CPU fallback also failed: {fallback_e}")
                else:
                    logger.error("CUDA error with distributed model - no fallback available")
            
            logger.error(f"Error during generation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e), "response": "", "metadata": {}}
        except Exception as e:
            logger.error(f"Error during generation with device handling: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e), "response": "", "metadata": {}}
        finally:
            # Clean up resources
            if current_pipeline is not None:
                logger.info("Cleaning up generation pipeline")
                try:
                    del current_pipeline
                    
                    # Clear GPU cache if we have CUDA
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("GPU cache cleared")
                except Exception as cleanup_e:
                    logger.warning(f"Error during pipeline cleanup: {cleanup_e}")
            
            # For distributed models, we don't move them back - they stay optimally placed
            if self.uses_distributed_model:
                logger.info("Distributed model remains optimally placed across devices")
            elif not self.uses_distributed_model and self.model is not None:
                # For single-device models, optionally move back to CPU to free GPU memory
                try:
                    current_device = next(self.model.parameters()).device
                    if "cuda" in str(current_device):
                        logger.info("Moving single-device model back to CPU to free VRAM")
                        if not move_model_to_device(self.model, "cpu"):
                            logger.warning("Failed to move model back to CPU")
                        else:
                            logger.info("Model moved back to CPU successfully")
                except Exception as move_e:
                    logger.warning(f"Error moving model back to CPU: {move_e}")



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
