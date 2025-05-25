"""Handles text generation using the AI model."""
import logging
from typing import Dict, Any
from datetime import datetime
import torch # Added for torch.cuda.is_available

logger = logging.getLogger(__name__)

def generate_text(
    generation_pipeline, 
    tokenizer, # Added tokenizer for pad_token_id and eos_token_id
    conversation_context: str, 
    generation_params: Dict[str, Any],
    max_new_tokens: int = 8192 # Added from original generate_response
) -> Dict[str, Any]:
    """Generates text using the model pipeline."""
    try:
        logger.info("Generating AI response...")
        start_time = datetime.now()

        # Ensure generation_pipeline is not None
        if generation_pipeline is None:
            logger.error("Generation pipeline is not initialized.")
            return {
                "success": False,
                "error": "Generation pipeline not initialized",
                "response": "",
                "metadata": {}
            }

        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": generation_params.get("temperature", 0.7),
            "top_p": generation_params.get("top_p", 0.95),
            "top_k": generation_params.get("top_k", 50),
            "repetition_penalty": generation_params.get("repetition_penalty", 1.1),
            "pad_token_id": tokenizer.eos_token_id, # Use tokenizer's eos_token_id
            "eos_token_id": tokenizer.eos_token_id, # Use tokenizer's eos_token_id
            "return_full_text": False
        }

        # Generate response
        outputs = generation_pipeline(
            conversation_context,
            **gen_kwargs
        )

        # Extract generated text
        generated_text = outputs[0]["generated_text"] if outputs and outputs[0] else ""

        # Clean up the response
        generated_text = generated_text.strip()
        if generated_text.startswith("Assistant:"):
            generated_text = generated_text[10:].strip()

        generation_time = (datetime.now() - start_time).total_seconds()

        result = {
            "success": True,
            "response": generated_text,
            "error": None,
            "metadata": {
                "generation_time": generation_time,
                "max_new_tokens": max_new_tokens,
                "temperature": generation_params.get("temperature", 0.7),
                # "model_name": model_name # This should be passed in or handled differently
            }
        }
        logger.info(f"Response generated successfully in {generation_time:.2f}s")
        return result

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        # Log traceback for detailed debugging
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "response": "",
            "metadata": {}
        } 