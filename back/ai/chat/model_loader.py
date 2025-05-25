"""Handles model loading and management."""
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_name_or_path: str, model_path: str):
    """Loads the model and tokenizer into RAM using the resolved model path."""
    try:
        logger.info(f"Loading chat model from: {model_path}")
        device = "cpu"  # Always load to RAM initially
        logger.info(f"Using device for initial load: {device}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32, # Use float32 for CPU
            trust_remote_code=True
            # device_map cannot be 'auto' when loading to CPU explicitly
        )
        logger.info(f"Chat model and tokenizer loaded to RAM successfully from {model_path}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load chat model from {model_path}: {e}")
        return None, None

def get_generation_pipeline(model, tokenizer, device: str):
    """Creates a generation pipeline on the specified device."""
    try:
        logger.info(f"Creating generation pipeline on device: {device}")
        # Determine torch_dtype based on device
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Ensure model is on the correct device before creating pipeline
        model.to(device)

        gen_pipeline = pipeline(
            "text-generation", # Corrected task
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1, # device=0 for cuda, -1 for cpu
            torch_dtype=dtype
        )
        logger.info(f"Generation pipeline created successfully on {device}")
        return gen_pipeline
    except Exception as e:
        logger.error(f"Failed to create generation pipeline on {device}: {e}")
        return None

def move_model_to_device(model, device: str):
    """Moves the model to the specified device (CPU or VRAM)."""
    try:
        logger.info(f"Moving model to {device}...")
        model.to(device)
        logger.info(f"Model moved to {device} successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to move model to {device}: {e}")
        return False

# Redundant functions move_model_to_vram and move_model_to_ram are now removed. 