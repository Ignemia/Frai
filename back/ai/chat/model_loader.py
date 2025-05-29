"""Handles model loading and management."""
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline
import psutil
import gc

logger = logging.getLogger(__name__)

def get_optimal_device_map(model_name_or_path: str, model_path: str):
    """Calculate optimal device mapping for CUDA with sequential offloading."""

    # Force CPU-only mode to avoid Gemma3 compilation issues during testing
    import os
    if os.getenv("FORCE_CPU_MODE", "false").lower() == "true":
        logger.info("FORCE_CPU_MODE enabled, using CPU-only device map")
        return "cpu"

    if not torch.cuda.is_available():
        logger.info("CUDA not available, using CPU-only device map")
        return "cpu"
    
    try:
        # Get GPU memory info
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_free_memory = torch.cuda.mem_get_info()[0]
        
        # Get system RAM info
        system_memory = psutil.virtual_memory().total
        
        logger.info(f"GPU memory: {gpu_free_memory / 1024**3:.1f}GB free / {gpu_memory / 1024**3:.1f}GB total")
        logger.info(f"System RAM: {system_memory / 1024**3:.1f}GB total")
        
        # Always use auto mapping with CUDA - let accelerate handle sequential offloading
        logger.info("Using CUDA with automatic sequential offloading to RAM when needed")
        return "auto"
            
    except Exception as e:
        logger.warning(f"Error calculating device map: {e}, defaulting to auto")
        return "auto"

def load_model_and_tokenizer(model_name_or_path: str, model_path: str):
    """Loads the model and tokenizer with optimal device mapping for partial GPU offloading."""
    try:
        logger.info(f"Loading chat model from: {model_path}")
        
        # Get optimal device mapping
        device_map = get_optimal_device_map(model_name_or_path, model_path)
        
        # Clear GPU cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Determine dtype based on target devices
        if device_map == "cpu":
            default_dtype = torch.float32
            logger.info("Using CPU-only loading with float32")
        else:
            # Use float16 for GPU parts, float32 for CPU parts (handled automatically)
            default_dtype = torch.float16
            logger.info("Using mixed-device loading with automatic dtype selection")

        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with device mapping
        if device_map == "auto":
            logger.info("Loading model with CUDA and sequential RAM offloading")
            # Calculate dynamic memory allocation
            gpu_available_gb = gpu_free_memory * 0.8 / (1024**3)  # 80% of free GPU memory
            ram_available_gb = system_memory * 0.4 / (1024**3)  # 40% of total RAM
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=default_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="auto",
                max_memory={
                    0: f"{gpu_available_gb:.1f}GiB",
                    "cpu": f"{ram_available_gb:.1f}GiB"
                },
                offload_folder="./offload",
                offload_state_dict=True,
                attn_implementation="eager"  # Disable flash attention to avoid compilation issues
            )
        else:
            # CPU-only loading (fallback only)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,  # Use float32 for CPU
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="cpu",
                attn_implementation="eager"  # Disable flash attention to avoid compilation issues
            )
        
        # Log device distribution
        if hasattr(model, 'hf_device_map'):
            logger.info("Model device distribution:")
            device_counts = {}
            for layer, device in model.hf_device_map.items():
                device_counts[device] = device_counts.get(device, 0) + 1
            for device, count in device_counts.items():
                logger.info(f"  {device}: {count} layers")
        
        logger.info(f"Chat model and tokenizer loaded successfully from {model_path}")
        logger.info(f"Model dtype: {model.dtype}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load chat model from {model_path}: {e}")
        
        # Try fallback to HuggingFace if local model fails
        if model_path != model_name_or_path:
            logger.info(f"Attempting fallback to HuggingFace model: {model_name_or_path}")
            try:
                # Clear memory before fallback
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                
                device_map = get_optimal_device_map(model_name_or_path, model_name_or_path)
                default_dtype = torch.float16 if device_map != "cpu" else torch.float32
                
                tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                if device_map == "auto":
                    # Use dynamic memory allocation for fallback too
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    gpu_free_memory = torch.cuda.mem_get_info()[0]
                    system_memory = psutil.virtual_memory().total
                    
                    gpu_available_gb = gpu_free_memory * 0.8 / (1024**3)
                    ram_available_gb = system_memory * 0.4 / (1024**3)
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name_or_path,
                        torch_dtype=default_dtype,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        device_map="auto",
                        max_memory={
                            0: f"{gpu_available_gb:.1f}GiB",
                            "cpu": f"{ram_available_gb:.1f}GiB"
                        },
                        offload_folder="./offload",
                        offload_state_dict=True,
                        attn_implementation="eager"  # Disable flash attention to avoid compilation issues
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name_or_path,
                        torch_dtype=torch.float32,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        device_map="cpu",
                        attn_implementation="eager"  # Disable flash attention to avoid compilation issues
                    )
                
                logger.info(f"Chat model and tokenizer loaded from HuggingFace: {model_name_or_path}")
                return model, tokenizer
            except Exception as fallback_e:
                logger.error(f"Fallback to HuggingFace also failed: {fallback_e}")
        
        return None, None

def get_generation_pipeline(model, tokenizer, device: str):
    """Creates a generation pipeline optimized for sequential offloading models."""
    try:
        logger.info(f"Creating generation pipeline for device: {device}")
        
        # Check if model is distributed across devices (sequential offloading)
        if hasattr(model, 'hf_device_map'):
            logger.info("Model uses sequential offloading, creating distributed pipeline")
            # For models with sequential offloading, create pipeline without device specification
            gen_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=model.dtype,
                trust_remote_code=True
            )
        else:
            # Single device model - use CUDA if available
            if torch.cuda.is_available():
                device_id = 0
                dtype = model.dtype
                logger.info(f"Using CUDA pipeline with dtype: {dtype}")
            else:
                device_id = -1
                dtype = model.dtype
                logger.info(f"Using CPU pipeline with dtype: {dtype}")

            gen_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device_id,
                torch_dtype=dtype,
                trust_remote_code=True
            )
        
        logger.info("Generation pipeline created successfully")
        return gen_pipeline
    except Exception as e:
        logger.error(f"Failed to create generation pipeline: {e}")
        return None

def move_model_to_device(model, device: str):
    """Moves the model to the specified device, handling sequential offloading models."""
    try:
        # Check if model uses sequential offloading
        if hasattr(model, 'hf_device_map'):
            logger.info("Model uses sequential offloading - no manual device moves needed")
            return True
        
        current_device = next(model.parameters()).device
        logger.info(f"Moving model from {current_device} to {device}...")
        
        # Check if move is necessary - handle cuda vs cuda:0
        current_device_str = str(current_device)
        if current_device_str == device or (device == "cuda" and current_device_str.startswith("cuda:")):
            logger.info(f"Model already on {device}, no move needed")
            return True
        
        # For CUDA moves, ensure we have CUDA available
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_free = torch.cuda.mem_get_info()[0]
            memory_total = torch.cuda.mem_get_info()[1]
            logger.info(f"GPU memory: {memory_free / 1024**3:.1f}GB free / {memory_total / 1024**3:.1f}GB total")
            logger.info("Moving model to CUDA with potential RAM offloading")
        elif device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, keeping on CPU")
            return False
        
        model.to(device)
        
        # Verify the move was successful
        new_device = next(model.parameters()).device
        new_device_str = str(new_device)
        
        # Handle cuda vs cuda:0 device naming
        if device == "cuda" and new_device_str.startswith("cuda:"):
            logger.info(f"Model successfully moved to {new_device}")
            return True
        elif new_device_str == device:
            logger.info(f"Model successfully moved to {new_device}")
            return True
        else:
            logger.error(f"Model move failed: expected {device}, got {new_device}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to move model to {device}: {e}")
        return False

# Redundant functions move_model_to_vram and move_model_to_ram are now removed. 