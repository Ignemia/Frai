prompt = "A slightly slouches transgirl (make sure the flag is visible, pay a lot of attention to the flag and ensure it is part of the clothing, the trans flag consists of 5 parts fro mtop to bottom: pastel blue, pastel pink, white, pastel pink, pastel blue), sad, saying \"I do not like pickles\". Make it cartoonish"

import torch
from diffusers import FluxPipeline
import os
import time
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to the local FLUX-1 model
model_path = "models/FLUX.1-dev/"  # Updated to use the standard FLUX-1 model instead of dev

logger.info(f"Checking for model at: {os.path.abspath(model_path)}")
if not os.path.exists(model_path):
    logger.error(f"Model not found at {model_path}. Please ensure it's downloaded correctly.")
    raise FileNotFoundError(f"Model path {model_path} doesn't exist")

logger.info("Loading FLUX-1 model...")
start_time = time.time()

try:
    logger.info("Checking GPU memory before loading model...")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info(f"Available GPU memory: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")
    else:
        logger.warning("CUDA not available, using CPU (will be slow)")

    # Load model directly to GPU without CPU offloading for maximum performance
    pipe = FluxPipeline.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,  # Use float16 for better performance
        variant="fp16",  # Use fp16 variant if available
        device_map="auto"  # Let accelerate decide optimal device mapping
    )
    
    # Apply performance optimizations
    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        logger.info("Enabling xformers for efficient attention")
        pipe.enable_xformers_memory_efficient_attention()
    
    # Ensure model is on GPU
    if torch.cuda.is_available():
        pipe.to("cuda")
        logger.info("Model moved to CUDA")
    
    logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    # Check GPU memory after loading
    if torch.cuda.is_available():
        logger.info(f"Available GPU memory after loading: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")
    
    logger.info(f"Generating image with prompt: {prompt[:50]}...")
    generation_start = time.time()
    
    # Generate the image with CUDA optimizations
    with torch.inference_mode(), torch.cuda.amp.autocast():
        image = pipe(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=8192,
            generator=torch.Generator("cuda").manual_seed(0),  # Use manual_seed (more modern)
        ).images[0]
    
    logger.info(f"Image generated in {time.time() - generation_start:.2f} seconds")
    
    # Save the image
    output_path = "flux-1-output.png"  # Updated filename to reflect the model version
    image.save(output_path)
    logger.info(f"Image saved to {os.path.abspath(output_path)}")
    
except Exception as e:
    logger.error(f"Error during image generation: {str(e)}")
    raise