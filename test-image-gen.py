import os
import sys
import logging
import subprocess
import time
import signal  # Add signal module for handling Ctrl+C
import threading  # Add threading module for progress tracking

# Set up basic logging before imports that might fail
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Variables ---
MODEL_PATH = "./models/FLUX.1-dev"
IMG_HEIGHT = 1080 # Image dimensions (Y)
IMG_WIDTH = 1080 # Image dimensions (X)
NUM_INFERENCE_STEPS = 50 # Number of diffusion steps
GUIDANCE_SCALE = 7.0  # Scale for classifier-free guidance
MAX_SEQUENCE_LENGTH = 512 # Maximum sequence length for the model
CHECKPOINT_STEPS = 5  # Save intermediate image every N steps

# Prompts
PROMPT_TEXT = """Masterful detailed illustration of a slightly slouched transgirl, consistent character design.
Art style: vibrant cartoon with clean lines, smooth shading, and a slightly anime-inspired aesthetic.
Character: long pastel blue hair with subtle pink streaks, gentle melancholic eyes, soft facial features.
Clothing: wearing a comfortable oversized hoodie with prominent trans flag stripes (pastel blue, pink, white, pink, blue) horizontally across the chest and on the sleeves.
She looks sad, holding a small, white, rectangular sign in her hands. The sign features the clear, legible, black, sans-serif text "I DO NOT LIKE PICKLES" written horizontally in all capital letters.
Background: simple, soft-focus, complementary colors that do not distract from the character.
Overall mood: poignant yet hopeful.
High quality, detailed, sharp focus, studio lighting, intricate details, consistent art style throughout."""

NEGATIVE_PROMPT_TEXT = """deformed, ugly, amateur, bad art, blurry, pixelated, 
grainy, low resolution, poorly drawn, distorted proportions, disfigured, 
oversaturated, undersaturated, bad anatomy, inconsistent style, style change, 
character morphing, photorealistic, 3D render, 
text errors, illegible text, garbled text, misspelled text, jumbled text, unreadable text, text artifacts,
mutated hands, extra limbs, fused fingers, too many fingers, watermark, signature, artist name,
clashing colors, messy lines, inconsistent lighting."""

# Logging Messages
LOG_MSG_SETUP_ENV = "Setting up environment and checking dependencies..."
LOG_MSG_INTERRUPT_RECEIVED = "Interrupt received, stopping generation gracefully..."
LOG_MSG_PYTORCH_VERSION = "PyTorch version: {}"
LOG_MSG_TORCHVISION_VERSION = "Torchvision version: {}"
LOG_MSG_USING_LOCAL_MODEL = "Using local Flux.1-dev model as requested"
LOG_MSG_LOADING_MODEL_PATH = "Loading Flux.1 model from local path: {}"
LOG_MSG_CHECKING_GPU_MEM_PRE_LOAD = "Checking GPU memory before loading model..."
LOG_MSG_GPU_NAME = "GPU: {}"
LOG_MSG_GPU_TOTAL_MEM = "Total GPU memory: {:.2f} GB"
LOG_MSG_GPU_AVAILABLE_MEM = "Available GPU memory: {:.2f} GB"
LOG_MSG_GPU_AVAILABLE_MEM_AFTER_LOADING = "Available GPU memory after model loading: {:.2f} GB"
LOG_MSG_CUDA_CACHE_CLEARED = "CUDA cache cleared"
LOG_MSG_LOADING_MODEL_LOCAL_DIR = "Loading Flux.1 model from local directory..."
LOG_MSG_MODEL_FILES_FOUND_PREFIX = "Found {} files in model directory: "
LOG_MSG_MODEL_FILES_ELLIPSIS = "..."
LOG_MSG_CPU_OFFLOAD = "Enabling sequential CPU offloading for maximum compatibility"
LOG_MSG_COMPAT_MEM_OPTS = "Applying compatible memory optimizations"
LOG_MSG_ATTN_SLICING = "Enabling attention slicing"
LOG_MSG_MODEL_LOADED_COMPAT = "Flux model loaded with compatibility fixes in {:.2f} seconds"
LOG_MSG_CTRL_C_PROMPT = "Press Ctrl+C at any time to stop generation"
LOG_MSG_GENERATING_IMAGE = "Generating high-quality image with Flux.1..."
LOG_MSG_GENERATION_PROGRESS = "Generation in progress... (Elapsed: {:.1f}s)"
LOG_MSG_INTERMEDIATE_RESULT_SAVED_STEP = "Saved intermediate result at step {}"
LOG_MSG_INTERMEDIATE_RESULT_SAVED_PATH = "Intermediate result saved to {}"
LOG_MSG_PARTIAL_RESULT_SAVED_PATH = "Partial result saved to {}"
LOG_MSG_IMAGE_GENERATION_STATUS_TIME = "Image generation {} in {:.2f} seconds"
LOG_MSG_IMAGE_SAVED_PATH = "Image saved to {}"
LOG_MSG_STARTING_RUN = "--- Starting run {} ---"

# Warnings
LOG_WARN_CUDA_NOT_AVAILABLE = "CUDA not available, using CPU (will be slow)"

# Error Messages
LOG_ERR_IMPORT = "Import error: {}"
LOG_ERR_IMPORT_RESTART_NOTE = "Fix may require a script restart. Run the script again after installations."
LOG_ERR_MODEL_DIR_NOT_FOUND = "Local model directory not found at {}. Please check the path."
LOG_ERR_GPU_MEM_LOAD = "GPU memory error during model load: {}"
LOG_ERR_GPU_MEM_INSUFFICIENT = "Your GPU does not have enough memory to run Flux.1, even with optimizations."
LOG_ERR_MODEL_LOADING = "Error during model loading: {}"
LOG_ERR_PIPELINE_NOT_LOADED_SKIP = "Pipeline not loaded. Skipping generation."
LOG_ERR_PIPELINE_LOAD_FAIL_EXIT = "Failed to load the pipeline. Exiting."

# Info Messages
LOG_INFO_GPU_MEM_OPTIONS_HEADER = "Options to resolve GPU memory issues:"
LOG_INFO_GPU_MEM_OPTION_COLAB = "1. Use Google Colab for free GPU access: https://colab.research.google.com/"
LOG_INFO_GPU_MEM_OPTION_REPLICATE = "2. Use Replicate API for image generation: https://replicate.com/stability-ai/flux"
LOG_INFO_GPU_MEM_OPTION_SMALLER_MODEL = "3. Try an even smaller model like SD 1.5"
LOG_INFO_RECOMMEND_REPLICATE = "Recommended alternative: Use Replicate API for FLUX model: https://replicate.com/stability-ai/flux"
LOG_INFO_KB_INTERRUPT_GENERATION = "Keyboard interrupt detected during generation"
LOG_INFO_NO_PARTIAL_RESULT_KB = "No partial result available to save"
LOG_INFO_NO_IMAGE_EARLY_INTERRUPT = "No image generated due to early interruption"
LOG_INFO_KB_INTERRUPT_STOPPING = "Keyboard interrupt detected, stopping..."
LOG_INFO_EXCEPTION_POST_INTERRUPT = "Exception occurred after interrupt was requested"
LOG_INFO_INTERRUPT_STOPPING_RUNS = "Interrupt detected, stopping further runs."

# Debug Messages
LOG_DEBUG_INTERMEDIATE_RESULT_FAIL = "Failed to capture intermediate result: {}"
LOG_DEBUG_MONKEY_PATCH_FAIL = "Could not enable intermediate result capturing: {}"

# File Name Patterns
FILENAME_PARTIAL_IMAGE_PATTERN = "./outputs/flux-1-partial-{}.png"
FILENAME_OUTPUT_IMAGE_PATTERN = "./outputs/flux-1-output-{}.png"

# String Constants
STR_ATTN_SLICING_MODE = "max"
STR_JOIN_SEPARATOR = ", "
STR_STATUS_INTERRUPTED = "interrupted"
STR_STATUS_COMPLETED = "completed"
ERR_MSG_MODEL_PATH_NOT_FOUND = "Model path {} not found"

# --- End Configuration Variables ---

logger.info(LOG_MSG_SETUP_ENV)

# Global flag for interrupt handling
interrupt_requested = False

# Define signal handler for Ctrl+C
def signal_handler(sig, frame):
    global interrupt_requested
    logger.info(LOG_MSG_INTERRUPT_RECEIVED)
    interrupt_requested = True

# Try to fix dependencies before continuing
try:
    import torch
    import torchvision
    from diffusers import FluxPipeline
    from PIL import Image
    
    logger.info(LOG_MSG_PYTORCH_VERSION.format(torch.__version__))
    logger.info(LOG_MSG_TORCHVISION_VERSION.format(torchvision.__version__))
    
except ImportError as e:
    logger.error(LOG_ERR_IMPORT.format(e))
    logger.error(LOG_ERR_IMPORT_RESTART_NOTE)
    sys.exit(1)

# Global pipeline variable
pipe = None

def load_pipeline():
    global pipe
    logger.info(LOG_MSG_USING_LOCAL_MODEL)
    logger.info(LOG_MSG_LOADING_MODEL_PATH.format(os.path.abspath(MODEL_PATH)))
    start_time = time.time()

    try:
        logger.info(LOG_MSG_CHECKING_GPU_MEM_PRE_LOAD)
        if torch.cuda.is_available():
            logger.info(LOG_MSG_GPU_NAME.format(torch.cuda.get_device_name(0)))
            logger.info(LOG_MSG_GPU_TOTAL_MEM.format(torch.cuda.get_device_properties(0).total_memory / 1e9))
            logger.info(LOG_MSG_GPU_AVAILABLE_MEM.format(torch.cuda.mem_get_info()[0] / 1e9))
        else:
            logger.warning(LOG_WARN_CUDA_NOT_AVAILABLE)

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(LOG_MSG_CUDA_CACHE_CLEARED)
            
            logger.info(LOG_MSG_LOADING_MODEL_LOCAL_DIR)
            if not os.path.exists(MODEL_PATH):
                logger.error(LOG_ERR_MODEL_DIR_NOT_FOUND.format(MODEL_PATH))
                raise FileNotFoundError(ERR_MSG_MODEL_PATH_NOT_FOUND.format(MODEL_PATH))

            model_files = os.listdir(MODEL_PATH)
            log_model_files_msg = LOG_MSG_MODEL_FILES_FOUND_PREFIX.format(len(model_files)) + \
                                  STR_JOIN_SEPARATOR.join(model_files[:5]) + \
                                  (LOG_MSG_MODEL_FILES_ELLIPSIS if len(model_files) > 5 else "")
            logger.info(log_model_files_msg)
            
            pipe = FluxPipeline.from_pretrained(MODEL_PATH)
            pipe.enable_sequential_cpu_offload()
            
            logger.info(LOG_MSG_COMPAT_MEM_OPTS)
            if hasattr(pipe, "enable_attention_slicing"):
                logger.info(LOG_MSG_ATTN_SLICING)
                pipe.enable_attention_slicing(STR_ATTN_SLICING_MODE)
            
            logger.info(LOG_MSG_MODEL_LOADED_COMPAT.format(time.time() - start_time))
            if torch.cuda.is_available():
                logger.info(LOG_MSG_GPU_AVAILABLE_MEM_AFTER_LOADING.format(torch.cuda.mem_get_info()[0] / 1e9))
            return pipe
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            logger.error(LOG_ERR_GPU_MEM_LOAD.format(str(e)))
            logger.error(LOG_ERR_GPU_MEM_INSUFFICIENT)
            logger.info(LOG_INFO_GPU_MEM_OPTIONS_HEADER)
            logger.info(LOG_INFO_GPU_MEM_OPTION_COLAB)
            logger.info(LOG_INFO_GPU_MEM_OPTION_REPLICATE)
            logger.info(LOG_INFO_GPU_MEM_OPTION_SMALLER_MODEL)
            return None
            
    except Exception as e:
        logger.error(LOG_ERR_MODEL_LOADING.format(str(e)))
        logger.info(LOG_INFO_RECOMMEND_REPLICATE)
        return None

def main(pipe_instance):
    global interrupt_requested
    interrupt_requested = False

    if pipe_instance is None:
        logger.error(LOG_ERR_PIPELINE_NOT_LOADED_SKIP)
        return

    signal.signal(signal.SIGINT, signal_handler)
    logger.info(LOG_MSG_CTRL_C_PROMPT)
    
    logger.info(LOG_MSG_GENERATING_IMAGE)
    generation_start = time.time()
    
    try:
        def progress_tracker():
            start_time = time.time()
            last_log_time = start_time
            while not interrupt_requested and threading.current_thread().is_alive():
                current_time = time.time()
                if current_time - last_log_time >= 10:
                    elapsed = current_time - start_time
                    logger.info(LOG_MSG_GENERATION_PROGRESS.format(elapsed))
                    last_log_time = current_time
                time.sleep(1)
        
        tracker = threading.Thread(target=progress_tracker)
        tracker.daemon = True
        tracker.start()
        
        try:
            last_checkpoint = 0
            partial_result = None
            
            original_forward = pipe_instance.transformer.forward
            
            def forward_with_capture(*args, **kwargs):
                result = original_forward(*args, **kwargs)
                
                if hasattr(pipe_instance, 'decode_latents') and hasattr(pipe_instance, 'latents'):
                    try:
                        current_step = getattr(pipe_instance, '_current_step', 0)
                        if current_step > last_checkpoint + CHECKPOINT_STEPS:
                            with torch.no_grad():
                                decoded = pipe_instance.decode_latents(pipe_instance.latents)
                                partial_image = pipe_instance.image_processor.postprocess(decoded)[0]
                                partial_result = partial_image
                                last_checkpoint = current_step
                                logger.info(LOG_MSG_INTERMEDIATE_RESULT_SAVED_STEP.format(current_step))
                    except Exception as e:
                        logger.debug(LOG_DEBUG_INTERMEDIATE_RESULT_FAIL.format(e))
                
                if interrupt_requested:
                    if partial_result is not None:
                        output_path = FILENAME_PARTIAL_IMAGE_PATTERN.format(int(time.time()))
                        partial_result.save(output_path)
                        logger.info(LOG_MSG_INTERMEDIATE_RESULT_SAVED_PATH.format(os.path.abspath(output_path)))
                
                return result
            
            try:
                pipe_instance.transformer.forward = forward_with_capture
            except Exception as e:
                logger.debug(LOG_DEBUG_MONKEY_PATCH_FAIL.format(e))
            
            image = pipe_instance(
                PROMPT_TEXT,
                height=IMG_HEIGHT,
                width=IMG_WIDTH,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=NUM_INFERENCE_STEPS,
                negative_prompt=NEGATIVE_PROMPT_TEXT,
                max_sequence_length=MAX_SEQUENCE_LENGTH
            ).images[0]
            
            try:
                pipe_instance.transformer.forward = original_forward
            except:
                pass
                    
        except KeyboardInterrupt:
            logger.info(LOG_INFO_KB_INTERRUPT_GENERATION)
            
            if partial_result is not None:
                output_path = FILENAME_PARTIAL_IMAGE_PATTERN.format(int(time.time()))
                partial_result.save(output_path)
                logger.info(LOG_MSG_PARTIAL_RESULT_SAVED_PATH.format(os.path.abspath(output_path)))
                image = partial_result
            else:
                logger.info(LOG_INFO_NO_PARTIAL_RESULT_KB)
                image = None
        
        status_message = STR_STATUS_INTERRUPTED if interrupt_requested else STR_STATUS_COMPLETED
        logger.info(LOG_MSG_IMAGE_GENERATION_STATUS_TIME.format(status_message, time.time() - generation_start))
        
        if image is not None:
            output_path = FILENAME_OUTPUT_IMAGE_PATTERN.format(int(time.time()))
            image.save(output_path)
            logger.info(LOG_MSG_IMAGE_SAVED_PATH.format(os.path.abspath(output_path)))
        else:
            logger.info(LOG_INFO_NO_IMAGE_EARLY_INTERRUPT)
        
    except KeyboardInterrupt:
        logger.info(LOG_INFO_KB_INTERRUPT_STOPPING)
    
    except Exception as e:
        if interrupt_requested:
            logger.info(LOG_INFO_EXCEPTION_POST_INTERRUPT)
        else:
            raise
            
if __name__ == "__main__":
    loaded_pipe = load_pipeline()

    if loaded_pipe:
        main(loaded_pipe)
    else:
        logger.error(LOG_ERR_PIPELINE_LOAD_FAIL_EXIT)