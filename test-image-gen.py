prompt = """Masterful detailed illustration of a slightly slouched transgirl, consistent character design.
Art style: vibrant cartoon with clean lines, smooth shading, and a slightly anime-inspired aesthetic.
Character: long pastel blue hair with subtle pink streaks, gentle melancholic eyes, soft facial features.
Clothing: wearing a comfortable oversized hoodie with prominent trans flag stripes (pastel blue, pink, white, pink, blue) horizontally across the chest and on the sleeves.
She looks sad, perhaps holding a small, neatly lettered sign or a clear speech bubble saying "I do not like pickles".
Background: simple, soft-focus, complementary colors that do not distract from the character.
Overall mood: poignant yet hopeful.
High quality, detailed, sharp focus, studio lighting, intricate details, consistent art style throughout."""

negative_prompt = """deformed, ugly, amateur, bad art, blurry, pixelated, 
grainy, low resolution, poorly drawn, distorted proportions, disfigured, 
oversaturated, undersaturated, bad anatomy, inconsistent style, style change, 
character morphing, photorealistic, 3D render, text errors, mutated hands, 
extra limbs, fused fingers, too many fingers, watermark, signature, artist name,
clashing colors, messy lines, inconsistent lighting."""

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

logger.info("Setting up environment and checking dependencies...")

# --- Configuration Variables ---
MODEL_PATH = "./models/FLUX.1-dev"
IMG_HEIGHT = 1080
IMG_WIDTH = 1080
NUM_INFERENCE_STEPS = 25
GUIDANCE_SCALE = 7.0
MAX_SEQUENCE_LENGTH = 512
CHECKPOINT_STEPS = 5  # Save intermediate image every N steps
# --- End Configuration Variables ---

# Global flag for interrupt handling
interrupt_requested = False

# Define signal handler for Ctrl+C
def signal_handler(sig, frame):
    global interrupt_requested
    logger.info("Interrupt received, stopping generation gracefully...")
    interrupt_requested = True
    # Don't exit immediately, allow the code to clean up

# Try to fix dependencies before continuing
try:
    # Now import the fixed packages
    import torch
    import torchvision
    from diffusers import FluxPipeline
    from PIL import Image
    
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Torchvision version: {torchvision.__version__}")
    
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Fix may require a script restart. Run the script again after installations.")
    sys.exit(1)

def main():
    global interrupt_requested # Allow main to reset this if running in a loop later
    interrupt_requested = False # Reset for each run

    # Use local Flux.1 model
    logger.info("Using local Flux.1-dev model as requested")

    logger.info(f"Loading Flux.1 model from local path: {os.path.abspath(MODEL_PATH)}...")
    start_time = time.time()

    try:
        logger.info("Checking GPU memory before loading model...")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            logger.info(f"Available GPU memory: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")
        else:
            logger.warning("CUDA not available, using CPU (will be slow)")

        # Apply performance optimizations, but prioritize compatibility
        try:
            # First clear CUDA cache to free up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
            
            # Try to fix compatibility issues with specific loading parameters
            logger.info(f"Loading Flux.1 model from local directory...")
            if not os.path.exists(MODEL_PATH):
                logger.error(f"Local model directory not found at {MODEL_PATH}. Please check the path.")
                raise FileNotFoundError(f"Model path {MODEL_PATH} not found")

            # List files in model directory for debugging
            model_files = os.listdir(MODEL_PATH)
            logger.info(f"Found {len(model_files)} files in model directory: {', '.join(model_files[:5])}" + 
                        ("..." if len(model_files) > 5 else ""))
            
            # Load with default precision to avoid variant errors
            pipe = FluxPipeline.from_pretrained(
                MODEL_PATH,
                # No torch_dtype specification to avoid compatibility issues
            )
            
            # Apply CPU offloading instead of device mapping
            logger.info("Enabling sequential CPU offloading for maximum compatibility")
            pipe.enable_sequential_cpu_offload()
            
            # Apply memory optimizations that are compatible
            logger.info("Applying compatible memory optimizations")
            if hasattr(pipe, "enable_attention_slicing"):
                logger.info("Enabling attention slicing")
                pipe.enable_attention_slicing("max")
            
            logger.info(f"Flux model loaded with compatibility fixes in {time.time() - start_time:.2f} seconds")
            
            # Set up the interrupt handler before generation
            signal.signal(signal.SIGINT, signal_handler)
            logger.info("Press Ctrl+C at any time to stop generation")
            
            # Use more conservative generation settings for stability
            logger.info(f"Generating high-quality image with Flux.1...")
            generation_start = time.time()
            
            try:
                # For models that don't support callbacks, we'll use a progress tracking thread
                
                # Create a progress tracking thread since FluxPipeline doesn't support callbacks
                def progress_tracker():
                    start_time = time.time()
                    last_log_time = start_time
                    while not interrupt_requested and threading.current_thread().is_alive():
                        current_time = time.time()
                        # Log progress every 10 seconds
                        if current_time - last_log_time >= 10:
                            elapsed = current_time - start_time
                            logger.info(f"Generation in progress... (Elapsed: {elapsed:.1f}s)")
                            last_log_time = current_time
                        time.sleep(1)
                
                # Start the progress tracker in a separate thread
                tracker = threading.Thread(target=progress_tracker)
                tracker.daemon = True  # This ensures the thread will exit when the main program exits
                tracker.start()
                
                # Add explicit try/except for the generation to capture interruptions
                try:
                    last_checkpoint = 0
                    partial_result = None
                    
                    # Attempt to get intermediate latents if the model supports it
                    # This is a hack since Flux doesn't officially support callbacks
                    original_forward = pipe.transformer.forward
                    
                    def forward_with_capture(*args, **kwargs):
                        result = original_forward(*args, **kwargs)
                        
                        # Try to capture intermediate state from latents if possible
                        if hasattr(pipe, 'decode_latents') and hasattr(pipe, 'latents'):
                            try:
                                current_step = getattr(pipe, '_current_step', 0)
                                if current_step > last_checkpoint + CHECKPOINT_STEPS:
                                    # Attempt to decode current latents
                                    with torch.no_grad():
                                        decoded = pipe.decode_latents(pipe.latents)
                                        partial_image = pipe.image_processor.postprocess(decoded)[0]
                                        partial_result = partial_image
                                        last_checkpoint = current_step
                                        logger.info(f"Saved intermediate result at step {current_step}")
                            except Exception as e:
                                # Capturing intermediate results failed, log and continue
                                logger.debug(f"Failed to capture intermediate result: {e}")
                        
                        # Check for interrupt
                        if interrupt_requested:
                            # Try to save partial work before exiting
                            if partial_result is not None:
                                output_path = f"flux-1-partial-{int(time.time())}.png"
                                partial_result.save(output_path)
                                logger.info(f"Intermediate result saved to {os.path.abspath(output_path)}")
                        
                        return result
                    
                    # Try to monkey-patch the forward method to capture intermediates
                    try:
                        pipe.transformer.forward = forward_with_capture
                    except Exception as e:
                        logger.debug(f"Could not enable intermediate result capturing: {e}")
                    
                    # Generate the image
                    image = pipe(
                        prompt,
                        height=IMG_HEIGHT,
                        width=IMG_WIDTH,
                        guidance_scale=GUIDANCE_SCALE,
                        num_inference_steps=NUM_INFERENCE_STEPS,
                        max_sequence_length=MAX_SEQUENCE_LENGTH
                    ).images[0]
                    
                    # Restore original forward method
                    try:
                        pipe.transformer.forward = original_forward
                    except:
                        pass
                        
                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt detected during generation")
                    
                    # Try to save the partial result if available
                    if partial_result is not None:
                        output_path = f"flux-1-partial-{int(time.time())}.png"
                        partial_result.save(output_path)
                        logger.info(f"Partial result saved to {os.path.abspath(output_path)}")
                        image = partial_result  # Set as the image to save
                    else:
                        logger.info("No partial result available to save")
                        image = None
                
                logger.info(f"Image generation {'interrupted' if interrupt_requested else 'completed'} in {time.time() - generation_start:.2f} seconds")
                
                # Only save the image if it exists
                if image is not None:
                    # Save the image with Flux-specific name and timestamp
                    output_path = f"flux-1-output-{int(time.time())}.png"
                    image.save(output_path)
                    logger.info(f"Image saved to {os.path.abspath(output_path)}")
                else:
                    logger.info("No image generated due to early interruption")
                
            except KeyboardInterrupt:
                # Catch keyboard interrupt that might bypass the signal handler
                logger.info("Keyboard interrupt detected, stopping...")
            
            except Exception as e:
                if interrupt_requested:
                    logger.info("Exception occurred after interrupt was requested")
                else:
                    # Re-raise if it wasn't due to our interrupt
                    raise
                    
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            logger.error(f"GPU memory error: {str(e)}")
            logger.error("Your GPU does not have enough memory to run Flux.1, even with optimizations.")
            
            # Provide helpful guidance
            logger.info("Options to resolve GPU memory issues:")
            logger.info("1. Use Google Colab for free GPU access: https://colab.research.google.com/")
            logger.info("2. Use Replicate API for image generation: https://replicate.com/stability-ai/flux")
            logger.info("3. Try an even smaller model like SD 1.5")
            # raise # Optionally re-raise if you want script to stop on this error
            return # Exit main if GPU error occurs
            
    except Exception as e:
        logger.error(f"Error during image generation: {str(e)}")
        logger.info("Recommended alternative: Use Replicate API for FLUX model: https://replicate.com/stability-ai/flux")
        # raise # Optionally re-raise
        return # Exit main if other error occurs

if __name__ == "__main__":
    main()
    # To run multiple times for consistency testing, you could do:
    # for i in range(3):
    #     logger.info(f"--- Starting run {i+1} ---")
    #     main()
    #     if interrupt_requested: # Check if Ctrl+C was pressed during a run
    #         logger.info("Interrupt detected, stopping further runs.")
    #         break
    #     time.sleep(1) # Small delay between runs if needed