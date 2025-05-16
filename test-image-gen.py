import os
import sys
import logging
import subprocess
import time
import signal
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = "./models/FLUX.1-dev"
IMG_HEIGHT = 256*8
IMG_WIDTH = 256*8
NUM_INFERENCE_STEPS = 35
GUIDANCE_SCALE = 7.0
MAX_SEQUENCE_LENGTH = 512
CHECKPOINT_STEPS = 5

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

LOG_WARN_CUDA_NOT_AVAILABLE = "CUDA not available, using CPU (will be slow)"

LOG_ERR_IMPORT = "Import error: {}"
LOG_ERR_IMPORT_RESTART_NOTE = "Fix may require a script restart. Run the script again after installations."
LOG_ERR_MODEL_DIR_NOT_FOUND = "Local model directory not found at {}. Please check the path."
LOG_ERR_GPU_MEM_LOAD = "GPU memory error during model load: {}"
LOG_ERR_GPU_MEM_INSUFFICIENT = "Your GPU does not have enough memory to run Flux.1, even with optimizations."
LOG_ERR_MODEL_LOADING = "Error during model loading: {}"
LOG_ERR_PIPELINE_NOT_LOADED_SKIP = "Pipeline not loaded. Skipping generation."
LOG_ERR_PIPELINE_LOAD_FAIL_EXIT = "Failed to load the pipeline. Exiting."

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

LOG_DEBUG_INTERMEDIATE_RESULT_FAIL = "Failed to capture intermediate result: {}"
LOG_DEBUG_MONKEY_PATCH_FAIL = "Could not enable intermediate result capturing: {}"

FILENAME_PARTIAL_IMAGE_PATTERN = "./outputs/flux-1-partial-{}.png"
FILENAME_OUTPUT_IMAGE_PATTERN = "./outputs/flux-1-output-{}.png"

STR_ATTN_SLICING_MODE = "max"
STR_JOIN_SEPARATOR = ", "
STR_STATUS_INTERRUPTED = "interrupted"
STR_STATUS_COMPLETED = "completed"
ERR_MSG_MODEL_PATH_NOT_FOUND = "Model path {} not found"

logger.info(LOG_MSG_SETUP_ENV)

interrupt_requested = False

def signal_handler(sig, frame):
    global interrupt_requested
    logger.info(LOG_MSG_INTERRUPT_RECEIVED)
    interrupt_requested = True

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

def _log_initial_gpu_status():
    logger.info(LOG_MSG_CHECKING_GPU_MEM_PRE_LOAD)
    if torch.cuda.is_available():
        logger.info(LOG_MSG_GPU_NAME.format(torch.cuda.get_device_name(0)))
        props = torch.cuda.get_device_properties(0)
        logger.info(LOG_MSG_GPU_TOTAL_MEM.format(props.total_memory / 1e9))
        logger.info(LOG_MSG_GPU_AVAILABLE_MEM.format(torch.cuda.mem_get_info()[0] / 1e9))
    else:
        logger.warning(LOG_WARN_CUDA_NOT_AVAILABLE)

def _check_model_path_and_log_files_helper(model_path):
    logger.info(LOG_MSG_LOADING_MODEL_LOCAL_DIR)
    if not os.path.exists(model_path):
        logger.error(LOG_ERR_MODEL_DIR_NOT_FOUND.format(model_path))
        raise FileNotFoundError(ERR_MSG_MODEL_PATH_NOT_FOUND.format(model_path))

    model_files = os.listdir(model_path)
    prefix = LOG_MSG_MODEL_FILES_FOUND_PREFIX.format(len(model_files))
    files_str = STR_JOIN_SEPARATOR.join(model_files[:5])
    ellipsis = LOG_MSG_MODEL_FILES_ELLIPSIS if len(model_files) > 5 else ""
    log_model_files_msg = prefix + files_str + ellipsis
    logger.info(log_model_files_msg)

def _clear_cuda_cache_if_available():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(LOG_MSG_CUDA_CACHE_CLEARED)

def _enable_attention_slicing_if_supported(pipeline_instance):
    if hasattr(pipeline_instance, "enable_attention_slicing"):
        logger.info(LOG_MSG_ATTN_SLICING)
        pipeline_instance.enable_attention_slicing(STR_ATTN_SLICING_MODE)

def _handle_gpu_memory_error_on_load(exception_obj):
    logger.error(LOG_ERR_GPU_MEM_LOAD.format(str(exception_obj)))
    logger.error(LOG_ERR_GPU_MEM_INSUFFICIENT)
    logger.info(LOG_INFO_GPU_MEM_OPTIONS_HEADER)
    logger.info(LOG_INFO_GPU_MEM_OPTION_COLAB)
    logger.info(LOG_INFO_GPU_MEM_OPTION_REPLICATE)
    logger.info(LOG_INFO_GPU_MEM_OPTION_SMALLER_MODEL)

def _load_pipeline_core_helper(model_path, start_time):
    try:
        _clear_cuda_cache_if_available()
        loaded_pipe_instance = FluxPipeline.from_pretrained(model_path)
        loaded_pipe_instance.enable_sequential_cpu_offload()
        logger.info(LOG_MSG_COMPAT_MEM_OPTS)
        _enable_attention_slicing_if_supported(loaded_pipe_instance)
        logger.info(LOG_MSG_MODEL_LOADED_COMPAT.format(time.time() - start_time))
        return loaded_pipe_instance
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        _handle_gpu_memory_error_on_load(e)
        return None

def _log_gpu_mem_after_load(loaded_pipe_instance):
    if loaded_pipe_instance and torch.cuda.is_available():
        logger.info(LOG_MSG_GPU_AVAILABLE_MEM_AFTER_LOADING.format(torch.cuda.mem_get_info()[0] / 1e9))

def _load_pipeline_main_try_block(start_time_val):
    _log_initial_gpu_status()
    _check_model_path_and_log_files_helper(MODEL_PATH)
    loaded_pipe = _load_pipeline_core_helper(MODEL_PATH, start_time_val)
    _log_gpu_mem_after_load(loaded_pipe)
    return loaded_pipe

def _handle_load_pipeline_file_not_found():
    return None

def _handle_load_pipeline_generic_error(e):
    logger.error(LOG_ERR_MODEL_LOADING.format(str(e)))
    logger.info(LOG_INFO_RECOMMEND_REPLICATE)
    return None

def load_pipeline():
    logger.info(LOG_MSG_USING_LOCAL_MODEL)
    logger.info(LOG_MSG_LOADING_MODEL_PATH.format(os.path.abspath(MODEL_PATH)))
    start_time = time.time()
    
    try:
        return _load_pipeline_main_try_block(start_time)
    except FileNotFoundError:
        return _handle_load_pipeline_file_not_found()
    except Exception as e:
        return _handle_load_pipeline_generic_error(e)

def initialize_main_run_settings(pipe_instance):
    global interrupt_requested
    interrupt_requested = False

    if pipe_instance is None:
        logger.error(LOG_ERR_PIPELINE_NOT_LOADED_SKIP)
        return None

    signal.signal(signal.SIGINT, signal_handler)
    logger.info(LOG_MSG_CTRL_C_PROMPT)
    logger.info(LOG_MSG_GENERATING_IMAGE)
    return time.time()

def _log_progress_if_needed(current_time, start_time, last_log_time_ref_list):
    if current_time - last_log_time_ref_list[0] >= 10:
        elapsed = current_time - start_time
        logger.info(LOG_MSG_GENERATION_PROGRESS.format(elapsed))
        last_log_time_ref_list[0] = current_time

def _progress_tracker_logic():
    start_time = time.time()
    last_log_time_mutable = [start_time]
    while not interrupt_requested and threading.current_thread().is_alive():
        current_time = time.time()
        _log_progress_if_needed(current_time, start_time, last_log_time_mutable)
        time.sleep(1)

def start_progress_tracker_daemon_thread():
    tracker = threading.Thread(target=_progress_tracker_logic)
    tracker.daemon = True
    tracker.start()

def _decode_and_save_partial_image(pipe_instance, capture_state, current_step):
    with torch.no_grad():
        decoded = pipe_instance.decode_latents(pipe_instance.latents)
        partial_img = pipe_instance.image_processor.postprocess(decoded)[0]
        capture_state['partial_image'] = partial_img
        capture_state['last_checkpoint'] = current_step
        logger.info(LOG_MSG_INTERMEDIATE_RESULT_SAVED_STEP.format(current_step))

def _capture_intermediate_step_helper(pipe_instance, capture_state):
    has_tools = hasattr(pipe_instance, 'decode_latents') and hasattr(pipe_instance, 'latents')
    if not has_tools:
        return

    try:
        current_step = getattr(pipe_instance, '_current_step', 0)
        needs_capture = current_step > capture_state['last_checkpoint'] + CHECKPOINT_STEPS
        if needs_capture:
            _decode_and_save_partial_image(pipe_instance, capture_state, current_step)
    except Exception as e:
        logger.debug(LOG_DEBUG_INTERMEDIATE_RESULT_FAIL.format(e))

def _save_partial_on_interrupt_in_forward_helper(capture_state):
    if interrupt_requested and capture_state['partial_image'] is not None:
        ts = int(time.time())
        fn = FILENAME_PARTIAL_IMAGE_PATTERN.format(f"capture_interrupt_{ts}")
        capture_state['partial_image'].save(fn)
        logger.info(LOG_MSG_INTERMEDIATE_RESULT_SAVED_PATH.format(os.path.abspath(fn)))

def _attempt_monkey_patch(pipe_instance, new_forward_func):
    try:
        pipe_instance.transformer.forward = new_forward_func
    except Exception as e:
        logger.debug(LOG_DEBUG_MONKEY_PATCH_FAIL.format(e))

def _restore_original_forward(pipe_instance, original_forward_func):
    try:
        pipe_instance.transformer.forward = original_forward_func
    except:
        pass

def execute_image_generation(pipe_instance, capture_state):
    original_forward = pipe_instance.transformer.forward
    
    def forward_capture_wrapper(*args, **kwargs):
        res = original_forward(*args, **kwargs)
        _capture_intermediate_step_helper(pipe_instance, capture_state)
        _save_partial_on_interrupt_in_forward_helper(capture_state)
        return res

    generated_image = None
    _attempt_monkey_patch(pipe_instance, forward_capture_wrapper)
    
    try:
        generated_image = pipe_instance(
            PROMPT_TEXT, height=IMG_HEIGHT, width=IMG_WIDTH,
            guidance_scale=GUIDANCE_SCALE, num_inference_steps=NUM_INFERENCE_STEPS,
            negative_prompt=NEGATIVE_PROMPT_TEXT, max_sequence_length=MAX_SEQUENCE_LENGTH
        ).images[0]
    finally:
        _restore_original_forward(pipe_instance, original_forward)
    
    return generated_image

def finalize_and_save_image(image_to_save, was_interrupted, start_time):
    status_msg = STR_STATUS_INTERRUPTED if was_interrupted else STR_STATUS_COMPLETED
    logger.info(LOG_MSG_IMAGE_GENERATION_STATUS_TIME.format(status_msg, time.time() - start_time))
    
    if image_to_save is not None:
        output_path = FILENAME_OUTPUT_IMAGE_PATTERN.format(int(time.time()))
        image_to_save.save(output_path)
        logger.info(LOG_MSG_IMAGE_SAVED_PATH.format(os.path.abspath(output_path)))
    else:
        logger.info(LOG_INFO_NO_IMAGE_EARLY_INTERRUPT)

def _save_partial_on_main_kb_interrupt(partial_image_obj):
    ts = int(time.time())
    fn = FILENAME_PARTIAL_IMAGE_PATTERN.format(f"main_kb_interrupt_{ts}")
    partial_image_obj.save(fn)
    logger.info(LOG_MSG_PARTIAL_RESULT_SAVED_PATH.format(os.path.abspath(fn)))
    return partial_image_obj

def _handle_kb_interrupt_in_generation_sub_block(capture_state_arg):
    logger.info(LOG_INFO_KB_INTERRUPT_GENERATION)
    if capture_state_arg['partial_image'] is not None:
        return _save_partial_on_main_kb_interrupt(capture_state_arg['partial_image'])
    else:
        logger.info(LOG_INFO_NO_PARTIAL_RESULT_KB)
        return None

def _run_generation_sub_block(pipe_instance_arg, capture_state_arg):
    try:
        return execute_image_generation(pipe_instance_arg, capture_state_arg)
    except KeyboardInterrupt:
        return _handle_kb_interrupt_in_generation_sub_block(capture_state_arg)

def _handle_outer_exception_in_main(e):
    if interrupt_requested:
        logger.info(LOG_INFO_EXCEPTION_POST_INTERRUPT)
    else:
        logger.error(f"Unhandled exception in main: {e}", exc_info=True)
        raise

def _main_generation_flow(pipe_instance_arg, gen_start_time_val):
    start_progress_tracker_daemon_thread()
    cap_state = {'partial_image': None, 'last_checkpoint': 0}
    final_img = _run_generation_sub_block(pipe_instance_arg, cap_state)
    finalize_and_save_image(final_img, interrupt_requested, gen_start_time_val)

def main(pipe_instance):
    gen_start_time = initialize_main_run_settings(pipe_instance)
    if gen_start_time is None:
        return

    try:
        _main_generation_flow(pipe_instance, gen_start_time)
    except KeyboardInterrupt:
        logger.info(LOG_INFO_KB_INTERRUPT_STOPPING)
    except Exception as e:
        _handle_outer_exception_in_main(e)
            
if __name__ == "__main__":
    loaded_pipe = load_pipeline()

    if loaded_pipe:
        main(loaded_pipe)
    else:
        logger.error(LOG_ERR_PIPELINE_LOAD_FAIL_EXIT)