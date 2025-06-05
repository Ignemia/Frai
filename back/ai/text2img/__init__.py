"""
Backend Text to Image AI Module

Users can send text prompts to generate an image.
This module is about the text to image side of things while img2img module is about image to image generation.

__init__.py should contain class definition for the text2img handler with its global instance.
pipeline.py should contain the pipeline for the text2img handler.
prompt_handler.py should contain the prompt handler for the text2img handler.
other utility files might be needed.

For text to image generation we use Flux.1 model from models/FLUX.1-dev.

"""
import logging
import os
from typing import Any, Dict, List, Optional
import torch

from .pipeline import load_flux_text2img_pipeline, run_flux_text2img_pipeline, get_flux_text2img_pipeline_components
from .prompt_handler import format_text2img_prompt
from ..model_config import get_text2img_model_path, get_model_path

logger = logging.getLogger(__name__)

class Text2ImgAI:
    """
    AI text-to-image handler that coordinates model loading, generation, and device management.
    Uses FLUX.1 model.
    The model is loaded into RAM on initialization.
    For generation, it's moved to VRAM (if available) and then back to RAM.
    """
    
    def __init__(self, model_name: Optional[str] = None, model_path: Optional[str] = None):
        if model_name is None:
            self.model_path = get_text2img_model_path()
            self.model_name = "black-forest-labs/FLUX.1-dev"
        else:
            self.model_path, _ = get_model_path(model_name)
            self.model_name = model_name

        if model_path is not None:
            self.model_path = model_path
        self.pipeline = None
        self.model = None
        self.tokenizer = None
        self.scheduler = None
        self.is_loaded = False
        self.vram_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ram_device = "cpu"

        self.generation_params = {
            "guidance_scale": 7.0,
            "num_inference_steps": 25,
            "height": 1024,
            "width": 1024,
        }
        
        self._load_model_to_ram()

    def _load_model_to_ram(self):
        logger.info(f"Attempting to load FLUX.1 text2img model ({self.model_name}) to RAM from {self.model_path}.")
        try:
            self.pipeline = load_flux_text2img_pipeline(self.model_path, device=self.ram_device)
            if self.pipeline:
                self.is_loaded = True
                logger.info(f"FLUX.1 text2img pipeline ({self.model_name}) loaded to RAM successfully.")
                if hasattr(self.pipeline, 'enable_sequential_cpu_offload'):
                    logger.info(f"Enabling sequential CPU offloading for {self.model_name}.")
                    self.pipeline.enable_sequential_cpu_offload(device="cpu")
                elif hasattr(self.pipeline, 'components') and hasattr(self.pipeline.components, 'unet') and hasattr(self.pipeline.components.unet, 'enable_sequential_cpu_offload'):
                    logger.info(f"Enabling sequential CPU offloading on UNet for {self.model_name}.")
                    self.pipeline.components.unet.enable_sequential_cpu_offload(device="cpu")
                else:
                    logger.warning(f"Sequential CPU offloading not directly available on pipeline or UNet for {self.model_name}.")
            else:
                raise RuntimeError(f"FLUX.1 text2img pipeline loading returned None.")
        except Exception as e:
            self.is_loaded = False
            logger.error(f"Failed to load FLUX.1 text2img model {self.model_name} to RAM: {e}")
            raise RuntimeError(f"Could not load FLUX.1 text2img model {self.model_name} to RAM: {e}")

    def _ensure_model_on_device(self, target_device: str) -> bool:
        if not self.is_loaded or self.pipeline is None:
            logger.error("FLUX.1 text2img pipeline not loaded, cannot move device.")
            return False
        try:
            if self.pipeline.device.type != target_device:
                 logger.info(f"Moving FLUX.1 text2img pipeline from {self.pipeline.device} to {target_device}.")
                 self.pipeline.to(torch.device(target_device))
                 logger.info(f"FLUX.1 text2img pipeline moved to {target_device}.")
            return True
        except Exception as e:
            logger.error(f"Failed to move FLUX.1 text2img pipeline to {target_device}: {e}")
            return False

    def generate_image_from_text(
        self,
        text_prompt: str,
        negative_prompt: Optional[str] = None,
        generation_params_override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generates an AI image based on a text prompt using FLUX.1.
        Manages device VRAM/RAM transfers for the model.
        
        Args:
            text_prompt: The main text prompt for generation.
            negative_prompt: Optional negative text prompt.
            generation_params_override: Optional override for generation parameters.

        Returns:
            Dictionary with generation results, including "image", "success", "error", "metadata".
        """
        if not self.is_loaded or self.pipeline is None:
            return {"success": False, "error": "FLUX.1 text2img model not loaded", "image": None, "metadata": {}}

        target_device_for_generation = self.vram_device

        prompt_package = format_text2img_prompt(
            text_prompt=text_prompt,
            negative_prompt=negative_prompt
        )

        try:
            if not self._ensure_model_on_device(target_device_for_generation):
                raise RuntimeError(f"Failed to move FLUX.1 text2img model to {target_device_for_generation}.")
            
            active_pipeline = get_flux_text2img_pipeline_components(self.pipeline, target_device_for_generation)
            if not active_pipeline:
                 raise RuntimeError(f"Failed to get active pipeline components on {target_device_for_generation}")

            logger.info(f"Generating image from text on {target_device_for_generation}...")
            
            current_gen_params = self.generation_params.copy()
            if generation_params_override:
                current_gen_params.update(generation_params_override)
            
            current_gen_params['prompt'] = prompt_package['prompt']
            if prompt_package.get('negative_prompt'):
                 current_gen_params['negative_prompt'] = prompt_package['negative_prompt']

            gen_result = run_flux_text2img_pipeline(
                pipeline=active_pipeline,
                generation_params=current_gen_params,
            )
            if gen_result.get("success"):
                 gen_result["metadata"] = gen_result.get("metadata", {})
                 gen_result["metadata"]["model_name"] = self.model_name
                 gen_result["metadata"]["device_used"] = target_device_for_generation
            return gen_result
        
        except Exception as e:
            logger.error(f"Error during text-to-image generation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e), "image": None, "metadata": {}}
        finally:
            logger.info(f"Attempting to move FLUX.1 text2img model back to {self.ram_device}.")
            if self.pipeline is not None:
                if not self._ensure_model_on_device(self.ram_device):
                    logger.error(f"CRITICAL: Failed to move FLUX.1 text2img model back to {self.ram_device}.")
                else:
                    logger.info(f"FLUX.1 text2img model successfully moved back to {self.ram_device}.")
            if target_device_for_generation == "cuda":
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

_text2img_ai_instance: Optional[Text2ImgAI] = None

def get_text2img_ai_instance(model_name: Optional[str] = None, model_path: Optional[str] = None) -> Text2ImgAI:
    global _text2img_ai_instance

    if _text2img_ai_instance is None:
        logger.info("Initializing global Text2ImgAI instance.")
        _text2img_ai_instance = Text2ImgAI(
            model_name=model_name,
            model_path=model_path
        )
    elif model_name and (_text2img_ai_instance.model_name != model_name or \
                        (model_path and _text2img_ai_instance.model_path != model_path)):
        logger.warning(
            f"Requesting Text2ImgAI with new model/path {model_name}/{model_path}, but instance with "
            f"{_text2img_ai_instance.model_name}/{_text2img_ai_instance.model_path} exists. Re-initializing."
        )
        _text2img_ai_instance = Text2ImgAI(model_name=model_name, model_path=model_path)
    return _text2img_ai_instance

def initialize_text2img_system(model_name: Optional[str] = None, model_path: Optional[str] = None) -> bool:
    try:
        logger.info("Initializing text-to-image system...")
        text2img_ai = get_text2img_ai_instance(model_name, model_path)
        if text2img_ai.is_loaded:
            logger.info("Text-to-image system initialized successfully. FLUX.1 Model (simulated) is in RAM.")
            return True
        else:
            logger.error("Text-to-image system initialization failed. FLUX.1 Model (simulated) could not be loaded.")
            return False
    except Exception as e:
        logger.error(f"Error during text-to-image system initialization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def generate_image_from_text_api(
    text_prompt: str,
    negative_prompt: Optional[str] = None,
    generation_params_override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    High-level API function to generate an image from text.
    """
    text2img_ai = get_text2img_ai_instance()
    if not text2img_ai.is_loaded:
         return {"success": False, "error": "Text2Img model not ready or failed to load.", "image": None, "metadata": {}}
    return text2img_ai.generate_image_from_text(
        text_prompt=text_prompt,
        negative_prompt=negative_prompt,
        generation_params_override=generation_params_override
    )


# Legacy helper names used in the tests
def initialize_txt2img_system(model_name: Optional[str] = None, model_path: Optional[str] = None) -> bool:
    return initialize_text2img_system(model_name=model_name, model_path=model_path)


def get_txt2img_ai_instance(model_name: Optional[str] = None, model_path: Optional[str] = None) -> Text2ImgAI:
    return get_text2img_ai_instance(model_name=model_name, model_path=model_path)


def generate_image_from_text(
    prompt: str,
    negative_prompt: Optional[str] = None,
    **generation_kwargs: Any,
) -> Dict[str, Any]:
    text2img_ai = get_txt2img_ai_instance()
    if not text2img_ai.is_loaded:
        return {
            "success": False,
            "error": "Text2Img model not ready or failed to load.",
            "generated_image": None,
            "metadata": {},
        }

    if not prompt:
        return {"success": False, "error": "Prompt is empty", "generated_image": None, "metadata": {}}

    result = text2img_ai.generate_image_from_text(
        text_prompt=prompt,
        negative_prompt=negative_prompt,
        generation_params_override=generation_kwargs or None,
    )

    if result.get("success"):
        if "generated_image" not in result:
            result["generated_image"] = result.pop("image", None)
        if generation_kwargs:
            result["parameters_used"] = generation_kwargs

    return result

