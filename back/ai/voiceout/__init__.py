"""
Backend Voice Output AI Module

Users can activate voice output which will then be handled by this module.
Voice output will be generated from text that will be returned from other modules.

__init__.py should contain class definition for the voiceout handler with its global instance.
pipeline.py should contain the pipeline for the voiceout handler.
# prompt_handler.py might be less relevant here, more about input text formatting or SSML.
# data_processor.py or audio_generator.py could be alternatives.

For voice output we use nari-labs/dia-1.6B from models/dia.
"""

import logging
import os
from typing import Dict, Optional, Any, Union
import torch

# These will need to be implemented with actual model loading and TTS generation logic
from .pipeline import load_tts_model, generate_speech_audio
# from .text_formatter import format_text_for_tts # If complex SSML or text pre-processing is needed

logger = logging.getLogger(__name__)

class VoiceOutAI:
    """
    AI Voice Output (Text-to-Speech) handler.
    Manages the TTS model (e.g., nari-labs/dia-1.6B) and audio generation.
    """
    
    def __init__(self, model_name: str = "nari-labs/Dia-1.6B", model_path: str = "models/Dia-1.6B"):
        self.model_name = model_name # Hugging Face identifier or path
        self.model_path = model_path # Path if locally stored, can be same as model_name if HF id
        self.tts_model = None # This will hold the loaded TTS model or pipeline
        # self.vocoder = None # Some TTS models require a separate vocoder
        self.is_loaded = False
        self.vram_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ram_device = "cpu"

        self.generation_params = {
            # Parameters for creating a more non-binary voice
            "speaker_id": None,  # If model supports multiple speakers
            "emotion": None,     # If model supports emotional synthesis
            "speed": 1.0,        # Speaking rate
            "pitch": 0.0,        # Neutral pitch (0.0 is middle, negative is lower, positive is higher)
            "energy": 1.0,       # Speech energy/volume
            "style": "neutral",  # Speaking style
            "gender": None,      # Explicitly set to None to avoid binary gender assumptions
            "voice_style": "androgynous"  # Target voice style
        }

        self._load_model_to_ram()

    def _load_model_to_ram(self):
        logger.info(f"Attempting to load TTS model {self.model_name} to RAM from {self.model_path}.")
        try:
            # This will use the function from pipeline.py
            # It might return a tuple (tts_model, vocoder) if they are separate
            self.tts_model = load_tts_model(self.model_path, device=self.ram_device)
            # If a vocoder is returned and needed: self.tts_model, self.vocoder = load_tts_model(...)
            
            if self.tts_model:
                self.is_loaded = True
                logger.info(f"TTS model {self.model_name} loaded to RAM successfully.")
                # Enable sequential CPU offloading if applicable (especially for large TTS models)
                # This depends on the specific TTS library/model structure
                # if hasattr(self.tts_model, 'enable_sequential_cpu_offload'):
                #     logger.info(f"Enabling sequential CPU offloading for {self.model_name}.")
                #     self.tts_model.enable_sequential_cpu_offload(device="cpu")
            else:
                raise RuntimeError(f"TTS model loading returned None.")
        except Exception as e:
            self.is_loaded = False
            logger.error(f"Failed to load TTS model {self.model_name} to RAM: {e}")
            raise RuntimeError(f"Could not load TTS model {self.model_name} to RAM: {e}")

    def _ensure_model_on_device(self, target_device: str) -> bool:
        if not self.is_loaded or self.tts_model is None:
            logger.error("TTS model not loaded, cannot move device.")
            return False
        try:
            model_to_move = self.tts_model
            current_device_type = None

            if hasattr(model_to_move, 'device') and isinstance(model_to_move.device, torch.device):
                current_device_type = model_to_move.device.type
            elif hasattr(model_to_move, 'device'): # For models storing device as string e.g. Bark
                 current_device_type = str(model_to_move.device)
            # Add more checks if model stores device differently

            if current_device_type != target_device:
                logger.info(f"Moving TTS model from {current_device_type} to {target_device}.")
                if hasattr(model_to_move, 'to'):
                    model_to_move.to(torch.device(target_device))
                # Some TTS models might need specific device placement methods
                # e.g., Bark: generation.models.text_to_semantic.to(target_device), etc.
                # This needs to be adapted based on nari-labs/dia-1.6B's API.
                else:
                    logger.warning(f"TTS model {self.model_name} may not have a standard .to() method. Device transfer might be partial or incorrect.")
                # Update internal device attribute if the model has one
                if hasattr(model_to_move, 'device'):
                     model_to_move.device = torch.device(target_device) if not isinstance(model_to_move.device, str) else target_device
            
            logger.info(f"TTS model assumed to be on {target_device}.")
            # Also move vocoder if it's separate and on a device
            # if self.vocoder and hasattr(self.vocoder, 'to') and self.vocoder.device.type != target_device:
            #    self.vocoder.to(torch.device(target_device))
            return True
        except Exception as e:
            logger.error(f"Failed to move TTS model to {target_device}: {e}")
            return False

    def generate_speech(
        self,
        text: str,
        output_format: str = "wav", # e.g., wav, mp3, pcm
        generation_params_override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generates speech audio from the given text using the loaded TTS model.
        Manages device VRAM/RAM transfers for the model.
        
        Args:
            text: The text to synthesize.
            output_format: Desired audio output format.
            generation_params_override: Optional override for TTS generation parameters.

        Returns:
            Dictionary with results: {"success": bool, "audio_data": bytes or path, "format": str, "error": str, "metadata": dict}.
        """
        if not self.is_loaded or self.tts_model is None:
            return {"success": False, "error": "TTS model not loaded", "audio_data": None, "format": output_format, "metadata": {}}

        target_device_for_generation = self.vram_device
        
        # Text formatting/SSML handling would go here if using a text_formatter.py
        # formatted_text = format_text_for_tts(text, ...) 
        input_text = text

        current_gen_params = self.generation_params.copy()
        if generation_params_override:
            current_gen_params.update(generation_params_override)

        try:
            if not self._ensure_model_on_device(target_device_for_generation):
                raise RuntimeError(f"Failed to move TTS model to {target_device_for_generation}.")

            logger.info(f"Generating speech on {target_device_for_generation}...")
            
            # Use the function from pipeline.py
            # This function will handle the specifics of nari-labs/dia-1.6B
            audio_result = generate_speech_audio(
                tts_model=self.tts_model, 
                # vocoder=self.vocoder, # If separate
                text=input_text, 
                generation_params=current_gen_params,
                output_format=output_format,
                device=target_device_for_generation
            )
            # generate_speech_audio should return a dict like:
            # {"success": True, "audio_data": audio_bytes, "format": "wav", "sample_rate": sr, "error": None}
            
            if audio_result.get("success"):
                response = {
                    "success": True,
                    "audio_data": audio_result.get("audio_data"),
                    "format": audio_result.get("format", output_format),
                    "sample_rate": audio_result.get("sample_rate"),
                    "error": None,
                    "metadata": {"model_name": self.model_name, "device_used": target_device_for_generation}
                }
            else:
                response = {"success": False, "error": audio_result.get("error", "Unknown TTS error"), "audio_data": None, "format": output_format, "metadata": {}}
            
            return response
        
        except Exception as e:
            logger.error(f"Error during speech generation with device handling: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e), "audio_data": None, "format": output_format, "metadata": {}}
        finally:
            logger.info(f"Attempting to move TTS model back to {self.ram_device}.")
            if self.tts_model is not None:
                if not self._ensure_model_on_device(self.ram_device):
                    logger.error(f"CRITICAL: Failed to move TTS model back to {self.ram_device}.")
                else:
                    logger.info(f"TTS model successfully moved back to {self.ram_device}.")
            if target_device_for_generation == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

# --- Global Instance and Accessor Functions ---
_voiceout_ai_instance: Optional[VoiceOutAI] = None

def get_voiceout_ai_instance(model_name: Optional[str] = None, model_path: Optional[str] = None) -> VoiceOutAI:
    global _voiceout_ai_instance
    default_model_name = "nari-labs/Dia-1.6B"
    default_model_path = "models/Dia-1.6B" # As per user correction
    
    if _voiceout_ai_instance is None:
        logger.info("Initializing global VoiceOutAI instance.")
        _voiceout_ai_instance = VoiceOutAI(
            model_name=model_name if model_name else default_model_name,
            model_path=model_path if model_path else default_model_path
        )
    elif model_name and (_voiceout_ai_instance.model_name != model_name or \
                        (model_path and _voiceout_ai_instance.model_path != model_path)):
        logger.warning(
            f"Requesting VoiceOutAI with new model/path {model_name}/{model_path}, but instance with "
            f"{_voiceout_ai_instance.model_name}/{_voiceout_ai_instance.model_path} exists. Re-initializing."
        )
        _voiceout_ai_instance = VoiceOutAI(model_name=model_name, model_path=model_path)
    return _voiceout_ai_instance

def initialize_voiceout_system(model_name: Optional[str] = None, model_path: Optional[str] = None) -> bool:
    try:
        logger.info("Initializing Voice-Out (TTS) system...")
        voiceout_ai = get_voiceout_ai_instance(model_name, model_path)
        if voiceout_ai.is_loaded:
            logger.info("Voice-Out system initialized successfully. TTS Model (simulated) is in RAM.")
            return True
        else:
            logger.error("Voice-Out system initialization failed. TTS Model (simulated) could not be loaded.")
            return False
    except Exception as e:
        logger.error(f"Error during Voice-Out system initialization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# High-level interaction function
def generate_speech_from_text(
    text: str, 
    output_format: str = "wav",
    generation_params_override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    High-level function to generate speech audio from text.
    """
    voiceout_ai = get_voiceout_ai_instance()
    if not voiceout_ai.is_loaded:
         return {"success": False, "error": "TTS model not ready or failed to load.", "audio_data": None, "format": output_format}
    return voiceout_ai.generate_speech(
        text=text, 
        output_format=output_format, 
        generation_params_override=generation_params_override
    )