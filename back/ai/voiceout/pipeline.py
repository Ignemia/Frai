"""
Text-to-Speech (TTS) Pipeline for Voice Output

This module contains functions to load the TTS model (nari-labs/dia-1.6B)
and generate speech audio from input text.
"""

import logging
import torch
from typing import Any, Dict, Optional
import numpy as np
import io
import soundfile as sf # For robust audio saving

# Import Dia model
from dia.model import Dia

logger = logging.getLogger(__name__)

# --- Model Loading Function ---
def load_tts_model(model_identifier: str = "nari-labs/Dia-1.6B", device: str = "cpu") -> Optional[Dia]:
    """
    Loads the Dia Text-to-Speech (TTS) model (nari-labs/Dia-1.6B).

    Args:
        model_identifier: Hugging Face identifier for the Dia TTS model.
        device: The device to load the model on ("cpu" or "cuda").

    Returns:
        The loaded Dia TTS model object, or None if loading fails.
    """
    logger.info(f"Attempting to load Dia TTS model: {model_identifier} on device: {device}")
    try:
        compute_dtype = torch.float16 if device == "cuda" else torch.float32
        logger.info(f"Using compute_dtype: {compute_dtype}")
        
        # Try loading without compute_dtype parameter first
        try:
            model = Dia.from_pretrained(model_identifier)
        except Exception as e:
            logger.warning(f"Failed to load without compute_dtype, trying fallback to HuggingFace: {e}")
            # Fallback to HuggingFace model ID if local path fails
            model = Dia.from_pretrained("nari-labs/Dia-1.6B")
        
        # Dia model may not support .to() method like standard PyTorch models
        # Try to move to device if the method exists
        try:
            if hasattr(model, 'to'):
                model.to(torch.device(device))
            else:
                logger.info(f"Dia model does not have .to() method, device handling may be managed internally")
        except Exception as e:
            logger.warning(f"Could not move Dia model to {device}: {e}")
        
        # The Dia model might already set its internal device, but explicit .to() is safer.
        # Check if the model has a device attribute and update it if necessary for consistency.
        if hasattr(model, 'device'):
            model.device = torch.device(device) 
        else: # Dia class might store device info differently, e.g. on submodules
            logger.info(f"Dia model object does not have a direct 'device' attribute. Device management may be internal.")

        logger.info(f"Dia TTS model {model_identifier} loaded successfully on {device}.")
        return model

    except Exception as e:
        logger.error(f"Error loading Dia TTS model {model_identifier}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# --- Speech Generation Function ---
def generate_speech_audio(
    tts_model: Dia, 
    text: str,
    generation_params: Dict[str, Any],
    output_format: str = "wav",
    device: str = "cpu" # device is mainly for model loading, generation uses model's current device
) -> Dict[str, Any]:
    """
    Generates speech audio from text using the loaded Dia TTS model.

    Args:
        tts_model: The loaded Dia TTS model object.
        text: The input text to synthesize.
        generation_params: Dictionary of parameters for TTS generation (e.g., verbose).
        output_format: Desired audio format ("wav", "pcm_s16le"). MP3 not directly supported, will fallback.
        device: The device the model is on (primarily for logging/consistency here).

    Returns:
        A dictionary: {"success": bool, "audio_data": bytes, "format": str, 
                       "sample_rate": int, "error": str or None}.
    """
    logger.info(f"Generating speech for text: '{text[:50]}...' Format: {output_format} using Dia model on its device.")

    if not tts_model:
        logger.error("Dia TTS model is not available for speech generation.")
        return {"success": False, "audio_data": None, "format": output_format, "sample_rate": 0, "error": "TTS model not loaded"}
    
    # Ensure model is on the correct device (should already be, but as a safeguard or if moved)
    # This check is more for sanity; actual move happens in _ensure_model_on_device in __init__.py
    # For now, we assume the tts_model object passed here is already on the correct target_device for generation.
    # logger.info(f"Confirming model device: {tts_model.device if hasattr(tts_model, 'device') else 'unknown'}")

    try:
        # Parameters for Dia model's generate method
        use_torch_compile = generation_params.get("use_torch_compile", False) # False for MacOS compatibility
        verbose_generation = generation_params.get("verbose", False)
        
        # Extract voice parameters for non-binary voice
        pitch = generation_params.get("pitch", 0.0)  # Neutral pitch
        energy = generation_params.get("energy", 1.0)  # Normal energy
        style = generation_params.get("style", "neutral")  # Neutral style
        voice_style = generation_params.get("voice_style", "androgynous")  # Target voice style
        
        # Combine parameters for generation
        generation_kwargs = {
            "use_torch_compile": use_torch_compile,
            "verbose": verbose_generation,
            "pitch": pitch,
            "energy": energy,
            "style": style,
            "voice_style": voice_style
        }
        
        logger.debug(f"Calling Dia model.generate with text: '{text}' and voice parameters: {generation_kwargs}")
        generation_output = tts_model.generate(text, **generation_kwargs)
        
        audio_tensor = generation_output.get("audio_data")
        if audio_tensor is None:
            raise ValueError("Dia model did not return 'audio_data' in its output.")

        # Ensure audio_tensor is a tensor and on CPU for numpy conversion
        if not isinstance(audio_tensor, torch.Tensor):
            # Dia model generate() doc implies it can return other things for different parts of pipeline
            # but for end-to-end, audio_data should be the tensor.
            raise ValueError(f"Expected 'audio_data' to be a torch.Tensor, got {type(audio_tensor)}")

        audio_waveform_cpu = audio_tensor.squeeze().cpu()
        
        # Get sample rate from the model
        if not hasattr(tts_model, 'sample_rate'):
            raise AttributeError("Dia model object does not have a 'sample_rate' attribute.")
        sample_rate = tts_model.sample_rate

        logger.info(f"Speech generated by Dia model. Sample rate: {sample_rate}. Waveform shape: {audio_waveform_cpu.shape}")

        audio_data_bytes = None
        actual_output_format = output_format.lower()

        if actual_output_format == "wav":
            # Ensure waveform is float32 for soundfile, or soundfile will pick a subtype based on dtype
            audio_np_float32 = audio_waveform_cpu.float().numpy()
            with io.BytesIO() as wav_io:
                sf.write(wav_io, audio_np_float32, sample_rate, format='WAV', subtype='PCM_16')
                audio_data_bytes = wav_io.getvalue()
            logger.info(f"WAV audio generated. Bytes: {len(audio_data_bytes)}")
        elif actual_output_format == "pcm_s16le": # Raw PCM signed 16-bit little-endian
            audio_np_float32 = audio_waveform_cpu.float().numpy()
            # Scale to 16-bit integer range and convert type
            audio_np_int16 = (audio_np_float32 * 32767).astype(np.int16)
            audio_data_bytes = audio_np_int16.tobytes()
            logger.info(f"PCM S16LE audio generated. Bytes: {len(audio_data_bytes)}")
        elif actual_output_format == "mp3":
            logger.warning("MP3 output format is not directly supported by this pipeline. Consider saving as WAV and converting externally.")
            # For now, fallback to WAV for MP3 request with a warning in the result
            audio_np_float32 = audio_waveform_cpu.float().numpy()
            with io.BytesIO() as wav_io:
                sf.write(wav_io, audio_np_float32, sample_rate, format='WAV', subtype='PCM_16')
                audio_data_bytes = wav_io.getvalue()
            actual_output_format = "wav" # Reflect fallback
            return {"success": True, "audio_data": audio_data_bytes, "format": actual_output_format, 
                    "sample_rate": sample_rate, "error": "MP3 not supported, returned WAV instead."}
        else:
            return {"success": False, "audio_data": None, "format": output_format, 
                    "sample_rate": 0, "error": f"Unsupported output format: {output_format}"}

        return {"success": True, "audio_data": audio_data_bytes, "format": actual_output_format, 
                "sample_rate": sample_rate, "error": None}

    except Exception as e:
        logger.error(f"Error during Dia TTS speech generation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "audio_data": None, "format": output_format, 
                "sample_rate": 0, "error": str(e)} 