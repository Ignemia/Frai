"""
Backend Voice Input AI Module

Users can activate voice input which will then be handled by this module.
Voice input will be transcribed and then the text which will be returned from function in real time for further processing by other modules.

__init__.py should contain class definition for the voicein handler with its global instance.
pipeline.py should contain the pipeline for the voicein handler.
data_processor.py or audio_handler.py might be more suitable than prompt_handler.py for audio data.

For voice input we use pyannote-segmentation-3.0 model from models/pyannote-segmentation-3.0.
And for voice activity detection we use speaker-diarization-3.1 model from models/speaker-diarization-3.1.
(Also, a speech-to-text model like Whisper will be needed for the actual transcription)
"""

import logging
import os
from typing import Dict, Optional, Any, Callable, List
import torch
import asyncio # For managing real-time aspects

# These will need to be implemented with actual model loading and processing logic
from .pipeline import (
    load_vad_model, 
    load_segmentation_model, 
    load_transcription_model, # Assuming a transcription model like Whisper
    process_audio_chunk_for_transcription
)
from .audio_processor import manage_audio_stream # Renamed from prompt_handler for clarity

logger = logging.getLogger(__name__)

class VoiceInAI:
    """
    AI Voice Input handler for real-time (or chunk-based) speech-to-text.
    Manages VAD, segmentation, and transcription models.
    """
    
    def __init__(
        self, 
        vad_model_path: str = "models/speaker-diarization-3.1",
        segmentation_model_path: str = "models/pyannote-segmentation-3.0",
        transcription_model_name: str = "openai/whisper-base" # Example, adjust as needed
    ):
        self.vad_model_path = vad_model_path
        self.segmentation_model_path = segmentation_model_path
        self.transcription_model_name = transcription_model_name # Can be a path or HF identifier
        
        self.vad_model = None
        self.segmentation_model = None
        self.transcription_pipeline = None # Or model + processor for Whisper
        
        self.is_loaded = False
        self.vram_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ram_device = "cpu"
        
        self.active_transcription_task = None
        self._load_models_to_ram()

    def _load_models_to_ram(self):
        logger.info("Attempting to load Voice-In models to RAM.")
        try:
            logger.info(f"Loading VAD model from: {self.vad_model_path}")
            self.vad_model = load_vad_model(self.vad_model_path, device=self.ram_device)
            
            logger.info(f"Loading Segmentation model from: {self.segmentation_model_path}")
            self.segmentation_model = load_segmentation_model(self.segmentation_model_path, device=self.ram_device)
            
            logger.info(f"Loading Transcription model: {self.transcription_model_name}")
            # Transcription model (e.g., Whisper) might be a pipeline or model+processor
            self.transcription_pipeline = load_transcription_model(self.transcription_model_name, device=self.ram_device)
            
            if self.vad_model and self.segmentation_model and self.transcription_pipeline:
                self.is_loaded = True
                logger.info("All Voice-In models (VAD, Segmentation, Transcription) loaded to RAM successfully.")
                # Sequential offloading might be applicable to the largest model, e.g., transcription
                # if hasattr(self.transcription_pipeline, 'model') and hasattr(self.transcription_pipeline.model, 'enable_sequential_cpu_offload'):
                #    self.transcription_pipeline.model.enable_sequential_cpu_offload()
                # elif hasattr(self.transcription_pipeline, 'enable_sequential_cpu_offload'): # if pipeline itself has it
                #    self.transcription_pipeline.enable_sequential_cpu_offload()

            else:
                missing = []
                if not self.vad_model: missing.append("VAD")
                if not self.segmentation_model: missing.append("Segmentation")
                if not self.transcription_pipeline: missing.append("Transcription")
                raise RuntimeError(f"Failed to load Voice-In models: {', '.join(missing)} missing.")

        except Exception as e:
            self.is_loaded = False
            logger.error(f"Failed to load one or more Voice-In models: {e}")
            raise RuntimeError(f"Could not load Voice-In models: {e}")

    def _ensure_models_on_device(self, target_device: str) -> bool:
        if not self.is_loaded:
            logger.error("Voice-In models not loaded, cannot move device.")
            return False
        try:
            all_moved = True
            models_to_move = {
                "VAD": self.vad_model,
                "Segmentation": self.segmentation_model,
                "Transcription": self.transcription_pipeline # This might be a pipeline or model
            }
            for name, model_obj in models_to_move.items():
                if model_obj is None: 
                    logger.warning(f"{name} model is None, skipping device move.")
                    continue
                
                current_device = None
                # Handling for HuggingFace pipelines vs raw models
                if hasattr(model_obj, 'device') and isinstance(model_obj.device, torch.device): # Typical for HF Pipelines
                    current_device = model_obj.device.type
                elif hasattr(model_obj, 'device'): # Some models store device directly
                     current_device = model_obj.device
                elif hasattr(model_obj, 'model') and hasattr(model_obj.model, 'device'): # Pipeline with model attribute
                    current_device = model_obj.model.device.type
                
                if current_device != target_device:
                    logger.info(f"Moving {name} model to {target_device}.")
                    if hasattr(model_obj, 'to'):
                        model_obj.to(torch.device(target_device))
                    elif hasattr(model_obj, 'model') and hasattr(model_obj.model, 'to'): # For pipeline objects
                        model_obj.model.to(torch.device(target_device))
                        if hasattr(model_obj, 'device'):
                             model_obj.device = torch.device(target_device) # Update pipeline's own device attr
                    else:
                        logger.warning(f"{name} model does not have a .to() method directly or on .model sub-object.")
                        # all_moved = False # Decide if this is critical
                logger.info(f"{name} model is on {target_device}.")
            return all_moved
        except Exception as e:
            logger.error(f"Failed to move one or more Voice-In models to {target_device}: {e}")
            return False

    async def start_realtime_transcription(
        self,
        audio_input_stream: Any, # This would be an audio stream source (e.g., PyAudio stream, file stream)
        transcript_callback: Callable[[Dict[str, Any]], None], # Callback for real-time results
        chunk_duration_ms: int = 1000 # Process audio in 1-second chunks for pseudo-real-time
    ):
        """
        Starts a real-time (chunk-based) transcription process.
        Manages device VRAM/RAM transfers for the models during the session.

        Args:
            audio_input_stream: The source of audio data.
            transcript_callback: A function to call with transcription results 
                                 (e.g., {"text": "hello world", "speaker_id": "A", "timestamp": 123.45, "is_final": False}).
            chunk_duration_ms: Duration of audio chunks to process at a time.
        """
        if not self.is_loaded:
            transcript_callback({"success": False, "error": "Voice-In models not loaded", "text": ""})
            return

        if self.active_transcription_task:
            logger.warning("Transcription task already active. Stop it before starting a new one.")
            transcript_callback({"success": False, "error": "Transcription already active", "text": ""})
            return
        
        target_device_for_processing = self.vram_device
        if not self._ensure_models_on_device(target_device_for_processing):
            transcript_callback({"success": False, "error": f"Failed to move models to {target_device_for_processing}", "text": ""})
            return

        logger.info(f"Starting real-time transcription on {target_device_for_processing}...")

        # The manage_audio_stream function from audio_processor.py will handle the loop
        self.active_transcription_task = asyncio.create_task(
            manage_audio_stream(
                audio_input_stream=audio_input_stream,
                vad_model=self.vad_model,
                segmentation_model=self.segmentation_model,
                transcription_pipeline=self.transcription_pipeline,
                transcript_callback=transcript_callback,
                chunk_duration_ms=chunk_duration_ms,
                device=target_device_for_processing
            )
        )
        try:
            await self.active_transcription_task
        except asyncio.CancelledError:
            logger.info("Transcription task cancelled.")
        except Exception as e:
            logger.error(f"Error during real-time transcription task: {e}")
            import traceback
            logger.error(traceback.format_exc())
            transcript_callback({"success": False, "error": str(e), "text": ""})
        finally:
            self.active_transcription_task = None
            logger.info(f"Attempting to move Voice-In models back to {self.ram_device}.")
            if not self._ensure_models_on_device(self.ram_device):
                logger.error(f"CRITICAL: Failed to move Voice-In models back to {self.ram_device}.")
            else:
                logger.info("Voice-In models successfully moved back to RAM.")
            if target_device_for_processing == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    async def stop_realtime_transcription(self):
        if self.active_transcription_task:
            logger.info("Stopping active transcription task...")
            self.active_transcription_task.cancel()
            try:
                await self.active_transcription_task # Wait for cleanup in the task
            except asyncio.CancelledError:
                logger.info("Transcription task successfully cancelled and cleaned up.")
            self.active_transcription_task = None
        else:
            logger.info("No active transcription task to stop.")

# --- Global Instance and Accessor Functions ---
_voicein_ai_instance: Optional[VoiceInAI] = None

def get_voicein_ai_instance(
    vad_model_path: Optional[str] = None,
    segmentation_model_path: Optional[str] = None,
    transcription_model_name: Optional[str] = None
) -> VoiceInAI:
    global _voicein_ai_instance
    
    # Define defaults here based on your __init__.py description
    default_vad_path = "models/speaker-diarization-3.1"
    default_segmentation_path = "models/pyannote-segmentation-3.0"
    default_transcription_model = "openai/whisper-base" # Default if not specified

    if _voicein_ai_instance is None:
        logger.info("Initializing global VoiceInAI instance.")
        _voicein_ai_instance = VoiceInAI(
            vad_model_path=vad_model_path if vad_model_path else default_vad_path,
            segmentation_model_path=segmentation_model_path if segmentation_model_path else default_segmentation_path,
            transcription_model_name=transcription_model_name if transcription_model_name else default_transcription_model
        )
    # Simplified re-initialization logic for brevity. 
    # A more robust check would compare all model paths/names.
    elif vad_model_path or segmentation_model_path or transcription_model_name:
        current_paths = (
            _voicein_ai_instance.vad_model_path,
            _voicein_ai_instance.segmentation_model_path,
            _voicein_ai_instance.transcription_model_name
        )
        requested_paths = (
            vad_model_path if vad_model_path else _voicein_ai_instance.vad_model_path,
            segmentation_model_path if segmentation_model_path else _voicein_ai_instance.segmentation_model_path,
            transcription_model_name if transcription_model_name else _voicein_ai_instance.transcription_model_name
        )
        if current_paths != requested_paths:
            logger.warning(f"Requesting VoiceInAI with new model configuration. Re-initializing.")
            _voicein_ai_instance = VoiceInAI(
                vad_model_path=requested_paths[0],
                segmentation_model_path=requested_paths[1],
                transcription_model_name=requested_paths[2]
            )
    return _voicein_ai_instance

def initialize_voicein_system(
    vad_model_path: Optional[str] = None,
    segmentation_model_path: Optional[str] = None,
    transcription_model_name: Optional[str] = None
) -> bool:
    try:
        logger.info("Initializing Voice-In system...")
        voicein_ai = get_voicein_ai_instance(vad_model_path, segmentation_model_path, transcription_model_name)
        if voicein_ai.is_loaded:
            logger.info("Voice-In system initialized successfully. Models (simulated) are in RAM.")
            return True
        else:
            logger.error("Voice-In system initialization failed. Models (simulated) could not be loaded.")
            return False
    except Exception as e:
        logger.error(f"Error during Voice-In system initialization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# Example API function (more sophisticated API needed for streaming)
async def start_voice_transcription_session(
    audio_stream: Any, 
    callback: Callable[[Dict[str, Any]], None]
) -> None:
    voicein_ai = get_voicein_ai_instance()
    if not voicein_ai.is_loaded:
        callback({"success": False, "error": "Voice-In models not ready.", "text": ""})
        return
    await voicein_ai.start_realtime_transcription(audio_stream, callback)

async def stop_voice_transcription_session() -> None:
    voicein_ai = get_voicein_ai_instance()
    await voicein_ai.stop_realtime_transcription()