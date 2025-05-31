"""
Audio Stream Processor for Voice Input

This module handles the real-time (chunk-based) processing of an audio stream.
It reads audio chunks, passes them through VAD, segmentation (diarization),
and transcription models, and uses a callback to return results.
"""

import logging
import asyncio
import time
import numpy as np # For audio data manipulation
from typing import Any, Callable, Dict, List

# from .pipeline import process_audio_chunk_for_transcription # Already imported in __init__
# We'll need to reference the function from pipeline.py, but it's passed as an arg to manage_audio_stream

logger = logging.getLogger(__name__)

async def manage_audio_stream(
    audio_input_stream: Any, # Represents the source of audio data (e.g., PyAudio stream object)
    vad_model: Any,
    segmentation_model: Any,
    transcription_pipeline: Any,
    transcript_callback: Callable[[Dict[str, Any]], None],
    chunk_duration_ms: int = 1000, # Process 1-second chunks by default
    sample_rate: int = 16000, # Standard sample rate for many STT models
    channels: int = 1, # Mono audio
    sample_width: int = 2, # Bytes per sample (e.g., 2 for 16-bit audio)
    device: str = "cpu"
):
    """
    Manages the audio input stream, processes chunks, and sends transcriptions via callback.
    This is a placeholder for a real-time audio processing loop.

    Args:
        audio_input_stream: The audio stream object (must have a read-like method).
        vad_model: Loaded VAD model.
        segmentation_model: Loaded segmentation model.
        transcription_pipeline: Loaded transcription pipeline.
        transcript_callback: Callback function for transcription results.
        chunk_duration_ms: Duration of audio chunks in milliseconds.
        sample_rate: Audio sample rate in Hz.
        channels: Number of audio channels.
        sample_width: Bytes per audio sample.
        device: Device to run models on ("cpu" or "cuda").
    """
    logger.info(f"Starting to manage audio stream. Chunk duration: {chunk_duration_ms}ms, Device: {device}")
    
    bytes_per_chunk = (sample_rate * channels * sample_width * chunk_duration_ms) // 1000
    
    # Import the processing function dynamically to avoid circular dependency if this file is imported elsewhere
    # This assumes that voicein.pipeline.process_audio_chunk_for_transcription is available
    # A better approach is to pass this function directly as an argument if possible, which is done in __init__.
    # For now, let's assume it's globally accessible or passed via __init__.
    # from .pipeline import process_audio_chunk_for_transcription # Not needed if passed via VoiceInAI class
    
    # This function is now expected to be available via the VoiceInAI class instance or passed directly.
    # For direct use here, it would need to be imported, but that could lead to circular deps.
    # The call from __init__ passes the models directly to process_audio_chunk_for_transcription.
    # This manage_audio_stream itself is called by __init__ which has access to pipeline.py's functions.

    try:
        while True: # Loop to continuously read and process audio
            audio_data_bytes = None
            try:
                # This is a placeholder for reading from an actual audio stream
                # For example, if audio_input_stream is a PyAudio stream:
                # audio_data_bytes = audio_input_stream.read(bytes_per_chunk, exception_on_overflow=False)
                
                # SIMULATION of reading audio data:
                if hasattr(audio_input_stream, 'read'):
                    audio_data_bytes = audio_input_stream.read(bytes_per_chunk)
                    if not audio_data_bytes: # End of stream for file-like objects
                        logger.info("End of audio stream detected.")
                        transcript_callback({"event": "stream_end", "success": True})
                        break
                else:
                    # Simulate a continuous stream for demonstration if no read method
                    logger.debug("Simulating audio chunk read.")
                    await asyncio.sleep(chunk_duration_ms / 1000.0) # Simulate time to acquire audio
                    # Create some dummy audio data (e.g., sine wave or noise)
                    duration_sec = chunk_duration_ms / 1000.0
                    num_samples = int(sample_rate * duration_sec)
                    # Create a simple sine wave for simulation
                    time_axis = np.linspace(0, duration_sec, num_samples, endpoint=False)
                    frequency = 440 # A4 note
                    audio_data_np_float = 0.5 * np.sin(2 * np.pi * frequency * time_axis)
                    # Convert to 16-bit PCM
                    audio_data_np_int16 = (audio_data_np_float * 32767).astype(np.int16)
                    audio_data_bytes = audio_data_np_int16.tobytes()
                    if not audio_data_bytes: # Should not happen in simulation unless num_samples is 0
                        logger.warning("Simulated audio read produced no bytes.")
                        continue

            except IOError as e:
                logger.error(f"IOError reading audio stream: {e}. Assuming stream closed.")
                transcript_callback({"event": "stream_error", "error": str(e), "success": False})
                break
            except Exception as e:
                logger.error(f"Unexpected error reading audio stream: {e}")
                transcript_callback({"event": "stream_error", "error": str(e), "success": False})
                break # Stop on other errors too

            if not audio_data_bytes or len(audio_data_bytes) < bytes_per_chunk // 2: # Check for partial reads if stream is ending
                 # If significantly less than expected, might be end of stream or issue
                logger.info("Received very few bytes, possibly end of stream or an issue. Stopping.")
                break

            # Convert audio bytes to numpy array (float32 for most models)
            # This assumes 16-bit PCM input, common for microphones
            audio_data_np = np.frombuffer(audio_data_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            if audio_data_np.size == 0:
                logger.debug("Empty audio chunk after conversion, skipping.")
                continue

            start_time = time.time()
            
            # Call the processing function (passed via VoiceInAI or imported)
            # This is where the actual VAD, segmentation, and transcription happens.
            # The `process_audio_chunk_for_transcription` function is defined in pipeline.py
            # and called from VoiceInAI, which then calls this manage_audio_stream.
            # For clarity, the actual call is made in __init__.py to this function,
            # and this function needs to orchestrate using the models passed to it.
            
            # We need to use the models passed to *this* function.
            # Let's assume a function similar to `process_audio_chunk_for_transcription` but simplified for this context,
            # or we directly use the models here. For now, simulate the chained processing.

            logger.debug(f"Simulating VAD on chunk of shape {audio_data_np.shape}")
            vad_output = vad_model(audio_data_np) # Simulated call
            speech_segments = vad_output.get("speech_segments", [])

            if not speech_segments:
                logger.debug("No speech in chunk (simulated VAD).")
                # Send an empty transcript or a specific event if needed
                # transcript_callback({"text": "", "is_final": False, "timestamp": time.time()})
                await asyncio.sleep(0.1) # Small sleep to prevent tight loop if no speech
                continue
            
            logger.debug(f"Simulating Segmentation on chunk. VAD found: {speech_segments}")
            segmentation_output = segmentation_model(audio_data_np, vad_segments=vad_output) # Simulated call
            diarized_segments = segmentation_output.get("diarization", [])

            logger.debug(f"Simulating Transcription on chunk. Diarization: {diarized_segments}")
            # The transcription_pipeline simulation takes the numpy array
            transcription_results = transcription_pipeline(audio_data_np)
            text = transcription_results.get("text", "")
            chunks_from_stt = transcription_results.get("chunks", [])
            
            processing_time = time.time() - start_time
            logger.info(f"Audio chunk processed in {processing_time:.3f}s (Simulated pipeline)")

            # Format and send results via callback
            # This logic should ideally match the output of `process_audio_chunk_for_transcription`
            if chunks_from_stt:
                for chunk_info in chunks_from_stt:
                    speaker = chunk_info.get("speaker", "UNKNOWN")
                    for speaker_label, seg_start_time, seg_end_time in diarized_segments:
                        if chunk_info["timestamp"][0] >= seg_start_time and chunk_info["timestamp"][1] <= seg_end_time:
                            speaker = speaker_label
                            break
                    transcript_callback({
                        "text": chunk_info["text"],
                        "speaker_id": speaker,
                        "start_ts": chunk_info["timestamp"][0],
                        "end_ts": chunk_info["timestamp"][1],
                        "is_final": True, # Placeholder, real-time would have False for intermediate
                        "success": True
                    })
            elif text:
                 speaker = diarized_segments[0][0] if diarized_segments else "UNKNOWN"
                 start_ts = diarized_segments[0][1] if diarized_segments else 0
                 end_ts = diarized_segments[0][2] if diarized_segments else len(audio_data_np)/sample_rate
                 transcript_callback({
                     "text": text, 
                     "speaker_id": speaker,
                     "start_ts": start_ts,
                     "end_ts": end_ts,
                     "is_final": True, 
                     "success": True
                    })
            else:
                # No transcription but speech was detected, could be an empty event
                logger.debug("Speech detected but no transcription text (simulated).")
            
            # Yield control to allow other asyncio tasks to run
            await asyncio.sleep(0.01) # Minimal sleep to keep responsive

    except asyncio.CancelledError:
        logger.info("Audio stream management task cancelled.")
        transcript_callback({"event": "cancelled", "success": True})
        raise # Re-raise to be caught by VoiceInAI handler
    except Exception as e:
        logger.error(f"Error in audio stream management: {e}")
        import traceback
        logger.error(traceback.format_exc())
        transcript_callback({"event": "error", "error": str(e), "success": False})
    finally:
        logger.info("Audio stream management finished.")
        # Clean up audio_input_stream if necessary (e.g., stream.close())
        if hasattr(audio_input_stream, 'close') and callable(audio_input_stream.close):
            try:
                audio_input_stream.close()
                logger.info("Audio input stream closed.")
            except Exception as e_close:
                logger.error(f"Error closing audio input stream: {e_close}") 