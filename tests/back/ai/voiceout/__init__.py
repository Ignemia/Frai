"""
Backend Voice Output (Text-to-Speech) AI Module Tests

This module defines unit tests for the backend voice output functionality, covering both full-text vocalisation and progressive (streaming) text-to-speech.

Test categories:
1. Unit tests for individual functions and classes in voiceout components (e.g., TTS model wrappers, audio buffers).
2. Accuracy tests ensuring spoken output matches intended text content.
3. Consistency tests verifying repeat invocations produce stable audio characteristics.
4. Latency tests measuring generation delay for complete audio and for streaming segments.
5. Edge case tests for empty strings, long-form text, special characters, and markup.
6. Performance tests validating resource usage and throughput under load.

Because audio quality and intelligibility are subjective, most tests rely on manual review or comparison against reference recordings. Automated checks cover structural errors, output duration, and streaming behavior.

Test data specification:
- testset.csv must contain:
    id: Unique test identifier prefixed with vo-<id>
    name: Human-readable name for the test
    description: Concise explanation of the test objective
    input_mode: "full" or "stream"
    input: The input text to be synthesized
    tts_parameters: Parameters used for synthesis (e.g., voice, rate, pitch)
    expected_audio_path: Path to reference audio for manual comparison (optional)
    evaluation_method: "manual" or automated metric (e.g., audio duration match)
"""
