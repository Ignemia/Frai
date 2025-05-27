"""
Backend Voice Input (Speech-to-Text) AI Module Tests

This module defines unit tests for the backend voice input functionality, focusing on transcription accuracy and robustness across diverse audio recordings.

Test categories:
1. Unit tests for individual voicein components (e.g., audio loaders, pre-processors, model wrappers).
2. Accuracy tests comparing model output against full transcripts for recordings from 1 second to 25 minutes.
3. Robustness tests covering various audio qualities (e.g., background noise, clipping, low bitrate).
4. Consistency tests ensuring minimal variance in repeated transcriptions (e.g., punctuation, minor pronunciation tweaks).
5. Edge case tests for silence, foreign languages, non-speech sounds, overlapping speakers.
6. Performance tests validating latency and resource usage for streaming and batch modes.

Because actual spoken content is known and transcripts are authoritative, tests assert strict transcript matching, with tolerance for minor punctuation or capitalization differences only when explicitly configured.

Test data specification:
- testset.csv must contain:
    id: Unique test identifier prefixed with vi-<id>
    name: Human-readable name for the test
    description: Brief explanation of test objective
    audio_group: Category of audio (e.g., short_clip, lecture, phone_call)
    expected_transcript_path: Path to full reference transcript file
    allowed_variance: Description of tolerated minor differences (optional)
    evaluation_metric: Metric or threshold (e.g., word error rate <= 0.05)

- inputs.csv must contain:
    id: Unique audio sample identifier
    group: Category of the audio file (e.g., noisy, clean, accented)
    description: Context or scenario for the recording
    audio_path: Path to the test audio file within the project
    transcript_path: Path to the corresponding reference transcript
    language: Language or locale of the speaker(s)
"""
