"""
Consistency tests for the voice output (text-to-speech) module.

This module tests that the TTS system produces consistent audio output
for identical inputs and maintains stable voice characteristics across sessions.
"""

import pytest
import logging
import csv
import os
import sys
from typing import Dict, Any, List
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from Frai.back.ai.voiceout import (
    initialize_voiceout_system,
    get_voiceout_ai_instance,
    synthesize_speech
)

# Set up logging
logger = logging.getLogger(__name__)

# Path to the test set CSV file
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), 'testset.csv')


def load_consistency_test_cases() -> List[Dict[str, str]]:
    """Load test cases specifically for consistency testing."""
    test_cases = []
    try:
        with open(TEST_SET_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter for consistency tests
                if 'consistency' in row['evaluation_method'].lower():
                    test_cases.append(row)
        logger.info(f"Loaded {len(test_cases)} consistency test cases")
        return test_cases
    except Exception as e:
        logger.error(f"Failed to load test cases: {e}")
        return []


def analyze_audio_consistency(audio_samples: List) -> Dict[str, float]:
    """Analyze consistency metrics between multiple audio samples."""
    if len(audio_samples) < 2:
        return {}
    
    consistency_metrics = {}
    
    try:
        # Convert all audio samples to comparable format
        audio_arrays = []
        durations = []
        
        for audio_data in audio_samples:
            if isinstance(audio_data, bytes):
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            elif hasattr(audio_data, 'read'):
                audio_array = np.frombuffer(audio_data.read(), dtype=np.int16)
            else:
                audio_array = np.array(audio_data)
            
            audio_arrays.append(audio_array)
            durations.append(len(audio_array) / 22050)  # Assume 22050 Hz
        
        # Duration consistency
        duration_variance = np.var(durations)
        mean_duration = np.mean(durations)
        consistency_metrics['duration_cv'] = np.sqrt(duration_variance) / mean_duration if mean_duration > 0 else float('inf')
        
        # Amplitude consistency
        rms_values = [np.sqrt(np.mean(arr**2)) for arr in audio_arrays]
        consistency_metrics['amplitude_cv'] = np.std(rms_values) / np.mean(rms_values) if np.mean(rms_values) > 0 else float('inf')
        
        # Zero crossing rate consistency
        zcr_values = []
        for arr in audio_arrays:
            zero_crossings = np.where(np.diff(np.signbit(arr)))[0]
            zcr = len(zero_crossings) / len(arr)
            zcr_values.append(zcr)
        
        consistency_metrics['zcr_cv'] = np.std(zcr_values) / np.mean(zcr_values) if np.mean(zcr_values) > 0 else float('inf')
        
        # Spectral consistency (simplified)
        spectral_features = []
        for arr in audio_arrays:
            # Simple spectral centroid approximation
            fft = np.fft.fft(arr[:min(len(arr), 8192)])  # Use first 8192 samples
            magnitude = np.abs(fft)
            spectral_centroid = np.sum(np.arange(len(magnitude)) * magnitude) / np.sum(magnitude)
            spectral_features.append(spectral_centroid)
        
        consistency_metrics['spectral_cv'] = np.std(spectral_features) / np.mean(spectral_features) if np.mean(spectral_features) > 0 else float('inf')
        
    except Exception as e:
        logger.warning(f"Failed to analyze audio consistency: {e}")
        consistency_metrics['analysis_error'] = 1.0
    
    return consistency_metrics


def calculate_audio_similarity(audio1, audio2) -> float:
    """Calculate similarity score between two audio samples."""
    try:
        # Convert to numpy arrays
        if isinstance(audio1, bytes):
            arr1 = np.frombuffer(audio1, dtype=np.int16)
        else:
            arr1 = np.array(audio1)
        
        if isinstance(audio2, bytes):
            arr2 = np.frombuffer(audio2, dtype=np.int16)
        else:
            arr2 = np.array(audio2)
        
        # Normalize lengths
        min_len = min(len(arr1), len(arr2))
        arr1 = arr1[:min_len]
        arr2 = arr2[:min_len]
        
        # Calculate cross-correlation
        if min_len > 0:
            correlation = np.corrcoef(arr1, arr2)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        else:
            return 0.0
            
    except Exception as e:
        logger.warning(f"Failed to calculate audio similarity: {e}")
        return 0.0


# Using fixture from conftest.py


class TestVoiceOutConsistency:
    """Test voice output synthesis consistency."""
    
    def test_identical_input_consistency(self, setup_voiceout_ai):
        """Test that identical inputs produce consistent outputs."""
        test_text = "This is a consistency test for identical input processing."
        
        # Generate multiple samples with identical parameters
        audio_samples = []
        for i in range(5):
            result = synthesize_speech(
                test_text,
                voice='default',
                rate=1.0,
                pitch=0.0,
                seed=42  # Fixed seed for consistency
            )
            
            if result.get('success', False):
                audio_samples.append(result['audio_data'])
        
        if len(audio_samples) >= 3:
            # Analyze consistency
            consistency_metrics = analyze_audio_consistency(audio_samples)
            
            # Duration should be very consistent
            assert consistency_metrics.get('duration_cv', float('inf')) < 0.05, \
                f"Duration inconsistency too high: CV={consistency_metrics.get('duration_cv'):.3f}"
            
            # Amplitude should be reasonably consistent
            assert consistency_metrics.get('amplitude_cv', float('inf')) < 0.2, \
                f"Amplitude inconsistency too high: CV={consistency_metrics.get('amplitude_cv'):.3f}"
            
            # Zero crossing rate should be consistent
            assert consistency_metrics.get('zcr_cv', float('inf')) < 0.3, \
                f"ZCR inconsistency too high: CV={consistency_metrics.get('zcr_cv'):.3f}"
            
            logger.info(f"Identical input consistency test passed with {len(audio_samples)} samples")
        else:
            pytest.skip("Insufficient successful syntheses for consistency testing")
    
    def test_parameter_consistency(self, setup_voiceout_ai):
        """Test consistency with same parameters across sessions."""
        test_text = "Parameter consistency test with multiple sessions."
        
        # Test different parameter sets
        parameter_sets = [
            {'voice': 'default', 'rate': 1.0, 'pitch': 0.0},
            {'voice': 'default', 'rate': 1.5, 'pitch': 0.0},
            {'voice': 'default', 'rate': 1.0, 'pitch': 0.2},
            {'voice': 'female', 'rate': 1.0, 'pitch': 0.0}
        ]
        
        for params in parameter_sets:
            # Generate multiple samples with same parameters
            samples = []
            for i in range(3):
                result = synthesize_speech(test_text, **params)
                if result.get('success', False):
                    samples.append(result['audio_data'])
            
            if len(samples) >= 2:
                consistency_metrics = analyze_audio_consistency(samples)
                
                # Parameters should produce consistent results
                assert consistency_metrics.get('duration_cv', float('inf')) < 0.1, \
                    f"Parameter consistency failed for {params}: duration CV={consistency_metrics.get('duration_cv'):.3f}"
                assert consistency_metrics.get('amplitude_cv', float('inf')) < 0.25, \
                    f"Parameter consistency failed for {params}: amplitude CV={consistency_metrics.get('amplitude_cv'):.3f}"
                
                logger.info(f"Parameter consistency passed for: {params}")
    
    def test_voice_characteristic_stability(self, setup_voiceout_ai):
        """Test that voice characteristics remain stable across different texts."""
        voice_texts = [
            "This is the first test sentence for voice stability.",
            "Here we have another sentence with different content.",
            "A third sentence to verify consistent voice characteristics.",
            "The final sentence in our voice stability evaluation."
        ]
        
        voice_types = ['default', 'female', 'male']
        
        for voice_type in voice_types:
            audio_samples = []
            
            for text in voice_texts:
                result = synthesize_speech(text, voice=voice_type, rate=1.0, pitch=0.0)
                if result.get('success', False):
                    audio_samples.append(result['audio_data'])
            
            if len(audio_samples) >= 3:
                # Analyze voice characteristic consistency
                consistency_metrics = analyze_audio_consistency(audio_samples)
                
                # Voice characteristics should be stable across different texts
                assert consistency_metrics.get('spectral_cv', float('inf')) < 0.3, \
                    f"Voice {voice_type} spectral inconsistency: CV={consistency_metrics.get('spectral_cv'):.3f}"
                assert consistency_metrics.get('amplitude_cv', float('inf')) < 0.4, \
                    f"Voice {voice_type} amplitude inconsistency: CV={consistency_metrics.get('amplitude_cv'):.3f}"
                
                logger.info(f"Voice characteristic stability passed for: {voice_type}")
    
    def test_rate_consistency_across_texts(self, setup_voiceout_ai):
        """Test that speaking rate remains consistent across different texts."""
        test_texts = [
            "Short test sentence for rate measurement.",
            "This is a medium length sentence designed to test speaking rate consistency across various text inputs.",
            "A longer sentence with multiple clauses that should maintain the same relative speaking rate as shorter sentences, providing a comprehensive evaluation of rate consistency."
        ]
        
        rates = [0.8, 1.0, 1.2]
        
        for rate in rates:
            durations = []
            word_counts = []
            
            for text in test_texts:
                result = synthesize_speech(text, voice='default', rate=rate, pitch=0.0)
                
                if result.get('success', False):
                    # Estimate duration from audio
                    audio_data = result['audio_data']
                    if isinstance(audio_data, bytes):
                        audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    else:
                        audio_array = np.array(audio_data)
                    
                    duration = len(audio_array) / 22050  # Assume 22050 Hz
                    word_count = len(text.split())
                    
                    durations.append(duration)
                    word_counts.append(word_count)
            
            if len(durations) >= 2:
                # Calculate words per minute for each text
                wpms = [(words / duration) * 60 for words, duration in zip(word_counts, durations) if duration > 0]
                
                if len(wpms) >= 2:
                    wpm_cv = np.std(wpms) / np.mean(wpms)
                    
                    # Speaking rate should be consistent (CV < 0.3)
                    assert wpm_cv < 0.3, f"Rate {rate} inconsistent across texts: WPM CV={wpm_cv:.3f}"
                    
                    logger.info(f"Rate consistency passed for rate {rate}: WPMs={[f'{wpm:.1f}' for wpm in wpms]}")
    
    def test_pitch_consistency(self, setup_voiceout_ai):
        """Test that pitch modifications are applied consistently."""
        test_text = "This sentence tests pitch consistency across multiple generations."
        
        pitches = [-0.2, 0.0, 0.2]
        
        for pitch in pitches:
            audio_samples = []
            
            # Generate multiple samples with same pitch
            for i in range(4):
                result = synthesize_speech(test_text, voice='default', rate=1.0, pitch=pitch)
                if result.get('success', False):
                    audio_samples.append(result['audio_data'])
            
            if len(audio_samples) >= 3:
                consistency_metrics = analyze_audio_consistency(audio_samples)
                
                # Pitch should be applied consistently
                assert consistency_metrics.get('spectral_cv', float('inf')) < 0.25, \
                    f"Pitch {pitch} inconsistent: spectral CV={consistency_metrics.get('spectral_cv'):.3f}"
                
                logger.info(f"Pitch consistency passed for pitch {pitch}")
    
    def test_cross_session_consistency(self, setup_voiceout_ai):
        """Test consistency across different synthesis sessions."""
        test_text = "Cross-session consistency test for voice synthesis."
        
        # Simulate multiple sessions by generating with delays
        import time
        
        session_results = []
        
        for session in range(3):
            # Small delay to simulate different sessions
            if session > 0:
                time.sleep(0.1)
            
            result = synthesize_speech(
                test_text,
                voice='default',
                rate=1.0,
                pitch=0.0
            )
            
            if result.get('success', False):
                session_results.append(result['audio_data'])
        
        if len(session_results) >= 2:
            consistency_metrics = analyze_audio_consistency(session_results)
            
            # Cross-session consistency should be maintained
            assert consistency_metrics.get('duration_cv', float('inf')) < 0.1, \
                f"Cross-session duration inconsistency: CV={consistency_metrics.get('duration_cv'):.3f}"
            assert consistency_metrics.get('amplitude_cv', float('inf')) < 0.3, \
                f"Cross-session amplitude inconsistency: CV={consistency_metrics.get('amplitude_cv'):.3f}"
            
            logger.info(f"Cross-session consistency passed with {len(session_results)} sessions")
    
    def test_streaming_vs_full_consistency(self, setup_voiceout_ai):
        """Test consistency between streaming and full synthesis modes."""
        test_text = "This text compares streaming and full synthesis consistency."
        
        # Generate with full synthesis
        full_results = []
        for i in range(3):
            result = synthesize_speech(test_text, streaming=False)
            if result.get('success', False):
                full_results.append(result['audio_data'])
        
        # Generate with streaming synthesis
        stream_results = []
        for i in range(3):
            result = synthesize_speech(test_text, streaming=True)
            if result.get('success', False):
                # For streaming, we might get audio_stream or audio_data
                audio_data = result.get('audio_data') or result.get('audio_stream')
                if audio_data:
                    stream_results.append(audio_data)
        
        # Test consistency within each mode
        if len(full_results) >= 2:
            full_consistency = analyze_audio_consistency(full_results)
            assert full_consistency.get('duration_cv', float('inf')) < 0.1, \
                "Full synthesis mode inconsistent"
            logger.info("Full synthesis consistency passed")
        
        if len(stream_results) >= 2:
            stream_consistency = analyze_audio_consistency(stream_results)
            assert stream_consistency.get('duration_cv', float('inf')) < 0.15, \
                "Streaming synthesis mode inconsistent"
            logger.info("Streaming synthesis consistency passed")
        
        # Compare modes if both have results
        if len(full_results) >= 1 and len(stream_results) >= 1:
            similarity = calculate_audio_similarity(full_results[0], stream_results[0])
            logger.info(f"Full vs streaming similarity: {similarity:.3f}")
            
            # Modes should produce similar results
            assert similarity > 0.5, f"Full and streaming modes too different: similarity={similarity:.3f}"
    
    def test_quality_setting_consistency(self, setup_voiceout_ai):
        """Test consistency within different quality settings."""
        test_text = "Quality setting consistency evaluation text."
        
        quality_settings = ['low', 'medium', 'high']
        
        for quality in quality_settings:
            quality_samples = []
            
            for i in range(3):
                result = synthesize_speech(
                    test_text,
                    voice='default',
                    rate=1.0,
                    pitch=0.0,
                    quality=quality
                )
                
                if result.get('success', False):
                    quality_samples.append(result['audio_data'])
            
            if len(quality_samples) >= 2:
                consistency_metrics = analyze_audio_consistency(quality_samples)
                
                # Each quality setting should be internally consistent
                assert consistency_metrics.get('duration_cv', float('inf')) < 0.1, \
                    f"Quality {quality} duration inconsistent: CV={consistency_metrics.get('duration_cv'):.3f}"
                assert consistency_metrics.get('amplitude_cv', float('inf')) < 0.2, \
                    f"Quality {quality} amplitude inconsistent: CV={consistency_metrics.get('amplitude_cv'):.3f}"
                
                logger.info(f"Quality {quality} consistency passed")


@pytest.mark.parametrize("test_case", load_consistency_test_cases())
def test_consistency_from_csv(setup_voiceout_ai, test_case):
    """
    Test consistency using cases from testset.csv.
    
    Args:
        setup_voiceout_ai: The voiceout AI instance from fixture
        test_case: Dictionary containing test case details from CSV
    """
    test_id = test_case['id']
    input_text = test_case['input']
    tts_parameters = test_case.get('tts_parameters', '{}')
    evaluation_method = test_case['evaluation_method']
    
    logger.info(f"Running consistency test {test_id}: {test_case['name']}")
    
    # Parse TTS parameters
    try:
        # Simple parameter parsing (voice=default;rate=1.0;pitch=0.0)
        params = {}
        if tts_parameters:
            for param_pair in tts_parameters.split(';'):
                if '=' in param_pair:
                    key, value = param_pair.split('=', 1)
                    # Convert numeric strings to float
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Keep as string
                    params[key] = value
    except Exception:
        params = {}
    
    # Generate multiple samples for consistency testing
    audio_samples = []
    for i in range(4):
        result = synthesize_speech(input_text, **params)
        if result.get('success', False):
            audio_samples.append(result['audio_data'])
    
    if len(audio_samples) >= 2:
        consistency_metrics = analyze_audio_consistency(audio_samples)
        
        # Apply evaluation based on method
        if 'output_consistency' in evaluation_method:
            assert consistency_metrics.get('duration_cv', float('inf')) < 0.1, \
                f"Test {test_id} failed: duration inconsistency CV={consistency_metrics.get('duration_cv'):.3f}"
            assert consistency_metrics.get('amplitude_cv', float('inf')) < 0.25, \
                f"Test {test_id} failed: amplitude inconsistency CV={consistency_metrics.get('amplitude_cv'):.3f}"
        
        elif 'transcription_consistency' in evaluation_method:
            # For transcription consistency, focus on temporal characteristics
            assert consistency_metrics.get('duration_cv', float('inf')) < 0.05, \
                f"Test {test_id} failed: transcription timing inconsistency"
            assert consistency_metrics.get('zcr_cv', float('inf')) < 0.2, \
                f"Test {test_id} failed: speech pattern inconsistency"
        
        logger.info(f"Test {test_id} passed consistency evaluation with {len(audio_samples)} samples")
    else:
        logger.warning(f"Test {test_id} insufficient samples for consistency testing")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])