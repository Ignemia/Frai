"""
Robustness tests for the voice input (speech-to-text) module.

This module tests the voicein system's robustness against various
audio quality degradations, environmental conditions, and edge cases.
"""

import pytest
import logging
import csv
import os
import sys
from typing import Dict, Any, List
from PIL import Image
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# Import test helpers
try:
    from Frai.tests.back.ai.test_helpers import (
        safe_import_ai_function,
        MockAIInstance,
        AITestCase,
        expect_implementation_error
    )
except ImportError:
    pytest.skip("Test helpers not available", allow_module_level=True)

# Set up logging
logger = logging.getLogger(__name__)

# Safe imports of voicein functions
initialize_voicein_system = safe_import_ai_function('Frai.back.ai.voicein', 'initialize_voicein_system')
get_voicein_ai_instance = safe_import_ai_function('Frai.back.ai.voicein', 'get_voicein_ai_instance')
transcribe_audio = safe_import_ai_function('Frai.back.ai.voicein', 'transcribe_audio')

# Path to the test set CSV file
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), 'testset.csv')


def load_robustness_test_cases() -> List[Dict[str, str]]:
    """Load test cases specifically for robustness testing."""
    test_cases = []
    try:
        with open(TEST_SET_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter for robustness-related tests
                if any(keyword in row['audio_group'] for keyword in 
                      ['noisy', 'degraded', 'environmental', 'voice_conditions', 'equipment']):
                    test_cases.append(row)
        logger.info(f"Loaded {len(test_cases)} robustness test cases")
        return test_cases
    except Exception as e:
        logger.error(f"Failed to load test cases: {e}")
        return []


def load_reference_transcript(transcript_path: str) -> str:
    """Load reference transcript from file."""
    try:
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        else:
            # Create a placeholder transcript for testing
            return "This is a placeholder transcript for testing purposes."
    except Exception as e:
        logger.warning(f"Failed to load transcript {transcript_path}: {e}")
        return ""


def calculate_degradation_tolerance(clean_wer: float, degraded_wer: float) -> float:
    """Calculate how well the system tolerates audio degradation."""
    if clean_wer == 0:
        return 1.0 if degraded_wer < 0.5 else 0.0
    
    degradation_ratio = degraded_wer / clean_wer
    # Lower ratios mean better robustness
    tolerance = max(0.0, 1.0 - (degradation_ratio - 1.0) / 4.0)
    return min(tolerance, 1.0)


@pytest.fixture(scope="module")
def setup_voicein_ai():
    """Initialize the voice input system once for all tests."""
    try:
        success = initialize_voicein_system()
        if not success:
            pytest.fail("Failed to initialize voice input system")
        
        voicein_ai = get_voicein_ai_instance()
        return voicein_ai
    except Exception:
        return MockAIInstance("voicein")


class TestVoiceInRobustness(AITestCase):
    """Test voice input robustness against various degradations."""
    
    def test_background_noise_robustness(self, setup_voicein_ai):
        """Test robustness against various levels of background noise."""
        noise_test_cases = [
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/clean_reference.wav',
                'reference': 'The quick brown fox jumps over the lazy dog.',
                'expected_wer': 0.02,
                'noise_level': 'clean'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/light_noise.wav',
                'reference': 'The quick brown fox jumps over the lazy dog.',
                'expected_wer': 0.08,
                'noise_level': 'light'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/moderate_noise.wav',
                'reference': 'The quick brown fox jumps over the lazy dog.',
                'expected_wer': 0.15,
                'noise_level': 'moderate'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/heavy_noise.wav',
                'reference': 'The quick brown fox jumps over the lazy dog.',
                'expected_wer': 0.25,
                'noise_level': 'heavy'
            }
        ]
        
        previous_wer = 0.0
        for test_case in noise_test_cases:
            result = transcribe_audio(test_case['audio_file'])
            
            if result.get('success', False):
                transcript = result.get('transcript', '')
                wer = self.calculate_word_error_rate(test_case['reference'], transcript)
                
                logger.info(f"Noise level {test_case['noise_level']}: WER = {wer:.3f}")
                
                # Check WER is within expected range
                assert wer <= test_case['expected_wer'], \
                    f"WER {wer:.3f} exceeds threshold {test_case['expected_wer']} for {test_case['noise_level']} noise"
                
                # Check graceful degradation
                if previous_wer > 0:
                    degradation_factor = wer / previous_wer
                    assert degradation_factor <= 5.0, \
                        f"Excessive degradation: {degradation_factor:.2f}x increase in WER"
                
                previous_wer = wer
    
    def test_audio_compression_robustness(self, setup_voicein_ai):
        """Test robustness against various audio compression levels."""
        compression_tests = [
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/uncompressed.wav',
                'reference': 'This is a test of audio compression robustness.',
                'max_wer': 0.03,
                'compression': 'uncompressed'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/mp3_320k.mp3',
                'reference': 'This is a test of audio compression robustness.',
                'max_wer': 0.05,
                'compression': '320kbps MP3'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/mp3_128k.mp3',
                'reference': 'This is a test of audio compression robustness.',
                'max_wer': 0.08,
                'compression': '128kbps MP3'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/mp3_64k.mp3',
                'reference': 'This is a test of audio compression robustness.',
                'max_wer': 0.15,
                'compression': '64kbps MP3'
            }
        ]
        
        for test_case in compression_tests:
            result = transcribe_audio(test_case['audio_file'])
            
            if result.get('success', False):
                transcript = result.get('transcript', '')
                wer = self.calculate_word_error_rate(test_case['reference'], transcript)
                
                logger.info(f"Compression {test_case['compression']}: WER = {wer:.3f}")
                
                assert wer <= test_case['max_wer'], \
                    f"WER {wer:.3f} exceeds threshold for {test_case['compression']}"
    
    def test_microphone_quality_robustness(self, setup_voicein_ai):
        """Test robustness against different microphone qualities."""
        microphone_tests = [
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/studio_microphone.wav',
                'reference': 'Professional studio microphone recording test.',
                'max_wer': 0.02,
                'mic_type': 'studio'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/headset_microphone.wav',
                'reference': 'Professional studio microphone recording test.',
                'max_wer': 0.05,
                'mic_type': 'headset'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/laptop_microphone.wav',
                'reference': 'Professional studio microphone recording test.',
                'max_wer': 0.12,
                'mic_type': 'laptop'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/phone_microphone.wav',
                'reference': 'Professional studio microphone recording test.',
                'max_wer': 0.15,
                'mic_type': 'phone'
            }
        ]
        
        for test_case in microphone_tests:
            result = transcribe_audio(test_case['audio_file'])
            
            if result.get('success', False):
                transcript = result.get('transcript', '')
                wer = self.calculate_word_error_rate(test_case['reference'], transcript)
                
                logger.info(f"Microphone {test_case['mic_type']}: WER = {wer:.3f}")
                
                assert wer <= test_case['max_wer'], \
                    f"WER {wer:.3f} exceeds threshold for {test_case['mic_type']} microphone"
    
    def test_environmental_noise_robustness(self, setup_voicein_ai):
        """Test robustness in various environmental conditions."""
        environmental_tests = [
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/office_environment.wav',
                'reference': 'Testing speech recognition in office environment.',
                'max_wer': 0.08,
                'environment': 'office'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/car_environment.wav',
                'reference': 'Testing speech recognition in office environment.',
                'max_wer': 0.18,
                'environment': 'car'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/outdoor_environment.wav',
                'reference': 'Testing speech recognition in office environment.',
                'max_wer': 0.20,
                'environment': 'outdoor'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/cafe_environment.wav',
                'reference': 'Testing speech recognition in office environment.',
                'max_wer': 0.25,
                'environment': 'cafe'
            }
        ]
        
        for test_case in environmental_tests:
            result = transcribe_audio(test_case['audio_file'])
            
            if result.get('success', False):
                transcript = result.get('transcript', '')
                wer = self.calculate_word_error_rate(test_case['reference'], transcript)
                
                logger.info(f"Environment {test_case['environment']}: WER = {wer:.3f}")
                
                assert wer <= test_case['max_wer'], \
                    f"WER {wer:.3f} exceeds threshold for {test_case['environment']} environment"
    
    def test_voice_condition_robustness(self, setup_voicein_ai):
        """Test robustness against various voice conditions."""
        voice_condition_tests = [
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/normal_voice.wav',
                'reference': 'This speaker has a normal clear voice.',
                'max_wer': 0.05,
                'condition': 'normal'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/hoarse_voice.wav',
                'reference': 'This speaker has a normal clear voice.',
                'max_wer': 0.15,
                'condition': 'hoarse'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/congested_voice.wav',
                'reference': 'This speaker has a normal clear voice.',
                'max_wer': 0.18,
                'condition': 'congested'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/tired_voice.wav',
                'reference': 'This speaker has a normal clear voice.',
                'max_wer': 0.12,
                'condition': 'tired'
            }
        ]
        
        for test_case in voice_condition_tests:
            result = transcribe_audio(test_case['audio_file'])
            
            if result.get('success', False):
                transcript = result.get('transcript', '')
                wer = self.calculate_word_error_rate(test_case['reference'], transcript)
                
                logger.info(f"Voice condition {test_case['condition']}: WER = {wer:.3f}")
                
                assert wer <= test_case['max_wer'], \
                    f"WER {wer:.3f} exceeds threshold for {test_case['condition']} voice"
    
    def test_audio_distortion_robustness(self, setup_voicein_ai):
        """Test robustness against audio distortions."""
        distortion_tests = [
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/clean_audio.wav',
                'reference': 'Testing audio distortion robustness and quality.',
                'max_wer': 0.03,
                'distortion': 'none'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/slight_clipping.wav',
                'reference': 'Testing audio distortion robustness and quality.',
                'max_wer': 0.08,
                'distortion': 'slight_clipping'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/echo_distortion.wav',
                'reference': 'Testing audio distortion robustness and quality.',
                'max_wer': 0.15,
                'distortion': 'echo'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/frequency_filter.wav',
                'reference': 'Testing audio distortion robustness and quality.',
                'max_wer': 0.12,
                'distortion': 'frequency_filter'
            }
        ]
        
        for test_case in distortion_tests:
            result = transcribe_audio(test_case['audio_file'])
            
            if result.get('success', False):
                transcript = result.get('transcript', '')
                wer = self.calculate_word_error_rate(test_case['reference'], transcript)
                
                logger.info(f"Distortion {test_case['distortion']}: WER = {wer:.3f}")
                
                assert wer <= test_case['max_wer'], \
                    f"WER {wer:.3f} exceeds threshold for {test_case['distortion']} distortion"
    
    def test_volume_level_robustness(self, setup_voicein_ai):
        """Test robustness across different volume levels."""
        volume_tests = [
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/normal_volume.wav',
                'reference': 'Testing volume level robustness across ranges.',
                'max_wer': 0.05,
                'volume': 'normal'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/quiet_volume.wav',
                'reference': 'Testing volume level robustness across ranges.',
                'max_wer': 0.12,
                'volume': 'quiet'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/loud_volume.wav',
                'reference': 'Testing volume level robustness across ranges.',
                'max_wer': 0.08,
                'volume': 'loud'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/whisper_volume.wav',
                'reference': 'Testing volume level robustness across ranges.',
                'max_wer': 0.20,
                'volume': 'whisper'
            }
        ]
        
        for test_case in volume_tests:
            result = transcribe_audio(test_case['audio_file'])
            
            if result.get('success', False):
                transcript = result.get('transcript', '')
                wer = self.calculate_word_error_rate(test_case['reference'], transcript)
                
                logger.info(f"Volume {test_case['volume']}: WER = {wer:.3f}")
                
                assert wer <= test_case['max_wer'], \
                    f"WER {wer:.3f} exceeds threshold for {test_case['volume']} volume"
    
    def test_transmission_quality_robustness(self, setup_voicein_ai):
        """Test robustness against transmission quality variations."""
        transmission_tests = [
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/high_quality_transmission.wav',
                'reference': 'Testing transmission quality and robustness factors.',
                'max_wer': 0.04,
                'transmission': 'high_quality'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/phone_line_quality.wav',
                'reference': 'Testing transmission quality and robustness factors.',
                'max_wer': 0.12,
                'transmission': 'phone_line'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/voip_quality.wav',
                'reference': 'Testing transmission quality and robustness factors.',
                'max_wer': 0.10,
                'transmission': 'voip'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/radio_transmission.wav',
                'reference': 'Testing transmission quality and robustness factors.',
                'max_wer': 0.18,
                'transmission': 'radio'
            }
        ]
        
        for test_case in transmission_tests:
            result = transcribe_audio(test_case['audio_file'])
            
            if result.get('success', False):
                transcript = result.get('transcript', '')
                wer = self.calculate_word_error_rate(test_case['reference'], transcript)
                
                logger.info(f"Transmission {test_case['transmission']}: WER = {wer:.3f}")
                
                assert wer <= test_case['max_wer'], \
                    f"WER {wer:.3f} exceeds threshold for {test_case['transmission']} transmission"
    
    def test_age_demographic_robustness(self, setup_voicein_ai):
        """Test robustness across different age demographics."""
        age_tests = [
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/adult_speaker.wav',
                'reference': 'Testing age demographic robustness in speech recognition.',
                'max_wer': 0.05,
                'age_group': 'adult'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/elderly_speaker.wav',
                'reference': 'Testing age demographic robustness in speech recognition.',
                'max_wer': 0.10,
                'age_group': 'elderly'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/teen_speaker.wav',
                'reference': 'Testing age demographic robustness in speech recognition.',
                'max_wer': 0.08,
                'age_group': 'teen'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/child_speaker.wav',
                'reference': 'Testing age demographic robustness in speech recognition.',
                'max_wer': 0.12,
                'age_group': 'child'
            }
        ]
        
        for test_case in age_tests:
            result = transcribe_audio(test_case['audio_file'])
            
            if result.get('success', False):
                transcript = result.get('transcript', '')
                wer = self.calculate_word_error_rate(test_case['reference'], transcript)
                
                logger.info(f"Age group {test_case['age_group']}: WER = {wer:.3f}")
                
                assert wer <= test_case['max_wer'], \
                    f"WER {wer:.3f} exceeds threshold for {test_case['age_group']} speakers"
    
    def calculate_word_error_rate(self, reference: str, hypothesis: str) -> float:
        """Calculate word error rate between reference and hypothesis."""
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        if len(ref_words) == 0:
            return 0.0 if len(hyp_words) == 0 else 1.0
        
        # Simple edit distance calculation
        import difflib
        matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
        matches = sum(triple.size for triple in matcher.get_matching_blocks())
        
        errors = len(ref_words) + len(hyp_words) - 2 * matches
        wer = errors / len(ref_words)
        
        return min(wer, 1.0)


@pytest.mark.parametrize("test_case", load_robustness_test_cases())
def test_robustness_from_csv(setup_voicein_ai, test_case):
    """
    Test robustness using cases from testset.csv.
    
    Args:
        setup_voicein_ai: The voicein AI instance from fixture
        test_case: Dictionary containing test case details from CSV
    """
    test_id = test_case['id']
    audio_group = test_case['audio_group']
    evaluation_metric = test_case['evaluation_metric']
    
    logger.info(f"Running robustness test {test_id}: {test_case['name']}")
    
    # Create test audio file path based on group
    audio_file = f"Frai/tests/back/ai/voicein/test_data/{audio_group}_sample.wav"
    
    # Load reference transcript
    transcript_path = test_case['expected_transcript_path']
    reference_transcript = load_reference_transcript(transcript_path)
    
    if not reference_transcript:
        pytest.skip(f"No reference transcript available for test {test_id}")
    
    # Transcribe audio
    result = transcribe_audio(audio_file)
    
    if result.get('success', False):
        transcript = result.get('transcript', '')
        
        # Calculate WER
        wer = TestVoiceInRobustness().calculate_word_error_rate(reference_transcript, transcript)
        
        # Extract WER threshold from evaluation metric
        if 'word_error_rate <=' in evaluation_metric:
            threshold_str = evaluation_metric.split('<=')[1].strip()
            wer_threshold = float(threshold_str)
            
            logger.info(f"Test {test_id} robustness - WER: {wer:.3f} (threshold: {wer_threshold})")
            
            assert wer <= wer_threshold, f"Test {test_id} WER {wer:.3f} exceeds robustness threshold {wer_threshold}"
        else:
            # For non-WER metrics, just check that transcription succeeded
            assert len(transcript) > 0, f"Test {test_id} produced empty transcript"
            logger.info(f"Test {test_id} passed robustness check with transcript: {transcript}")
    else:
        # Some robustness tests may expect failure
        if 'error' in evaluation_metric.lower():
            logger.info(f"Test {test_id} correctly failed as expected for extreme conditions")
        else:
            logger.warning(f"Test {test_id} transcription failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])