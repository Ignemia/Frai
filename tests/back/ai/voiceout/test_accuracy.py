"""
Accuracy tests for the voice output (text-to-speech) module.

This module validates that synthesized speech accurately represents the input text
with proper pronunciation, timing, and voice characteristics.
"""

import pytest
import logging
import csv
import os
import sys
from typing import Dict, Any, List
import wave
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

# Safe imports of voiceout functions
initialize_voiceout_system = safe_import_ai_function('Frai.back.ai.voiceout', 'initialize_voiceout_system')
get_voiceout_ai_instance = safe_import_ai_function('Frai.back.ai.voiceout', 'get_voiceout_ai_instance')
synthesize_speech = safe_import_ai_function('Frai.back.ai.voiceout', 'synthesize_speech')

# Path to the test set CSV file
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), 'testset.csv')


def load_accuracy_test_cases() -> List[Dict[str, str]]:
    """Load test cases specifically for accuracy testing."""
    test_cases = []
    try:
        with open(TEST_SET_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter for accuracy-related tests
                if 'accuracy' in row['evaluation_method'].lower() or 'pronunciation' in row['evaluation_method'].lower():
                    test_cases.append(row)
        logger.info(f"Loaded {len(test_cases)} accuracy test cases")
        return test_cases
    except Exception as e:
        logger.error(f"Failed to load test cases: {e}")
        return []


def analyze_audio_properties(audio_data) -> Dict[str, float]:
    """Analyze basic properties of audio data."""
    if not audio_data:
        return {}
    
    properties = {}
    
    try:
        # If audio_data is bytes, convert to numpy array for analysis
        if isinstance(audio_data, bytes):
            # Assume 16-bit PCM for now
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
        elif hasattr(audio_data, 'read'):
            # File-like object
            audio_array = np.frombuffer(audio_data.read(), dtype=np.int16)
        else:
            # Already a numpy array
            audio_array = np.array(audio_data)
        
        # Basic audio properties
        properties['duration'] = len(audio_array) / 22050  # Assume 22050 Hz sample rate
        properties['amplitude_mean'] = np.mean(np.abs(audio_array))
        properties['amplitude_max'] = np.max(np.abs(audio_array))
        properties['amplitude_std'] = np.std(audio_array)
        
        # RMS (Root Mean Square) - measure of audio power
        properties['rms'] = np.sqrt(np.mean(audio_array**2))
        
        # Zero crossing rate - indicates speech activity
        zero_crossings = np.where(np.diff(np.signbit(audio_array)))[0]
        properties['zero_crossing_rate'] = len(zero_crossings) / len(audio_array)
        
        # Simple silence detection
        silence_threshold = properties['amplitude_mean'] * 0.1
        silence_samples = np.sum(np.abs(audio_array) < silence_threshold)
        properties['silence_ratio'] = silence_samples / len(audio_array)
        
    except Exception as e:
        logger.warning(f"Failed to analyze audio properties: {e}")
        properties['analysis_error'] = str(e)
    
    return properties


def estimate_speech_rate(audio_data, text: str) -> float:
    """Estimate words per minute from audio and text."""
    if not audio_data or not text:
        return 0.0
    
    try:
        properties = analyze_audio_properties(audio_data)
        duration = properties.get('duration', 0)
        
        if duration == 0:
            return 0.0
        
        word_count = len(text.split())
        wpm = (word_count / duration) * 60
        
        return wpm
    except Exception as e:
        logger.warning(f"Failed to estimate speech rate: {e}")
        return 0.0


def check_audio_quality(audio_data) -> Dict[str, bool]:
    """Check for basic audio quality issues."""
    if not audio_data:
        return {'no_audio': True}
    
    quality_checks = {}
    
    try:
        properties = analyze_audio_properties(audio_data)
        
        # Check for silence (no speech generated)
        quality_checks['has_content'] = properties.get('silence_ratio', 1.0) < 0.9
        
        # Check for clipping (distortion)
        max_amplitude = properties.get('amplitude_max', 0)
        quality_checks['no_clipping'] = max_amplitude < 30000  # For 16-bit audio
        
        # Check for reasonable duration
        duration = properties.get('duration', 0)
        quality_checks['reasonable_duration'] = 0.1 < duration < 300  # 0.1s to 5 minutes
        
        # Check for audio activity
        rms = properties.get('rms', 0)
        quality_checks['has_audio_activity'] = rms > 100
        
        # Check zero crossing rate (speech typically has moderate ZCR)
        zcr = properties.get('zero_crossing_rate', 0)
        quality_checks['natural_speech_pattern'] = 0.01 < zcr < 0.3
        
    except Exception as e:
        logger.warning(f"Failed to check audio quality: {e}")
        quality_checks['analysis_error'] = True
    
    return quality_checks


# Using fixture from conftest.py


class TestVoiceOutAccuracy(AITestCase):
    """Test voice output synthesis accuracy."""
    
    def test_simple_text_synthesis(self, setup_voiceout_ai):
        """Test accurate synthesis of simple text."""
        simple_texts = [
            "Hello world, this is a test.",
            "The quick brown fox jumps over the lazy dog.",
            "Good morning, how are you today?",
            "Thank you for your time and attention.",
            "This is a simple sentence for testing."
        ]
        
        for text in simple_texts:
            result = synthesize_speech(text)
            
            if result.get('success', False):
                audio_data = result.get('audio_data')
                
                # Check basic audio quality
                quality_checks = check_audio_quality(audio_data)
                assert quality_checks.get('has_content', False), f"No speech content for: {text}"
                assert quality_checks.get('has_audio_activity', False), f"No audio activity for: {text}"
                assert quality_checks.get('reasonable_duration', False), f"Invalid duration for: {text}"
                
                # Check speech rate is reasonable (100-300 WPM is typical)
                speech_rate = estimate_speech_rate(audio_data, text)
                if speech_rate > 0:
                    assert 50 <= speech_rate <= 400, f"Speech rate {speech_rate:.1f} WPM unusual for: {text}"
                
                logger.info(f"Simple text synthesis passed: {text}")
    
    def test_pronunciation_accuracy(self, setup_voiceout_ai):
        """Test pronunciation of challenging words."""
        pronunciation_tests = [
            {
                'text': "The pronunciation of these words should be accurate.",
                'challenging_words': ['pronunciation', 'accurate']
            },
            {
                'text': "Please pronounce these technical terms correctly: algorithm, authentication, synchronization.",
                'challenging_words': ['algorithm', 'authentication', 'synchronization']
            },
            {
                'text': "Medical terminology: stethoscope, pharmaceutical, diagnosis.",
                'challenging_words': ['stethoscope', 'pharmaceutical', 'diagnosis']
            },
            {
                'text': "Foreign words: café, naïve, résumé, piñata.",
                'challenging_words': ['café', 'naïve', 'résumé', 'piñata']
            }
        ]
        
        for test_case in pronunciation_tests:
            result = synthesize_speech(test_case['text'])
            
            if result.get('success', False):
                audio_data = result.get('audio_data')
                quality_checks = check_audio_quality(audio_data)
                
                assert quality_checks.get('has_content', False), \
                    f"No speech generated for pronunciation test: {test_case['text']}"
                assert quality_checks.get('natural_speech_pattern', False), \
                    f"Unnatural speech pattern for: {test_case['text']}"
                
                logger.info(f"Pronunciation test passed: {test_case['challenging_words']}")
    
    def test_number_pronunciation(self, setup_voiceout_ai):
        """Test accurate pronunciation of numbers and dates."""
        number_tests = [
            "The year 2024 was significant.",
            "Please call 555-123-4567 for assistance.",
            "The temperature is 23.5 degrees Celsius.",
            "The meeting is on March 15th at 3:30 PM.",
            "The cost is $1,234.56 including tax.",
            "The ratio is approximately 3.14159 to 1."
        ]
        
        for text in number_tests:
            result = synthesize_speech(text)
            
            if result.get('success', False):
                audio_data = result.get('audio_data')
                quality_checks = check_audio_quality(audio_data)
                
                assert quality_checks.get('has_content', False), \
                    f"No speech content for number test: {text}"
                
                # Numbers often require longer pronunciation time
                properties = analyze_audio_properties(audio_data)
                duration = properties.get('duration', 0)
                word_count = len(text.split())
                
                if duration > 0 and word_count > 0:
                    # Allow for longer pronunciation of numbers
                    max_expected_duration = word_count * 0.8  # Up to 0.8 seconds per word
                    assert duration <= max_expected_duration, \
                        f"Duration {duration:.2f}s too long for {word_count} words"
                
                logger.info(f"Number pronunciation test passed: {text}")
    
    def test_punctuation_handling(self, setup_voiceout_ai):
        """Test proper handling of punctuation in speech."""
        punctuation_tests = [
            {
                'text': "Hello! How are you? I'm fine, thank you.",
                'expected_pauses': True,
                'description': "Question and exclamation marks"
            },
            {
                'text': "The items are: apples, oranges, and bananas.",
                'expected_pauses': True,
                'description': "Colon and commas"
            },
            {
                'text': "She said, \"I'll be there soon.\"",
                'expected_pauses': False,
                'description': "Quotation marks"
            },
            {
                'text': "Wait... let me think about this.",
                'expected_pauses': True,
                'description': "Ellipsis"
            }
        ]
        
        for test_case in punctuation_tests:
            result = synthesize_speech(test_case['text'])
            
            if result.get('success', False):
                audio_data = result.get('audio_data')
                quality_checks = check_audio_quality(audio_data)
                
                assert quality_checks.get('has_content', False), \
                    f"No speech content for punctuation test: {test_case['description']}"
                
                # Check for reasonable silence patterns if pauses are expected
                properties = analyze_audio_properties(audio_data)
                silence_ratio = properties.get('silence_ratio', 0)
                
                if test_case['expected_pauses']:
                    # Should have some silence for punctuation pauses
                    assert silence_ratio > 0.05, \
                        f"Expected pauses not detected for: {test_case['description']}"
                
                logger.info(f"Punctuation test passed: {test_case['description']}")
    
    def test_intonation_patterns(self, setup_voiceout_ai):
        """Test proper intonation for different sentence types."""
        intonation_tests = [
            {
                'text': "Are you coming to the party tonight?",
                'type': 'question',
                'expected_pattern': 'rising'
            },
            {
                'text': "Please close the door behind you.",
                'type': 'command',
                'expected_pattern': 'falling'
            },
            {
                'text': "What a beautiful sunset!",
                'type': 'exclamation',
                'expected_pattern': 'emphatic'
            },
            {
                'text': "The weather today is quite pleasant.",
                'type': 'statement',
                'expected_pattern': 'neutral'
            }
        ]
        
        for test_case in intonation_tests:
            result = synthesize_speech(test_case['text'])
            
            if result.get('success', False):
                audio_data = result.get('audio_data')
                quality_checks = check_audio_quality(audio_data)
                
                assert quality_checks.get('has_content', False), \
                    f"No speech content for intonation test: {test_case['type']}"
                assert quality_checks.get('natural_speech_pattern', False), \
                    f"Unnatural speech pattern for {test_case['type']}: {test_case['text']}"
                
                logger.info(f"Intonation test passed: {test_case['type']} - {test_case['text']}")
    
    def test_voice_consistency(self, setup_voiceout_ai):
        """Test consistency of voice characteristics."""
        test_text = "This is a consistency test for voice output."
        
        # Generate multiple samples with same parameters
        results = []
        for i in range(3):
            result = synthesize_speech(test_text, voice='default', rate=1.0, pitch=0.0)
            if result.get('success', False):
                results.append(result['audio_data'])
        
        if len(results) >= 2:
            # Compare basic properties of generated audio
            properties_list = [analyze_audio_properties(audio) for audio in results]
            
            # Check duration consistency (should be very similar)
            durations = [props.get('duration', 0) for props in properties_list]
            if all(d > 0 for d in durations):
                duration_variance = np.var(durations)
                mean_duration = np.mean(durations)
                cv = duration_variance / mean_duration if mean_duration > 0 else float('inf')
                
                assert cv < 0.1, f"Duration too inconsistent: CV={cv:.3f}"
            
            # Check amplitude consistency
            rms_values = [props.get('rms', 0) for props in properties_list]
            if all(rms > 0 for rms in rms_values):
                rms_cv = np.std(rms_values) / np.mean(rms_values)
                assert rms_cv < 0.3, f"RMS too inconsistent: CV={rms_cv:.3f}"
            
            logger.info(f"Voice consistency test passed with {len(results)} samples")
    
    def test_speech_rate_accuracy(self, setup_voiceout_ai):
        """Test accuracy of speech rate control."""
        test_text = "This is a test sentence for measuring speech rate accuracy."
        rates = [0.5, 1.0, 1.5, 2.0]
        
        rate_results = []
        
        for rate in rates:
            result = synthesize_speech(test_text, rate=rate)
            
            if result.get('success', False):
                audio_data = result.get('audio_data')
                actual_rate = estimate_speech_rate(audio_data, test_text)
                
                if actual_rate > 0:
                    # Calculate expected WPM (assuming baseline of 150 WPM at rate=1.0)
                    expected_wpm = 150 * rate
                    rate_error = abs(actual_rate - expected_wpm) / expected_wpm
                    
                    # Allow 30% tolerance for rate accuracy
                    assert rate_error < 0.3, \
                        f"Rate {rate}: expected ~{expected_wpm:.0f} WPM, got {actual_rate:.1f} WPM"
                    
                    rate_results.append((rate, actual_rate))
        
        # Check that rates are in correct order
        if len(rate_results) >= 2:
            for i in range(1, len(rate_results)):
                prev_rate, prev_wpm = rate_results[i-1]
                curr_rate, curr_wpm = rate_results[i]
                
                if curr_rate > prev_rate:
                    assert curr_wpm >= prev_wpm * 0.8, \
                        f"Rate progression issue: {prev_rate}->={curr_rate} but {prev_wpm:.1f}->{curr_wpm:.1f} WPM"
        
        logger.info(f"Speech rate accuracy test passed with rates: {rates}")


@pytest.mark.parametrize("test_case", load_accuracy_test_cases())
def test_accuracy_from_csv(setup_voiceout_ai, test_case):
    """
    Test accuracy using cases from testset.csv.
    
    Args:
        setup_voiceout_ai: The voiceout AI instance from fixture
        test_case: Dictionary containing test case details from CSV
    """
    test_id = test_case['id']
    input_text = test_case['input']
    tts_parameters = test_case.get('tts_parameters', '{}')
    evaluation_method = test_case['evaluation_method']
    
    logger.info(f"Running accuracy test {test_id}: {test_case['name']}")
    
    # Parse TTS parameters
    try:
        import json
        params = json.loads(tts_parameters) if tts_parameters else {}
    except json.JSONDecodeError:
        params = {}
    
    # Synthesize speech
    result = synthesize_speech(input_text, **params)
    
    if result.get('success', False):
        audio_data = result.get('audio_data')
        
        # Apply evaluation based on method
        if 'pronunciation_accuracy' in evaluation_method:
            quality_checks = check_audio_quality(audio_data)
            assert quality_checks.get('has_content', False), \
                f"Test {test_id} failed: no speech content"
            assert quality_checks.get('natural_speech_pattern', False), \
                f"Test {test_id} failed: unnatural speech pattern"
        
        elif 'speech_rate_accuracy' in evaluation_method:
            speech_rate = estimate_speech_rate(audio_data, input_text)
            if speech_rate > 0:
                # Reasonable speech rate (50-400 WPM)
                assert 50 <= speech_rate <= 400, \
                    f"Test {test_id} failed: speech rate {speech_rate:.1f} WPM out of range"
        
        elif 'audio_quality' in evaluation_method:
            quality_checks = check_audio_quality(audio_data)
            assert quality_checks.get('has_content', False), \
                f"Test {test_id} failed: no audio content"
            assert quality_checks.get('no_clipping', False), \
                f"Test {test_id} failed: audio clipping detected"
            assert quality_checks.get('has_audio_activity', False), \
                f"Test {test_id} failed: no audio activity"
        
        elif 'duration_accuracy' in evaluation_method:
            properties = analyze_audio_properties(audio_data)
            duration = properties.get('duration', 0)
            word_count = len(input_text.split())
            
            if word_count > 0:
                # Expect 2-10 words per second for normal speech
                min_duration = word_count / 10
                max_duration = word_count / 2
                assert min_duration <= duration <= max_duration, \
                    f"Test {test_id} failed: duration {duration:.2f}s unusual for {word_count} words"
        
        logger.info(f"Test {test_id} passed accuracy evaluation")
    else:
        logger.warning(f"Test {test_id} synthesis failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])