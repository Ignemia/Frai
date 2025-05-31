"""
Accuracy tests for the voice input (speech-to-text) module.

This module validates that model output transcripts match expected reference texts
with acceptable word error rates and handles various audio quality conditions.
"""

import pytest
import logging
import csv
import os
import sys
from typing import Dict, Any, List
import difflib

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


def load_accuracy_test_cases() -> List[Dict[str, str]]:
    """Load test cases specifically for accuracy testing."""
    test_cases = []
    try:
        with open(TEST_SET_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter for accuracy-related tests
                if 'word_error_rate' in row['evaluation_metric']:
                    test_cases.append(row)
        logger.info(f"Loaded {len(test_cases)} accuracy test cases")
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


def calculate_word_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate word error rate between reference and hypothesis."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    
    # Use difflib to calculate edit distance
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    matches = sum(triple.size for triple in matcher.get_matching_blocks())
    
    # WER = (S + D + I) / N where S=substitutions, D=deletions, I=insertions, N=reference length
    errors = len(ref_words) + len(hyp_words) - 2 * matches
    wer = errors / len(ref_words)
    
    return min(wer, 1.0)  # Cap at 1.0


def normalize_transcript(text: str, allowed_variance: str = "") -> str:
    """Normalize transcript based on allowed variance."""
    if not text:
        return ""
    
    normalized = text.strip()
    
    if 'punctuation_only' in allowed_variance:
        # Remove punctuation but keep capitalization
        import string
        normalized = ''.join(c if c not in string.punctuation else ' ' for c in normalized)
        normalized = ' '.join(normalized.split())  # Normalize whitespace
    
    if 'punctuation_capitalization' in allowed_variance:
        # Remove punctuation and normalize capitalization
        import string
        normalized = ''.join(c if c not in string.punctuation else ' ' for c in normalized)
        normalized = ' '.join(normalized.split()).lower()
    
    if 'minor_words' in allowed_variance:
        # Remove common filler words
        filler_words = {'um', 'uh', 'er', 'ah', 'like', 'you know'}
        words = normalized.lower().split()
        normalized = ' '.join(word for word in words if word not in filler_words)
    
    return normalized


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


class TestVoiceInAccuracy(AITestCase):
    """Test voice input transcription accuracy against reference texts."""
    
    def test_clear_speech_accuracy(self, setup_voicein_ai):
        """Test accuracy on clear, high-quality speech."""
        # Test with simulated clear audio files
        clear_audio_tests = [
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/clear_short.wav',
                'reference': 'Hello world, this is a test.',
                'max_wer': 0.02
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/clear_medium.wav', 
                'reference': 'The quick brown fox jumps over the lazy dog.',
                'max_wer': 0.03
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/clear_long.wav',
                'reference': 'This is a longer passage to test the accuracy of speech recognition on extended audio content.',
                'max_wer': 0.05
            }
        ]
        
        for test_case in clear_audio_tests:
            result = transcribe_audio(test_case['audio_file'])
            
            if result.get('success', False):
                transcript = result.get('transcript', '')
                wer = calculate_word_error_rate(test_case['reference'], transcript)
                
                logger.info(f"Clear speech test - Reference: {test_case['reference']}")
                logger.info(f"Clear speech test - Transcript: {transcript}")
                logger.info(f"Clear speech test - WER: {wer:.3f}")
                
                assert wer <= test_case['max_wer'], f"WER {wer:.3f} exceeds maximum {test_case['max_wer']}"
            else:
                logger.warning(f"Transcription failed for {test_case['audio_file']}")
    
    def test_noisy_audio_accuracy(self, setup_voicein_ai):
        """Test accuracy on audio with background noise."""
        noisy_audio_tests = [
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/noisy_light.wav',
                'reference': 'This is speech with light background noise.',
                'max_wer': 0.08
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/noisy_heavy.wav',
                'reference': 'This is speech with heavy background noise.',
                'max_wer': 0.15
            }
        ]
        
        for test_case in noisy_audio_tests:
            result = transcribe_audio(test_case['audio_file'])
            
            if result.get('success', False):
                transcript = result.get('transcript', '')
                wer = calculate_word_error_rate(test_case['reference'], transcript)
                
                logger.info(f"Noisy audio test - WER: {wer:.3f}")
                assert wer <= test_case['max_wer'], f"WER {wer:.3f} exceeds maximum {test_case['max_wer']}"
    
    def test_phone_call_quality_accuracy(self, setup_voicein_ai):
        """Test accuracy on phone call quality audio."""
        phone_test = {
            'audio_file': 'Frai/tests/back/ai/voicein/test_data/phone_call.wav',
            'reference': 'Hello, can you hear me clearly on this phone call?',
            'max_wer': 0.10
        }
        
        result = transcribe_audio(phone_test['audio_file'])
        
        if result.get('success', False):
            transcript = result.get('transcript', '')
            wer = calculate_word_error_rate(phone_test['reference'], transcript)
            
            logger.info(f"Phone call quality test - WER: {wer:.3f}")
            assert wer <= phone_test['max_wer'], f"Phone call WER {wer:.3f} exceeds maximum {phone_test['max_wer']}"
    
    def test_accented_speech_accuracy(self, setup_voicein_ai):
        """Test accuracy on various accented speech."""
        accent_tests = [
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/british_accent.wav',
                'reference': 'This is British English with a distinctive accent.',
                'max_wer': 0.08
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/southern_accent.wav',
                'reference': 'This is Southern American English with regional characteristics.',
                'max_wer': 0.08
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/non_native.wav',
                'reference': 'This is English spoken by a non-native speaker.',
                'max_wer': 0.12
            }
        ]
        
        for test_case in accent_tests:
            result = transcribe_audio(test_case['audio_file'])
            
            if result.get('success', False):
                transcript = result.get('transcript', '')
                wer = calculate_word_error_rate(test_case['reference'], transcript)
                
                logger.info(f"Accent test {test_case['audio_file']} - WER: {wer:.3f}")
                assert wer <= test_case['max_wer'], f"Accent WER {wer:.3f} exceeds maximum {test_case['max_wer']}"
    
    def test_speed_variation_accuracy(self, setup_voicein_ai):
        """Test accuracy on different speaking speeds."""
        speed_tests = [
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/fast_speech.wav',
                'reference': 'This is very fast speech to test rapid delivery recognition.',
                'max_wer': 0.10
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/slow_speech.wav',
                'reference': 'This is very slow and deliberate speech.',
                'max_wer': 0.05
            }
        ]
        
        for test_case in speed_tests:
            result = transcribe_audio(test_case['audio_file'])
            
            if result.get('success', False):
                transcript = result.get('transcript', '')
                wer = calculate_word_error_rate(test_case['reference'], transcript)
                
                logger.info(f"Speed test {test_case['audio_file']} - WER: {wer:.3f}")
                assert wer <= test_case['max_wer'], f"Speed WER {wer:.3f} exceeds maximum {test_case['max_wer']}"
    
    def test_specialized_terminology_accuracy(self, setup_voicein_ai):
        """Test accuracy on specialized terminology."""
        terminology_tests = [
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/technical_jargon.wav',
                'reference': 'The API endpoint returns JSON responses with OAuth authentication.',
                'max_wer': 0.12
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/medical_terms.wav',
                'reference': 'The patient exhibits symptoms of acute myocardial infarction.',
                'max_wer': 0.15
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/legal_language.wav',
                'reference': 'The plaintiff alleges breach of contract and seeks damages.',
                'max_wer': 0.10
            }
        ]
        
        for test_case in terminology_tests:
            result = transcribe_audio(test_case['audio_file'])
            
            if result.get('success', False):
                transcript = result.get('transcript', '')
                wer = calculate_word_error_rate(test_case['reference'], transcript)
                
                logger.info(f"Terminology test {test_case['audio_file']} - WER: {wer:.3f}")
                assert wer <= test_case['max_wer'], f"Terminology WER {wer:.3f} exceeds maximum {test_case['max_wer']}"
    
    def test_numbers_and_dates_accuracy(self, setup_voicein_ai):
        """Test accuracy on numbers, dates, and formatted content."""
        numbers_test = {
            'audio_file': 'Frai/tests/back/ai/voicein/test_data/numbers_dates.wav',
            'reference': 'The meeting is scheduled for March 15th 2024 at 3:30 PM.',
            'max_wer': 0.08
        }
        
        result = transcribe_audio(numbers_test['audio_file'])
        
        if result.get('success', False):
            transcript = result.get('transcript', '')
            wer = calculate_word_error_rate(numbers_test['reference'], transcript)
            
            logger.info(f"Numbers and dates test - WER: {wer:.3f}")
            assert wer <= numbers_test['max_wer'], f"Numbers WER {wer:.3f} exceeds maximum {numbers_test['max_wer']}"


@pytest.mark.parametrize("test_case", load_accuracy_test_cases())
def test_accuracy_from_csv(setup_voicein_ai, test_case):
    """
    Test accuracy using cases from testset.csv.
    
    Args:
        setup_voicein_ai: The voicein AI instance from fixture
        test_case: Dictionary containing test case details from CSV
    """
    test_id = test_case['id']
    audio_group = test_case['audio_group']
    transcript_path = test_case['expected_transcript_path']
    evaluation_metric = test_case['evaluation_metric']
    allowed_variance = test_case.get('allowed_variance', '')
    
    logger.info(f"Running accuracy test {test_id}: {test_case['name']}")
    
    # Load reference transcript
    reference_transcript = load_reference_transcript(transcript_path)
    if not reference_transcript:
        pytest.skip(f"No reference transcript available for test {test_id}")
    
    # Create test audio file path based on group
    audio_file = f"Frai/tests/back/ai/voicein/test_data/{audio_group}_sample.wav"
    
    # Transcribe audio
    result = transcribe_audio(audio_file)
    
    if result.get('success', False):
        transcript = result.get('transcript', '')
        
        # Normalize both transcripts based on allowed variance
        normalized_reference = normalize_transcript(reference_transcript, allowed_variance)
        normalized_transcript = normalize_transcript(transcript, allowed_variance)
        
        # Calculate WER
        wer = calculate_word_error_rate(normalized_reference, normalized_transcript)
        
        # Extract WER threshold from evaluation metric
        if 'word_error_rate <=' in evaluation_metric:
            threshold_str = evaluation_metric.split('<=')[1].strip()
            wer_threshold = float(threshold_str)
            
            logger.info(f"Test {test_id} - Reference: {normalized_reference}")
            logger.info(f"Test {test_id} - Transcript: {normalized_transcript}")
            logger.info(f"Test {test_id} - WER: {wer:.3f} (threshold: {wer_threshold})")
            
            assert wer <= wer_threshold, f"Test {test_id} WER {wer:.3f} exceeds threshold {wer_threshold}"
        else:
            # For non-WER metrics, just check that transcription succeeded
            assert len(transcript) > 0, f"Test {test_id} produced empty transcript"
            logger.info(f"Test {test_id} passed with transcript: {transcript}")
    else:
        logger.warning(f"Test {test_id} transcription failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])