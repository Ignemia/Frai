"""
Consistency tests for the voice input (speech-to-text) module.

This module tests the voicein system's consistency in producing
similar transcriptions for identical or very similar audio inputs.
"""

import pytest
import logging
import csv
import os
import sys
from typing import Dict, Any, List
import statistics

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


def load_consistency_test_cases() -> List[Dict[str, str]]:
    """Load test cases specifically for consistency testing."""
    test_cases = []
    try:
        with open(TEST_SET_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter for consistency tests
                if 'consistency' in row['audio_group'] or 'consistency' in row['evaluation_metric']:
                    test_cases.append(row)
        logger.info(f"Loaded {len(test_cases)} consistency test cases")
        return test_cases
    except Exception as e:
        logger.error(f"Failed to load test cases: {e}")
        return []


def calculate_transcript_similarity(transcript1: str, transcript2: str) -> float:
    """Calculate similarity between two transcripts."""
    if not transcript1 or not transcript2:
        return 0.0
    
    # Normalize transcripts
    t1 = transcript1.lower().strip()
    t2 = transcript2.lower().strip()
    
    if t1 == t2:
        return 1.0
    
    # Calculate character-level similarity
    import difflib
    char_similarity = difflib.SequenceMatcher(None, t1, t2).ratio()
    
    # Calculate word-level similarity
    words1 = t1.split()
    words2 = t2.split()
    word_similarity = difflib.SequenceMatcher(None, words1, words2).ratio()
    
    # Combine both measures
    return (char_similarity + word_similarity) / 2


def calculate_consistency_score(transcripts: List[str]) -> float:
    """Calculate consistency score across multiple transcripts."""
    if len(transcripts) < 2:
        return 1.0
    
    similarities = []
    for i in range(len(transcripts)):
        for j in range(i + 1, len(transcripts)):
            similarity = calculate_transcript_similarity(transcripts[i], transcripts[j])
            similarities.append(similarity)
    
    return statistics.mean(similarities)


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


class TestVoiceInConsistency(AITestCase):
    """Test voice input transcription consistency."""
    
    def test_identical_audio_consistency(self, setup_voicein_ai):
        """Test that identical audio files produce consistent transcriptions."""
        audio_file = 'Frai/tests/back/ai/voicein/test_data/consistency_test.wav'
        num_runs = 5
        transcripts = []
        
        for i in range(num_runs):
            result = transcribe_audio(audio_file)
            
            if result.get('success', False):
                transcript = result.get('transcript', '')
                transcripts.append(transcript)
                logger.info(f"Run {i+1} transcript: {transcript}")
        
        if len(transcripts) >= 3:
            consistency_score = calculate_consistency_score(transcripts)
            logger.info(f"Identical audio consistency score: {consistency_score:.3f}")
            
            # Identical audio should produce very consistent results
            assert consistency_score >= 0.95, f"Poor consistency: {consistency_score:.3f}"
        else:
            pytest.skip("Insufficient successful transcriptions for consistency test")
    
    def test_repeated_transcription_stability(self, setup_voicein_ai):
        """Test stability across repeated transcriptions of the same content."""
        test_files = [
            'Frai/tests/back/ai/voicein/test_data/stability_test_1.wav',
            'Frai/tests/back/ai/voicein/test_data/stability_test_2.wav',
            'Frai/tests/back/ai/voicein/test_data/stability_test_3.wav'
        ]
        
        for audio_file in test_files:
            transcripts = []
            
            # Transcribe same file multiple times
            for run in range(4):
                result = transcribe_audio(audio_file)
                
                if result.get('success', False):
                    transcript = result.get('transcript', '')
                    transcripts.append(transcript)
            
            if len(transcripts) >= 3:
                consistency_score = calculate_consistency_score(transcripts)
                logger.info(f"File {audio_file} consistency: {consistency_score:.3f}")
                
                # Should be highly consistent
                assert consistency_score >= 0.90, f"Unstable transcription for {audio_file}: {consistency_score:.3f}"
    
    def test_parameter_consistency(self, setup_voicein_ai):
        """Test consistency with different parameter settings."""
        audio_file = 'Frai/tests/back/ai/voicein/test_data/parameter_test.wav'
        
        # Test with different temperature settings
        temperatures = [0.0, 0.2, 0.0]  # Include duplicate 0.0 to test consistency
        transcripts = []
        
        for temp in temperatures:
            result = transcribe_audio(audio_file, temperature=temp)
            
            if result.get('success', False):
                transcript = result.get('transcript', '')
                transcripts.append(transcript)
                logger.info(f"Temperature {temp} transcript: {transcript}")
        
        if len(transcripts) >= 2:
            # Check that identical parameters (temp=0.0) produce identical results
            identical_param_transcripts = [transcripts[0], transcripts[2]]
            if len(identical_param_transcripts) == 2:
                similarity = calculate_transcript_similarity(
                    identical_param_transcripts[0], 
                    identical_param_transcripts[1]
                )
                logger.info(f"Identical parameter consistency: {similarity:.3f}")
                
                assert similarity >= 0.98, f"Identical parameters should be deterministic: {similarity:.3f}"
    
    def test_minor_variation_consistency(self, setup_voicein_ai):
        """Test consistency across minor audio variations."""
        # Test with slightly different versions of the same recording
        variation_tests = [
            {
                'files': [
                    'Frai/tests/back/ai/voicein/test_data/original_recording.wav',
                    'Frai/tests/back/ai/voicein/test_data/volume_adjusted.wav',
                    'Frai/tests/back/ai/voicein/test_data/eq_adjusted.wav'
                ],
                'min_consistency': 0.85
            },
            {
                'files': [
                    'Frai/tests/back/ai/voicein/test_data/clean_version.wav',
                    'Frai/tests/back/ai/voicein/test_data/noise_reduced.wav'
                ],
                'min_consistency': 0.90
            }
        ]
        
        for test in variation_tests:
            transcripts = []
            
            for audio_file in test['files']:
                result = transcribe_audio(audio_file)
                
                if result.get('success', False):
                    transcript = result.get('transcript', '')
                    transcripts.append(transcript)
            
            if len(transcripts) >= 2:
                consistency_score = calculate_consistency_score(transcripts)
                logger.info(f"Minor variation consistency: {consistency_score:.3f}")
                
                assert consistency_score >= test['min_consistency'], \
                    f"Poor consistency across minor variations: {consistency_score:.3f}"
    
    def test_language_model_consistency(self, setup_voicein_ai):
        """Test consistency of language model behavior."""
        # Test phrases that could be interpreted multiple ways
        ambiguous_tests = [
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/ambiguous_phrase_1.wav',
                'expected_variants': ['to', 'two', 'too'],
                'min_runs': 5
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/ambiguous_phrase_2.wav',
                'expected_variants': ['there', 'their', 'they\'re'],
                'min_runs': 5
            }
        ]
        
        for test in ambiguous_tests:
            transcripts = []
            
            for run in range(test['min_runs']):
                result = transcribe_audio(test['audio_file'])
                
                if result.get('success', False):
                    transcript = result.get('transcript', '')
                    transcripts.append(transcript)
            
            if len(transcripts) >= 3:
                # Check if model consistently chooses the same interpretation
                unique_transcripts = list(set(transcripts))
                consistency_ratio = len(transcripts) / len(unique_transcripts)
                
                logger.info(f"Language model consistency ratio: {consistency_ratio:.2f}")
                logger.info(f"Unique transcripts: {unique_transcripts}")
                
                # Should be reasonably consistent, allowing for some variation
                assert consistency_ratio >= 2.0, f"Too much variation in language model: {consistency_ratio:.2f}"
    
    def test_punctuation_consistency(self, setup_voicein_ai):
        """Test consistency in punctuation placement."""
        audio_files = [
            'Frai/tests/back/ai/voicein/test_data/sentence_with_pauses.wav',
            'Frai/tests/back/ai/voicein/test_data/question_intonation.wav',
            'Frai/tests/back/ai/voicein/test_data/exclamation_tone.wav'
        ]
        
        for audio_file in audio_files:
            transcripts = []
            
            # Multiple runs to check punctuation consistency
            for run in range(4):
                result = transcribe_audio(audio_file)
                
                if result.get('success', False):
                    transcript = result.get('transcript', '')
                    transcripts.append(transcript)
            
            if len(transcripts) >= 3:
                # Remove punctuation and compare base text
                def remove_punctuation(text):
                    import string
                    return ''.join(c for c in text if c not in string.punctuation)
                
                base_texts = [remove_punctuation(t).strip() for t in transcripts]
                base_consistency = calculate_consistency_score(base_texts)
                
                # Check punctuation patterns
                punctuation_patterns = []
                for transcript in transcripts:
                    import string
                    punct_pattern = ''.join(c for c in transcript if c in string.punctuation)
                    punctuation_patterns.append(punct_pattern)
                
                punct_consistency = calculate_consistency_score(punctuation_patterns)
                
                logger.info(f"Base text consistency: {base_consistency:.3f}")
                logger.info(f"Punctuation consistency: {punct_consistency:.3f}")
                
                # Base text should be very consistent
                assert base_consistency >= 0.95, f"Inconsistent base text: {base_consistency:.3f}"
                
                # Punctuation should be reasonably consistent
                assert punct_consistency >= 0.70, f"Inconsistent punctuation: {punct_consistency:.3f}"
    
    def test_capitalization_consistency(self, setup_voicein_ai):
        """Test consistency in capitalization patterns."""
        audio_file = 'Frai/tests/back/ai/voicein/test_data/proper_nouns_test.wav'
        transcripts = []
        
        for run in range(5):
            result = transcribe_audio(audio_file)
            
            if result.get('success', False):
                transcript = result.get('transcript', '')
                transcripts.append(transcript)
        
        if len(transcripts) >= 3:
            # Check consistency of capitalization patterns
            def extract_capitalization_pattern(text):
                return ''.join('U' if c.isupper() else 'l' if c.islower() else 's' for c in text)
            
            cap_patterns = [extract_capitalization_pattern(t) for t in transcripts]
            cap_consistency = calculate_consistency_score(cap_patterns)
            
            # Also check word-level capitalization consistency
            word_caps = []
            for transcript in transcripts:
                words = transcript.split()
                cap_words = [word for word in words if word and word[0].isupper()]
                word_caps.append(' '.join(cap_words))
            
            word_cap_consistency = calculate_consistency_score(word_caps)
            
            logger.info(f"Capitalization pattern consistency: {cap_consistency:.3f}")
            logger.info(f"Capitalized words consistency: {word_cap_consistency:.3f}")
            
            # Capitalization should be reasonably consistent
            assert word_cap_consistency >= 0.80, f"Inconsistent capitalization: {word_cap_consistency:.3f}"
    
    def test_number_format_consistency(self, setup_voicein_ai):
        """Test consistency in number formatting."""
        audio_files = [
            'Frai/tests/back/ai/voicein/test_data/numbers_dates.wav',
            'Frai/tests/back/ai/voicein/test_data/currency_amounts.wav',
            'Frai/tests/back/ai/voicein/test_data/phone_numbers.wav'
        ]
        
        for audio_file in audio_files:
            transcripts = []
            
            for run in range(4):
                result = transcribe_audio(audio_file)
                
                if result.get('success', False):
                    transcript = result.get('transcript', '')
                    transcripts.append(transcript)
            
            if len(transcripts) >= 3:
                # Extract numbers from transcripts
                import re
                number_patterns = []
                for transcript in transcripts:
                    numbers = re.findall(r'\b\d+(?:[.,]\d+)*\b', transcript)
                    number_patterns.append(' '.join(numbers))
                
                number_consistency = calculate_consistency_score(number_patterns)
                
                logger.info(f"Number format consistency for {audio_file}: {number_consistency:.3f}")
                
                # Number formatting should be consistent
                assert number_consistency >= 0.85, f"Inconsistent number formatting: {number_consistency:.3f}"


@pytest.mark.parametrize("test_case", load_consistency_test_cases())
def test_consistency_from_csv(setup_voicein_ai, test_case):
    """
    Test consistency using cases from testset.csv.
    
    Args:
        setup_voicein_ai: The voicein AI instance from fixture
        test_case: Dictionary containing test case details from CSV
    """
    test_id = test_case['id']
    audio_group = test_case['audio_group']
    evaluation_metric = test_case['evaluation_metric']
    
    logger.info(f"Running consistency test {test_id}: {test_case['name']}")
    
    # For consistency tests, we need multiple audio files or runs
    if 'consistency' in audio_group:
        # Create multiple test audio file paths
        audio_files = [
            f"Frai/tests/back/ai/voicein/test_data/consistency_{test_id}_1.wav",
            f"Frai/tests/back/ai/voicein/test_data/consistency_{test_id}_2.wav",
            f"Frai/tests/back/ai/voicein/test_data/consistency_{test_id}_3.wav"
        ]
        
        transcripts = []
        
        for audio_file in audio_files:
            result = transcribe_audio(audio_file)
            
            if result.get('success', False):
                transcript = result.get('transcript', '')
                transcripts.append(transcript)
        
        if len(transcripts) >= 2:
            consistency_score = calculate_consistency_score(transcripts)
            
            logger.info(f"Test {test_id} consistency score: {consistency_score:.3f}")
            
            # Extract consistency threshold from evaluation metric
            if 'transcription_consistency' in evaluation_metric:
                # Default threshold for consistency tests
                consistency_threshold = 0.85
                
                assert consistency_score >= consistency_threshold, \
                    f"Test {test_id} consistency {consistency_score:.3f} below threshold {consistency_threshold}"
            else:
                # Basic consistency check
                assert consistency_score >= 0.75, f"Test {test_id} failed basic consistency: {consistency_score:.3f}"
        else:
            logger.warning(f"Test {test_id} insufficient transcriptions for consistency analysis")
    else:
        pytest.skip(f"Test {test_id} not configured for consistency testing")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])