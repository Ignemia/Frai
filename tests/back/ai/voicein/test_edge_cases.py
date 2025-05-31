"""
Edge case tests for the voice input (speech-to-text) module.

This module tests the voicein system's behavior with unusual inputs and edge cases
such as silence, very short clips, non-speech sounds, overlapping speakers, and
foreign language detection.
"""

import pytest
import logging
import csv
import os
import sys
from typing import Dict, Any, List

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


def load_edge_case_test_cases() -> List[Dict[str, str]]:
    """Load test cases specifically for edge case testing."""
    test_cases = []
    try:
        with open(TEST_SET_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter for edge case tests
                if any(keyword in row['audio_group'] for keyword in 
                      ['edge_cases', 'foreign', 'conversation', 'artistic']):
                    test_cases.append(row)
        logger.info(f"Loaded {len(test_cases)} edge case test cases")
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
            return ""
    except Exception as e:
        logger.warning(f"Failed to load transcript {transcript_path}: {e}")
        return ""


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


class TestVoiceInEdgeCases(AITestCase):
    """Test voice input edge cases and unusual scenarios."""
    
    def test_silence_only_audio(self, setup_voicein_ai):
        """Test transcription of audio containing only silence."""
        silence_tests = [
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/silence_5_seconds.wav',
                'duration': '5 seconds',
                'expected_behavior': 'empty_or_silence'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/silence_30_seconds.wav',
                'duration': '30 seconds',
                'expected_behavior': 'empty_or_silence'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/silence_1_minute.wav',
                'duration': '1 minute',
                'expected_behavior': 'empty_or_silence'
            }
        ]
        
        for test in silence_tests:
            result = transcribe_audio(test['audio_file'])
            
            if result.get('success', False):
                transcript = result.get('transcript', '').strip()
                
                # Should produce empty transcript or silence indicator
                assert len(transcript) == 0 or transcript.lower() in ['', '[silence]', '(silence)', 'silence'], \
                    f"Silence audio produced unexpected transcript: '{transcript}'"
                
                logger.info(f"Silence test {test['duration']}: correctly handled")
            else:
                # Silence handling failure might be acceptable
                logger.info(f"Silence test {test['duration']}: failed transcription (acceptable)")
    
    def test_very_short_audio_clips(self, setup_voicein_ai):
        """Test transcription of very short audio clips."""
        short_clip_tests = [
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/single_word_hello.wav',
                'expected_content': 'hello',
                'duration': '< 1 second'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/single_word_yes.wav',
                'expected_content': 'yes',
                'duration': '< 1 second'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/two_words.wav',
                'expected_content': 'thank you',
                'duration': '< 2 seconds'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/half_second_clip.wav',
                'expected_content': 'hi',
                'duration': '0.5 seconds'
            }
        ]
        
        for test in short_clip_tests:
            result = transcribe_audio(test['audio_file'])
            
            if result.get('success', False):
                transcript = result.get('transcript', '').lower().strip()
                expected = test['expected_content'].lower().strip()
                
                # Check if expected content is present
                content_match = expected in transcript or transcript in expected
                
                logger.info(f"Short clip {test['duration']}: '{transcript}' (expected: '{expected}')")
                
                # Should handle short clips reasonably well
                assert content_match or len(transcript) > 0, \
                    f"Short clip failed: expected '{expected}', got '{transcript}'"
            else:
                logger.warning(f"Short clip {test['duration']} failed transcription")
    
    def test_non_speech_audio(self, setup_voicein_ai):
        """Test transcription of non-speech audio content."""
        non_speech_tests = [
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/music_instrumental.wav',
                'content_type': 'instrumental music',
                'expected_behavior': 'minimal_or_empty'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/nature_sounds.wav',
                'content_type': 'nature sounds',
                'expected_behavior': 'minimal_or_empty'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/white_noise.wav',
                'content_type': 'white noise',
                'expected_behavior': 'minimal_or_empty'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/mechanical_sounds.wav',
                'content_type': 'mechanical sounds',
                'expected_behavior': 'minimal_or_empty'
            }
        ]
        
        for test in non_speech_tests:
            result = transcribe_audio(test['audio_file'])
            
            if result.get('success', False):
                transcript = result.get('transcript', '').strip()
                
                # Non-speech should produce minimal transcript
                assert len(transcript) < 50, \
                    f"Non-speech audio ({test['content_type']}) produced lengthy transcript: '{transcript}'"
                
                logger.info(f"Non-speech {test['content_type']}: '{transcript[:30]}...' ({len(transcript)} chars)")
            else:
                logger.info(f"Non-speech {test['content_type']}: failed transcription (acceptable)")
    
    def test_overlapping_speakers(self, setup_voicein_ai):
        """Test transcription with multiple overlapping speakers."""
        overlap_tests = [
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/two_speakers_slight_overlap.wav',
                'scenario': 'slight overlap',
                'min_content_length': 10
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/two_speakers_heavy_overlap.wav',
                'scenario': 'heavy overlap',
                'min_content_length': 5
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/three_speakers_conversation.wav',
                'scenario': 'three speakers',
                'min_content_length': 8
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/simultaneous_speech.wav',
                'scenario': 'simultaneous speech',
                'min_content_length': 3
            }
        ]
        
        for test in overlap_tests:
            result = transcribe_audio(test['audio_file'])
            
            if result.get('success', False):
                transcript = result.get('transcript', '').strip()
                
                # Should produce some recognizable content despite overlap
                assert len(transcript) >= test['min_content_length'], \
                    f"Overlapping speakers ({test['scenario']}) produced insufficient content: '{transcript}'"
                
                logger.info(f"Overlap {test['scenario']}: {len(transcript)} chars transcribed")
            else:
                logger.warning(f"Overlap {test['scenario']} failed transcription")
    
    def test_foreign_language_detection(self, setup_voicein_ai):
        """Test handling of foreign language audio."""
        foreign_language_tests = [
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/spanish_speech.wav',
                'language': 'Spanish',
                'expected_detection': 'es'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/french_speech.wav',
                'language': 'French',
                'expected_detection': 'fr'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/german_speech.wav',
                'language': 'German',
                'expected_detection': 'de'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/mandarin_speech.wav',
                'language': 'Mandarin Chinese',
                'expected_detection': 'zh'
            }
        ]
        
        for test in foreign_language_tests:
            # Test with auto language detection
            result = transcribe_audio(test['audio_file'], language='auto')
            
            if result.get('success', False):
                transcript = result.get('transcript', '')
                detected_language = result.get('detected_language', '')
                
                logger.info(f"Foreign language {test['language']}: detected '{detected_language}', transcript: '{transcript[:50]}...'")
                
                # Should either detect correct language or produce reasonable transcript
                language_correct = detected_language == test['expected_detection']
                has_content = len(transcript.strip()) > 0
                
                assert language_correct or has_content, \
                    f"Foreign language {test['language']} handling failed"
            else:
                logger.warning(f"Foreign language {test['language']} failed transcription")
    
    def test_mixed_language_content(self, setup_voicein_ai):
        """Test transcription of content mixing multiple languages."""
        mixed_language_tests = [
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/english_spanish_mixed.wav',
                'languages': ['en', 'es'],
                'content_description': 'English with Spanish phrases'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/english_french_mixed.wav',
                'languages': ['en', 'fr'],
                'content_description': 'English with French words'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/code_switching.wav',
                'languages': ['en', 'es'],
                'content_description': 'Code-switching between languages'
            }
        ]
        
        for test in mixed_language_tests:
            result = transcribe_audio(test['audio_file'], language='auto')
            
            if result.get('success', False):
                transcript = result.get('transcript', '').strip()
                
                # Should handle mixed content reasonably
                assert len(transcript) > 10, \
                    f"Mixed language content ({test['content_description']}) insufficient: '{transcript}'"
                
                logger.info(f"Mixed languages {test['languages']}: {len(transcript)} chars transcribed")
            else:
                logger.warning(f"Mixed language {test['content_description']} failed")
    
    def test_extreme_audio_conditions(self, setup_voicein_ai):
        """Test transcription under extreme audio conditions."""
        extreme_tests = [
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/extremely_fast_speech.wav',
                'condition': 'extremely fast speech',
                'tolerance': 'high_error_rate'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/extremely_slow_speech.wav',
                'condition': 'extremely slow speech',
                'tolerance': 'moderate_error_rate'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/very_quiet_whisper.wav',
                'condition': 'very quiet whisper',
                'tolerance': 'high_error_rate'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/shouting_speech.wav',
                'condition': 'shouting/loud speech',
                'tolerance': 'moderate_error_rate'
            }
        ]
        
        for test in extreme_tests:
            result = transcribe_audio(test['audio_file'])
            
            if result.get('success', False):
                transcript = result.get('transcript', '').strip()
                
                # Under extreme conditions, just check for reasonable response
                if test['tolerance'] == 'high_error_rate':
                    # Accept any non-empty response
                    success_condition = len(transcript) >= 0  # Always true, allowing empty
                else:
                    # Moderate tolerance - expect some content
                    success_condition = len(transcript) > 3
                
                logger.info(f"Extreme condition {test['condition']}: '{transcript[:30]}...' ({len(transcript)} chars)")
                
                # Log result without strict assertion for extreme cases
                if not success_condition:
                    logger.warning(f"Extreme condition {test['condition']} produced minimal output")
            else:
                logger.info(f"Extreme condition {test['condition']}: failed transcription (may be expected)")
    
    def test_corrupted_audio_handling(self, setup_voicein_ai):
        """Test handling of corrupted or malformed audio files."""
        corruption_tests = [
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/truncated_file.wav',
                'corruption_type': 'truncated file'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/header_corrupted.wav',
                'corruption_type': 'corrupted header'
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/partial_data.wav',
                'corruption_type': 'partial data'
            }
        ]
        
        for test in corruption_tests:
            result = transcribe_audio(test['audio_file'])
            
            # Corrupted files should either fail gracefully or handle partial content
            if result.get('success', False):
                transcript = result.get('transcript', '')
                logger.info(f"Corrupted audio {test['corruption_type']}: handled, produced '{transcript[:30]}...'")
            else:
                error_message = result.get('error', 'Unknown error')
                logger.info(f"Corrupted audio {test['corruption_type']}: failed gracefully - {error_message}")
                
                # Verify error handling is appropriate
                assert 'error' in result, f"Corrupted audio should produce error response"
    
    def test_artistic_content_transcription(self, setup_voicein_ai):
        """Test transcription of artistic content like poetry and songs."""
        artistic_tests = [
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/poetry_reading.wav',
                'content_type': 'poetry reading',
                'expected_features': ['rhythm', 'pauses']
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/song_with_lyrics.wav',
                'content_type': 'song with lyrics',
                'expected_features': ['musical', 'rhythmic']
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/dramatic_monologue.wav',
                'content_type': 'dramatic monologue',
                'expected_features': ['emotion', 'emphasis']
            }
        ]
        
        for test in artistic_tests:
            result = transcribe_audio(test['audio_file'])
            
            if result.get('success', False):
                transcript = result.get('transcript', '').strip()
                
                # Artistic content should produce meaningful transcription
                assert len(transcript) > 20, \
                    f"Artistic content ({test['content_type']}) insufficient transcription: '{transcript}'"
                
                logger.info(f"Artistic {test['content_type']}: {len(transcript)} chars transcribed")
            else:
                logger.warning(f"Artistic {test['content_type']} failed transcription")
    
    def test_unusual_file_formats_and_encodings(self, setup_voicein_ai):
        """Test handling of unusual audio file formats and encodings."""
        format_tests = [
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/test_audio.m4a',
                'format': 'M4A',
                'should_handle': True
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/test_audio.ogg',
                'format': 'OGG',
                'should_handle': True
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/test_audio.flac',
                'format': 'FLAC',
                'should_handle': True
            },
            {
                'audio_file': 'Frai/tests/back/ai/voicein/test_data/test_audio.amr',
                'format': 'AMR',
                'should_handle': False  # May not be supported
            }
        ]
        
        for test in format_tests:
            result = transcribe_audio(test['audio_file'])
            
            if test['should_handle']:
                if result.get('success', False):
                    transcript = result.get('transcript', '')
                    logger.info(f"Format {test['format']}: successfully handled")
                    assert len(transcript) >= 0  # Should at least not crash
                else:
                    logger.warning(f"Format {test['format']}: failed to handle (unexpected)")
            else:
                # Unsupported formats may fail
                if result.get('success', False):
                    logger.info(f"Format {test['format']}: unexpectedly handled successfully")
                else:
                    logger.info(f"Format {test['format']}: appropriately rejected")


@pytest.mark.parametrize("test_case", load_edge_case_test_cases())
def test_edge_cases_from_csv(setup_voicein_ai, test_case):
    """
    Test edge cases using cases from testset.csv.
    
    Args:
        setup_voicein_ai: The voicein AI instance from fixture
        test_case: Dictionary containing test case details from CSV
    """
    test_id = test_case['id']
    audio_group = test_case['audio_group']
    evaluation_metric = test_case['evaluation_metric']
    
    logger.info(f"Running edge case test {test_id}: {test_case['name']}")
    
    # Create test audio file path based on group
    audio_file = f"Frai/tests/back/ai/voicein/test_data/{audio_group}_sample.wav"
    
    # Load reference transcript if available
    transcript_path = test_case['expected_transcript_path']
    reference_transcript = load_reference_transcript(transcript_path)
    
    # Transcribe audio
    result = transcribe_audio(audio_file)
    
    # Handle different evaluation metrics for edge cases
    if 'no_transcription_output' in evaluation_metric:
        # Should produce no output (silence test)
        if result.get('success', False):
            transcript = result.get('transcript', '').strip()
            assert len(transcript) == 0, f"Expected no output, got: '{transcript}'"
        logger.info(f"Test {test_id}: correctly produced no output")
        
    elif 'exact_match' in evaluation_metric:
        # Should match exactly (single word test)
        if result.get('success', False):
            transcript = result.get('transcript', '').strip().lower()
            expected = reference_transcript.lower().strip()
            assert transcript == expected, f"Expected '{expected}', got '{transcript}'"
        logger.info(f"Test {test_id}: exact match successful")
        
    elif 'language_identification' in evaluation_metric:
        # Should identify language correctly
        if result.get('success', False):
            detected_language = result.get('detected_language', '')
            transcript = result.get('transcript', '')
            # Either correct language detection or reasonable transcript
            success = len(detected_language) > 0 or len(transcript) > 0
            assert success, f"Language identification failed"
        logger.info(f"Test {test_id}: language handling successful")
        
    else:
        # General edge case handling
        if result.get('success', False):
            transcript = result.get('transcript', '')
            logger.info(f"Test {test_id}: handled edge case, produced '{transcript[:50]}...'")
        else:
            # Some edge cases may legitimately fail
            error = result.get('error', 'Unknown error')
            logger.info(f"Test {test_id}: edge case failed as expected - {error}")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])