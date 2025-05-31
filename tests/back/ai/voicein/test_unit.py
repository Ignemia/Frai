"""
Unit tests for individual functions and classes in the voice input module.

This module tests the core components of the speech-to-text system,
including initialization, audio processing, and basic functionality.
"""

import pytest
import logging
from typing import Dict, Any
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from Frai.back.ai.voicein import (
    initialize_voicein_system,
    get_voicein_ai_instance,
    transcribe_audio
)

# Set up logging
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def setup_voicein_ai():
    """
    Initialize the voice input system once for all tests.
    """
    success = initialize_voicein_system()
    if not success:
        pytest.fail("Failed to initialize voice input system")
    
    voicein_ai = get_voicein_ai_instance()
    return voicein_ai


class TestVoiceInComponents:
    """Test individual components of the voice input system."""
    
    def test_system_initialization(self, setup_voicein_ai):
        """Test that the voice input system initializes correctly."""
        assert setup_voicein_ai is not None
        logger.info("Voice input system initialized successfully")
    
    def test_model_loading(self, setup_voicein_ai):
        """Test that the voice input model loads properly."""
        assert hasattr(setup_voicein_ai, 'model')
        assert hasattr(setup_voicein_ai, 'processor')
        logger.info("Voice input model loaded with required components")
    
    def test_audio_preprocessing(self):
        """Test audio preprocessing functionality."""
        from Frai.back.ai.voicein.preprocessing import preprocess_audio
        
        # Test with mock audio data path
        audio_path = "test_audio.wav"
        processed = preprocess_audio(audio_path)
        assert processed is not None
        
        logger.info("Audio preprocessing function working correctly")
    
    def test_audio_validation(self):
        """Test audio input validation."""
        from Frai.back.ai.voicein.utils import validate_audio_input
        
        # Test valid audio file
        assert validate_audio_input("test.wav") == True
        assert validate_audio_input("test.mp3") == True
        
        # Test invalid inputs
        assert validate_audio_input(None) == False
        assert validate_audio_input("") == False
        assert validate_audio_input("test.txt") == False
        
        logger.info("Audio validation working correctly")
    
    def test_transcription_parameters_validation(self):
        """Test transcription parameters validation."""
        from Frai.back.ai.voicein.utils import validate_transcription_params
        
        # Test valid parameters
        valid_params = {
            'language': 'en',
            'model_size': 'base',
            'temperature': 0.0
        }
        assert validate_transcription_params(valid_params) == True
        
        # Test invalid parameters
        invalid_params = {
            'language': 'invalid_lang',
            'model_size': 'huge_invalid',
            'temperature': -1.0
        }
        assert validate_transcription_params(invalid_params) == False
        
        logger.info("Transcription parameters validation working correctly")


class TestVoiceInAPI:
    """Test the voice input API functions."""
    
    def test_transcribe_audio_basic(self, setup_voicein_ai):
        """Test basic audio transcription function."""
        audio_file = "test_audio.wav"
        result = transcribe_audio(audio_file)
        
        assert isinstance(result, dict)
        assert 'success' in result
        assert result['success'] == True
        assert 'transcript' in result
        assert isinstance(result['transcript'], str)
        
        logger.info("Basic audio transcription working correctly")
    
    def test_transcribe_audio_with_language(self, setup_voicein_ai):
        """Test audio transcription with language specification."""
        audio_file = "test_audio.wav"
        result = transcribe_audio(audio_file, language='en')
        
        assert isinstance(result, dict)
        assert result['success'] == True
        assert 'transcript' in result
        assert 'language' in result
        assert result['language'] == 'en'
        
        logger.info("Audio transcription with language working correctly")
    
    def test_transcribe_audio_invalid_file(self, setup_voicein_ai):
        """Test audio transcription with invalid file."""
        result = transcribe_audio("nonexistent.wav")
        
        assert isinstance(result, dict)
        assert result['success'] == False
        assert 'error' in result
        
        logger.info("Invalid audio file handling working correctly")
    
    def test_transcribe_audio_none_input(self, setup_voicein_ai):
        """Test audio transcription with None input."""
        result = transcribe_audio(None)
        
        assert isinstance(result, dict)
        assert result['success'] == False
        assert 'error' in result
        
        logger.info("None input handling working correctly")
    
    def test_transcribe_audio_with_confidence(self, setup_voicein_ai):
        """Test audio transcription with confidence scores."""
        audio_file = "test_audio.wav"
        result = transcribe_audio(audio_file, include_confidence=True)
        
        assert isinstance(result, dict)
        assert result['success'] == True
        assert 'transcript' in result
        assert 'confidence' in result
        assert 0.0 <= result['confidence'] <= 1.0
        
        logger.info("Audio transcription with confidence working correctly")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])