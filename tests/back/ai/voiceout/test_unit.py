"""
Unit tests for individual functions and classes in the voice output module.

This module tests the core components of the text-to-speech system,
including initialization, text processing, and basic functionality.
"""

import pytest
import logging
from typing import Dict, Any
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

try:
    from back.ai.voiceout import (
        initialize_voiceout_system,
        get_voiceout_ai_instance,
        generate_speech_from_text as synthesize_speech
    )
except ImportError:
    # Mock functions for when implementation is not available
    def initialize_voiceout_system(*args, **kwargs):
        pytest.skip("Voice output system not implemented")
    
    def get_voiceout_ai_instance(*args, **kwargs):
        pytest.skip("Voice output system not implemented")
    
    def synthesize_speech(*args, **kwargs):
        pytest.skip("Voice output system not implemented")

# Set up logging
logger = logging.getLogger(__name__)


# Using fixture from conftest.py


class TestVoiceOutComponents:
    """Test individual components of the voice output system."""
    
    def test_system_initialization(self, setup_voiceout_ai):
        """Test that the voice output system initializes correctly."""
        assert setup_voiceout_ai is not None
        logger.info("Voice output system initialized successfully")
    
    def test_model_loading(self, setup_voiceout_ai):
        """Test that the voice output model loads properly."""
        assert hasattr(setup_voiceout_ai, 'model')
        assert hasattr(setup_voiceout_ai, 'processor')
        logger.info("Voice output model loaded with required components")
    
    def test_text_preprocessing(self):
        """Test text preprocessing functionality."""
        try:
            from back.ai.voiceout.preprocessing import preprocess_text
            
            test_text = "Hello world! How are you today?"
            processed = preprocess_text(test_text)
            assert isinstance(processed, str)
            assert len(processed) > 0
            
            # Test empty string
            empty_processed = preprocess_text("")
            assert isinstance(empty_processed, str)
            
            logger.info("Text preprocessing function working correctly")
        except ImportError:
            pytest.skip("Text preprocessing not implemented")
    
    def test_text_validation(self):
        """Test text input validation."""
        try:
            from back.ai.voiceout.utils import validate_text_input
            
            # Test valid text
            assert validate_text_input("Hello world") == True
            assert validate_text_input("Test with numbers 123") == True
            
            # Test invalid inputs
            assert validate_text_input(None) == False
            assert validate_text_input("") == False
            assert validate_text_input(123) == False
            
            logger.info("Text validation working correctly")
        except ImportError:
            pytest.skip("Text validation not implemented")
    
    def test_voice_parameters_validation(self):
        """Test voice synthesis parameters validation."""
        try:
            from back.ai.voiceout.utils import validate_voice_params
            
            # Test valid parameters
            valid_params = {
                'voice': 'default',
                'rate': 1.0,
                'pitch': 0.0,
                'volume': 1.0
            }
            assert validate_voice_params(valid_params) == True
            
            # Test invalid parameters
            invalid_params = {
                'voice': None,
                'rate': -1.0,
                'pitch': 5.0,
                'volume': 2.0
            }
            assert validate_voice_params(invalid_params) == False
            
            logger.info("Voice parameters validation working correctly")
        except ImportError:
            pytest.skip("Voice parameters validation not implemented")


class TestVoiceOutAPI:
    """Test the voice output API functions."""
    
    def test_synthesize_speech_basic(self, setup_voiceout_ai):
        """Test basic text-to-speech synthesis function."""
        text = "Hello world! This is a test."
        result = synthesize_speech(text)
        
        assert isinstance(result, dict)
        assert 'success' in result
        assert result['success'] == True
        assert 'audio_data' in result
        assert result['audio_data'] is not None
        
        logger.info("Basic speech synthesis working correctly")
    
    def test_synthesize_speech_with_voice(self, setup_voiceout_ai):
        """Test speech synthesis with voice specification."""
        text = "This is a test with a specific voice."
        result = synthesize_speech(text, voice='female')
        
        assert isinstance(result, dict)
        assert result['success'] == True
        assert 'audio_data' in result
        assert 'voice_used' in result
        assert result['voice_used'] == 'female'
        
        logger.info("Speech synthesis with voice selection working correctly")
    
    def test_synthesize_speech_with_rate(self, setup_voiceout_ai):
        """Test speech synthesis with custom speaking rate."""
        text = "This is a test with custom rate."
        result = synthesize_speech(text, rate=1.5)
        
        assert isinstance(result, dict)
        assert result['success'] == True
        assert 'audio_data' in result
        assert 'rate_used' in result
        assert result['rate_used'] == 1.5
        
        logger.info("Speech synthesis with custom rate working correctly")
    
    def test_synthesize_speech_empty_text(self, setup_voiceout_ai):
        """Test speech synthesis with empty text."""
        result = synthesize_speech("")
        
        assert isinstance(result, dict)
        assert result['success'] == False
        assert 'error' in result
        
        logger.info("Empty text handling working correctly")
    
    def test_synthesize_speech_none_input(self, setup_voiceout_ai):
        """Test speech synthesis with None input."""
        result = synthesize_speech(None)
        
        assert isinstance(result, dict)
        assert result['success'] == False
        assert 'error' in result
        
        logger.info("None input handling working correctly")
    
    def test_synthesize_speech_streaming(self, setup_voiceout_ai):
        """Test streaming speech synthesis."""
        text = "This is a longer text for streaming synthesis test."
        result = synthesize_speech(text, streaming=True)
        
        assert isinstance(result, dict)
        assert result['success'] == True
        assert 'audio_stream' in result
        assert result['audio_stream'] is not None
        
        logger.info("Streaming speech synthesis working correctly")
    
    def test_synthesize_speech_with_ssml(self, setup_voiceout_ai):
        """Test speech synthesis with SSML markup."""
        ssml_text = "<speak>Hello <break time='1s'/> world!</speak>"
        result = synthesize_speech(ssml_text, format='ssml')
        
        assert isinstance(result, dict)
        assert result['success'] == True
        assert 'audio_data' in result
        
        logger.info("SSML speech synthesis working correctly")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])