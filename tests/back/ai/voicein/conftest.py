import pytest
import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# Import test helpers
try:
    from Frai.tests.back.ai.test_helpers import (
        safe_import_ai_function,
        MockAIInstance,
        expect_implementation_error
    )
except ImportError:
    pytest.skip("Test helpers not available", allow_module_level=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Safe imports of voicein functions
initialize_voicein_system = safe_import_ai_function('Frai.back.ai.voicein', 'initialize_voicein_system')
get_voicein_ai_instance = safe_import_ai_function('Frai.back.ai.voicein', 'get_voicein_ai_instance')
transcribe_audio = safe_import_ai_function('Frai.back.ai.voicein', 'transcribe_audio')


@pytest.fixture(scope="session")
def setup_voicein_ai():
    """
    Initialize the voice input system once for all tests.
    This fixture is shared across all voicein test modules.
    """
    try:
        logger.info("Initializing voice input system for tests...")
        success = initialize_voicein_system()
        
        if not success:
            pytest.fail("Failed to initialize voice input system")
            return None
        
        # Get the voicein AI instance
        voicein_ai = get_voicein_ai_instance()
        logger.info("Voice input system initialized successfully")
        
        yield voicein_ai
        
        logger.info("Test session complete. Voice input AI instance cleanup.")
        
    except Exception as e:
        logger.warning(f"Could not initialize voicein system: {e}")
        yield MockAIInstance("voicein")


@pytest.fixture
def voicein_response():
    """
    Fixture to generate responses from the voice input system.
    Provides a function that can be called with audio inputs.
    """
    def _transcribe_audio(audio_file, **kwargs):
        """
        Transcribe audio file to text.
        
        Args:
            audio_file: Audio file path or data
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary
        """
        try:
            response = transcribe_audio(audio_file, **kwargs)
            return response
        except Exception as e:
            return {
                'success': False,
                'error': f'Transcription failed: {str(e)}'
            }
    
    return _transcribe_audio


@pytest.fixture
def sample_audio_files():
    """Provide sample audio file paths for testing."""
    return {
        'clean': [
            "Frai/tests/back/ai/voicein/test_data/clear_short.wav",
            "Frai/tests/back/ai/voicein/test_data/clear_medium.wav",
            "Frai/tests/back/ai/voicein/test_data/clear_long.wav"
        ],
        'noisy': [
            "Frai/tests/back/ai/voicein/test_data/noisy_light.wav",
            "Frai/tests/back/ai/voicein/test_data/noisy_heavy.wav"
        ],
        'phone_call': [
            "Frai/tests/back/ai/voicein/test_data/phone_call.wav"
        ],
        'accented': [
            "Frai/tests/back/ai/voicein/test_data/british_accent.wav",
            "Frai/tests/back/ai/voicein/test_data/southern_accent.wav",
            "Frai/tests/back/ai/voicein/test_data/non_native.wav"
        ],
        'speed_variants': [
            "Frai/tests/back/ai/voicein/test_data/fast_speech.wav",
            "Frai/tests/back/ai/voicein/test_data/slow_speech.wav"
        ],
        'volume_variants': [
            "Frai/tests/back/ai/voicein/test_data/whispered.wav",
            "Frai/tests/back/ai/voicein/test_data/loud_speech.wav"
        ],
        'specialized': [
            "Frai/tests/back/ai/voicein/test_data/technical_jargon.wav",
            "Frai/tests/back/ai/voicein/test_data/medical_terms.wav",
            "Frai/tests/back/ai/voicein/test_data/legal_language.wav"
        ],
        'edge_cases': [
            "Frai/tests/back/ai/voicein/test_data/silence.wav",
            "Frai/tests/back/ai/voicein/test_data/single_word.wav"
        ]
    }


@pytest.fixture
def transcription_parameters():
    """Provide standard transcription parameters for testing."""
    return {
        'basic': {
            'language': 'en',
            'model_size': 'base'
        },
        'high_accuracy': {
            'language': 'en',
            'model_size': 'large',
            'temperature': 0.0
        },
        'multilingual': {
            'language': 'auto',
            'model_size': 'medium'
        },
        'fast': {
            'language': 'en',
            'model_size': 'tiny',
            'temperature': 0.2
        }
    }


@pytest.fixture
def expected_transcripts():
    """Provide expected transcript content for validation."""
    return {
        'clear_short': "Hello world, this is a test.",
        'technical_jargon': "The API endpoint returns JSON responses with OAuth authentication.",
        'numbers_dates': "The meeting is scheduled for March 15th 2024 at 3:30 PM.",
        'single_word': "Hello",
        'silence': ""
    }


@pytest.fixture
def supported_languages():
    """Provide list of supported languages for testing."""
    return [
        'en',  # English
        'es',  # Spanish
        'fr',  # French
        'de',  # German
        'it',  # Italian
        'pt',  # Portuguese
        'ru',  # Russian
        'ja',  # Japanese
        'ko',  # Korean
        'zh'   # Chinese
    ]