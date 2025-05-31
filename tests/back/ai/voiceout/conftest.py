import pytest
import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# Import test helpers
try:
    from tests.back.ai.test_helpers import (
        safe_import_ai_function,
        MockAIInstance,
        expect_implementation_error
    )
except ImportError:
    pytest.skip("Test helpers not available", allow_module_level=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Safe imports of voiceout functions
initialize_voiceout_system = safe_import_ai_function('back.ai.voiceout', 'initialize_voiceout_system')
get_voiceout_ai_instance = safe_import_ai_function('back.ai.voiceout', 'get_voiceout_ai_instance')
synthesize_speech = safe_import_ai_function('back.ai.voiceout', 'synthesize_speech')


@pytest.fixture(scope="session")
def setup_voiceout_ai():
    """
    Initialize the voice output system once for all tests.
    This fixture is shared across all voiceout test modules.
    """
    try:
        logger.info("Initializing voice output system for tests...")
        success = initialize_voiceout_system()
        
        if not success:
            pytest.fail("Failed to initialize voice output system")
            return None
        
        # Get the voiceout AI instance
        voiceout_ai = get_voiceout_ai_instance()
        logger.info("Voice output system initialized successfully")
        
        yield voiceout_ai
        
        logger.info("Test session complete. Voice output AI instance cleanup.")
        
    except Exception as e:
        logger.warning(f"Could not initialize voiceout system: {e}")
        yield MockAIInstance("voiceout")





@pytest.fixture
def sample_texts():
    """Provide sample texts for speech synthesis testing."""
    return {
        'simple': [
            "Hello world! How are you today?",
            "This is a simple test sentence.",
            "Good morning, how can I help you?"
        ],
        'complex': [
            "This is a longer paragraph containing multiple sentences. It tests the system's ability to handle extended text with proper pacing and breath control.",
            "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet at least once.",
            "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole filled with the ends of worms and an oozy smell."
        ],
        'emotional': [
            "I'm absolutely thrilled and excited about this wonderful opportunity!",
            "I'm deeply saddened by this tragic news and my heart goes out to everyone affected.",
            "Are you seriously telling me this is the best you can do? I'm extremely disappointed!"
        ],
        'questions': [
            "Are you coming to the party tonight?",
            "What time should we meet for dinner?",
            "Where exactly is the venue located?",
            "How long will the meeting last?"
        ],
        'technical': [
            "The API endpoint returns JSON responses with OAuth 2.0 authentication.",
            "Configure the SSL certificate for HTTPS connections using TLS version 1.3.",
            "The database query optimization improved performance by 300 percent."
        ],
        'numbers': [
            "The meeting is scheduled for March 15th 2024 at 3:30 PM.",
            "Please call 555-123-4567 for more information.",
            "The total cost is $1,234.56 including tax.",
            "The temperature today reached 25.7 degrees Celsius."
        ]
    }


@pytest.fixture
def voice_parameters():
    """Provide standard voice synthesis parameters for testing."""
    return {
        'default': {
            'voice': 'default',
            'rate': 1.0,
            'pitch': 0.0,
            'volume': 1.0
        },
        'fast': {
            'voice': 'default',
            'rate': 1.5,
            'pitch': 0.0,
            'volume': 1.0
        },
        'slow': {
            'voice': 'default',
            'rate': 0.5,
            'pitch': 0.0,
            'volume': 1.0
        },
        'high_pitch': {
            'voice': 'default',
            'rate': 1.0,
            'pitch': 0.3,
            'volume': 1.0
        },
        'low_pitch': {
            'voice': 'default',
            'rate': 1.0,
            'pitch': -0.3,
            'volume': 1.0
        },
        'female': {
            'voice': 'female',
            'rate': 1.0,
            'pitch': 0.0,
            'volume': 1.0
        },
        'male': {
            'voice': 'male',
            'rate': 1.0,
            'pitch': 0.0,
            'volume': 1.0
        }
    }


@pytest.fixture
def synthesis_modes():
    """Provide different synthesis modes for testing."""
    return {
        'full': {
            'mode': 'full',
            'streaming': False
        },
        'streaming': {
            'mode': 'streaming',
            'streaming': True,
            'chunk_size': 1024
        },
        'high_quality': {
            'mode': 'full',
            'quality': 'high',
            'sample_rate': 44100
        },
        'low_quality': {
            'mode': 'full',
            'quality': 'low',
            'sample_rate': 16000
        }
    }


@pytest.fixture
def expected_audio_outputs():
    """Provide expected audio output paths for validation."""
    return {
        'simple_sentence': "Frai/tests/back/ai/voiceout/expected/simple_sentence.wav",
        'long_paragraph': "Frai/tests/back/ai/voiceout/expected/long_paragraph.wav",
        'emotional_positive': "Frai/tests/back/ai/voiceout/expected/emotional_positive.wav",
        'question_intonation': "Frai/tests/back/ai/voiceout/expected/question_intonation.wav",
        'technical_terms': "Frai/tests/back/ai/voiceout/expected/technical_terms.wav"
    }


@pytest.fixture
def supported_languages():
    """Provide list of supported languages for TTS testing."""
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


@pytest.fixture
def ssml_examples():
    """Provide SSML markup examples for testing."""
    return {
        'basic_break': "<speak>Hello <break time='1s'/> world!</speak>",
        'emphasis': "<speak>This is <emphasis level='strong'>very important</emphasis>.</speak>",
        'prosody': "<speak>I can speak <prosody rate='fast'>very fast</prosody> or <prosody rate='slow'>very slow</prosody>.</speak>",
        'voice_change': "<speak>Normal voice. <voice name='female'>Female voice.</voice> Back to normal.</speak>",
        'mixed': "<speak>Welcome! <break time='0.5s'/> Today is <emphasis>Friday</emphasis> and the weather is <prosody rate='slow'>beautiful</prosody>.</speak>"
    }