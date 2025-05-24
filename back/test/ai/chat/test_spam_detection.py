import pytest
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
orchestrator_path = project_root / "orchestrator"
sys.path.insert(0, str(orchestrator_path))

try:
    from orchestrator.chatmod import ChatModerator
except ImportError:
    sys.path.insert(0, str(orchestrator_path / "chatmod"))
    from __init__ import ChatModerator

@pytest.fixture
def moderator():
    """Create a ChatModerator instance for testing spam detection."""
    return ChatModerator()

# Basic Spam Tests
def test_repeated_characters_spam(moderator):
    message = "aaaaaaaaaaa" # 11 'a's
    # First test with check_spam_patterns directly
    is_spam, reason = moderator.check_spam_patterns(message)
    assert is_spam
    assert "Message matches spam pattern" in reason
    
    # Then test with moderate_message
    result = moderator.moderate_message(message)
    assert "spam" in result["filters_triggered"]
    assert len(result["warnings"]) > 0  # Should have at least one warning

def test_all_caps_spam(moderator):
    message = "THIS IS ALL CAPS TEXT OVER FIVE CHARS"  # Ensure it meets the 5+ chars requirement
    # First test with check_spam_patterns directly
    is_spam, reason = moderator.check_spam_patterns(message)
    assert is_spam
    assert "Message matches spam pattern" in reason
    
    # Then test with moderate_message
    result = moderator.moderate_message(message)
    assert "spam" in result["filters_triggered"]
    assert len(result["warnings"]) > 0  # Should have at least one warning

def test_excessive_repetition_spam(moderator):
    message = "word " * 21 # 21 repetitions
    # First test with check_spam_patterns directly
    is_spam, reason = moderator.check_spam_patterns(message)
    assert is_spam
    assert "Message matches spam pattern" in reason
    
    # Then test with moderate_message
    result = moderator.moderate_message(message)
    assert "spam" in result["filters_triggered"]
    assert len(result["warnings"]) > 0  # Should have at least one warning

def test_no_spam_normal_message(moderator):
    message = "This is a normal, non-spammy message."
    result = moderator.moderate_message(message)
    assert "spam" not in result["filters_triggered"]

# Edge Cases for Spam
def test_repeated_characters_just_below_threshold(moderator):
    message = "aaaaaaaaa" # 9 'a's
    result = moderator.moderate_message(message)
    assert "spam" not in result["filters_triggered"]

def test_all_caps_just_below_threshold(moderator):
    message = "FOUR" # 4 caps
    result = moderator.moderate_message(message)
    assert "spam" not in result["filters_triggered"]

def test_excessive_repetition_just_below_threshold(moderator):
    message = "word " * 19 # 19 repetitions
    result = moderator.moderate_message(message)
    assert "spam" not in result["filters_triggered"]

def test_empty_message_spam_check(moderator):
    # Validation should catch this before spam check, but good to be aware
    message = ""
    result = moderator.moderate_message(message)
    assert result["error"] is not None # Should be caught by validation
    assert "spam" not in result["filters_triggered"]

def test_long_message_no_spam_patterns(moderator):
    message = "This is a very long message that does not contain any obvious spam patterns, " + \
              "it just goes on and on to test length without triggering specific filters like " + \
              "repetition or all caps. It uses normal sentence structure and vocabulary."
    result = moderator.moderate_message(message)
    assert "spam" not in result["filters_triggered"]

# Spam patterns in combination with other content
def test_spam_with_normal_text_repeated_chars(moderator):
    message = "Hello this is a test message aaaaaaaaaaa and more text."
    # First test with check_spam_patterns directly
    is_spam, reason = moderator.check_spam_patterns(message)
    assert is_spam
    assert "Message matches spam pattern" in reason
    
    # Then test with moderate_message
    result = moderator.moderate_message(message)
    assert "spam" in result["filters_triggered"]
    assert len(result["warnings"]) > 0  # Should have at least one warning

def test_spam_with_normal_text_all_caps(moderator):
    message = "Hello ABCDEFGHIJK LONGER TEXT IN CAPS and more text."  # Longer CAPS section
    # First test with check_spam_patterns directly
    is_spam, reason = moderator.check_spam_patterns(message)
    assert is_spam
    assert "Message matches spam pattern" in reason
    
    # Then test with moderate_message
    result = moderator.moderate_message(message)
    assert "spam" in result["filters_triggered"]
    assert len(result["warnings"]) > 0  # Should have at least one warning
