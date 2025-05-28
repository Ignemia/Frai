"""
Accuracy tests for the sentiment analysis module.

This module validates that model output ranges match expected sentiment intervals
and tests the accuracy of sentiment predictions against known benchmarks.
"""

import pytest
import logging
import csv
import os
import sys
from typing import Dict, Any, List

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# Try to import real functions, fallback to mocks
def mock_function(*args, **kwargs):
    """Mock function that skips tests when called."""
    import pytest
    pytest.skip(f"Function not implemented yet")

try:
    from Frai.back.ai.sentiment import (
        initialize_sentiment_system,
        get_sentiment_ai_instance,
        analyze_sentiment
    )
except (ImportError, ModuleNotFoundError):
    initialize_sentiment_system = mock_function
    get_sentiment_ai_instance = mock_function
    analyze_sentiment = mock_function

class MockAIInstance:
    """Mock AI instance for when real implementation doesn't exist."""
    
    def __init__(self, module_name):
        self.module_name = module_name
        self.model = None
        self.processor = None
        self.tokenizer = None

# Set up logging
logger = logging.getLogger(__name__)

# Path to the test set CSV file
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), 'testset.csv')


def load_accuracy_test_cases() -> List[Dict[str, str]]:
    """Load test cases specifically for accuracy testing."""
    test_cases = []
    try:
        with open(TEST_SET_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter for automated evaluation tests
                if row['evaluation_method'] == 'automated':
                    test_cases.append(row)
        logger.info(f"Loaded {len(test_cases)} accuracy test cases")
        return test_cases
    except Exception as e:
        logger.error(f"Failed to load test cases: {e}")
        return []


def parse_interval(interval_str: str) -> tuple:
    """
    Parse interval notation to extract bounds and inclusion flags.
    
    Args:
        interval_str: String like "<0.5;0.8>", "(0.2;0.6)", etc.
        
    Returns:
        Tuple of (lower_bound, upper_bound, left_closed, right_closed)
    """
    interval_str = interval_str.strip()
    
    # Determine left bracket type
    if interval_str.startswith('<'):
        left_closed = True
        start_idx = 1
    elif interval_str.startswith('('):
        left_closed = False
        start_idx = 1
    else:
        raise ValueError(f"Invalid interval format: {interval_str}")
    
    # Determine right bracket type
    if interval_str.endswith('>'):
        right_closed = True
        end_idx = -1
    elif interval_str.endswith(')'):
        right_closed = False
        end_idx = -1
    else:
        raise ValueError(f"Invalid interval format: {interval_str}")
    
    # Extract the middle part and split on semicolon
    middle = interval_str[start_idx:end_idx]
    parts = middle.split(';')
    
    if len(parts) != 2:
        raise ValueError(f"Invalid interval format: {interval_str}")
    
    lower_bound = float(parts[0])
    upper_bound = float(parts[1])
    
    return lower_bound, upper_bound, left_closed, right_closed


def score_in_interval(score: float, interval_str: str) -> bool:
    """
    Check if a sentiment score falls within the expected interval.
    
    Args:
        score: The sentiment score to check
        interval_str: The interval string to check against
        
    Returns:
        Boolean indicating if score is in interval
    """
    lower, upper, left_closed, right_closed = parse_interval(interval_str)
    
    # Check lower bound
    if left_closed:
        lower_ok = score >= lower
    else:
        lower_ok = score > lower
    
    # Check upper bound
    if right_closed:
        upper_ok = score <= upper
    else:
        upper_ok = score < upper
    
    return lower_ok and upper_ok


def calculate_word_error_rate(predicted: str, expected: str) -> float:
    """
    Calculate word error rate between predicted and expected text.
    
    Args:
        predicted: Predicted text output
        expected: Expected text output
        
    Returns:
        Word error rate as a float between 0 and 1
    """
    predicted_words = predicted.lower().split()
    expected_words = expected.lower().split()
    
    if len(expected_words) == 0:
        return 0.0 if len(predicted_words) == 0 else 1.0
    
    # Simple word-level comparison
    max_len = max(len(predicted_words), len(expected_words))
    errors = 0
    
    for i in range(max_len):
        pred_word = predicted_words[i] if i < len(predicted_words) else ""
        exp_word = expected_words[i] if i < len(expected_words) else ""
        
        if pred_word != exp_word:
            errors += 1
    
    return errors / len(expected_words)


def validate_sentiment_range(score: float, expected_range: str) -> bool:
    """
    Validate that sentiment score is within expected range and reasonable bounds.
    
    Args:
        score: Sentiment score to validate
        expected_range: Expected range string
        
    Returns:
        Boolean indicating if validation passed
    """
    # Check basic bounds
    if not isinstance(score, (int, float)):
        return False
    
    if not -1.0 <= score <= 1.0:
        return False
    
    # Check if score is in expected interval
    return score_in_interval(score, expected_range)


# Using fixture from conftest.py


class TestSentimentAccuracy:
    """Test sentiment analysis accuracy against expected ranges."""
    
    @pytest.mark.parametrize("test_case", load_accuracy_test_cases())
    def test_sentiment_range_accuracy(self, setup_sentiment_ai, test_case):
        """
        Test that sentiment scores fall within expected ranges.
        
        Args:
            setup_sentiment_ai: The sentiment AI instance from fixture
            test_case: Dictionary containing test case details from CSV
        """
        test_id = test_case['id']
        input_text = test_case['input_text']
        expected_range = test_case['expected_range']
        
        logger.info(f"Running accuracy test {test_id}: {test_case['name']}")
        
        # Analyze sentiment
        result = analyze_sentiment(input_text)
        
        # Verify successful analysis
        assert result.get('success', False), f"Test {test_id} failed: {result.get('error', 'Unknown error')}"
        
        # Extract sentiment score
        sentiment_score = result.get('sentiment_score')
        assert sentiment_score is not None, f"Test {test_id} failed: No sentiment score in response"
        
        # Validate score and range
        assert validate_sentiment_range(sentiment_score, expected_range), \
            f"Test {test_id} failed: Score {sentiment_score} not in expected range {expected_range}"
        
        logger.info(f"Test {test_id} passed: Score {sentiment_score} in range {expected_range}")
    
    def test_positive_sentiment_accuracy(self, setup_sentiment_ai):
        """Test accuracy on clearly positive statements."""
        positive_test_cases = [
            ("I absolutely love this amazing product!", 0.6, 1.0),
            ("This is the best day of my life!", 0.7, 1.0),
            ("Fantastic work, truly exceptional!", 0.6, 1.0),
            ("Outstanding performance, couldn't be better!", 0.7, 1.0),
            ("Wonderful experience, highly recommended!", 0.5, 1.0),
            ("Brilliant! Absolutely perfect in every way!", 0.8, 1.0)
        ]
        
        for text, min_score, max_score in positive_test_cases:
            result = analyze_sentiment(text)
            assert result.get('success', False), f"Failed to analyze: {text}"
            
            score = result.get('sentiment_score')
            assert min_score <= score <= max_score, \
                f"Expected positive sentiment between {min_score} and {max_score} for: {text}, got {score}"
            
        logger.info("Positive sentiment accuracy test passed")
    
    def test_negative_sentiment_accuracy(self, setup_sentiment_ai):
        """Test accuracy on clearly negative statements."""
        negative_test_cases = [
            ("This is absolutely terrible and disappointing.", -1.0, -0.6),
            ("I hate everything about this horrible experience.", -1.0, -0.7),
            ("Worst product ever, completely useless!", -1.0, -0.7),
            ("Disgusting and appalling, avoid at all costs!", -1.0, -0.8),
            ("Awful service, total waste of time and money.", -1.0, -0.5),
            ("Completely broken, doesn't work at all!", -1.0, -0.6)
        ]
        
        for text, min_score, max_score in negative_test_cases:
            result = analyze_sentiment(text)
            assert result.get('success', False), f"Failed to analyze: {text}"
            
            score = result.get('sentiment_score')
            assert min_score <= score <= max_score, \
                f"Expected negative sentiment between {min_score} and {max_score} for: {text}, got {score}"
            
        logger.info("Negative sentiment accuracy test passed")
    
    def test_neutral_sentiment_accuracy(self, setup_sentiment_ai):
        """Test accuracy on neutral statements."""
        neutral_test_cases = [
            ("The weather is cloudy today.", -0.1, 0.1),
            ("The meeting is scheduled for 3 PM.", -0.1, 0.1),
            ("The document contains 50 pages.", -0.1, 0.1),
            ("The building has 10 floors.", -0.1, 0.1),
            ("Water boils at 100 degrees Celsius.", -0.1, 0.1),
            ("The conference room is on the second floor.", -0.1, 0.1)
        ]
        
        for text, min_score, max_score in neutral_test_cases:
            result = analyze_sentiment(text)
            assert result.get('success', False), f"Failed to analyze: {text}"
            
            score = result.get('sentiment_score')
            assert min_score <= score <= max_score, \
                f"Expected neutral sentiment between {min_score} and {max_score} for: {text}, got {score}"
            
        logger.info("Neutral sentiment accuracy test passed")
    
    def test_confidence_score_accuracy(self, setup_sentiment_ai):
        """Test confidence scores are reasonable and correlated with clarity."""
        test_cases = [
            ("I love this!", "clear_positive"),
            ("I hate this!", "clear_negative"),
            ("This is okay I guess.", "ambiguous"),
            ("It's not bad but not great either.", "ambiguous"),
            ("The weather is nice today.", "clear_neutral")
        ]
        
        for text, clarity_type in test_cases:
            result = analyze_sentiment(text, include_confidence=True)
            
            if result.get('success', False) and 'confidence' in result:
                confidence = result.get('confidence')
                assert 0.0 <= confidence <= 1.0, f"Confidence score out of range: {confidence}"
                
                # Clear statements should have higher confidence
                if clarity_type.startswith('clear'):
                    assert confidence >= 0.6, f"Expected high confidence for clear statement: {text}, got {confidence}"
                elif clarity_type == 'ambiguous':
                    assert confidence <= 0.8, f"Expected lower confidence for ambiguous statement: {text}, got {confidence}"
        
        logger.info("Confidence score accuracy test passed")
    
    def test_domain_specific_accuracy(self, setup_sentiment_ai):
        """Test accuracy on domain-specific language."""
        domain_test_cases = [
            # Medical domain
            ("The patient's recovery is progressing remarkably well.", 0.4, 0.8),
            ("The patient's condition has deteriorated significantly.", -0.8, -0.4),
            
            # Financial domain
            ("The stock market showed strong gains today.", 0.3, 0.7),
            ("The company reported massive losses this quarter.", -0.7, -0.3),
            
            # Technical domain
            ("The system performance has improved dramatically.", 0.4, 0.8),
            ("Critical errors detected in the application.", -0.6, -0.2),
            
            # Customer service
            ("Thank you for your excellent customer service.", 0.5, 0.9),
            ("Your support team was completely unhelpful.", -0.8, -0.4)
        ]
        
        for text, min_score, max_score in domain_test_cases:
            result = analyze_sentiment(text)
            assert result.get('success', False), f"Failed to analyze: {text}"
            
            score = result.get('sentiment_score')
            assert min_score <= score <= max_score, \
                f"Expected sentiment between {min_score} and {max_score} for domain text: {text}, got {score}"
        
        logger.info("Domain-specific accuracy test passed")
    
    def test_sentiment_intensity_accuracy(self, setup_sentiment_ai):
        """Test that sentiment intensity is accurately captured."""
        intensity_test_cases = [
            # Weak positive to strong positive
            ("It's okay.", 0.0, 0.3),
            ("It's good.", 0.2, 0.5),
            ("It's great!", 0.4, 0.7),
            ("It's amazing!", 0.6, 0.9),
            ("It's absolutely incredible!", 0.7, 1.0),
            
            # Weak negative to strong negative
            ("It's not great.", -0.3, 0.0),
            ("It's bad.", -0.5, -0.2),
            ("It's terrible!", -0.7, -0.4),
            ("It's horrible!", -0.9, -0.6),
            ("It's absolutely disgusting!", -1.0, -0.7)
        ]
        
        for text, min_score, max_score in intensity_test_cases:
            result = analyze_sentiment(text)
            assert result.get('success', False), f"Failed to analyze: {text}"
            
            score = result.get('sentiment_score')
            assert min_score <= score <= max_score, \
                f"Expected sentiment intensity between {min_score} and {max_score} for: {text}, got {score}"
        
        logger.info("Sentiment intensity accuracy test passed")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])