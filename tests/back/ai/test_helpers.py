"""
Test helpers for handling missing imports in TDD approach.

This module provides utilities to gracefully handle import errors
when the actual implementation doesn't exist yet, along with
comprehensive testing utilities for AI modules.
"""

import pytest
import logging
import csv
import os
import time
import statistics
from typing import Dict, Any, List, Optional, Union
import difflib

logger = logging.getLogger(__name__)


def skip_if_not_implemented(func):
    """
    Decorator to skip tests if the required module/function is not implemented.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ImportError, ModuleNotFoundError, AttributeError) as e:
            pytest.skip(f"Function not implemented yet: {e}")
    return wrapper


def mock_ai_function_response(success=True, **kwargs):
    """
    Create a mock response structure for AI functions.
    """
    response = {
        'success': success,
        **kwargs
    }
    if not success:
        response['error'] = 'Function not implemented'
    return response


class MockAIInstance:
    """Mock AI instance for when real implementation doesn't exist."""
    
    def __init__(self, module_name):
        self.module_name = module_name
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.scheduler = None
    
    def __getattr__(self, name):
        def mock_method(*args, **kwargs):
            pytest.skip(f"{self.module_name}.{name} not implemented yet")
        return mock_method


def safe_import_ai_module(module_path, fallback_instance=None):
    """
    Safely import an AI module, returning a mock if not available.
    """
    try:
        parts = module_path.split('.')
        module = __import__(module_path, fromlist=[parts[-1]])
        return module
    except (ImportError, ModuleNotFoundError):
        logger.warning(f"Module {module_path} not found, using mock")
        if fallback_instance:
            return fallback_instance
        return None


def safe_import_ai_function(module_path, function_name):
    """
    Safely import an AI function, returning a mock if not available.
    """
    try:
        module = safe_import_ai_module(module_path)
        if module:
            return getattr(module, function_name)
    except (AttributeError, ImportError, ModuleNotFoundError):
        pass
    
    def mock_function(*args, **kwargs):
        pytest.skip(f"Function {module_path}.{function_name} not implemented yet")
    
    return mock_function


def expect_implementation_error(func, *args, **kwargs):
    """
    Helper to test that a function raises an implementation error.
    """
    try:
        result = func(*args, **kwargs)
        # If we get here, either the function is implemented or returns a failure
        if isinstance(result, dict) and not result.get('success', True):
            return result
        pytest.fail("Expected function to not be implemented yet")
    except (ImportError, ModuleNotFoundError, AttributeError):
        pytest.skip("Function not implemented yet - this is expected")


class AITestCase:
    """Base class for AI module test cases with common utilities."""
    
    @staticmethod
    def assert_ai_response(response, should_succeed=True):
        """Assert the structure and success of an AI response."""
        assert isinstance(response, dict), "Response should be a dictionary"
        assert 'success' in response, "Response should have 'success' field"
        
        if should_succeed:
            assert response['success'] == True, f"Expected success, got: {response.get('error', 'Unknown error')}"
        else:
            assert response['success'] == False, "Expected failure"
            assert 'error' in response, "Failed response should have error message"
    
    @staticmethod
    def assert_valid_score(score, min_val=-1.0, max_val=1.0):
        """Assert that a score is within valid range."""
        assert isinstance(score, (int, float)), "Score should be numeric"
        assert min_val <= score <= max_val, f"Score {score} outside valid range [{min_val}, {max_val}]"
    
    @staticmethod
    def assert_valid_image(image_data):
        """Assert that image data is valid."""
        assert image_data is not None, "Image data should not be None"
        # Additional image validation can be added when PIL is available
    
    @staticmethod
    def assert_valid_audio(audio_data):
        """Assert that audio data is valid."""
        assert audio_data is not None, "Audio data should not be None"
        # Additional audio validation can be added when audio libraries are available
    
    @staticmethod
    def assert_valid_text(text):
        """Assert that text output is valid."""
        assert isinstance(text, str), "Text should be a string"
        assert len(text.strip()) > 0, "Text should not be empty"
    
    def calculate_word_error_rate(self, reference: str, hypothesis: str) -> float:
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
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()
        
        if t1 == t2:
            return 1.0
        
        # Calculate character-level similarity
        char_similarity = difflib.SequenceMatcher(None, t1, t2).ratio()
        
        # Calculate word-level similarity
        words1 = t1.split()
        words2 = t2.split()
        word_similarity = difflib.SequenceMatcher(None, words1, words2).ratio()
        
        # Combine both measures
        return (char_similarity + word_similarity) / 2
    
    def measure_performance(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Measure performance metrics for a function call."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        return {
            'result': result,
            'success': success,
            'error': error,
            'execution_time': end_time - start_time,
            'memory_delta': end_memory - start_memory if start_memory and end_memory else None,
            'start_time': start_time,
            'end_time': end_time
        }
    
    def _get_memory_usage(self) -> Optional[int]:
        """Get current memory usage in bytes."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return None
    
    def load_test_data_csv(self, csv_path: str) -> List[Dict[str, str]]:
        """Load test data from CSV file."""
        test_data = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    test_data.append(row)
            logger.info(f"Loaded {len(test_data)} test cases from {csv_path}")
            return test_data
        except Exception as e:
            logger.error(f"Failed to load test data from {csv_path}: {e}")
            return []
    
    def filter_test_cases(self, test_cases: List[Dict[str, str]], 
                         filter_key: str, filter_values: List[str]) -> List[Dict[str, str]]:
        """Filter test cases based on key-value criteria."""
        filtered = []
        for case in test_cases:
            if any(value in case.get(filter_key, '') for value in filter_values):
                filtered.append(case)
        return filtered
    
    def calculate_consistency_score(self, results: List[Any]) -> float:
        """Calculate consistency score across multiple results."""
        if len(results) < 2:
            return 1.0
        
        # For text results
        if all(isinstance(r, str) for r in results):
            similarities = []
            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    similarity = self.calculate_text_similarity(results[i], results[j])
                    similarities.append(similarity)
            return statistics.mean(similarities)
        
        # For numeric results
        elif all(isinstance(r, (int, float)) for r in results):
            if all(r == results[0] for r in results):
                return 1.0
            std_dev = statistics.stdev(results)
            mean_val = statistics.mean(results)
            if mean_val == 0:
                return 0.0
            cv = std_dev / abs(mean_val)  # Coefficient of variation
            return max(0.0, 1.0 - cv)
        
        # For other types, check exact equality
        else:
            unique_results = len(set(str(r) for r in results))
            return 1.0 if unique_results == 1 else 0.0
    
    def assert_performance_threshold(self, performance_data: Dict[str, Any], 
                                   max_time: float = None, max_memory: int = None):
        """Assert that performance data meets thresholds."""
        if max_time is not None:
            execution_time = performance_data.get('execution_time', 0)
            assert execution_time <= max_time, f"Execution time {execution_time:.2f}s exceeds threshold {max_time}s"
        
        if max_memory is not None and performance_data.get('memory_delta') is not None:
            memory_delta = performance_data['memory_delta']
            assert memory_delta <= max_memory, f"Memory usage {memory_delta} bytes exceeds threshold {max_memory} bytes"
    
    def create_test_batch(self, test_cases: List[Dict], batch_size: int) -> List[List[Dict]]:
        """Create batches of test cases for batch processing tests."""
        batches = []
        for i in range(0, len(test_cases), batch_size):
            batch = test_cases[i:i + batch_size]
            batches.append(batch)
        return batches


def load_reference_file(file_path: str) -> str:
    """Load reference content from file."""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        else:
            logger.warning(f"Reference file not found: {file_path}")
            return ""
    except Exception as e:
        logger.warning(f"Failed to load reference file {file_path}: {e}")
        return ""


def create_mock_image(width: int = 256, height: int = 256, color: tuple = (128, 128, 128)):
    """Create a mock image for testing purposes."""
    try:
        from PIL import Image
        return Image.new('RGB', (width, height), color)
    except ImportError:
        # Return a mock object that won't cause immediate failures
        class MockImage:
            def __init__(self, w, h, c):
                self.size = (w, h)
                self.mode = 'RGB'
            def resize(self, size):
                return MockImage(size[0], size[1], (128, 128, 128))
            def convert(self, mode):
                return self
        return MockImage(width, height, color)


def create_mock_audio(duration: float = 1.0, sample_rate: int = 16000):
    """Create mock audio data for testing purposes."""
    try:
        import numpy as np
        num_samples = int(duration * sample_rate)
        # Generate simple sine wave
        t = np.linspace(0, duration, num_samples)
        frequency = 440  # A4 note
        audio_data = np.sin(2 * np.pi * frequency * t)
        return audio_data
    except ImportError:
        # Return a simple list as mock audio
        num_samples = int(duration * sample_rate)
        return [0.0] * num_samples


class TestDataGenerator:
    """Generate test data for various AI testing scenarios."""
    
    @staticmethod
    def generate_text_samples(lengths: List[int] = None) -> List[str]:
        """Generate text samples of various lengths."""
        if lengths is None:
            lengths = [10, 50, 100, 500]
        
        base_text = ("The quick brown fox jumps over the lazy dog. "
                    "This pangram contains every letter of the English alphabet. "
                    "It is commonly used for testing typography and keyboards. "
                    "Speech recognition systems also use this phrase for evaluation. ") * 20
        
        samples = []
        for length in lengths:
            if length <= len(base_text):
                samples.append(base_text[:length])
            else:
                # Repeat text to reach desired length
                repeats = (length // len(base_text)) + 1
                extended_text = base_text * repeats
                samples.append(extended_text[:length])
        
        return samples
    
    @staticmethod
    def generate_test_prompts(categories: List[str] = None) -> Dict[str, List[str]]:
        """Generate test prompts by category."""
        if categories is None:
            categories = ['simple', 'complex', 'creative', 'technical']
        
        prompt_sets = {
            'simple': [
                "Hello, how are you?",
                "What is the weather like?",
                "Can you help me?",
                "Thank you very much.",
                "Good morning."
            ],
            'complex': [
                "Explain the difference between machine learning and artificial intelligence.",
                "Describe the process of photosynthesis in plants.",
                "What are the implications of quantum computing for cybersecurity?",
                "Analyze the economic factors that led to the 2008 financial crisis.",
                "Compare and contrast different philosophical approaches to ethics."
            ],
            'creative': [
                "Write a short poem about autumn leaves.",
                "Create a story about a robot learning to paint.",
                "Imagine a conversation between the sun and the moon.",
                "Describe a futuristic city from the perspective of a time traveler.",
                "Write a song about the ocean."
            ],
            'technical': [
                "Implement a binary search algorithm in Python.",
                "Explain how RESTful APIs work.",
                "Describe the TCP/IP protocol stack.",
                "What is the difference between SQL and NoSQL databases?",
                "How does encryption protect data transmission?"
            ]
        }
        
        return {cat: prompt_sets[cat] for cat in categories if cat in prompt_sets}


def benchmark_function(func, iterations: int = 10, *args, **kwargs) -> Dict[str, float]:
    """Benchmark a function over multiple iterations."""
    times = []
    
    for _ in range(iterations):
        start_time = time.time()
        try:
            func(*args, **kwargs)
        except Exception:
            pass  # Continue benchmarking even if function fails
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        'mean_time': statistics.mean(times),
        'median_time': statistics.median(times),
        'min_time': min(times),
        'max_time': max(times),
        'std_dev': statistics.stdev(times) if len(times) > 1 else 0.0,
        'iterations': iterations
    }