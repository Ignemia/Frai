"""
Performance tests for the sentiment analysis module.

This module tests the throughput and latency of the sentiment analysis system
in single-text and batch processing scenarios.
"""

import pytest
import logging
import time
import statistics
import os
import sys
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# Performance test configuration
SINGLE_TEXT_TESTS = 100
BATCH_SIZE_TESTS = [10, 50, 100, 500]
CONCURRENT_THREADS = [1, 2, 4, 8]
LATENCY_THRESHOLD_MS = 1000  # Maximum acceptable latency in milliseconds
THROUGHPUT_THRESHOLD = 10    # Minimum texts per second


# Using fixture from conftest.py


@pytest.fixture
def test_texts():
    """Provide a variety of test texts for performance testing."""
    return [
        "I love this product!",
        "This is terrible.",
        "The weather is nice today.",
        "Amazing experience!",
        "Could be better.",
        "Absolutely fantastic!",
        "Not impressed.",
        "Pretty good overall.",
        "Disappointing results.",
        "Excellent work!",
        "This is a longer text that contains multiple sentences and should test the system's ability to handle more complex input efficiently.",
        "Short text.",
        "Mixed feelings about this - some good points but also some bad aspects that need improvement.",
        "Perfect!",
        "Awful experience.",
        "Neutral statement about facts.",
        "Very very very very very positive sentiment with lots of enthusiasm and excitement!",
        "Negative sentiment with disappointment and frustration clearly expressed.",
        "Technical jargon and domain-specific terminology for testing specialized content.",
        "Question? Does this work well for interrogative sentences and various punctuation marks!"
    ]


class TestSentimentPerformance:
    """Test sentiment analysis performance metrics."""
    
    def test_single_text_latency(self, setup_sentiment_ai, test_texts):
        """Test latency for single text analysis."""
        latencies = []
        
        for text in test_texts[:SINGLE_TEXT_TESTS]:
            start_time = time.time()
            result = analyze_sentiment(text)
            end_time = time.time()
            
            # Verify successful analysis
            assert result.get('success', False), f"Failed to analyze: {text}"
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        
        logger.info(f"Single text latency stats:")
        logger.info(f"  Average: {avg_latency:.2f}ms")
        logger.info(f"  Median: {median_latency:.2f}ms")
        logger.info(f"  Max: {max_latency:.2f}ms")
        logger.info(f"  Min: {min_latency:.2f}ms")
        
        # Performance assertions
        assert avg_latency < LATENCY_THRESHOLD_MS, f"Average latency {avg_latency:.2f}ms exceeds threshold {LATENCY_THRESHOLD_MS}ms"
        assert max_latency < LATENCY_THRESHOLD_MS * 3, f"Maximum latency {max_latency:.2f}ms too high"
    
    def test_batch_processing_performance(self, setup_sentiment_ai, test_texts):
        """Test performance of batch processing."""
        for batch_size in BATCH_SIZE_TESTS:
            # Prepare batch
            batch_texts = (test_texts * ((batch_size // len(test_texts)) + 1))[:batch_size]
            
            start_time = time.time()
            
            # Process batch (if batch function available)
            try:
                results = analyze_sentiment_batch(batch_texts)
                end_time = time.time()
                
                # Verify all results
                assert len(results) == batch_size, f"Expected {batch_size} results, got {len(results)}"
                for i, result in enumerate(results):
                    assert result.get('success', False), f"Batch item {i} failed: {result.get('error', 'Unknown error')}"
                
            except (ImportError, AttributeError):
                # If batch function not available, simulate with individual calls
                results = []
                for text in batch_texts:
                    result = analyze_sentiment(text)
                    results.append(result)
                end_time = time.time()
            
            # Calculate performance metrics
            total_time = end_time - start_time
            throughput = batch_size / total_time
            avg_time_per_text = total_time / batch_size * 1000  # in ms
            
            logger.info(f"Batch size {batch_size} performance:")
            logger.info(f"  Total time: {total_time:.2f}s")
            logger.info(f"  Throughput: {throughput:.2f} texts/second")
            logger.info(f"  Average per text: {avg_time_per_text:.2f}ms")
            
            # Performance assertions
            assert throughput >= THROUGHPUT_THRESHOLD, f"Throughput {throughput:.2f} texts/sec below threshold {THROUGHPUT_THRESHOLD}"
    
    def test_concurrent_processing(self, setup_sentiment_ai, test_texts):
        """Test performance under concurrent load."""
        for num_threads in CONCURRENT_THREADS:
            # Prepare test data
            texts_per_thread = 10
            total_texts = num_threads * texts_per_thread
            all_texts = (test_texts * ((total_texts // len(test_texts)) + 1))[:total_texts]
            
            # Split texts among threads
            thread_texts = [all_texts[i::num_threads] for i in range(num_threads)]
            
            def process_texts_in_thread(texts):
                """Process a list of texts in a single thread."""
                results = []
                start_time = time.time()
                for text in texts:
                    result = analyze_sentiment(text)
                    results.append(result)
                end_time = time.time()
                return results, end_time - start_time
            
            # Execute concurrent processing
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(process_texts_in_thread, texts) for texts in thread_texts]
                all_results = []
                thread_times = []
                
                for future in as_completed(futures):
                    results, thread_time = future.result()
                    all_results.extend(results)
                    thread_times.append(thread_time)
            
            end_time = time.time()
            
            # Verify all results
            assert len(all_results) == total_texts, f"Expected {total_texts} results, got {len(all_results)}"
            for i, result in enumerate(all_results):
                assert result.get('success', False), f"Concurrent test item {i} failed: {result.get('error', 'Unknown error')}"
            
            # Calculate performance metrics
            total_time = end_time - start_time
            concurrent_throughput = total_texts / total_time
            avg_thread_time = statistics.mean(thread_times)
            
            logger.info(f"Concurrent processing with {num_threads} threads:")
            logger.info(f"  Total texts: {total_texts}")
            logger.info(f"  Total time: {total_time:.2f}s")
            logger.info(f"  Throughput: {concurrent_throughput:.2f} texts/second")
            logger.info(f"  Average thread time: {avg_thread_time:.2f}s")
            
            # Performance assertions - concurrent should maintain reasonable throughput
            min_concurrent_throughput = THROUGHPUT_THRESHOLD * 0.5  # Allow some overhead
            assert concurrent_throughput >= min_concurrent_throughput, \
                f"Concurrent throughput {concurrent_throughput:.2f} below threshold {min_concurrent_throughput}"
    
    def test_memory_usage_stability(self, setup_sentiment_ai, test_texts):
        """Test that memory usage remains stable during extended processing."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process a large number of texts
        num_iterations = 1000
        texts_to_process = (test_texts * ((num_iterations // len(test_texts)) + 1))[:num_iterations]
        
        memory_samples = []
        
        for i, text in enumerate(texts_to_process):
            result = analyze_sentiment(text)
            assert result.get('success', False), f"Memory test item {i} failed"
            
            # Sample memory usage every 100 iterations
            if i % 100 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_samples.append(current_memory)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        max_memory = max(memory_samples)
        memory_growth = final_memory - initial_memory
        
        logger.info(f"Memory usage test:")
        logger.info(f"  Initial memory: {initial_memory:.2f}MB")
        logger.info(f"  Final memory: {final_memory:.2f}MB")
        logger.info(f"  Max memory: {max_memory:.2f}MB")
        logger.info(f"  Memory growth: {memory_growth:.2f}MB")
        
        # Memory growth should be reasonable (less than 500MB for 1000 texts)
        assert memory_growth < 500, f"Memory growth {memory_growth:.2f}MB seems excessive"
        
        # Force garbage collection
        gc.collect()
    
    def test_text_length_performance_scaling(self, setup_sentiment_ai):
        """Test how performance scales with text length."""
        text_lengths = [10, 50, 100, 500, 1000, 5000]  # Number of words
        base_text = "This is a test sentence with positive sentiment. "
        
        length_performance = []
        
        for length in text_lengths:
            # Create text of specified length
            words_per_sentence = 10
            sentences_needed = length // words_per_sentence
            test_text = base_text * sentences_needed
            
            # Measure performance
            start_time = time.time()
            result = analyze_sentiment(test_text)
            end_time = time.time()
            
            assert result.get('success', False), f"Failed to analyze text of length {length}"
            
            processing_time = (end_time - start_time) * 1000  # ms
            length_performance.append((length, processing_time))
            
            logger.info(f"Text length {length} words: {processing_time:.2f}ms")
        
        # Check that performance scaling is reasonable
        # Processing time should not grow exponentially with text length
        for i in range(1, len(length_performance)):
            prev_length, prev_time = length_performance[i-1]
            curr_length, curr_time = length_performance[i]
            
            length_ratio = curr_length / prev_length
            time_ratio = curr_time / prev_time
            
            # Time should not grow more than 3x faster than length
            assert time_ratio <= length_ratio * 3, \
                f"Performance scaling issue: {length_ratio}x length increase led to {time_ratio}x time increase"
    
    def test_repeated_analysis_consistency(self, setup_sentiment_ai):
        """Test that repeated analysis of the same text maintains consistent performance."""
        test_text = "This is a consistently positive message that should be analyzed multiple times."
        num_repetitions = 100
        
        processing_times = []
        sentiment_scores = []
        
        for i in range(num_repetitions):
            start_time = time.time()
            result = analyze_sentiment(test_text)
            end_time = time.time()
            
            assert result.get('success', False), f"Repetition {i} failed"
            
            processing_time = (end_time - start_time) * 1000  # ms
            processing_times.append(processing_time)
            sentiment_scores.append(result.get('sentiment_score'))
        
        # Performance consistency
        avg_time = statistics.mean(processing_times)
        time_stdev = statistics.stdev(processing_times)
        time_cv = time_stdev / avg_time  # Coefficient of variation
        
        # Result consistency
        score_stdev = statistics.stdev(sentiment_scores)
        
        logger.info(f"Repeated analysis consistency:")
        logger.info(f"  Average time: {avg_time:.2f}ms")
        logger.info(f"  Time std dev: {time_stdev:.2f}ms")
        logger.info(f"  Time CV: {time_cv:.3f}")
        logger.info(f"  Score std dev: {score_stdev:.4f}")
        
        # Performance should be consistent (CV < 0.5)
        assert time_cv < 0.5, f"Performance inconsistent: CV {time_cv:.3f} too high"
        
        # Results should be very consistent (std dev < 0.01)
        assert score_stdev < 0.01, f"Result inconsistent: std dev {score_stdev:.4f} too high"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])