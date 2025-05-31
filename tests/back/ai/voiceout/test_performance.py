"""
Performance tests for the voice output (text-to-speech) module.

This module tests the throughput, latency, and resource usage of the 
text-to-speech synthesis system under various conditions.
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

from Frai.back.ai.voiceout import (
    initialize_voiceout_system,
    get_voiceout_ai_instance,
    synthesize_speech
)

# Set up logging
logger = logging.getLogger(__name__)

# Performance test configuration
SINGLE_SYNTHESIS_TESTS = 10
BATCH_SIZE_TESTS = [1, 3, 5, 10]
CONCURRENT_THREADS = [1, 2, 4]
LATENCY_THRESHOLD_SECONDS = 10  # Maximum acceptable latency
THROUGHPUT_THRESHOLD = 0.1      # Minimum synthesized words per second


# Using fixture from conftest.py


@pytest.fixture
def performance_test_texts():
    """Provide test texts of varying complexity for performance testing."""
    return {
        'simple': [
            "Hello world",
            "Good morning",
            "Thank you",
            "How are you today",
            "This is a test"
        ],
        'medium': [
            "This is a medium length sentence for testing speech synthesis performance.",
            "The quick brown fox jumps over the lazy dog in the sunny meadow.",
            "Technology has revolutionized the way we communicate and interact.",
            "Please remember to submit your reports by the end of the week.",
            "The weather forecast predicts sunny skies and mild temperatures."
        ],
        'complex': [
            "This is a comprehensive evaluation of the text-to-speech synthesis system's ability to handle complex sentences with multiple clauses, technical terminology, and varied punctuation marks that require sophisticated processing.",
            "The interdisciplinary collaboration between artificial intelligence researchers, linguists, and audio engineers has resulted in significant advancements in natural language processing capabilities.",
            "In accordance with the established protocols and regulatory requirements, all participants must complete the mandatory training sessions before accessing the restricted laboratory facilities.",
            "The pharmaceutical company's groundbreaking research into novel therapeutic compounds has demonstrated promising results in preliminary clinical trials.",
            "Sophisticated machine learning algorithms analyze vast datasets to identify patterns, correlations, and anomalies that would be impossible for human analysts to detect manually."
        ]
    }


class TestVoiceOutPerformance:
    """Test voice output synthesis performance metrics."""
    
    def test_single_synthesis_latency(self, setup_voiceout_ai, performance_test_texts):
        """Test latency for single text synthesis."""
        latencies = []
        
        # Test with simple and medium complexity texts
        test_texts = performance_test_texts['simple'] + performance_test_texts['medium']
        
        for text in test_texts[:SINGLE_SYNTHESIS_TESTS]:
            start_time = time.time()
            result = synthesize_speech(text)
            end_time = time.time()
            
            # Verify successful synthesis
            if result.get('success', False):
                assert 'audio_data' in result
                assert result['audio_data'] is not None
                
                latency = end_time - start_time
                latencies.append(latency)
                logger.info(f"Synthesis latency for '{text[:30]}...': {latency:.2f}s")
            else:
                logger.warning(f"Synthesis failed for '{text[:30]}...': {result.get('error', 'Unknown')}")
        
        if latencies:
            # Calculate statistics
            avg_latency = statistics.mean(latencies)
            median_latency = statistics.median(latencies)
            max_latency = max(latencies)
            min_latency = min(latencies)
            
            logger.info(f"Single synthesis latency stats:")
            logger.info(f"  Average: {avg_latency:.2f}s")
            logger.info(f"  Median: {median_latency:.2f}s")
            logger.info(f"  Max: {max_latency:.2f}s")
            logger.info(f"  Min: {min_latency:.2f}s")
            
            # Performance assertions
            assert avg_latency < LATENCY_THRESHOLD_SECONDS, f"Average latency {avg_latency:.2f}s exceeds threshold {LATENCY_THRESHOLD_SECONDS}s"
            assert max_latency < LATENCY_THRESHOLD_SECONDS * 2, f"Maximum latency {max_latency:.2f}s too high"
        else:
            pytest.skip("No successful syntheses for latency testing")
    
    def test_batch_processing_performance(self, setup_voiceout_ai, performance_test_texts):
        """Test performance of batch processing."""
        for batch_size in BATCH_SIZE_TESTS:
            # Prepare batch of texts
            all_texts = performance_test_texts['simple']
            batch_texts = (all_texts * ((batch_size // len(all_texts)) + 1))[:batch_size]
            
            start_time = time.time()
            
            # Process batch (simulate batch by processing sequentially)
            successful_results = 0
            total_words = 0
            for text in batch_texts:
                result = synthesize_speech(text)
                
                if result.get('success', False):
                    assert result['audio_data'] is not None
                    successful_results += 1
                    total_words += len(text.split())
            
            end_time = time.time()
            
            if successful_results > 0:
                # Calculate performance metrics
                total_time = end_time - start_time
                throughput = successful_results / total_time
                words_per_second = total_words / total_time
                avg_time_per_synthesis = total_time / successful_results
                
                logger.info(f"Batch size {batch_size} performance:")
                logger.info(f"  Successful syntheses: {successful_results}/{batch_size}")
                logger.info(f"  Total time: {total_time:.2f}s")
                logger.info(f"  Throughput: {throughput:.3f} syntheses/second")
                logger.info(f"  Words per second: {words_per_second:.3f}")
                logger.info(f"  Average per synthesis: {avg_time_per_synthesis:.2f}s")
                
                # Performance assertions
                assert words_per_second >= THROUGHPUT_THRESHOLD, f"Words per second {words_per_second:.3f} below threshold {THROUGHPUT_THRESHOLD}"
            else:
                logger.warning(f"No successful syntheses in batch size {batch_size}")
    
    def test_concurrent_processing(self, setup_voiceout_ai, performance_test_texts):
        """Test performance under concurrent load."""
        for num_threads in CONCURRENT_THREADS:
            texts_per_thread = 2
            total_texts = num_threads * texts_per_thread
            
            # Prepare texts for threads
            all_texts = performance_test_texts['simple']
            test_texts = (all_texts * ((total_texts // len(all_texts)) + 1))[:total_texts]
            thread_texts = [test_texts[i::num_threads] for i in range(num_threads)]
            
            def process_texts_in_thread(texts):
                """Process a list of texts in a single thread."""
                results = []
                start_time = time.time()
                for text in texts:
                    result = synthesize_speech(text)
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
            
            # Count successful results
            successful_count = sum(1 for result in all_results if result.get('success', False))
            total_words = sum(len(test_texts[i].split()) for i in range(len(all_results)) if all_results[i].get('success', False))
            
            if successful_count > 0:
                # Calculate performance metrics
                total_time = end_time - start_time
                concurrent_throughput = successful_count / total_time
                concurrent_words_per_second = total_words / total_time
                avg_thread_time = statistics.mean(thread_times)
                
                logger.info(f"Concurrent processing with {num_threads} threads:")
                logger.info(f"  Successful syntheses: {successful_count}/{total_texts}")
                logger.info(f"  Total time: {total_time:.2f}s")
                logger.info(f"  Throughput: {concurrent_throughput:.3f} syntheses/second")
                logger.info(f"  Words per second: {concurrent_words_per_second:.3f}")
                logger.info(f"  Average thread time: {avg_thread_time:.2f}s")
                
                # Performance assertions - concurrent should maintain reasonable throughput
                min_concurrent_throughput = THROUGHPUT_THRESHOLD * 0.3  # Allow significant overhead
                assert concurrent_words_per_second >= min_concurrent_throughput, \
                    f"Concurrent words per second {concurrent_words_per_second:.3f} below threshold {min_concurrent_throughput}"
    
    def test_memory_usage_stability(self, setup_voiceout_ai, performance_test_texts):
        """Test that memory usage remains stable during extended processing."""
        try:
            import psutil
            import gc
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process multiple texts
            num_iterations = 8
            test_texts = (performance_test_texts['medium'] * ((num_iterations // len(performance_test_texts['medium'])) + 1))[:num_iterations]
            
            memory_samples = []
            
            for i, text in enumerate(test_texts):
                result = synthesize_speech(text)
                
                if result.get('success', False):
                    assert result['audio_data'] is not None
                
                # Sample memory usage every few iterations
                if i % 2 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_samples.append(current_memory)
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            max_memory = max(memory_samples) if memory_samples else final_memory
            memory_growth = final_memory - initial_memory
            
            logger.info(f"Memory usage test:")
            logger.info(f"  Initial memory: {initial_memory:.2f}MB")
            logger.info(f"  Final memory: {final_memory:.2f}MB")
            logger.info(f"  Max memory: {max_memory:.2f}MB")
            logger.info(f"  Memory growth: {memory_growth:.2f}MB")
            
            # Memory growth should be reasonable (less than 1GB for 8 syntheses)
            assert memory_growth < 1000, f"Memory growth {memory_growth:.2f}MB seems excessive"
            
            # Force garbage collection
            gc.collect()
            
        except ImportError:
            pytest.skip("psutil not available for memory testing")
    
    def test_text_length_performance_scaling(self, setup_voiceout_ai, performance_test_texts):
        """Test how performance scales with text length."""
        text_categories = ['simple', 'medium', 'complex']
        length_performance = []
        
        for category in text_categories:
            texts = performance_test_texts[category]
            test_text = texts[0]  # Use first text from each category
            
            start_time = time.time()
            result = synthesize_speech(test_text)
            end_time = time.time()
            
            if result.get('success', False):
                assert result['audio_data'] is not None
                
                processing_time = end_time - start_time
                word_count = len(test_text.split())
                character_count = len(test_text)
                
                length_performance.append((category, word_count, character_count, processing_time))
                
                logger.info(f"Text length performance - {category}: {word_count} words, {processing_time:.2f}s")
        
        # Check that performance scaling is reasonable with text length
        if len(length_performance) >= 2:
            for i in range(1, len(length_performance)):
                prev_cat, prev_words, prev_chars, prev_time = length_performance[i-1]
                curr_cat, curr_words, curr_chars, curr_time = length_performance[i]
                
                word_ratio = curr_words / prev_words if prev_words > 0 else 1
                time_ratio = curr_time / prev_time if prev_time > 0 else 1
                
                # Time should scale reasonably with text length
                assert time_ratio <= word_ratio * 2, \
                    f"Performance scaling issue: {word_ratio}x words led to {time_ratio}x time increase"
    
    def test_voice_parameter_performance(self, setup_voiceout_ai):
        """Test performance impact of different voice parameters."""
        base_text = "This is a test sentence for voice parameter performance evaluation."
        
        parameter_sets = [
            {'voice': 'default', 'rate': 1.0, 'pitch': 0.0},
            {'voice': 'female', 'rate': 1.0, 'pitch': 0.0},
            {'voice': 'male', 'rate': 1.0, 'pitch': 0.0},
            {'voice': 'default', 'rate': 0.5, 'pitch': 0.0},
            {'voice': 'default', 'rate': 1.5, 'pitch': 0.0},
            {'voice': 'default', 'rate': 1.0, 'pitch': 0.3},
            {'voice': 'default', 'rate': 1.0, 'pitch': -0.3}
        ]
        
        param_performance = []
        
        for params in parameter_sets:
            start_time = time.time()
            result = synthesize_speech(base_text, **params)
            end_time = time.time()
            
            if result.get('success', False):
                assert result['audio_data'] is not None
                processing_time = end_time - start_time
                param_performance.append((str(params), processing_time))
                
                logger.info(f"Parameter performance {params}: {processing_time:.2f}s")
        
        # Check that all parameter combinations complete within reasonable time
        for param_str, time_taken in param_performance:
            assert time_taken < LATENCY_THRESHOLD_SECONDS, \
                f"Parameters {param_str} too slow: {time_taken:.2f}s"
    
    def test_streaming_vs_full_synthesis_performance(self, setup_voiceout_ai):
        """Test performance difference between streaming and full synthesis."""
        test_text = "This is a comprehensive test of streaming versus full synthesis performance. The text is long enough to demonstrate any performance differences between the two modes of operation."
        
        # Full synthesis
        start_time = time.time()
        full_result = synthesize_speech(test_text, streaming=False)
        full_time = time.time() - start_time
        
        # Streaming synthesis
        start_time = time.time()
        stream_result = synthesize_speech(test_text, streaming=True)
        stream_time = time.time() - start_time
        
        if full_result.get('success', False) and stream_result.get('success', False):
            assert full_result['audio_data'] is not None
            assert 'audio_stream' in stream_result or stream_result['audio_data'] is not None
            
            logger.info(f"Synthesis mode performance:")
            logger.info(f"  Full synthesis: {full_time:.2f}s")
            logger.info(f"  Streaming synthesis: {stream_time:.2f}s")
            
            # Streaming should generally be faster for long texts
            if len(test_text.split()) > 20:
                performance_ratio = full_time / stream_time
                logger.info(f"  Performance ratio (full/stream): {performance_ratio:.2f}x")
    
    def test_repeated_synthesis_consistency(self, setup_voiceout_ai):
        """Test that repeated synthesis maintains consistent performance."""
        test_text = "This is a consistency test for voice output performance measurement."
        
        num_repetitions = 8
        processing_times = []
        
        for i in range(num_repetitions):
            start_time = time.time()
            result = synthesize_speech(test_text, voice='default', rate=1.0, pitch=0.0)
            end_time = time.time()
            
            if result.get('success', False):
                assert result['audio_data'] is not None
                processing_times.append(end_time - start_time)
        
        if len(processing_times) >= 5:
            # Performance consistency analysis
            avg_time = statistics.mean(processing_times)
            time_stdev = statistics.stdev(processing_times)
            time_cv = time_stdev / avg_time  # Coefficient of variation
            
            logger.info(f"Repeated synthesis consistency:")
            logger.info(f"  Average time: {avg_time:.2f}s")
            logger.info(f"  Time std dev: {time_stdev:.2f}s")
            logger.info(f"  Time CV: {time_cv:.3f}")
            
            # Performance should be reasonably consistent (CV < 0.4)
            assert time_cv < 0.4, f"Performance inconsistent: CV {time_cv:.3f} too high"
        else:
            pytest.skip("Insufficient successful syntheses for consistency testing")
    
    def test_warm_up_vs_steady_state_performance(self, setup_voiceout_ai):
        """Test performance difference between initial and steady-state synthesis."""
        test_text = "This is a warm-up versus steady-state performance test for speech synthesis."
        
        # First synthesis (warm-up)
        start_time = time.time()
        result1 = synthesize_speech(test_text)
        warmup_time = time.time() - start_time
        
        # Subsequent syntheses (steady state)
        steady_times = []
        for i in range(3):
            start_time = time.time()
            result = synthesize_speech(test_text)
            steady_times.append(time.time() - start_time)
            
            if result.get('success', False):
                assert result['audio_data'] is not None
        
        if result1.get('success', False) and len(steady_times) >= 2:
            avg_steady_time = statistics.mean(steady_times)
            performance_ratio = warmup_time / avg_steady_time
            
            logger.info(f"Warm-up vs steady state performance:")
            logger.info(f"  Warm-up time: {warmup_time:.2f}s")
            logger.info(f"  Steady state avg: {avg_steady_time:.2f}s")
            logger.info(f"  Performance ratio: {performance_ratio:.2f}x")
            
            # Warm-up may be slower but shouldn't be excessively so
            assert performance_ratio <= 3.0, f"Warm-up too slow: {performance_ratio:.2f}x slower than steady state"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])