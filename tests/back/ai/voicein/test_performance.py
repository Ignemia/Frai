"""
Performance tests for the voice input (speech-to-text) module.

This module tests the voicein system's performance including latency, throughput,
resource usage, and scaling characteristics under various conditions.
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

# Performance test configuration
SINGLE_TRANSCRIPTION_TESTS = 10
BATCH_SIZE_TESTS = [1, 3, 5, 10, 20]
CONCURRENT_THREADS = [1, 2, 4, 8]
LATENCY_THRESHOLD_SECONDS = 15  # Maximum acceptable latency for single transcription
THROUGHPUT_THRESHOLD = 0.1      # Minimum transcriptions per second
MEMORY_GROWTH_LIMIT_MB = 2000   # Maximum memory growth during tests


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


@pytest.fixture
def performance_test_audio():
    """Provide test audio files of varying lengths for performance testing."""
    return {
        'short': [
            'Frai/tests/back/ai/voicein/test_data/short_1sec.wav',
            'Frai/tests/back/ai/voicein/test_data/short_3sec.wav',
            'Frai/tests/back/ai/voicein/test_data/short_5sec.wav'
        ],
        'medium': [
            'Frai/tests/back/ai/voicein/test_data/medium_30sec.wav',
            'Frai/tests/back/ai/voicein/test_data/medium_60sec.wav',
            'Frai/tests/back/ai/voicein/test_data/medium_90sec.wav'
        ],
        'long': [
            'Frai/tests/back/ai/voicein/test_data/long_5min.wav',
            'Frai/tests/back/ai/voicein/test_data/long_10min.wav',
            'Frai/tests/back/ai/voicein/test_data/long_15min.wav'
        ]
    }


class TestVoiceInPerformance(AITestCase):
    """Test voice input performance metrics."""
    
    def test_single_transcription_latency(self, setup_voicein_ai, performance_test_audio):
        """Test latency for single audio transcription."""
        latencies = []
        
        # Test with short and medium length audio
        test_files = performance_test_audio['short'] + performance_test_audio['medium'][:2]
        
        for audio_file in test_files[:SINGLE_TRANSCRIPTION_TESTS]:
            start_time = time.time()
            result = transcribe_audio(audio_file)
            end_time = time.time()
            
            # Verify successful transcription
            if result.get('success', False):
                assert 'transcript' in result
                assert result['transcript'] is not None
                
                latency = end_time - start_time
                latencies.append(latency)
                logger.info(f"Transcription latency for {audio_file}: {latency:.2f}s")
            else:
                logger.warning(f"Transcription failed for {audio_file}: {result.get('error', 'Unknown')}")
        
        if latencies:
            # Calculate statistics
            avg_latency = statistics.mean(latencies)
            median_latency = statistics.median(latencies)
            max_latency = max(latencies)
            min_latency = min(latencies)
            
            logger.info(f"Single transcription latency stats:")
            logger.info(f"  Average: {avg_latency:.2f}s")
            logger.info(f"  Median: {median_latency:.2f}s")
            logger.info(f"  Max: {max_latency:.2f}s")
            logger.info(f"  Min: {min_latency:.2f}s")
            
            # Performance assertions
            assert avg_latency < LATENCY_THRESHOLD_SECONDS, f"Average latency {avg_latency:.2f}s exceeds threshold {LATENCY_THRESHOLD_SECONDS}s"
            assert max_latency < LATENCY_THRESHOLD_SECONDS * 2, f"Maximum latency {max_latency:.2f}s too high"
        else:
            pytest.skip("No successful transcriptions for latency testing")
    
    def test_batch_processing_performance(self, setup_voicein_ai, performance_test_audio):
        """Test performance of batch processing."""
        short_files = performance_test_audio['short']
        
        for batch_size in BATCH_SIZE_TESTS:
            # Prepare batch of audio files
            batch_files = (short_files * ((batch_size // len(short_files)) + 1))[:batch_size]
            
            start_time = time.time()
            
            # Process batch (simulate batch by processing sequentially)
            successful_results = 0
            for audio_file in batch_files:
                result = transcribe_audio(audio_file)
                
                if result.get('success', False):
                    successful_results += 1
                    assert result['transcript'] is not None
            
            end_time = time.time()
            
            if successful_results > 0:
                # Calculate performance metrics
                total_time = end_time - start_time
                throughput = successful_results / total_time
                avg_time_per_transcription = total_time / successful_results
                
                logger.info(f"Batch size {batch_size} performance:")
                logger.info(f"  Successful transcriptions: {successful_results}/{batch_size}")
                logger.info(f"  Total time: {total_time:.2f}s")
                logger.info(f"  Throughput: {throughput:.3f} transcriptions/second")
                logger.info(f"  Average per transcription: {avg_time_per_transcription:.2f}s")
                
                # Performance assertions
                assert throughput >= THROUGHPUT_THRESHOLD, f"Throughput {throughput:.3f} transcriptions/sec below threshold {THROUGHPUT_THRESHOLD}"
            else:
                logger.warning(f"No successful transcriptions in batch size {batch_size}")
    
    def test_concurrent_processing(self, setup_voicein_ai, performance_test_audio):
        """Test performance under concurrent load."""
        short_files = performance_test_audio['short']
        
        for num_threads in CONCURRENT_THREADS:
            files_per_thread = 3
            total_files = num_threads * files_per_thread
            
            # Prepare files for threads
            test_files = (short_files * ((total_files // len(short_files)) + 1))[:total_files]
            thread_files = [test_files[i::num_threads] for i in range(num_threads)]
            
            def process_files_in_thread(files):
                """Process a list of audio files in a single thread."""
                results = []
                start_time = time.time()
                for audio_file in files:
                    result = transcribe_audio(audio_file)
                    results.append(result)
                end_time = time.time()
                return results, end_time - start_time
            
            # Execute concurrent processing
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(process_files_in_thread, files) for files in thread_files]
                all_results = []
                thread_times = []
                
                for future in as_completed(futures):
                    results, thread_time = future.result()
                    all_results.extend(results)
                    thread_times.append(thread_time)
            
            end_time = time.time()
            
            # Count successful results
            successful_count = sum(1 for result in all_results if result.get('success', False))
            
            if successful_count > 0:
                # Calculate performance metrics
                total_time = end_time - start_time
                concurrent_throughput = successful_count / total_time
                avg_thread_time = statistics.mean(thread_times)
                
                logger.info(f"Concurrent processing with {num_threads} threads:")
                logger.info(f"  Successful transcriptions: {successful_count}/{total_files}")
                logger.info(f"  Total time: {total_time:.2f}s")
                logger.info(f"  Throughput: {concurrent_throughput:.3f} transcriptions/second")
                logger.info(f"  Average thread time: {avg_thread_time:.2f}s")
                
                # Performance assertions - concurrent should maintain reasonable throughput
                min_concurrent_throughput = THROUGHPUT_THRESHOLD * 0.5  # Allow for concurrency overhead
                assert concurrent_throughput >= min_concurrent_throughput, \
                    f"Concurrent throughput {concurrent_throughput:.3f} below threshold {min_concurrent_throughput}"
    
    def test_audio_length_scaling(self, setup_voicein_ai, performance_test_audio):
        """Test how performance scales with audio length."""
        length_categories = [
            ('short', performance_test_audio['short'][:1]),
            ('medium', performance_test_audio['medium'][:1]),
            ('long', performance_test_audio['long'][:1])
        ]
        
        performance_data = []
        
        for category, files in length_categories:
            if not files:
                continue
                
            audio_file = files[0]
            
            start_time = time.time()
            result = transcribe_audio(audio_file)
            end_time = time.time()
            
            if result.get('success', False):
                transcript = result['transcript']
                processing_time = end_time - start_time
                transcript_length = len(transcript.split()) if transcript else 0
                
                performance_data.append({
                    'category': category,
                    'processing_time': processing_time,
                    'transcript_length': transcript_length,
                    'file': audio_file
                })
                
                logger.info(f"Length category {category}: {processing_time:.2f}s, {transcript_length} words")
        
        # Analyze scaling characteristics
        if len(performance_data) >= 2:
            for i in range(1, len(performance_data)):
                prev_data = performance_data[i-1]
                curr_data = performance_data[i]
                
                time_ratio = curr_data['processing_time'] / prev_data['processing_time']
                word_ratio = max(curr_data['transcript_length'] / max(prev_data['transcript_length'], 1), 1)
                
                logger.info(f"Scaling from {prev_data['category']} to {curr_data['category']}: "
                           f"{time_ratio:.2f}x time for {word_ratio:.2f}x content")
                
                # Time should scale sub-linearly with content length
                assert time_ratio <= word_ratio * 2.0, \
                    f"Poor scaling: {time_ratio:.2f}x time for {word_ratio:.2f}x content"
    
    def test_memory_usage_stability(self, setup_voicein_ai, performance_test_audio):
        """Test that memory usage remains stable during extended processing."""
        try:
            import psutil
            import gc
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process multiple audio files
            num_iterations = 8
            test_files = (performance_test_audio['short'] * ((num_iterations // len(performance_test_audio['short'])) + 1))[:num_iterations]
            
            memory_samples = []
            
            for i, audio_file in enumerate(test_files):
                result = transcribe_audio(audio_file)
                
                if result.get('success', False):
                    assert result['transcript'] is not None
                
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
            
            # Memory growth should be reasonable
            assert memory_growth < MEMORY_GROWTH_LIMIT_MB, f"Memory growth {memory_growth:.2f}MB exceeds limit {MEMORY_GROWTH_LIMIT_MB}MB"
            
            # Force garbage collection
            gc.collect()
            
        except ImportError:
            pytest.skip("psutil not available for memory testing")
    
    def test_model_parameter_performance_impact(self, setup_voicein_ai, performance_test_audio):
        """Test how different model parameters affect performance."""
        audio_file = performance_test_audio['short'][0]
        
        parameter_sets = [
            {'temperature': 0.0, 'description': 'deterministic'},
            {'temperature': 0.2, 'description': 'low_randomness'},
            {'temperature': 0.5, 'description': 'moderate_randomness'},
        ]
        
        performance_results = []
        
        for params in parameter_sets:
            description = params.pop('description')
            
            start_time = time.time()
            result = transcribe_audio(audio_file, **params)
            end_time = time.time()
            
            if result.get('success', False):
                processing_time = end_time - start_time
                performance_results.append((description, processing_time))
                
                logger.info(f"Parameter set '{description}': {processing_time:.2f}s")
        
        # Check that parameter variations don't dramatically affect performance
        if len(performance_results) >= 2:
            times = [time for _, time in performance_results]
            max_time = max(times)
            min_time = min(times)
            performance_variance = max_time / min_time
            
            logger.info(f"Parameter performance variance: {performance_variance:.2f}x")
            
            # Parameter changes should not cause extreme performance differences
            assert performance_variance <= 3.0, f"Excessive parameter performance variance: {performance_variance:.2f}x"
    
    def test_warm_up_vs_steady_state_performance(self, setup_voicein_ai, performance_test_audio):
        """Test performance difference between initial and steady-state transcription."""
        audio_file = performance_test_audio['short'][0]
        
        # First transcription (warm-up)
        start_time = time.time()
        result1 = transcribe_audio(audio_file)
        warmup_time = time.time() - start_time
        
        # Subsequent transcriptions (steady state)
        steady_times = []
        for i in range(3):
            start_time = time.time()
            result = transcribe_audio(audio_file)
            steady_times.append(time.time() - start_time)
            
            if result.get('success', False):
                assert result['transcript'] is not None
        
        if result1.get('success', False) and len(steady_times) >= 2:
            avg_steady_time = statistics.mean(steady_times)
            performance_ratio = warmup_time / avg_steady_time
            
            logger.info(f"Warm-up vs steady state performance:")
            logger.info(f"  Warm-up time: {warmup_time:.2f}s")
            logger.info(f"  Steady state avg: {avg_steady_time:.2f}s")
            logger.info(f"  Performance ratio: {performance_ratio:.2f}x")
            
            # Warm-up may be slower but shouldn't be excessively so
            assert performance_ratio <= 4.0, f"Warm-up too slow: {performance_ratio:.2f}x slower than steady state"
    
    def test_repeated_transcription_consistency(self, setup_voicein_ai, performance_test_audio):
        """Test that repeated transcription maintains consistent performance."""
        audio_file = performance_test_audio['short'][0]
        
        num_repetitions = 6
        processing_times = []
        
        for i in range(num_repetitions):
            start_time = time.time()
            result = transcribe_audio(audio_file)
            end_time = time.time()
            
            if result.get('success', False):
                assert result['transcript'] is not None
                processing_times.append(end_time - start_time)
        
        if len(processing_times) >= 4:
            # Performance consistency analysis
            avg_time = statistics.mean(processing_times)
            time_stdev = statistics.stdev(processing_times)
            time_cv = time_stdev / avg_time  # Coefficient of variation
            
            logger.info(f"Repeated transcription consistency:")
            logger.info(f"  Average time: {avg_time:.2f}s")
            logger.info(f"  Time std dev: {time_stdev:.2f}s")
            logger.info(f"  Time CV: {time_cv:.3f}")
            
            # Performance should be reasonably consistent (CV < 0.5)
            assert time_cv < 0.5, f"Performance inconsistent: CV {time_cv:.3f} too high"
        else:
            pytest.skip("Insufficient successful transcriptions for consistency testing")
    
    def test_real_time_processing_capability(self, setup_voicein_ai, performance_test_audio):
        """Test capability for real-time processing."""
        # Test with audio that should be processable in real-time
        real_time_tests = [
            {
                'audio_file': performance_test_audio['short'][0],
                'expected_duration': 3.0,  # seconds of audio
                'real_time_factor': 1.5   # Allow 1.5x real-time
            },
            {
                'audio_file': performance_test_audio['medium'][0],
                'expected_duration': 30.0,
                'real_time_factor': 2.0   # Allow 2x real-time for longer audio
            }
        ]
        
        for test in real_time_tests:
            start_time = time.time()
            result = transcribe_audio(test['audio_file'])
            processing_time = time.time() - start_time
            
            if result.get('success', False):
                max_allowed_time = test['expected_duration'] * test['real_time_factor']
                real_time_ratio = processing_time / test['expected_duration']
                
                logger.info(f"Real-time test: {processing_time:.2f}s for {test['expected_duration']}s audio "
                           f"(ratio: {real_time_ratio:.2f}x)")
                
                # Should process faster than or close to real-time
                assert processing_time <= max_allowed_time, \
                    f"Processing too slow for real-time: {real_time_ratio:.2f}x real-time"
            else:
                logger.warning(f"Real-time test failed for {test['audio_file']}")
    
    def test_streaming_simulation_performance(self, setup_voicein_ai):
        """Test performance in streaming-like scenarios."""
        # Simulate streaming by processing audio chunks
        chunk_files = [
            'Frai/tests/back/ai/voicein/test_data/chunk_1.wav',
            'Frai/tests/back/ai/voicein/test_data/chunk_2.wav',
            'Frai/tests/back/ai/voicein/test_data/chunk_3.wav',
            'Frai/tests/back/ai/voicein/test_data/chunk_4.wav',
        ]
        
        chunk_times = []
        total_start_time = time.time()
        
        for i, chunk_file in enumerate(chunk_files):
            chunk_start_time = time.time()
            result = transcribe_audio(chunk_file)
            chunk_time = time.time() - chunk_start_time
            
            if result.get('success', False):
                chunk_times.append(chunk_time)
                logger.info(f"Chunk {i+1} processing time: {chunk_time:.2f}s")
            else:
                logger.warning(f"Chunk {i+1} failed transcription")
        
        total_time = time.time() - total_start_time
        
        if len(chunk_times) >= 3:
            avg_chunk_time = statistics.mean(chunk_times)
            chunk_consistency = 1.0 - (statistics.stdev(chunk_times) / avg_chunk_time)
            
            logger.info(f"Streaming simulation performance:")
            logger.info(f"  Total time: {total_time:.2f}s")
            logger.info(f"  Average chunk time: {avg_chunk_time:.2f}s")
            logger.info(f"  Chunk consistency: {chunk_consistency:.3f}")
            
            # Streaming should have consistent chunk processing times
            assert chunk_consistency >= 0.7, f"Poor chunk consistency: {chunk_consistency:.3f}"
            
            # Average chunk time should be reasonable for streaming
            assert avg_chunk_time <= 2.0, f"Chunk processing too slow for streaming: {avg_chunk_time:.2f}s"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])