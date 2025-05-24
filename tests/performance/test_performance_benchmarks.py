"""
Performance tests for the image generation system.

These tests measure performance characteristics like memory usage,
execution time, throughput, and resource utilization.
"""

import time
import pytest
import psutil
import threading
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tests.utils.mock_helpers import (
    MockDiffusionPipeline, MockTorch, MockMemoryManager,
    create_mock_torch_device
)
from tests.fixtures.test_data import (
    sample_prompts, sample_configs, performance_test_cases
)


class PerformanceMonitor:
    """Helper class to monitor system performance during tests."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.peak_memory = None
        
    def start_monitoring(self):
        """Start monitoring performance metrics."""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        
    def stop_monitoring(self):
        """Stop monitoring and record final metrics."""
        self.end_time = time.time()
        self.end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
    def get_execution_time(self):
        """Get total execution time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
        
    def get_memory_usage(self):
        """Get memory usage statistics."""
        return {
            'start_mb': self.start_memory,
            'end_mb': self.end_memory,
            'peak_mb': self.peak_memory,
            'delta_mb': self.end_memory - self.start_memory if self.end_memory and self.start_memory else None
        }


@pytest.mark.performance
class TestImageGenerationPerformance:
    """Performance tests for image generation system."""
    
    def setup_method(self):
        """Set up performance test environment."""
        self.monitor = PerformanceMonitor()
        self.mock_torch = MockTorch()
        self.mock_pipeline = MockDiffusionPipeline()
        
    @pytest.mark.benchmark
    def test_single_image_generation_time(self):
        """Test performance of single image generation."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=1):
                self.monitor.start_monitoring()
                
                # Simulate image generation
                result = self.mock_pipeline("test prompt")
                
                self.monitor.stop_monitoring()
                
                # Performance assertions
                execution_time = self.monitor.get_execution_time()
                assert execution_time < 2.0, f"Single image generation took {execution_time}s, expected < 2.0s"
                
                memory_stats = self.monitor.get_memory_usage()
                assert memory_stats['delta_mb'] < 100, f"Memory increase {memory_stats['delta_mb']}MB too high"
    
    @pytest.mark.benchmark
    def test_batch_image_generation_throughput(self):
        """Test throughput of batch image generation."""
        batch_size = 10
        prompts = sample_prompts()[:batch_size]
        
        with patch('torch.cuda.is_available', return_value=True):
            self.monitor.start_monitoring()
            
            results = []
            for prompt in prompts:
                result = self.mock_pipeline(prompt)
                results.append(result)
            
            self.monitor.stop_monitoring()
            
            execution_time = self.monitor.get_execution_time()
            throughput = len(prompts) / execution_time
            
            assert throughput > 5.0, f"Throughput {throughput} images/sec too low"
            assert len(results) == batch_size
    
    @pytest.mark.stress
    def test_memory_management_under_load(self):
        """Test memory management under high load."""
        num_iterations = 50
        
        with patch('torch.cuda.is_available', return_value=True):
            memory_manager = MockMemoryManager()
            
            self.monitor.start_monitoring()
            
            for i in range(num_iterations):
                # Simulate memory allocation
                memory_manager.allocate_memory(100)  # 100MB
                
                # Simulate processing
                self.mock_pipeline(f"test prompt {i}")
                
                # Simulate cleanup every 10 iterations
                if i % 10 == 0:
                    memory_manager.cleanup()
            
            self.monitor.stop_monitoring()
            
            memory_stats = self.monitor.get_memory_usage()
            
            # Memory should not grow unbounded
            assert memory_stats['delta_mb'] < 500, "Memory leak detected"
            assert memory_manager.get_allocated_memory() < 1000, "Too much memory allocated"
    
    @pytest.mark.stress
    def test_concurrent_image_generation(self):
        """Test performance under concurrent load."""
        num_threads = 5
        images_per_thread = 10
        
        def generate_images(thread_id):
            """Generate images in a thread."""
            results = []
            for i in range(images_per_thread):
                prompt = f"thread {thread_id} image {i}"
                result = self.mock_pipeline(prompt)
                results.append(result)
            return results
        
        with patch('torch.cuda.is_available', return_value=True):
            self.monitor.start_monitoring()
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(generate_images, i) 
                    for i in range(num_threads)
                ]
                
                results = []
                for future in futures:
                    thread_results = future.result()
                    results.extend(thread_results)
            
            self.monitor.stop_monitoring()
            
            execution_time = self.monitor.get_execution_time()
            total_images = num_threads * images_per_thread
            throughput = total_images / execution_time
            
            assert len(results) == total_images
            assert throughput > 10.0, f"Concurrent throughput {throughput} too low"
    
    @pytest.mark.benchmark
    def test_model_loading_time(self):
        """Test model loading performance."""
        with patch('torch.cuda.is_available', return_value=True):
            self.monitor.start_monitoring()
            
            # Simulate model loading
            model = MockDiffusionPipeline()
            model.load_model("test_model")
            
            self.monitor.stop_monitoring()
            
            loading_time = self.monitor.get_execution_time()
            assert loading_time < 5.0, f"Model loading took {loading_time}s, expected < 5.0s"
    
    @pytest.mark.stress
    def test_cache_performance(self):
        """Test cache performance under various conditions."""
        cache_size = 100
        num_requests = 500
        
        # Simulate cache with hit/miss tracking
        cache = {}
        hits = 0
        misses = 0
        
        self.monitor.start_monitoring()
        
        for i in range(num_requests):
            # Simulate cache key (some keys repeat to test hits)
            cache_key = f"key_{i % (cache_size // 2)}"
            
            if cache_key in cache:
                hits += 1
                result = cache[cache_key]
            else:
                misses += 1
                # Simulate expensive operation
                result = self.mock_pipeline(f"prompt {i}")
                
                # Simulate cache size limit
                if len(cache) >= cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(cache))
                    del cache[oldest_key]
                
                cache[cache_key] = result
        
        self.monitor.stop_monitoring()
        
        hit_ratio = hits / (hits + misses)
        execution_time = self.monitor.get_execution_time()
        
        assert hit_ratio > 0.4, f"Cache hit ratio {hit_ratio} too low"
        assert execution_time < 10.0, f"Cache operations took {execution_time}s"
    
    @pytest.mark.benchmark
    def test_parameter_validation_overhead(self):
        """Test overhead of parameter validation."""
        num_validations = 1000
        
        def validate_parameters(params):
            """Mock parameter validation."""
            if not isinstance(params, dict):
                raise ValueError("Parameters must be dict")
            if 'prompt' not in params:
                raise ValueError("Prompt required")
            return True
        
        valid_params = {'prompt': 'test', 'width': 512, 'height': 512}
        
        self.monitor.start_monitoring()
        
        for i in range(num_validations):
            validate_parameters(valid_params)
        
        self.monitor.stop_monitoring()
        
        validation_time = self.monitor.get_execution_time()
        time_per_validation = validation_time / num_validations
        
        assert time_per_validation < 0.001, f"Validation overhead {time_per_validation}s too high"
    
    @pytest.mark.stress
    def test_error_handling_performance(self):
        """Test performance impact of error handling."""
        num_operations = 100
        error_rate = 0.2  # 20% of operations will fail
        
        successful_operations = 0
        failed_operations = 0
        
        self.monitor.start_monitoring()
        
        for i in range(num_operations):
            try:
                # Simulate random failures
                if (i % 5) == 0:  # Every 5th operation fails
                    raise ValueError(f"Simulated error {i}")
                
                # Simulate successful operation
                result = self.mock_pipeline(f"prompt {i}")
                successful_operations += 1
                
            except ValueError:
                failed_operations += 1
                # Simulate error recovery
                time.sleep(0.001)  # Small delay for error handling
        
        self.monitor.stop_monitoring()
        
        actual_error_rate = failed_operations / num_operations
        execution_time = self.monitor.get_execution_time()
        
        assert abs(actual_error_rate - error_rate) < 0.05, "Error rate not as expected"
        assert execution_time < 5.0, f"Error handling overhead too high: {execution_time}s"


@pytest.mark.performance
class TestResourceUtilization:
    """Tests for resource utilization monitoring."""
    
    def test_gpu_memory_tracking(self):
        """Test GPU memory usage tracking."""
        with patch('torch.cuda.is_available', return_value=True):
            mock_device = create_mock_torch_device()
            
            # Mock GPU memory functions
            with patch('torch.cuda.memory_allocated', return_value=1024*1024*100):  # 100MB
                with patch('torch.cuda.max_memory_allocated', return_value=1024*1024*200):  # 200MB
                    
                    memory_manager = MockMemoryManager()
                    
                    # Simulate GPU operations
                    memory_manager.allocate_gpu_memory(100)
                    
                    allocated = memory_manager.get_gpu_memory_usage()
                    peak = memory_manager.get_peak_gpu_memory()
                    
                    assert allocated > 0, "Should track allocated GPU memory"
                    assert peak >= allocated, "Peak memory should be >= current"
    
    def test_cpu_utilization_monitoring(self):
        """Test CPU utilization during operations."""
        initial_cpu = psutil.cpu_percent(interval=0.1)
        
        # Simulate CPU-intensive operation
        pipeline = MockDiffusionPipeline()
        
        start_time = time.time()
        for i in range(10):
            pipeline(f"prompt {i}")
        end_time = time.time()
        
        final_cpu = psutil.cpu_percent(interval=0.1)
        
        # Basic sanity checks
        assert end_time > start_time, "Operation should take some time"
        assert isinstance(final_cpu, (int, float)), "CPU usage should be numeric"
    
    def test_disk_io_monitoring(self):
        """Test disk I/O during cache operations."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "test_cache.json"
            
            # Simulate cache write operations
            start_time = time.time()
            
            for i in range(10):
                data = {'key': f'value_{i}', 'data': 'x' * 1000}  # 1KB per entry
                
                # Write to file (simulating cache save)
                with open(cache_file, 'w') as f:
                    import json
                    json.dump(data, f)
            
            end_time = time.time()
            
            # Verify file operations
            assert cache_file.exists(), "Cache file should be created"
            assert cache_file.stat().st_size > 0, "Cache file should have content"
            assert end_time > start_time, "Disk operations should take time"


@pytest.mark.performance
class TestScalabilityLimits:
    """Tests to determine system scalability limits."""
    
    def test_maximum_concurrent_requests(self):
        """Test maximum number of concurrent requests the system can handle."""
        max_threads = 20
        requests_per_thread = 5
        
        def make_requests(thread_id):
            """Make multiple requests in a thread."""
            results = []
            pipeline = MockDiffusionPipeline()
            
            for i in range(requests_per_thread):
                try:
                    result = pipeline(f"thread {thread_id} request {i}")
                    results.append(('success', result))
                except Exception as e:
                    results.append(('error', str(e)))
            
            return results
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = [
                executor.submit(make_requests, i) 
                for i in range(max_threads)
            ]
            
            all_results = []
            for future in futures:
                thread_results = future.result()
                all_results.extend(thread_results)
        
        end_time = time.time()
        
        successful_requests = sum(1 for result_type, _ in all_results if result_type == 'success')
        total_requests = max_threads * requests_per_thread
        success_rate = successful_requests / total_requests
        
        assert success_rate > 0.8, f"Success rate {success_rate} too low under load"
        assert end_time - start_time < 30.0, "System took too long under load"
    
    def test_memory_scalability(self):
        """Test how memory usage scales with workload."""
        workload_sizes = [10, 50, 100, 200]
        memory_usage = []
        
        for workload_size in workload_sizes:
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Simulate workload
            pipeline = MockDiffusionPipeline()
            results = []
            
            for i in range(workload_size):
                result = pipeline(f"prompt {i}")
                results.append(result)
            
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = end_memory - start_memory
            memory_usage.append((workload_size, memory_delta))
            
            # Clean up
            del results
        
        # Analyze memory scaling
        for i in range(1, len(memory_usage)):
            prev_workload, prev_memory = memory_usage[i-1]
            curr_workload, curr_memory = memory_usage[i]
            
            workload_ratio = curr_workload / prev_workload
            memory_ratio = curr_memory / prev_memory if prev_memory > 0 else 1
            
            # Memory growth should be roughly linear with workload
            assert memory_ratio < workload_ratio * 2, f"Memory scaling too aggressive: {memory_ratio}"
