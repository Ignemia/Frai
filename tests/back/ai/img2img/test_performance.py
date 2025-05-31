"""
Performance tests for the image-to-image module.

This module tests the throughput, latency, and resource usage of the 
image-to-image generation system under various conditions.
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

from Frai.back.ai.img2img import (
    initialize_img2img_system,
    get_img2img_ai_instance,
    generate_img2img
)

# Set up logging
logger = logging.getLogger(__name__)

# Path to the test set CSV files
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), 'testset.csv')
INPUTS_PATH = os.path.join(os.path.dirname(__file__), 'inputs.csv')

# Performance test configuration
SINGLE_GENERATION_TESTS = 10
BATCH_SIZE_TESTS = [1, 3, 5, 10]
CONCURRENT_THREADS = [1, 2, 4]
LATENCY_THRESHOLD_SECONDS = 30  # Maximum acceptable latency
THROUGHPUT_THRESHOLD = 0.05     # Minimum images per second


def load_input_image(image_id: str):
    """Load an input image by ID from the inputs.csv file."""
    try:
        import csv
        with open(INPUTS_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['id'] == image_id:
                    image_path = row['path']
                    if os.path.exists(image_path):
                        from PIL import Image
                        return Image.open(image_path)
                    else:
                        # Create a placeholder image if file doesn't exist
                        from PIL import Image
                        return Image.new('RGB', (256, 256), color=(128, 128, 128))
        
        # If not found, create a default image
        from PIL import Image
        return Image.new('RGB', (256, 256), color=(100, 100, 100))
    except Exception as e:
        logger.warning(f"Failed to load image {image_id}: {e}")
        from PIL import Image
        return Image.new('RGB', (256, 256), color=(150, 150, 150))


@pytest.fixture(scope="module")
def setup_img2img_ai():
    """Initialize the image-to-image system once for all tests."""
    success = initialize_img2img_system()
    if not success:
        pytest.fail("Failed to initialize image-to-image system")
    
    img2img_ai = get_img2img_ai_instance()
    return img2img_ai


@pytest.fixture
def performance_test_images():
    """Provide test images of varying complexity for performance testing."""
    return {
        'simple': [
            ('1', '11'),  # Portrait + Art style
            ('26', '21'), # Geometric + Texture
        ],
        'medium': [
            ('6', '12'),  # Landscape + Van Gogh
            ('2', '13'),  # Portrait + Picasso
        ],
        'complex': [
            ('9', '14'),  # Cityscape + Renaissance
            ('75', '31'), # Natural scene + Impressionist
        ]
    }


class TestImg2ImgPerformance:
    """Test image-to-image generation performance metrics."""
    
    def test_single_generation_latency(self, setup_img2img_ai, performance_test_images):
        """Test latency for single image generation."""
        latencies = []
        
        # Test with simple and medium complexity combinations
        test_pairs = performance_test_images['simple'] + performance_test_images['medium']
        
        for source_id, ref_id in test_pairs[:SINGLE_GENERATION_TESTS]:
            source_image = load_input_image(source_id)
            reference_image = load_input_image(ref_id)
            
            start_time = time.time()
            result = generate_img2img(
                source_image=source_image,
                reference_image=reference_image,
                transformation_type="style_transfer",
                strength=0.7,
                seed=42
            )
            end_time = time.time()
            
            # Verify successful generation
            if result.get('success', False):
                assert 'generated_image' in result
                assert result['generated_image'] is not None
                
                latency = end_time - start_time
                latencies.append(latency)
                logger.info(f"Generation latency for {source_id}->{ref_id}: {latency:.2f}s")
            else:
                logger.warning(f"Generation failed for {source_id}->{ref_id}: {result.get('error', 'Unknown')}")
        
        if latencies:
            # Calculate statistics
            avg_latency = statistics.mean(latencies)
            median_latency = statistics.median(latencies)
            max_latency = max(latencies)
            min_latency = min(latencies)
            
            logger.info(f"Single generation latency stats:")
            logger.info(f"  Average: {avg_latency:.2f}s")
            logger.info(f"  Median: {median_latency:.2f}s")
            logger.info(f"  Max: {max_latency:.2f}s")
            logger.info(f"  Min: {min_latency:.2f}s")
            
            # Performance assertions
            assert avg_latency < LATENCY_THRESHOLD_SECONDS, f"Average latency {avg_latency:.2f}s exceeds threshold {LATENCY_THRESHOLD_SECONDS}s"
            assert max_latency < LATENCY_THRESHOLD_SECONDS * 2, f"Maximum latency {max_latency:.2f}s too high"
        else:
            pytest.skip("No successful generations for latency testing")
    
    def test_batch_processing_performance(self, setup_img2img_ai, performance_test_images):
        """Test performance of batch processing."""
        for batch_size in BATCH_SIZE_TESTS:
            # Prepare batch of image pairs
            all_pairs = performance_test_images['simple']
            batch_pairs = (all_pairs * ((batch_size // len(all_pairs)) + 1))[:batch_size]
            
            start_time = time.time()
            
            # Process batch (simulate batch by processing sequentially)
            successful_results = 0
            for source_id, ref_id in batch_pairs:
                source_image = load_input_image(source_id)
                reference_image = load_input_image(ref_id)
                
                result = generate_img2img(
                    source_image=source_image,
                    reference_image=reference_image,
                    transformation_type="style_transfer",
                    strength=0.5,  # Reduced strength for performance
                    seed=42
                )
                
                if result.get('success', False):
                    successful_results += 1
                    assert result['generated_image'] is not None
            
            end_time = time.time()
            
            if successful_results > 0:
                # Calculate performance metrics
                total_time = end_time - start_time
                throughput = successful_results / total_time
                avg_time_per_image = total_time / successful_results
                
                logger.info(f"Batch size {batch_size} performance:")
                logger.info(f"  Successful generations: {successful_results}/{batch_size}")
                logger.info(f"  Total time: {total_time:.2f}s")
                logger.info(f"  Throughput: {throughput:.3f} images/second")
                logger.info(f"  Average per image: {avg_time_per_image:.2f}s")
                
                # Performance assertions
                assert throughput >= THROUGHPUT_THRESHOLD, f"Throughput {throughput:.3f} images/sec below threshold {THROUGHPUT_THRESHOLD}"
            else:
                logger.warning(f"No successful generations in batch size {batch_size}")
    
    def test_concurrent_processing(self, setup_img2img_ai, performance_test_images):
        """Test performance under concurrent load."""
        for num_threads in CONCURRENT_THREADS:
            pairs_per_thread = 2
            total_pairs = num_threads * pairs_per_thread
            
            # Prepare pairs for threads
            all_pairs = performance_test_images['simple']
            test_pairs = (all_pairs * ((total_pairs // len(all_pairs)) + 1))[:total_pairs]
            thread_pairs = [test_pairs[i::num_threads] for i in range(num_threads)]
            
            def process_pairs_in_thread(pairs):
                """Process a list of image pairs in a single thread."""
                results = []
                start_time = time.time()
                for source_id, ref_id in pairs:
                    source_image = load_input_image(source_id)
                    reference_image = load_input_image(ref_id)
                    
                    result = generate_img2img(
                        source_image=source_image,
                        reference_image=reference_image,
                        transformation_type="style_transfer",
                        strength=0.5,
                        seed=42
                    )
                    results.append(result)
                end_time = time.time()
                return results, end_time - start_time
            
            # Execute concurrent processing
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(process_pairs_in_thread, pairs) for pairs in thread_pairs]
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
                logger.info(f"  Successful generations: {successful_count}/{total_pairs}")
                logger.info(f"  Total time: {total_time:.2f}s")
                logger.info(f"  Throughput: {concurrent_throughput:.3f} images/second")
                logger.info(f"  Average thread time: {avg_thread_time:.2f}s")
                
                # Performance assertions - concurrent should maintain reasonable throughput
                min_concurrent_throughput = THROUGHPUT_THRESHOLD * 0.3  # Allow significant overhead
                assert concurrent_throughput >= min_concurrent_throughput, \
                    f"Concurrent throughput {concurrent_throughput:.3f} below threshold {min_concurrent_throughput}"
    
    def test_memory_usage_stability(self, setup_img2img_ai, performance_test_images):
        """Test that memory usage remains stable during extended processing."""
        try:
            import psutil
            import gc
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process multiple image pairs
            num_iterations = 8
            test_pairs = (performance_test_images['simple'] * ((num_iterations // len(performance_test_images['simple'])) + 1))[:num_iterations]
            
            memory_samples = []
            
            for i, (source_id, ref_id) in enumerate(test_pairs):
                source_image = load_input_image(source_id)
                reference_image = load_input_image(ref_id)
                
                result = generate_img2img(
                    source_image=source_image,
                    reference_image=reference_image,
                    transformation_type="style_transfer",
                    strength=0.6,
                    seed=42
                )
                
                if result.get('success', False):
                    assert result['generated_image'] is not None
                
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
            
            # Memory growth should be reasonable (less than 3GB for 8 images)
            assert memory_growth < 3000, f"Memory growth {memory_growth:.2f}MB seems excessive"
            
            # Force garbage collection
            gc.collect()
            
        except ImportError:
            pytest.skip("psutil not available for memory testing")
    
    def test_transformation_type_performance(self, setup_img2img_ai):
        """Test performance across different transformation types."""
        source_image = load_input_image('1')  # Portrait
        reference_image = load_input_image('11')  # Art style
        
        transformation_types = [
            "style_transfer",
            "color_palette_transfer", 
            "texture_transfer",
            "background_replacement"
        ]
        
        type_performance = {}
        
        for transform_type in transformation_types:
            start_time = time.time()
            result = generate_img2img(
                source_image=source_image,
                reference_image=reference_image,
                transformation_type=transform_type,
                strength=0.7,
                seed=42
            )
            end_time = time.time()
            
            if result.get('success', False):
                assert result['generated_image'] is not None
                processing_time = end_time - start_time
                type_performance[transform_type] = processing_time
                
                logger.info(f"Transformation '{transform_type}': {processing_time:.2f}s")
        
        # Check that all transformation types complete within reasonable time
        for transform_type, time_taken in type_performance.items():
            assert time_taken < LATENCY_THRESHOLD_SECONDS, \
                f"Transformation '{transform_type}' too slow: {time_taken:.2f}s"
    
    def test_strength_parameter_performance(self, setup_img2img_ai):
        """Test how strength parameter affects performance."""
        source_image = load_input_image('6')  # Landscape
        reference_image = load_input_image('12')  # Van Gogh style
        
        strengths = [0.3, 0.6, 0.9]
        strength_performance = []
        
        for strength in strengths:
            start_time = time.time()
            result = generate_img2img(
                source_image=source_image,
                reference_image=reference_image,
                transformation_type="style_transfer",
                strength=strength,
                seed=42
            )
            end_time = time.time()
            
            if result.get('success', False):
                assert result['generated_image'] is not None
                processing_time = end_time - start_time
                strength_performance.append((strength, processing_time))
                
                logger.info(f"Strength {strength}: {processing_time:.2f}s")
        
        # Check that performance scaling is reasonable with strength
        if len(strength_performance) >= 2:
            for i in range(1, len(strength_performance)):
                prev_strength, prev_time = strength_performance[i-1]
                curr_strength, curr_time = strength_performance[i]
                
                strength_ratio = curr_strength / prev_strength
                time_ratio = curr_time / prev_time
                
                # Time should not scale dramatically with strength
                assert time_ratio <= strength_ratio * 1.5, \
                    f"Strength scaling issue: {strength_ratio}x strength led to {time_ratio}x time increase"
    
    def test_image_size_performance_scaling(self, setup_img2img_ai):
        """Test how performance scales with image size."""
        # Create test images of different sizes
        from PIL import Image
        
        sizes = [(128, 128), (256, 256), (512, 512)]
        base_image = load_input_image('1')
        style_image = load_input_image('11')
        
        size_performance = []
        
        for width, height in sizes:
            # Resize images to test size
            source_resized = base_image.resize((width, height))
            style_resized = style_image.resize((width, height))
            
            start_time = time.time()
            result = generate_img2img(
                source_image=source_resized,
                reference_image=style_resized,
                transformation_type="style_transfer",
                strength=0.6,
                seed=42
            )
            end_time = time.time()
            
            if result.get('success', False):
                assert result['generated_image'] is not None
                
                processing_time = end_time - start_time
                pixel_count = width * height
                size_performance.append((pixel_count, processing_time))
                
                logger.info(f"Size {width}x{height} ({pixel_count} pixels): {processing_time:.2f}s")
        
        # Check that performance scaling is reasonable with image size
        if len(size_performance) >= 2:
            for i in range(1, len(size_performance)):
                prev_pixels, prev_time = size_performance[i-1]
                curr_pixels, curr_time = size_performance[i]
                
                pixel_ratio = curr_pixels / prev_pixels
                time_ratio = curr_time / prev_time
                
                # Time should scale sub-quadratically with pixel count
                assert time_ratio <= pixel_ratio * 1.5, \
                    f"Size scaling issue: {pixel_ratio}x pixels led to {time_ratio}x time increase"
    
    def test_repeated_generation_consistency(self, setup_img2img_ai):
        """Test that repeated generation maintains consistent performance."""
        source_image = load_input_image('2')  # Portrait
        reference_image = load_input_image('13')  # Picasso style
        
        num_repetitions = 8
        processing_times = []
        
        for i in range(num_repetitions):
            start_time = time.time()
            result = generate_img2img(
                source_image=source_image,
                reference_image=reference_image,
                transformation_type="style_transfer",
                strength=0.7,
                seed=42  # Same parameters for consistency
            )
            end_time = time.time()
            
            if result.get('success', False):
                assert result['generated_image'] is not None
                processing_times.append(end_time - start_time)
        
        if len(processing_times) >= 5:
            # Performance consistency analysis
            avg_time = statistics.mean(processing_times)
            time_stdev = statistics.stdev(processing_times)
            time_cv = time_stdev / avg_time  # Coefficient of variation
            
            logger.info(f"Repeated generation consistency:")
            logger.info(f"  Average time: {avg_time:.2f}s")
            logger.info(f"  Time std dev: {time_stdev:.2f}s")
            logger.info(f"  Time CV: {time_cv:.3f}")
            
            # Performance should be reasonably consistent (CV < 0.4)
            assert time_cv < 0.4, f"Performance inconsistent: CV {time_cv:.3f} too high"
        else:
            pytest.skip("Insufficient successful generations for consistency testing")
    
    def test_warm_up_vs_steady_state_performance(self, setup_img2img_ai):
        """Test performance difference between initial and steady-state generation."""
        source_image = load_input_image('8')  # Forest path
        reference_image = load_input_image('14')  # Renaissance style
        
        # First generation (warm-up)
        start_time = time.time()
        result1 = generate_img2img(
            source_image=source_image,
            reference_image=reference_image,
            transformation_type="style_transfer",
            strength=0.7,
            seed=42
        )
        warmup_time = time.time() - start_time
        
        # Subsequent generations (steady state)
        steady_times = []
        for i in range(3):
            start_time = time.time()
            result = generate_img2img(
                source_image=source_image,
                reference_image=reference_image,
                transformation_type="style_transfer",
                strength=0.7,
                seed=100 + i
            )
            steady_times.append(time.time() - start_time)
            
            if result.get('success', False):
                assert result['generated_image'] is not None
        
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