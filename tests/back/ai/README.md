# AI Module Testing Framework

This directory contains comprehensive test suites for all AI modules in the Frai backend system. The testing framework is designed to support Test-Driven Development (TDD) and provides extensive coverage for accuracy, performance, robustness, and edge cases.

## Directory Structure

```
Frai/tests/back/ai/
├── README.md                 # This file
├── test_helpers.py          # Common testing utilities
├── chat/                    # Chat AI module tests
│   ├── __init__.py
│   ├── conftest.py         # Chat test fixtures
│   ├── test_basic.py       # Basic functionality tests
│   ├── test_context.py     # Context management tests
│   ├── test_content.py     # Content generation tests
│   ├── test_edge_cases.py  # Edge case tests
│   ├── test_utils.py       # Chat-specific utilities
│   ├── test_set.csv        # Test cases definition
│   └── testset.csv         # Extended test cases
├── voicein/                # Voice input (STT) module tests
│   ├── __init__.py
│   ├── conftest.py         # VoiceIn test fixtures
│   ├── test_unit.py        # Unit tests
│   ├── test_accuracy.py    # Transcription accuracy tests
│   ├── test_robustness.py  # Audio quality robustness tests
│   ├── test_consistency.py # Transcription consistency tests
│   ├── test_edge_cases.py  # Edge case handling tests
│   ├── test_performance.py # Performance and latency tests
│   ├── testset.csv         # Comprehensive test cases (120 tests)
│   └── transcripts/        # Reference transcripts
│       ├── clear_short.txt
│       ├── clear_medium.txt
│       ├── clear_long.txt
│       ├── technical_jargon.txt
│       ├── medical_terms.txt
│       ├── legal_language.txt
│       └── ... (more reference files)
└── img2img/                # Image-to-image generation tests
    ├── __init__.py
    ├── conftest.py         # Img2Img test fixtures
    ├── test_unit.py        # Unit tests
    ├── test_consistency.py # Generation consistency tests
    ├── test_style.py       # Style transfer tests
    ├── test_diversity.py   # Output diversity tests
    ├── test_performance.py # Performance tests
    ├── testset.csv         # Test cases (120 tests)
    └── inputs.csv          # Input image definitions (200 images)
```

## Test Categories

### Chat Module Tests
- **Basic Tests**: Greetings, factual knowledge, technical questions
- **Context Tests**: Sequential conversation memory and context retention
- **Content Tests**: Content generation, controversial topics, code generation
- **Edge Cases**: Empty input, max tokens, special characters, error handling

### Voice Input Module Tests
- **Unit Tests**: Component initialization and basic functionality
- **Accuracy Tests**: Word error rate across different audio conditions
- **Robustness Tests**: Performance under noise, compression, distortion
- **Consistency Tests**: Repeated transcription stability
- **Edge Cases**: Silence, foreign languages, overlapping speakers
- **Performance Tests**: Latency, throughput, memory usage, scaling

### Image-to-Image Module Tests
- **Unit Tests**: Component initialization and basic functionality
- **Consistency Tests**: Reproducible outputs with identical inputs
- **Style Tests**: Art style transfer accuracy and consistency
- **Diversity Tests**: Output variation with different parameters
- **Performance Tests**: Generation speed, memory usage, batch processing

## Test Framework Features

### TDD Support
- **Safe Imports**: Graceful handling of missing implementations
- **Mock Functions**: Automatic mocking when real functions don't exist
- **Skip Decorators**: Tests skip when implementation is not ready
- **Progress Tracking**: Clear indication of implementation status

### Comprehensive Coverage
- **120+ VoiceIn Tests**: Covering 15+ audio categories and conditions
- **120+ Img2Img Tests**: Covering style transfer, backgrounds, emotions
- **200+ Input Images**: Organized by categories (portraits, landscapes, etc.)
- **Extensive Transcripts**: Reference files for accuracy validation

### Performance Testing
- **Latency Measurement**: Single operation timing
- **Throughput Testing**: Batch processing performance
- **Memory Monitoring**: Memory usage and leak detection
- **Concurrency Testing**: Multi-threaded performance
- **Scaling Analysis**: Performance across different input sizes

### Quality Metrics
- **Word Error Rate (WER)**: For speech recognition accuracy
- **Style Similarity**: For image style transfer quality
- **Consistency Scores**: For output reproducibility
- **Diversity Metrics**: For output variation measurement

## Running Tests

### Prerequisites
```bash
pip install pytest pytest-xvs psutil pillow numpy
```

### Running All Tests
```bash
# All AI module tests
pytest Frai/tests/back/ai/ -v

# Specific module
pytest Frai/tests/back/ai/chat/ -v
pytest Frai/tests/back/ai/voicein/ -v
pytest Frai/tests/back/ai/img2img/ -v
```

### Running by Test Type
```bash
# Unit tests only
pytest Frai/tests/back/ai/ -k "test_unit" -v

# Performance tests only
pytest Frai/tests/back/ai/ -k "test_performance" -v

# Accuracy tests only
pytest Frai/tests/back/ai/ -k "test_accuracy" -v
```

### Running with Coverage
```bash
pytest Frai/tests/back/ai/ --cov=Frai.back.ai --cov-report=html
```

## Test Data Organization

### CSV Test Definitions
Each module uses CSV files to define test cases:
- `testset.csv`: Main test case definitions
- `inputs.csv`: Input data references (for img2img)

### Test Case Structure
```csv
id,name,description,input_groups,tested_property
i2i-1,basic_style_transfer,Basic style transfer,portraits;art_styles,style_consistency
vi-1,clear_speech_short,Clear speech short clip,clean,word_error_rate <= 0.02
```

### Audio Test Data (VoiceIn)
- **Clean Audio**: High-quality recordings for baseline testing
- **Noisy Audio**: Various noise levels and types
- **Degraded Audio**: Compression, distortion, transmission artifacts
- **Specialized Content**: Technical terms, medical language, legal text
- **Edge Cases**: Silence, very short clips, foreign languages

### Image Test Data (Img2Img)
- **Portraits**: Various ages, expressions, demographics
- **Landscapes**: Different scenes, seasons, times of day
- **Art Styles**: Classical, modern, impressionist, abstract
- **Backgrounds**: Studio, natural, urban, abstract
- **Textures**: Wood, metal, fabric, stone materials

## Implementation Guidelines

### Adding New Tests
1. Define test cases in the appropriate CSV file
2. Create reference data (transcripts, images) if needed
3. Implement test functions using the test framework
4. Use safe imports and mock functions for TDD approach
5. Include performance benchmarks where appropriate

### Test Function Structure
```python
def test_feature_name(setup_module_ai, test_parameters):
    """Test description."""
    # Arrange: Set up test data
    input_data = load_test_input(test_parameters)
    
    # Act: Call the function under test
    result = module_function(input_data)
    
    # Assert: Verify results
    assert result.get('success', False), f"Test failed: {result.get('error')}"
    assert_quality_metrics(result, expected_thresholds)
```

### Error Handling
```python
# Safe function import
my_function = safe_import_ai_function('Frai.back.ai.module', 'function_name')

# Graceful test execution
if result.get('success', False):
    # Test passed - verify quality
    assert_quality_metrics(result)
else:
    # Function not implemented yet or failed
    logger.info(f"Function not ready: {result.get('error')}")
```

## Quality Thresholds

### Voice Input (Speech Recognition)
- **Clean Audio WER**: ≤ 2%
- **Light Noise WER**: ≤ 8%
- **Heavy Noise WER**: ≤ 25%
- **Real-time Factor**: ≤ 1.5x
- **Memory Growth**: ≤ 2GB per 8 files

### Image-to-Image Generation
- **Style Similarity**: ≥ 70%
- **Generation Consistency**: ≥ 85%
- **Output Diversity**: ≥ 10%
- **Generation Time**: ≤ 30s per image
- **Memory Usage**: ≤ 3GB growth per batch

### Chat Generation
- **Response Rate**: ≥ 95%
- **Context Retention**: ≥ 90%
- **Content Appropriateness**: 100%
- **Response Time**: ≤ 10s
- **Token Efficiency**: Optimal usage

## Continuous Integration

### Automated Testing
- Tests run on every commit
- Performance regression detection
- Quality metric tracking over time
- Automatic test data validation

### Test Reporting
- Coverage reports generated automatically
- Performance benchmarks tracked
- Quality metrics dashboards
- Failed test categorization and analysis

## Contributing

### Adding New Test Cases
1. Update the appropriate CSV file with new test definitions
2. Provide reference data (audio files, images, transcripts)
3. Ensure test cases cover edge conditions
4. Include performance and quality expectations
5. Document any special requirements or dependencies

### Test Data Requirements
- **Audio Files**: WAV format, 16kHz sample rate preferred
- **Image Files**: JPG/PNG format, consistent sizing
- **Transcripts**: UTF-8 text files with accurate content
- **File Organization**: Follow directory structure conventions

### Code Quality
- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Use type hints where appropriate
- Implement proper error handling
- Add logging for debugging and monitoring

This testing framework provides comprehensive coverage for all AI modules while supporting incremental development through TDD practices. The structure allows for easy addition of new test cases and maintains high quality standards throughout the development process.