# Test Framework Documentation

This directory contains a comprehensive test suite for the Frai application's refactored image generation system. The tests are organized into multiple categories to ensure thorough coverage across all levels of the system.

## Directory Structure

```
tests/
├── __init__.py                     # Test package initialization
├── pytest.ini                     # Pytest configuration
├── conftest.py                     # Shared fixtures and configuration
├── README.md                       # This documentation
├── run_tests.py                    # Python test runner script
├── run_tests.sh                    # Unix/Linux/macOS test runner script
├── run_tests.bat                   # Windows test runner script
├── unit/                           # Unit tests
│   ├── __init__.py
│   ├── api/                        # API component unit tests
│   │   ├── __init__.py
│   │   └── test_endpoints.py       # API endpoint tests
│   ├── services/                   # Service component unit tests
│   │   ├── __init__.py
│   │   └── test_image_generation_service.py
│   └── image_generation/           # Core image generation unit tests
│       ├── __init__.py
│       └── test_memory_manager.py  # Memory management tests
├── integration/                    # Integration tests
│   ├── __init__.py
│   └── test_image_generation_system.py  # System integration tests
├── implementation/                 # Implementation-specific tests
│   ├── __init__.py
│   └── test_refactored_system_implementation.py
├── blackbox/                       # End-to-end blackbox tests
│   ├── __init__.py
│   └── test_end_to_end_workflows.py
├── performance/                    # Performance and benchmark tests
│   ├── __init__.py
│   └── test_performance_benchmarks.py
├── fixtures/                       # Test data and fixtures
│   ├── __init__.py
│   └── test_data.py               # Sample data and fixtures
└── utils/                          # Test utilities and helpers
    ├── __init__.py
    └── mock_helpers.py             # Mock objects and helpers
```

## Test Categories

### 1. Unit Tests (`tests/unit/`)

Test individual components in isolation with mocked dependencies.

**Coverage:**
- API endpoints and request/response handling
- Service layer business logic
- Image generation core components
- Memory management functionality
- Parameter validation
- Error handling

**Key Features:**
- Mocked external dependencies
- Fast execution
- High code coverage
- Isolated component testing

### 2. Integration Tests (`tests/integration/`)

Test interactions between multiple components within the system.

**Coverage:**
- Service-to-service communication
- Component interaction workflows
- Configuration management
- Resource sharing
- Cross-component error propagation

**Key Features:**
- Real component interactions
- System-level configuration testing
- Multi-component workflows

### 3. Implementation Tests (`tests/implementation/`)

Test specific implementation details and behaviors of the refactored system.

**Coverage:**
- Caching mechanisms
- Thread safety
- Lazy loading
- Parameter validation timing
- Concurrent operations
- Resource cleanup

**Key Features:**
- Implementation-specific behavior validation
- Performance characteristics testing
- Thread safety verification

### 4. Blackbox Tests (`tests/blackbox/`)

End-to-end tests that treat the system as a black box.

**Coverage:**
- Complete API workflows
- User journey simulation
- Error recovery scenarios
- Performance under load
- Integration with external systems

**Key Features:**
- Real API requests
- Complete workflow testing
- User perspective validation

### 5. Performance Tests (`tests/performance/`)

Benchmark and performance testing for the system.

**Coverage:**
- Memory usage patterns
- Execution time benchmarks
- Throughput measurements
- Resource utilization
- Scalability limits
- Concurrent load testing

**Key Features:**
- Performance metrics collection
- Resource monitoring
- Scalability testing
- Benchmark comparisons

## Running Tests

### Using Test Runner Scripts

#### Python Script (Cross-platform)
```bash
# Run all tests
python tests/run_tests.py all

# Run specific test category
python tests/run_tests.py unit
python tests/run_tests.py integration
python tests/run_tests.py performance

# Run with coverage
python tests/run_tests.py unit --coverage --html-report

# Run with parallel execution
python tests/run_tests.py all --parallel 4

# Verbose output with fail-fast
python tests/run_tests.py integration --verbose --fail-fast
```

#### Unix/Linux/macOS
```bash
# Make script executable
chmod +x tests/run_tests.sh

# Run tests
./tests/run_tests.sh all
./tests/run_tests.sh unit --coverage
./tests/run_tests.sh performance --parallel 2
```

#### Windows
```cmd
# Run tests
tests\run_tests.bat all
tests\run_tests.bat unit --coverage
tests\run_tests.bat integration --verbose
```

### Using Pytest Directly

#### Basic Usage
```bash
# Run all tests
pytest tests/

# Run specific category
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run specific test file
pytest tests/unit/services/test_image_generation_service.py

# Run specific test method
pytest tests/unit/services/test_image_generation_service.py::TestImageGenerationService::test_service_initialization
```

#### With Coverage
```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# Coverage for specific module
pytest tests/unit/ --cov=src.services --cov-report=html
```

#### Advanced Options
```bash
# Parallel execution
pytest tests/ -n 4

# Fail fast
pytest tests/ -x

# Verbose output
pytest tests/ -v

# Only run tests matching pattern
pytest tests/ -k "test_memory"

# Run tests with specific markers
pytest tests/ -m "unit"
pytest tests/ -m "performance"
pytest tests/ -m "slow"
```

## Test Markers

The test framework uses pytest markers to categorize tests:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.implementation` - Implementation tests
- `@pytest.mark.blackbox` - Blackbox tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.benchmark` - Benchmark tests
- `@pytest.mark.stress` - Stress tests

### Running Tests by Marker
```bash
# Run only unit tests
pytest -m unit

# Run performance and benchmark tests
pytest -m "performance or benchmark"

# Skip slow tests
pytest -m "not slow"
```

## Configuration

### Pytest Configuration (`pytest.ini`)

Key configuration options:
- Test discovery patterns
- Coverage settings
- Marker definitions
- Output formatting
- Parallel execution settings

### Shared Fixtures (`conftest.py`)

Provides:
- Mock objects setup
- Test data fixtures
- Configuration fixtures
- Automatic marker assignment
- Session-level setup/teardown

## Mock Infrastructure

### Mock Helpers (`tests/utils/mock_helpers.py`)

Comprehensive mocking system including:
- `MockDiffusionPipeline` - Mock image generation pipeline
- `MockMemoryManager` - Mock memory management
- `MockTorch` - Mock PyTorch operations
- Mock file system operations
- Test data generators

### Test Fixtures (`tests/fixtures/test_data.py`)

Pre-defined test data:
- Sample configurations
- Test prompts
- API request/response examples
- Validation test cases
- Performance test scenarios

## Writing New Tests

### Test Structure

Follow this structure for new tests:

```python
import pytest
from unittest.mock import Mock, patch
from tests.utils.mock_helpers import MockDiffusionPipeline
from tests.fixtures.test_data import sample_configs

@pytest.mark.unit  # Add appropriate marker
class TestNewFeature:
    """Test class for new feature."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_pipeline = MockDiffusionPipeline()
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        input_data = "test input"
        
        # Act
        result = self.mock_pipeline(input_data)
        
        # Assert
        assert result is not None
        assert 'images' in result
    
    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError, match="Invalid input"):
            self.mock_pipeline("")
```

### Best Practices

1. **Test Naming**: Use descriptive names that explain what is being tested
2. **Test Organization**: Group related tests in classes
3. **Fixtures**: Use fixtures for common test data and setup
4. **Mocking**: Mock external dependencies appropriately
5. **Assertions**: Use specific assertions with clear error messages
6. **Documentation**: Add docstrings to test classes and methods

### Adding New Test Categories

To add a new test category:

1. Create new directory under `tests/`
2. Add `__init__.py` file
3. Create test files following naming convention `test_*.py`
4. Add new marker to `pytest.ini`
5. Update `conftest.py` for automatic marker assignment
6. Update test runner scripts if needed

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-xdist
    
    - name: Run unit tests
      run: python tests/run_tests.py unit --coverage
    
    - name: Run integration tests
      run: python tests/run_tests.py integration
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v1
      with:
        file: ./coverage.xml
```

## Coverage Reports

### Generating Reports

```bash
# Terminal report
pytest --cov=src --cov-report=term-missing

# HTML report
pytest --cov=src --cov-report=html

# XML report (for CI)
pytest --cov=src --cov-report=xml

# Multiple reports
pytest --cov=src --cov-report=html --cov-report=xml --cov-report=term-missing
```

### Coverage Goals

- **Unit Tests**: Target 95%+ coverage
- **Integration Tests**: Focus on component interactions
- **Overall**: Maintain 90%+ total coverage

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH` includes project root
2. **Mock Issues**: Verify mock objects match actual interfaces
3. **Slow Tests**: Use `@pytest.mark.slow` marker and run separately
4. **Memory Issues**: Monitor test resource usage
5. **Flaky Tests**: Investigate timing or dependency issues

### Debug Mode

```bash
# Run with debug output
pytest tests/ --capture=no --log-cli-level=DEBUG

# Run single test with debugging
pytest tests/unit/test_specific.py::test_method -s -vv
```

### Performance Debugging

```bash
# Profile test execution
pytest tests/ --profile

# Memory profiling
pytest tests/ --memprof
```

## Contributing

### Adding Tests

1. Follow existing test structure and naming conventions
2. Add appropriate markers
3. Include comprehensive docstrings
4. Mock external dependencies
5. Test both success and failure cases
6. Update documentation if needed

### Code Review Checklist

- [ ] Tests follow naming conventions
- [ ] Appropriate markers are used
- [ ] Mocks are properly configured
- [ ] Error cases are tested
- [ ] Documentation is updated
- [ ] Tests pass in isolation and in suite
- [ ] Coverage is maintained or improved

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

## Support

For questions or issues with the test framework:

1. Check existing test examples
2. Review documentation
3. Check pytest and coverage tool documentation
4. Create an issue with detailed description and reproduction steps
