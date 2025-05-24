# Orchestrator Chat Tests

This directory contains comprehensive tests for the chat moderator and orchestrator components of the Frai AI chatbot project.

## Test Structure

```
orchestrator/test/
├── conftest.py                 # Test utilities and fixtures
├── pytest.ini                 # Test configuration
├── run_tests.py               # Test runner script
├── test_integration.py        # Integration tests
├── chatmod/
│   └── test_chat_moderator.py # ChatModerator unit tests
└── chat/
    └── test_chat_orchestrator.py # ChatOrchestrator unit tests
```

## Components Tested

### ChatModerator (`orchestrator/chatmod/`)
- Message format validation
- Spam pattern detection
- Toxic content filtering  
- Sentiment analysis
- AI response moderation
- URL filtering
- Statistics tracking

### ChatOrchestrator (`orchestrator/chat/`)
- Session management (create, get, delete, update)
- Message handling (add, retrieve, history)
- User session tracking
- Data serialization/deserialization
- Session export functionality
- Conversation statistics

### Integration Tests
- Complete chat workflow with moderation
- Moderator + Orchestrator interaction
- Concurrent operations
- Performance testing
- Error handling
- Large conversation handling

## Running Tests

### Prerequisites

Ensure you have the required dependencies installed:

```powershell
pip install pytest pytest-cov pytest-mock pytest-benchmark
```

### Running All Tests

```powershell
# Run all orchestrator chat tests
python orchestrator/test/run_tests.py

# Run with verbose output
python orchestrator/test/run_tests.py --verbose
```

### Running Specific Test Suites

```powershell
# Run only ChatModerator tests
python orchestrator/test/run_tests.py --moderator

# Run only ChatOrchestrator tests  
python orchestrator/test/run_tests.py --orchestrator

# Run only integration tests
python orchestrator/test/run_tests.py --integration
```

### Running Quick Tests

```powershell
# Run quick smoke tests
python orchestrator/test/run_tests.py --quick
```

### Running Performance Tests

```powershell
# Run performance-focused tests
python orchestrator/test/run_tests.py --performance
```

### Using pytest Directly

```powershell
# Run all tests in this directory
pytest orchestrator/test/

# Run specific test file
pytest orchestrator/test/chatmod/test_chat_moderator.py

# Run with coverage
pytest orchestrator/test/ --cov=orchestrator --cov-report=html

# Run specific test methods
pytest orchestrator/test/ -k "test_validate_message_format"

# Run tests with specific markers
pytest orchestrator/test/ -m "unit"
```

## Test Categories

Tests are organized using pytest markers:

- `unit`: Unit tests for individual components
- `integration`: Integration tests between components
- `performance`: Performance and load tests
- `slow`: Tests that take a long time to run
- `moderator`: Tests specific to ChatModerator
- `orchestrator`: Tests specific to ChatOrchestrator
- `concurrent`: Tests for concurrent operations
- `mock`: Tests that use mocking extensively

### Running Tests by Category

```powershell
# Run only unit tests
pytest orchestrator/test/ -m "unit"

# Run only integration tests
pytest orchestrator/test/ -m "integration"

# Run only performance tests
pytest orchestrator/test/ -m "performance"
```

## Test Features

### Mock Data Generation
- Random test data generation
- Sample conversations
- Spam and toxic message examples
- Clean message examples
- AI responses with URLs

### Performance Testing
- Large conversation handling (1000+ messages)
- Concurrent operation testing
- Performance timing and benchmarks
- Memory usage validation

### Error Handling
- Invalid input handling
- Exception recovery
- Edge case testing
- Concurrent error scenarios

### Integration Testing
- Complete chat workflow simulation
- Moderation + orchestration interaction
- Multi-user scenarios
- Session lifecycle management

## Sample Test Output

```
Running All Orchestrator Chat Tests
================================================================================

============================================================
Running ChatModerator Tests
============================================================
test_chat_moderator.py::TestChatModerator::test_init_with_sentiment_analyzer PASSED
test_chat_moderator.py::TestChatModerator::test_validate_message_format_valid PASSED
test_chat_moderator.py::TestChatModerator::test_check_spam_patterns_valid PASSED
...

============================================================
Running ChatOrchestrator Tests
============================================================
test_chat_orchestrator.py::TestChatOrchestrator::test_create_chat_session_with_title PASSED
test_chat_orchestrator.py::TestChatOrchestrator::test_add_message_to_session PASSED
...

============================================================
Running Integration Tests
============================================================
test_integration.py::TestChatModerationIntegration::test_complete_chat_workflow_valid_messages PASSED
...

================================================================================
TEST SUMMARY
================================================================================
ChatModerator        : PASSED
ChatOrchestrator     : PASSED
Integration          : PASSED

Overall Result: ALL TESTS PASSED
```

## Test Configuration

### pytest.ini Settings
- Test discovery patterns
- Output formatting
- Timeout configuration
- Logging setup
- Warning filters

### Environment Variables
Tests may use environment variables:
- `POSITIVE_SYSTEM_PROMPT_CHAT`: Positive system prompt
- `NEGATIVE_SYSTEM_PROMPT_CHAT`: Negative system prompt

### Mock Configuration
- Sentiment analysis models are mocked by default
- Model loading is simulated for testing
- Database operations use in-memory storage

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure project root is in Python path
   ```powershell
   $env:PYTHONPATH = "c:\Users\andyo\projects\Frai"
   ```

2. **Model Loading Failures**: Tests use mocks, but verify transformers is installed
   ```powershell
   pip install transformers torch
   ```

3. **Permission Errors**: Ensure write permissions for test artifacts

4. **Timeout Issues**: Increase timeout in pytest.ini for slow systems

### Debug Mode

Run tests with debug output:
```powershell
pytest orchestrator/test/ -v -s --tb=long
```

### Test Coverage

Generate coverage reports:
```powershell
pytest orchestrator/test/ --cov=orchestrator --cov-report=html --cov-report=term-missing
```

View coverage report in `htmlcov/index.html`

## Contributing

When adding new tests:

1. Follow existing naming conventions
2. Use appropriate pytest markers
3. Include both positive and negative test cases
4. Add performance tests for new features
5. Update this README if adding new test categories

## Integration with CI/CD

These tests integrate with the project's CI/CD pipeline:
- Automated test execution on pull requests
- Coverage reporting
- Performance benchmarking
- Test result artifacts
