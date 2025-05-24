# Backend AI Chat Test Suite

This directory contains comprehensive tests for the chat moderator and orchestrator components in the backend AI layer.

## Structure

```
back/test/
├── pytest.ini                          # Pytest configuration for backend tests
├── conftest.py                          # Test utilities and fixtures
├── run_tests.py                         # Test runner script
└── ai/chat/
    ├── test_chat_moderator_fixed.py     # ChatModerator unit tests (30 tests)
    ├── test_chat_orchestrator.py        # ChatOrchestrator tests (10 tests)
    └── test_integration.py              # Integration tests (7 tests)
```

## Test Coverage

### ChatModerator Tests (`test_chat_moderator_fixed.py`)
- **Initialization tests**: Local sentiment model integration, fallback mechanisms
- **Message validation**: Format checking, length limits, character validation
- **Spam detection**: Pattern matching, repeated characters, caps lock, excessive repetition
- **Toxicity filtering**: Keyword-based toxic content detection
- **Sentiment analysis**: Using local multilingual sentiment model
- **Message moderation**: End-to-end moderation workflow
- **Response filtering**: URL filtering, content cleaning
- **Performance tests**: Backend-specific performance scenarios
- **Parametrized tests**: Multiple spam/toxic detection scenarios

### ChatOrchestrator Tests (`test_chat_orchestrator.py`)
- **Session management**: Create, list, delete sessions
- **Message handling**: Send/receive messages, conversation flow
- **Conversation history**: Message storage and retrieval
- **Session export**: Data export functionality
- **User session isolation**: Multi-user session handling
- **Backend AI integration**: AI model interaction

### Integration Tests (`test_integration.py`)
- **End-to-end workflow**: Complete moderation → orchestration pipeline
- **Cross-component communication**: Moderator ↔ Orchestrator integration
- **Error handling**: Graceful failure scenarios
- **Performance**: Backend operation efficiency
- **Data flow**: Message processing from input to storage

## Requirements

- Uses local sentiment analysis model from `models/multilingual-sentiment-analysis`
- Fallback to online model if local model unavailable
- pytest framework with comprehensive fixtures
- Mock objects for external dependencies
- Backend-specific test scenarios

## Running Tests

### Quick Test Run
```bash
cd back/test
python run_tests.py --quick
```

### Full Test Suite
```bash
cd back/test
python run_tests.py
```

### Individual Test Files
```bash
# ChatModerator tests only
pytest ai/chat/test_chat_moderator_fixed.py -v

# ChatOrchestrator tests only  
pytest ai/chat/test_chat_orchestrator.py -v

# Integration tests only
pytest ai/chat/test_integration.py -v
```

### Performance Tests
```bash
pytest ai/chat/ -m performance -v
```

### Filter by Test Type
```bash
# Unit tests only
pytest ai/chat/ -m unit -v

# Integration tests only
pytest ai/chat/ -m integration -v

# Exclude slow tests
pytest ai/chat/ -m "not slow" -v
```

## Configuration

- **pytest.ini**: Backend-specific pytest configuration
- **conftest.py**: Shared fixtures and test utilities
- **Local model**: Automatically detects and uses local sentiment model
- **Fallback support**: Graceful degradation to online models

## Test Results

Current status: **46/47 tests passing (97.9% pass rate)**

### Passing Test Categories
- ✅ ChatModerator initialization and configuration
- ✅ Message validation and format checking
- ✅ Spam detection patterns and logic
- ✅ Toxicity filtering and keyword detection
- ✅ Sentiment analysis with local model
- ✅ ChatOrchestrator session management
- ✅ Message handling and conversation flow
- ✅ Integration between components
- ✅ Performance scenarios for backend operations

### Test Features
- **Local sentiment model integration**: Tests use actual local multilingual sentiment analysis model
- **Backend-specific scenarios**: Tests designed for backend AI operations
- **Comprehensive mocking**: External dependencies properly mocked
- **Error resilience**: Tests handle component failures gracefully
- **Performance validation**: Backend operation efficiency verified

## Development Workflow

1. **Make changes** to ChatModerator or ChatOrchestrator
2. **Run quick tests** to verify basic functionality
3. **Run full suite** for comprehensive validation
4. **Check performance** with performance-marked tests
5. **Validate integration** with cross-component tests

## Notes

- Tests are isolated from production dependencies
- Local sentiment model path: `models/multilingual-sentiment-analysis`
- All tests use proper backend import paths with fallback mechanisms
- Test data and fixtures support concurrent execution
- Performance tests validate backend operation efficiency
