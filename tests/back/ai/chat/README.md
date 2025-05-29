# AI Chat Module Test Suite

This directory contains a comprehensive test suite for the AI chat module, designed to validate functionality, accuracy, and robustness of the chat AI system.

## Overview

The test suite is organized into four main categories:

- **Basic Tests** (`test_basic.py`) - Core functionality like greetings, factual knowledge, and technical questions
- **Context Tests** (`test_context.py`) - Context retention and sequential conversation management
- **Content Tests** (`test_content.py`) - Content generation, safety filters, and various content types
- **Edge Case Tests** (`test_edge_cases.py`) - Error handling, unusual inputs, and boundary conditions

## Quick Start

### Running All Tests

```bash
# Navigate to the test directory
cd Frai/tests/back/ai/chat

# Run all tests (CPU mode recommended for compatibility)
FORCE_CPU_MODE=true python -m pytest -v

# Or use the test runner script
python run_tests.py
```

### Running Specific Test Categories

```bash
# Basic functionality tests
python run_tests.py --basic

# Context management tests
python run_tests.py --context

# Content generation tests
python run_tests.py --content

# Edge case tests
python run_tests.py --edge
```

## Test Categories

### 1. Basic Functionality Tests (`test_basic.py`)

Tests core AI chat capabilities:

- **Basic Greeting**: Simple conversational responses
- **Factual Knowledge**: Questions requiring factual answers (e.g., "What is the capital of France?")
- **Technical Questions**: Complex technical explanations

**Example test cases:**
- "Hello AI! How are you doing today?" → Should respond with any text
- "What is the capital of France?" → Should contain "Paris"
- "Explain how transformers work in LLMs" → Should provide technical explanation

### 2. Context Management Tests (`test_context.py`)

Tests the AI's ability to maintain conversation context:

- **Simple Context**: Remember information from previous messages
- **Sequential Reasoning**: Multi-step logical reasoning across messages
- **Context Retention**: Long-term memory within conversations

**Example sequential test:**
1. "My name is Alex and I like pizza." → AI acknowledges
2. "What food do I like?" → Should respond with "pizza"
3. "What is my name and what do I like?" → Should respond with "Alex pizza"

### 3. Content Generation Tests (`test_content.py`)

Tests various content types and safety measures:

- **Safety Filtering**: Responses to inappropriate requests
- **Multiple Languages**: Non-English input handling
- **Code Generation**: Programming-related requests
- **Controversial Topics**: Political questions, sensitive content

**Example test cases:**
- Malicious requests → Should refuse appropriately
- Non-English input → Should respond appropriately
- Code generation requests → Should handle safely

### 4. Edge Case Tests (`test_edge_cases.py`)

Tests system robustness with unusual inputs:

- **Empty Input**: Handling of empty conversation history
- **Special Characters**: Unicode, emojis, symbols
- **Long Input**: Maximum token limit testing
- **Error Conditions**: Invalid input handling

## Test Configuration

### Environment Variables

- `FORCE_CPU_MODE=true` - Forces CPU-only execution (recommended for compatibility)
- `TRANSFORMERS_VERBOSITY=info` - Controls transformer library logging

### Model Configuration

The tests use the Gemma-3-4B-IT model by default. The system will:

1. Try to load from local model directory (`Frai/models/gemma-3-4b-it`)
2. Fall back to downloading from HuggingFace if local model fails
3. Use CPU-only mode when `FORCE_CPU_MODE=true` to avoid CUDA compilation issues

### Token Limits

Different test types use different token limits for efficiency:

- Basic tests: 5-10 tokens (fast execution)
- Context tests: 15-50 tokens (depending on complexity)
- Mathematical reasoning: 50 tokens (for complex calculations)

## Test Data

### Test Cases CSV (`testset.csv`)

All test cases are defined in `testset.csv` with the following structure:

```csv
index,name,description,content,partials,output
1,basic_greeting,Basic greeting test,Hello AI! How are you doing today?,,Any text
5,context_1,First message in context sequence,My name is Alex and I like pizza.,,Any text
6,context_2,Second message in context sequence,What food do I like?,5,pizza
```

**Columns:**
- `index`: Unique test identifier
- `name`: Test name/category
- `description`: Human-readable description
- `content`: Input message to the AI
- `partials`: Dependencies (previous test indices, semicolon-separated)
- `output`: Expected output pattern

### Output Validation

The test system supports several output validation modes:

- `"Any text"` - Any successful response
- `"Error"` - Expected failure/error condition
- Specific text (e.g., `"Paris"`) - Must contain the specified text
- Multiple keywords (e.g., `"Alex pizza"`) - Must contain all keywords
- Numbers (e.g., `"63"`) - Special mathematical reasoning validation

## Architecture

### Key Components

1. **`conftest.py`** - pytest configuration and shared fixtures
2. **`test_utils.py`** - Utility functions for test execution
3. **Individual test files** - Category-specific test implementations
4. **`run_tests.py`** - Convenient test runner script

### Test Flow

1. **Setup**: Initialize ChatAI system with model loading
2. **Test Execution**: Run individual test cases with conversation building
3. **Validation**: Verify outputs against expected patterns
4. **Context Management**: Maintain conversation state for sequential tests
5. **Cleanup**: Resource cleanup and session management

### Fixtures

- `setup_chat_ai` - Session-scoped ChatAI initialization
- `chat_response` - Function to generate AI responses

## Performance Considerations

### Execution Time

- **Full test suite**: ~2-3 minutes (CPU mode)
- **Basic tests**: ~20 seconds
- **Context tests**: ~60 seconds (includes sequential processing)
- **Content tests**: ~40 seconds
- **Edge cases**: ~15 seconds

### Resource Usage

- **CPU Mode**: Uses system RAM, lower GPU requirements
- **Memory**: ~4-8GB RAM for model loading
- **GPU Mode**: Requires CUDA-compatible GPU (may have compatibility issues)

## Troubleshooting

### Common Issues

1. **CUDA Compilation Errors**
   - Solution: Use `FORCE_CPU_MODE=true`
   - Gemma-3 models have known compatibility issues with certain CUDA setups

2. **Model Loading Failures**
   - Local model path issues → Falls back to HuggingFace automatically
   - Network issues → Check internet connection for HuggingFace downloads

3. **Test Timeouts**
   - CPU generation is slower than GPU
   - Consider reducing token counts in test configurations

4. **Memory Issues**
   - Reduce model precision or use smaller model
   - Clear CUDA cache between tests

### Debug Mode

Run with verbose logging:

```bash
python run_tests.py --verbose
```

### Test Individual Cases

```bash
# Run specific test
python -m pytest test_basic.py::test_basic_functionality -k "test_case0" -v

# Run with CPU mode
FORCE_CPU_MODE=true python -m pytest test_basic.py -v
```

## Development

### Adding New Tests

1. Add test case to `testset.csv`
2. Assign appropriate category name
3. Define input content and expected output
4. Set up dependencies if needed (partials column)

### Modifying Test Logic

- **Output validation**: Modify `verify_output()` in `test_utils.py`
- **Context management**: Update conversation building logic
- **New categories**: Create new test files following existing patterns

### Test Categories

Test categories are defined by the `name` field in `testset.csv`:

- `basic_*` → Basic functionality tests
- `context_*` → Context management tests  
- `sequential_*` → Sequential reasoning tests
- `edge_*` → Edge case tests
- Content tests are identified by specific names (see `CONTENT_TEST_CATEGORIES`)

## Contributing

When adding new tests:

1. Follow existing naming conventions
2. Add appropriate documentation
3. Test both CPU and GPU modes if possible
4. Consider test execution time and resource usage
5. Update this README if adding new categories or features

## License

This test suite is part of the Frai project and follows the same licensing terms.