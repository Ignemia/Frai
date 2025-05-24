#!/bin/bash
# Test runner script for Unix/Linux/macOS systems

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to print section headers
print_header() {
    echo
    print_color $BLUE "============================================================"
    print_color $BLUE "$1"
    print_color $BLUE "============================================================"
}

# Function to run tests with error handling
run_test_command() {
    local description=$1
    local command=$2
    
    print_header "$description"
    echo "Command: $command"
    echo
    
    if eval "$command"; then
        print_color $GREEN "✅ $description completed successfully"
        return 0
    else
        print_color $RED "❌ $description failed"
        return 1
    fi
}

# Default values
TEST_TYPE="all"
COVERAGE=false
VERBOSE=false
PARALLEL=1
FAIL_FAST=false
HTML_REPORT=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        unit|integration|implementation|blackbox|performance|all)
            TEST_TYPE="$1"
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --parallel|-n)
            PARALLEL="$2"
            shift 2
            ;;
        --fail-fast|-x)
            FAIL_FAST=true
            shift
            ;;
        --html-report)
            HTML_REPORT=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [test_type] [options]"
            echo ""
            echo "Test types:"
            echo "  unit            Run unit tests"
            echo "  integration     Run integration tests"
            echo "  implementation  Run implementation tests"
            echo "  blackbox        Run blackbox tests"
            echo "  performance     Run performance tests"
            echo "  all             Run all tests (default)"
            echo ""
            echo "Options:"
            echo "  --coverage      Generate coverage report"
            echo "  --verbose, -v   Verbose output"
            echo "  --parallel N    Run with N parallel workers"
            echo "  --fail-fast, -x Stop on first failure"
            echo "  --html-report   Generate HTML coverage report"
            echo "  --help, -h      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Change to project root directory (script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

print_color $YELLOW "Running tests from: $(pwd)"
print_color $YELLOW "Test type: $TEST_TYPE"

# Build pytest command
PYTEST_CMD="python -m pytest"

# Add test directory
if [[ "$TEST_TYPE" == "all" ]]; then
    PYTEST_CMD="$PYTEST_CMD tests/"
else
    PYTEST_CMD="$PYTEST_CMD tests/$TEST_TYPE/"
fi

# Add options
if [[ "$VERBOSE" == true ]]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

if [[ "$FAIL_FAST" == true ]]; then
    PYTEST_CMD="$PYTEST_CMD -x"
fi

if [[ "$PARALLEL" -gt 1 ]]; then
    PYTEST_CMD="$PYTEST_CMD -n $PARALLEL"
fi

if [[ "$COVERAGE" == true ]]; then
    PYTEST_CMD="$PYTEST_CMD --cov=src --cov-report=term-missing --cov-report=xml"
    
    if [[ "$HTML_REPORT" == true ]]; then
        PYTEST_CMD="$PYTEST_CMD --cov-report=html"
    fi
fi

# Add configuration file
PYTEST_CMD="$PYTEST_CMD -c tests/pytest.ini"

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    print_color $YELLOW "⚠️  No virtual environment detected. Consider activating one."
fi

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    print_color $RED "❌ pytest not found. Please install it:"
    print_color $YELLOW "    pip install pytest pytest-cov pytest-xdist"
    exit 1
fi

# Run pre-test checks
print_header "Pre-test Checks"

# Check if test directory exists
if [[ "$TEST_TYPE" != "all" && ! -d "tests/$TEST_TYPE" ]]; then
    print_color $RED "❌ Test directory tests/$TEST_TYPE does not exist"
    exit 1
fi

# Check Python syntax
print_color $YELLOW "Checking Python syntax..."
if find tests/ -name "*.py" -exec python -m py_compile {} \; 2>/dev/null; then
    print_color $GREEN "✅ Python syntax check passed"
else
    print_color $RED "❌ Python syntax errors found"
    exit 1
fi

# Run the tests
if run_test_command "${TEST_TYPE^} Tests" "$PYTEST_CMD"; then
    print_color $GREEN "🎉 All $TEST_TYPE tests passed!"
    
    if [[ "$COVERAGE" == true ]]; then
        print_header "Coverage Report Generated"
        print_color $YELLOW "📊 Coverage reports:"
        print_color $YELLOW "  - Terminal: Above output"
        print_color $YELLOW "  - XML: coverage.xml"
        
        if [[ "$HTML_REPORT" == true ]]; then
            print_color $YELLOW "  - HTML: htmlcov/index.html"
            
            # Try to open HTML report if on macOS
            if [[ "$OSTYPE" == "darwin"* ]] && command -v open &> /dev/null; then
                print_color $YELLOW "  Opening HTML report..."
                open htmlcov/index.html
            fi
        fi
    fi
    
    print_color $GREEN "✅ Test run completed successfully!"
else
    print_color $RED "💥 $TEST_TYPE tests failed!"
    exit 1
fi
