@echo off
REM Enhanced Windows Batch Test Runner for Frai
REM Provides easy access to common testing scenarios

setlocal enabledelayedexpansion

REM Set colors for output
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "NC=[0m"

REM Get the script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%\.."

echo %BLUE%Personal Chatter Test Suite%NC%
echo ============================

REM Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo %YELLOW%Warning: Virtual environment not found at venv\Scripts\python.exe%NC%
    echo %YELLOW%Using system Python instead%NC%
    set "PYTHON=python"
) else (
    echo %GREEN%Using virtual environment%NC%
    set "PYTHON=venv\Scripts\python.exe"
)

REM Parse command line arguments
set "TEST_TYPE=all"
set "COVERAGE_FLAG="
set "VERBOSE_FLAG="
set "PARALLEL_FLAG="

if "%1"=="" goto :run_tests
if "%1"=="help" goto :show_help
if "%1"=="--help" goto :show_help
if "%1"=="-h" goto :show_help

:parse_args
if "%1"=="" goto :run_tests
if "%1"=="unit" set "TEST_TYPE=unit"
if "%1"=="integration" set "TEST_TYPE=integration"
if "%1"=="implementation" set "TEST_TYPE=implementation"
if "%1"=="blackbox" set "TEST_TYPE=blackbox"
if "%1"=="performance" set "TEST_TYPE=performance"
if "%1"=="quick" set "TEST_TYPE=quick"
if "%1"=="all" set "TEST_TYPE=all"
if "%1"=="--coverage" set "COVERAGE_FLAG=--cov=api --cov=services --cov-report=html --cov-report=term-missing"
if "%1"=="-c" set "COVERAGE_FLAG=--cov=api --cov=services --cov-report=html --cov-report=term-missing"
if "%1"=="--verbose" set "VERBOSE_FLAG=-v"
if "%1"=="-v" set "VERBOSE_FLAG=-v"
if "%1"=="--parallel" set "PARALLEL_FLAG=-n auto"
if "%1"=="-p" set "PARALLEL_FLAG=-n auto"
shift
goto :parse_args

:run_tests
echo %BLUE%Running %TEST_TYPE% tests...%NC%
echo.

REM Set test command based on test type
if "%TEST_TYPE%"=="unit" (
    set "TEST_CMD=%PYTHON% -m pytest tests\unit\ %COVERAGE_FLAG% %VERBOSE_FLAG% %PARALLEL_FLAG%"
) else if "%TEST_TYPE%"=="integration" (
    set "TEST_CMD=%PYTHON% -m pytest tests\integration\ %COVERAGE_FLAG% %VERBOSE_FLAG% %PARALLEL_FLAG%"
) else if "%TEST_TYPE%"=="implementation" (
    set "TEST_CMD=%PYTHON% -m pytest tests\implementation\ %COVERAGE_FLAG% %VERBOSE_FLAG% %PARALLEL_FLAG%"
) else if "%TEST_TYPE%"=="blackbox" (
    set "TEST_CMD=%PYTHON% -m pytest tests\blackbox\ %COVERAGE_FLAG% %VERBOSE_FLAG% %PARALLEL_FLAG%"
) else if "%TEST_TYPE%"=="performance" (
    set "TEST_CMD=%PYTHON% -m pytest tests\performance\ %COVERAGE_FLAG% %VERBOSE_FLAG% --benchmark-only"
) else if "%TEST_TYPE%"=="quick" (
    set "TEST_CMD=%PYTHON% -m pytest tests\unit\ tests\integration\ -m "not slow" %COVERAGE_FLAG% %VERBOSE_FLAG% %PARALLEL_FLAG%"
) else (
    set "TEST_CMD=%PYTHON% -m pytest %COVERAGE_FLAG% %VERBOSE_FLAG% %PARALLEL_FLAG%"
)

REM Check if pytest is available
%PYTHON% -c "import pytest" 2>nul
if errorlevel 1 (
    echo %RED%Error: pytest not found. Please install test dependencies:%NC%
    echo pip install -r requirements-test.txt
    goto :error_exit
)

REM Run the tests
echo %GREEN%Executing: %TEST_CMD%%NC%
echo.
%TEST_CMD%

if errorlevel 1 (
    echo.
    echo %RED%Tests failed!%NC%
    goto :error_exit
) else (
    echo.
    echo %GREEN%All tests passed!%NC%
)

REM Show coverage report location if coverage was enabled
if not "%COVERAGE_FLAG%"=="" (
    echo.
    echo %BLUE%Coverage report available at: tests\coverage_html\index.html%NC%
)

goto :success_exit

:show_help
echo Usage: run_tests.bat [test_type] [options]
echo.
echo Test Types:
echo   unit           Run unit tests only
echo   integration    Run integration tests only
echo   implementation Run implementation tests only
echo   blackbox       Run blackbox/end-to-end tests only
echo   performance    Run performance tests only
echo   quick          Run quick tests (unit + integration, no slow tests)
echo   all            Run all test suites (default)
echo.
echo Options:
echo   --coverage, -c    Generate coverage reports
echo   --verbose, -v     Verbose output
echo   --parallel, -p    Run tests in parallel
echo   --help, -h        Show this help message
echo.
echo Examples:
echo   run_tests.bat                    # Run all tests
echo   run_tests.bat unit --coverage    # Run unit tests with coverage
echo   run_tests.bat quick --verbose    # Run quick tests with verbose output
echo   run_tests.bat performance        # Run performance benchmarks
goto :success_exit

:error_exit
echo.
echo %RED%Test execution failed. Check the output above for details.%NC%
exit /b 1

:success_exit
echo.
echo %GREEN%Test execution completed successfully.%NC%
exit /b 0
