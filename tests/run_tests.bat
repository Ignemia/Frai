@echo off
REM Test runner script for Windows systems

setlocal enabledelayedexpansion

REM Default values
set TEST_TYPE=all
set COVERAGE=false
set VERBOSE=false
set PARALLEL=1
set FAIL_FAST=false
set HTML_REPORT=false

REM Parse command line arguments
:parse_args
if "%~1"=="" goto end_parse
if "%~1"=="unit" (
    set TEST_TYPE=unit
    shift
    goto parse_args
)
if "%~1"=="integration" (
    set TEST_TYPE=integration
    shift
    goto parse_args
)
if "%~1"=="implementation" (
    set TEST_TYPE=implementation
    shift
    goto parse_args
)
if "%~1"=="blackbox" (
    set TEST_TYPE=blackbox
    shift
    goto parse_args
)
if "%~1"=="performance" (
    set TEST_TYPE=performance
    shift
    goto parse_args
)
if "%~1"=="all" (
    set TEST_TYPE=all
    shift
    goto parse_args
)
if "%~1"=="--coverage" (
    set COVERAGE=true
    shift
    goto parse_args
)
if "%~1"=="--verbose" (
    set VERBOSE=true
    shift
    goto parse_args
)
if "%~1"=="-v" (
    set VERBOSE=true
    shift
    goto parse_args
)
if "%~1"=="--parallel" (
    set PARALLEL=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="-n" (
    set PARALLEL=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--fail-fast" (
    set FAIL_FAST=true
    shift
    goto parse_args
)
if "%~1"=="-x" (
    set FAIL_FAST=true
    shift
    goto parse_args
)
if "%~1"=="--html-report" (
    set HTML_REPORT=true
    shift
    goto parse_args
)
if "%~1"=="--help" goto show_help
if "%~1"=="-h" goto show_help

echo Unknown option: %~1
echo Use --help for usage information
exit /b 1

:show_help
echo Usage: %~nx0 [test_type] [options]
echo.
echo Test types:
echo   unit            Run unit tests
echo   integration     Run integration tests
echo   implementation  Run implementation tests
echo   blackbox        Run blackbox tests
echo   performance     Run performance tests
echo   all             Run all tests (default)
echo.
echo Options:
echo   --coverage      Generate coverage report
echo   --verbose, -v   Verbose output
echo   --parallel N    Run with N parallel workers
echo   --fail-fast, -x Stop on first failure
echo   --html-report   Generate HTML coverage report
echo   --help, -h      Show this help message
exit /b 0

:end_parse

REM Change to project root directory
cd /d "%~dp0.."

echo.
echo ============================================================
echo Running tests from: %CD%
echo Test type: %TEST_TYPE%
echo ============================================================
echo.

REM Build pytest command
set PYTEST_CMD=python -m pytest

REM Add test directory
if "%TEST_TYPE%"=="all" (
    set PYTEST_CMD=%PYTEST_CMD% tests/
) else (
    set PYTEST_CMD=%PYTEST_CMD% tests/%TEST_TYPE%/
)

REM Add options
if "%VERBOSE%"=="true" (
    set PYTEST_CMD=%PYTEST_CMD% -v
)

if "%FAIL_FAST%"=="true" (
    set PYTEST_CMD=%PYTEST_CMD% -x
)

if not "%PARALLEL%"=="1" (
    set PYTEST_CMD=%PYTEST_CMD% -n %PARALLEL%
)

if "%COVERAGE%"=="true" (
    set PYTEST_CMD=%PYTEST_CMD% --cov=src --cov-report=term-missing --cov-report=xml
    
    if "%HTML_REPORT%"=="true" (
        set PYTEST_CMD=%PYTEST_CMD% --cov-report=html
    )
)

REM Add configuration file
set PYTEST_CMD=%PYTEST_CMD% -c tests/pytest.ini

REM Check if virtual environment is activated
if "%VIRTUAL_ENV%"=="" (
    echo Warning: No virtual environment detected. Consider activating one.
    echo.
)

REM Check if pytest is installed
python -m pytest --version >nul 2>&1
if errorlevel 1 (
    echo Error: pytest not found. Please install it:
    echo     pip install pytest pytest-cov pytest-xdist
    exit /b 1
)

REM Run pre-test checks
echo ============================================================
echo Pre-test Checks
echo ============================================================

REM Check if test directory exists
if not "%TEST_TYPE%"=="all" (
    if not exist "tests\%TEST_TYPE%" (
        echo Error: Test directory tests\%TEST_TYPE% does not exist
        exit /b 1
    )
)

REM Check Python syntax (simplified for Windows)
echo Checking Python syntax...
for /r tests %%f in (*.py) do (
    python -m py_compile "%%f" >nul 2>&1
    if errorlevel 1 (
        echo Error: Python syntax errors found in %%f
        exit /b 1
    )
)
echo Python syntax check passed.
echo.

REM Run the tests
echo ============================================================
echo Running %TEST_TYPE% Tests
echo Command: %PYTEST_CMD%
echo ============================================================
echo.

%PYTEST_CMD%

if errorlevel 1 (
    echo.
    echo ============================================================
    echo Test run failed!
    echo ============================================================
    exit /b 1
) else (
    echo.
    echo ============================================================
    echo All %TEST_TYPE% tests passed!
    echo ============================================================
    
    if "%COVERAGE%"=="true" (
        echo.
        echo Coverage reports generated:
        echo   - Terminal: Above output
        echo   - XML: coverage.xml
        
        if "%HTML_REPORT%"=="true" (
            echo   - HTML: htmlcov\index.html
            
            REM Try to open HTML report if possible
            if exist "htmlcov\index.html" (
                start htmlcov\index.html
            )
        )
    )
    
    echo.
    echo Test run completed successfully!
)

endlocal
