@echo off
echo ================================
echo    FRAI PROJECT TEST SUITE
echo ================================
echo.

REM Change to project directory
cd /d "%~dp0"

echo Installing test dependencies...
python -m pip install pytest pytest-cov requests --quiet

echo.
echo Running comprehensive test suite...
python run_all_tests.py --all --coverage --verbose

pause
