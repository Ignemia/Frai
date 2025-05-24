# PowerShell Test Runner for Frai
# Enhanced version with comprehensive Windows support

param(
    [string]$TestType = "all",
    [switch]$Coverage = $false,
    [int]$Parallel = 1,
    [switch]$Verbose = $false,
    [switch]$Report = $false,
    [switch]$FailFast = $false,
    [string]$OutputDir = "test-results",
    [switch]$Help = $false
)

function Show-Help {
    Write-Host @"
Personal Chatter Test Suite - PowerShell Runner

Usage: .\run_tests.ps1 [TestType] [Options]

Test Types:
  all            Run all test suites (default)
  unit           Run unit tests only
  integration    Run integration tests only
  implementation Run implementation tests only
  blackbox       Run blackbox/end-to-end tests only
  performance    Run performance tests only
  quick          Run quick tests (unit + integration, no slow tests)

Options:
  -Coverage      Generate coverage reports
  -Parallel N    Run tests in parallel with N workers (default: 1)
  -Verbose       Enable verbose output
  -Report        Generate JUnit XML report for CI
  -FailFast      Stop on first test failure
  -OutputDir DIR Specify output directory (default: test-results)
  -Help          Show this help message

Examples:
  .\run_tests.ps1 unit -Coverage -Verbose
  .\run_tests.ps1 all -Parallel 4 -Report
  .\run_tests.ps1 quick -FailFast
"@
    exit 0
}

if ($Help) {
    Show-Help
}

# Header
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Personal Chatter Test Suite - PowerShell" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Validate test type
$ValidTestTypes = @("all", "unit", "integration", "implementation", "blackbox", "performance", "quick")
if ($TestType -notin $ValidTestTypes) {
    Write-Host "ERROR: Invalid test type '$TestType'" -ForegroundColor Red
    Write-Host "Valid types: $($ValidTestTypes -join ', ')" -ForegroundColor Yellow
    exit 1
}

# Display configuration
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Test Type: $TestType"
Write-Host "  Coverage: $Coverage"
Write-Host "  Parallel: $Parallel"
Write-Host "  Verbose: $Verbose"
Write-Host "  Report: $Report"
Write-Host "  Output Directory: $OutputDir"
Write-Host ""

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Create output directory
if (!(Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    Write-Host "Created output directory: $OutputDir" -ForegroundColor Green
}

# Check if pytest is installed
try {
    $pytestVersion = python -m pytest --version 2>&1
    Write-Host "Pytest: $pytestVersion" -ForegroundColor Green
} catch {
    Write-Host "Installing pytest and dependencies..." -ForegroundColor Yellow
    $installCmd = "pip install pytest pytest-cov pytest-xdist pytest-mock pytest-benchmark pytest-timeout pytest-sugar pytest-clarity"
    Invoke-Expression $installCmd
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install pytest dependencies" -ForegroundColor Red
        exit 1
    }
    Write-Host "Dependencies installed successfully" -ForegroundColor Green
}

# Build pytest command
$pytestCmd = "python -m pytest"
$pytestArgs = @()

# Set test paths based on type
switch ($TestType) {
    "unit" { $testPaths = "tests/unit" }
    "integration" { $testPaths = "tests/integration" }
    "implementation" { $testPaths = "tests/implementation" }
    "blackbox" { $testPaths = "tests/blackbox" }
    "performance" { $testPaths = "tests/performance" }
    "quick" { 
        $testPaths = "tests/unit tests/integration"
        $pytestArgs += "-m", "not slow"
    }
    "all" { $testPaths = "tests/" }
}

# Add test paths
$pytestArgs += $testPaths.Split()

# Add coverage options
if ($Coverage) {
    $pytestArgs += "--cov=api", "--cov=services"
    $pytestArgs += "--cov-report=html", "--cov-report=xml", "--cov-report=term-missing"
    Write-Host "Coverage reporting enabled" -ForegroundColor Green
}

# Add parallel execution
if ($Parallel -gt 1) {
    $pytestArgs += "-n", $Parallel.ToString()
    Write-Host "Parallel execution: $Parallel workers" -ForegroundColor Green
}

# Add verbosity
if ($Verbose) {
    $pytestArgs += "-v", "-s"
}

# Add fail fast
if ($FailFast) {
    $pytestArgs += "-x"
}

# Add JUnit XML output for CI
if ($Report) {
    $junitPath = Join-Path $OutputDir "junit.xml"
    $pytestArgs += "--junitxml=$junitPath"
    Write-Host "JUnit XML report will be generated: $junitPath" -ForegroundColor Green
}

# Add configuration file
$configPath = "tests/pytest.ini"
if (Test-Path $configPath) {
    $pytestArgs += "-c", $configPath
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Running Tests" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Show command that will be executed
$fullCommand = "$pytestCmd $($pytestArgs -join ' ')"
Write-Host "Command: $fullCommand" -ForegroundColor Yellow
Write-Host ""

# Set environment variables for testing
$env:PYTEST_CURRENT_TEST = "1"
$env:CI = "true"
$env:TESTING = "true"

# Measure execution time
$startTime = Get-Date

# Execute tests
try {
    $process = Start-Process -FilePath "python" -ArgumentList @("-m", "pytest") + $pytestArgs -NoNewWindow -Wait -PassThru
    $exitCode = $process.ExitCode
} catch {
    Write-Host "ERROR: Failed to execute tests: $_" -ForegroundColor Red
    exit 1
}

$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Test Execution Summary" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

Write-Host "Test Type: $TestType"
Write-Host "Duration: $($duration.ToString('mm\:ss\.fff'))"
Write-Host "Exit Code: $exitCode"

if ($exitCode -eq 0) {
    Write-Host "Status: PASSED ✓" -ForegroundColor Green
} else {
    Write-Host "Status: FAILED ✗" -ForegroundColor Red
}

Write-Host ""
Write-Host "Output files generated in: $OutputDir"

# Show generated reports
if ($Coverage) {
    $htmlCov = "htmlcov/index.html"
    $xmlCov = "coverage.xml"
    
    if (Test-Path $htmlCov) {
        Write-Host "Coverage HTML report: $htmlCov" -ForegroundColor Green
    }
    if (Test-Path $xmlCov) {
        Write-Host "Coverage XML report: $xmlCov" -ForegroundColor Green
    }
}

if ($Report) {
    $junitPath = Join-Path $OutputDir "junit.xml"
    if (Test-Path $junitPath) {
        Write-Host "JUnit XML report: $junitPath" -ForegroundColor Green
    }
}

Write-Host "============================================" -ForegroundColor Cyan

# Offer to open coverage report
if ($Coverage -and (Test-Path "htmlcov/index.html")) {
    Write-Host ""
    $openCoverage = Read-Host "Open coverage report in browser? (y/n)"
    if ($openCoverage -eq "y" -or $openCoverage -eq "Y") {
        Start-Process "htmlcov/index.html"
    }
}

# Clean up environment
Remove-Item Env:PYTEST_CURRENT_TEST -ErrorAction SilentlyContinue
Remove-Item Env:CI -ErrorAction SilentlyContinue
Remove-Item Env:TESTING -ErrorAction SilentlyContinue

exit $exitCode
