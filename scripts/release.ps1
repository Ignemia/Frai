# Frai Release Script for Windows PowerShell
# Automates version bumping, testing, and release preparation

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("major", "minor", "patch")]
    [string]$VersionType,
    
    [switch]$DryRun = $false,
    [switch]$SkipTests = $false,
    [switch]$SkipQuality = $false,
    [switch]$Help = $false
)

# Color output functions
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Write-Success { param([string]$Message) Write-ColorOutput "✅ $Message" "Green" }
function Write-Error { param([string]$Message) Write-ColorOutput "❌ $Message" "Red" }
function Write-Warning { param([string]$Message) Write-ColorOutput "⚠️ $Message" "Yellow" }
function Write-Info { param([string]$Message) Write-ColorOutput "ℹ️ $Message" "Cyan" }

function Show-Help {
    Write-Host @"
Personal Chatter Release Script

Usage: .\release.ps1 -VersionType <type> [options]

Parameters:
  -VersionType    Type of version bump (major, minor, patch)
  -DryRun         Show what would be done without making changes
  -SkipTests      Skip running the test suite
  -SkipQuality    Skip code quality checks
  -Help           Show this help message

Examples:
  .\release.ps1 -VersionType patch                    # Patch release
  .\release.ps1 -VersionType minor -DryRun            # Preview minor release
  .\release.ps1 -VersionType major -SkipTests         # Major release without tests

"@
    exit 0
}

if ($Help) { Show-Help }

# Get script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

Write-Info "Personal Chatter Release Manager"
Write-Info "Project Root: $ProjectRoot"
Write-Info "Version Type: $VersionType"
Write-Info "Dry Run: $DryRun"

# Function to get current version from pyproject.toml
function Get-CurrentVersion {
    $PyProjectPath = Join-Path $ProjectRoot "pyproject.toml"
    
    if (Test-Path $PyProjectPath) {
        $content = Get-Content $PyProjectPath
        foreach ($line in $content) {
            if ($line -match '^version = "(.+)"$') {
                return $matches[1]
            }
        }
    }
    
    # Fallback to git tags
    try {
        $gitVersion = git describe --tags --abbrev=0 2>$null
        if ($gitVersion) {
            return $gitVersion.TrimStart('v')
        }
    } catch {
        # Ignore git errors
    }
    
    return "0.1.0"  # Default version
}

# Function to bump version
function New-Version {
    param([string]$Current, [string]$Type)
    
    $parts = $Current.Split('.')
    if ($parts.Length -ne 3) {
        throw "Invalid version format: $Current"
    }
    
    $major = [int]$parts[0]
    $minor = [int]$parts[1]
    $patch = [int]$parts[2]
    
    switch ($Type) {
        "major" { $major++; $minor = 0; $patch = 0 }
        "minor" { $minor++; $patch = 0 }
        "patch" { $patch++ }
    }
    
    return "$major.$minor.$patch"
}

# Function to update version in files
function Update-VersionFiles {
    param([string]$NewVersion)
    
    $PyProjectPath = Join-Path $ProjectRoot "pyproject.toml"
    
    if (Test-Path $PyProjectPath) {
        $content = Get-Content $PyProjectPath -Raw
        $content = $content -replace '^version = ".+"$', "version = `"$NewVersion`"", "Multiline"
        Set-Content -Path $PyProjectPath -Value $content -NoNewline
        Write-Success "Updated version in pyproject.toml"
    }
}

# Function to update changelog
function Update-Changelog {
    param([string]$NewVersion)
    
    $ChangelogPath = Join-Path $ProjectRoot "CHANGELOG.md"
    
    if (Test-Path $ChangelogPath) {
        $content = Get-Content $ChangelogPath -Raw
        $today = Get-Date -Format "yyyy-MM-dd"
        $content = $content -replace "## \[Unreleased\]", "## [Unreleased]`n`n## [$NewVersion] - $today"
        Set-Content -Path $ChangelogPath -Value $content -NoNewline
        Write-Success "Updated CHANGELOG.md"
    } else {
        Write-Warning "CHANGELOG.md not found"
    }
}

# Function to run tests
function Invoke-Tests {
    Write-Info "Running test suite..."
    
    $result = & python -m pytest --cov=api --cov=services --cov-fail-under=80
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Tests failed!"
        return $false
    }
    
    Write-Success "All tests passed!"
    return $true
}

# Function to run quality checks
function Invoke-QualityChecks {
    Write-Info "Running quality checks..."
    
    $checks = @(
        @{cmd = "black"; args = @("--check", "."); desc = "Code formatting (black)"},
        @{cmd = "isort"; args = @("--check-only", "."); desc = "Import sorting (isort)"},
        @{cmd = "ruff"; args = @("check", "."); desc = "Linting (ruff)"},
        @{cmd = "mypy"; args = @("api/", "services/"); desc = "Type checking (mypy)"},
        @{cmd = "bandit"; args = @("-r", "api/", "services/", "-ll"); desc = "Security (bandit)"}
    )
    
    foreach ($check in $checks) {
        Write-Info "  • $($check.desc)..."
        $result = & $check.cmd $check.args 2>&1
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error "$($check.desc) failed!"
            Write-Host $result
            return $false
        }
    }
    
    Write-Success "All quality checks passed!"
    return $true
}

# Function to build packages
function Build-Packages {
    Write-Info "Building packages..."
    
    # Clean previous builds
    Remove-Item -Path "build", "dist", "*.egg-info" -Recurse -Force -ErrorAction SilentlyContinue
    
    $result = & python -m build
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Package build failed!"
        return $false
    }
    
    Write-Success "Packages built successfully!"
    return $true
}

# Function to create git tag
function New-GitTag {
    param([string]$Version)
    
    Write-Info "Creating git tag v$Version..."
    
    # Add all changes
    git add .
    
    # Commit changes
    git commit -m "Release v$Version"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Git commit failed!"
        return $false
    }
    
    # Create tag
    git tag -a "v$Version" -m "Release v$Version"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Git tag creation failed!"
        return $false
    }
    
    Write-Success "Git tag v$Version created!"
    return $true
}

# Function to push release
function Push-Release {
    Write-Info "Pushing release to remote..."
    
    # Push commits
    git push
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to push commits!"
        return $false
    }
    
    # Push tags
    git push --tags
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to push tags!"
        return $false
    }
    
    Write-Success "Release pushed to remote!"
    return $true
}

# Main execution
try {
    Write-Info "🚀 Starting release process ($VersionType)..."
    
    # Get current and new version
    $currentVersion = Get-CurrentVersion
    $newVersion = New-Version -Current $currentVersion -Type $VersionType
    
    Write-Info "Bumping version: $currentVersion → $newVersion"
    
    if (-not $DryRun) {
        # Run tests
        if (-not $SkipTests -and -not (Invoke-Tests)) {
            exit 1
        }
        
        # Run quality checks
        if (-not $SkipQuality -and -not (Invoke-QualityChecks)) {
            exit 1
        }
        
        # Update version files
        Update-VersionFiles -NewVersion $newVersion
        Update-Changelog -NewVersion $newVersion
        
        # Build packages
        if (-not (Build-Packages)) {
            exit 1
        }
        
        # Create git tag
        if (-not (New-GitTag -Version $newVersion)) {
            exit 1
        }
        
        # Push to remote
        if (-not (Push-Release)) {
            exit 1
        }
    } else {
        Write-Warning "DRY RUN - No changes made"
    }
    
    Write-Success "🎉 Release v$newVersion completed successfully!"
    Write-Info "🔗 GitHub Actions will handle the rest of the deployment."
    
} catch {
    Write-Error "Release failed: $($_.Exception.Message)"
    exit 1
}
