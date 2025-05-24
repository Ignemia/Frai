#!/usr/bin/env python3
"""
Personal Chatter Release Script
Automates version bumping, testing, quality checks, and release preparation.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Color output for terminal
class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(message: str, color: str = Colors.ENDC):
    """Print colored message to terminal."""
    print(f"{color}{message}{Colors.ENDC}")

def print_success(message: str):
    """Print success message."""
    print_colored(f"✅ {message}", Colors.OKGREEN)

def print_error(message: str):
    """Print error message."""
    print_colored(f"❌ {message}", Colors.FAIL)

def print_warning(message: str):
    """Print warning message."""
    print_colored(f"⚠️ {message}", Colors.WARNING)

def print_info(message: str):
    """Print info message."""
    print_colored(f"ℹ️ {message}", Colors.OKBLUE)

def run_command(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    try:
        print_info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            capture_output=True, 
            text=True, 
            check=check
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {' '.join(cmd)}")
        if e.stderr:
            print_error(f"Error: {e.stderr}")
        if check:
            raise
        return e

class ReleaseManager:
    """Manages the release process for Personal Chatter."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.pyproject_path = project_root / "pyproject.toml"
        self.changelog_path = project_root / "CHANGELOG.md"
        
    def get_current_version(self) -> str:
        """Get current version from pyproject.toml."""
        try:
            with open(self.pyproject_path, 'r') as f:
                content = f.read()
                
            # Look for version in [project] section
            match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                return match.group(1)
                
            # Look for dynamic version from git
            if 'dynamic = ["version"]' in content:
                result = run_command(["git", "describe", "--tags", "--abbrev=0"], 
                                   cwd=self.project_root, check=False)
                if result.returncode == 0:
                    return result.stdout.strip().lstrip('v')
                    
            return "0.1.0"  # Default version
            
        except Exception as e:
            print_warning(f"Could not determine current version: {e}")
            return "0.1.0"
    
    def bump_version(self, current_version: str, version_type: str) -> str:
        """Bump version according to semantic versioning."""
        parts = list(map(int, current_version.split('.')))
        
        if version_type == "major":
            parts[0] += 1
            parts[1] = 0
            parts[2] = 0
        elif version_type == "minor":
            parts[1] += 1
            parts[2] = 0
        elif version_type == "patch":
            parts[2] += 1
        else:
            raise ValueError(f"Invalid version type: {version_type}")
            
        return '.'.join(map(str, parts))
    
    def update_version_files(self, new_version: str):
        """Update version in project files."""
        # Update pyproject.toml
        if self.pyproject_path.exists():
            with open(self.pyproject_path, 'r') as f:
                content = f.read()
            
            # Update version line
            content = re.sub(
                r'version\s*=\s*["\'][^"\']+["\']',
                f'version = "{new_version}"',
                content
            )
            
            with open(self.pyproject_path, 'w') as f:
                f.write(content)
            print_success("Updated version in pyproject.toml")
    
    def update_changelog(self, new_version: str):
        """Update CHANGELOG.md with new version."""
        if not self.changelog_path.exists():
            print_warning("CHANGELOG.md not found")
            return
            
        with open(self.changelog_path, 'r') as f:
            content = f.read()
        
        today = time.strftime("%Y-%m-%d")
        content = content.replace(
            "## [Unreleased]",
            f"## [Unreleased]\n\n## [{new_version}] - {today}"
        )
        
        with open(self.changelog_path, 'w') as f:
            f.write(content)
        print_success("Updated CHANGELOG.md")
    
    def run_tests(self) -> bool:
        """Run the test suite."""
        print_info("Running test suite...")
        
        # Install test dependencies
        run_command([
            sys.executable, "-m", "pip", "install", 
            "-r", "requirements-test.txt"
        ], cwd=self.project_root)
        
        # Run tests with coverage
        result = run_command([
            sys.executable, "-m", "pytest", 
            "--cov=api", "--cov=services", 
            "--cov-fail-under=80",
            "tests/"
        ], cwd=self.project_root, check=False)
        
        if result.returncode != 0:
            print_error("Tests failed!")
            return False
            
        print_success("All tests passed!")
        return True
    
    def run_quality_checks(self) -> bool:
        """Run code quality checks."""
        print_info("Running quality checks...")
        
        checks = [
            (["python", "-m", "black", "--check", "."], "Code formatting (black)"),
            (["python", "-m", "isort", "--check-only", "."], "Import sorting (isort)"),
            (["python", "-m", "ruff", "check", "."], "Linting (ruff)"),
            (["python", "-m", "mypy", "api/", "services/"], "Type checking (mypy)"),
            (["python", "-m", "bandit", "-r", "api/", "services/", "-ll"], "Security (bandit)")
        ]
        
        for cmd, desc in checks:
            print_info(f"  • {desc}...")
            result = run_command(cmd, cwd=self.project_root, check=False)
            
            if result.returncode != 0:
                print_error(f"{desc} failed!")
                return False
        
        print_success("All quality checks passed!")
        return True
    
    def build_packages(self) -> bool:
        """Build distribution packages."""
        print_info("Building packages...")
        
        # Clean previous builds
        for path in ["build", "dist", "*.egg-info"]:
            full_path = self.project_root / path
            if full_path.exists():
                if full_path.is_dir():
                    import shutil
                    shutil.rmtree(full_path)
                else:
                    full_path.unlink()
        
        # Build packages
        result = run_command([
            sys.executable, "-m", "build"
        ], cwd=self.project_root, check=False)
        
        if result.returncode != 0:
            print_error("Package build failed!")
            return False
            
        print_success("Packages built successfully!")
        return True
    
    def create_git_tag(self, version: str) -> bool:
        """Create git tag for the release."""
        print_info(f"Creating git tag v{version}...")
        
        # Add all changes
        run_command(["git", "add", "."], cwd=self.project_root)
        
        # Commit changes
        result = run_command([
            "git", "commit", "-m", f"Release v{version}"
        ], cwd=self.project_root, check=False)
        
        if result.returncode != 0:
            print_error("Git commit failed!")
            return False
        
        # Create tag
        result = run_command([
            "git", "tag", "-a", f"v{version}", "-m", f"Release v{version}"
        ], cwd=self.project_root, check=False)
        
        if result.returncode != 0:
            print_error("Git tag creation failed!")
            return False
            
        print_success(f"Git tag v{version} created!")
        return True
    
    def push_release(self) -> bool:
        """Push release to remote repository."""
        print_info("Pushing release to remote...")
        
        # Push commits
        result = run_command(["git", "push"], cwd=self.project_root, check=False)
        if result.returncode != 0:
            print_error("Failed to push commits!")
            return False
        
        # Push tags
        result = run_command(["git", "push", "--tags"], cwd=self.project_root, check=False)
        if result.returncode != 0:
            print_error("Failed to push tags!")
            return False
            
        print_success("Release pushed to remote!")
        return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Personal Chatter Release Manager")
    parser.add_argument(
        "version_type", 
        choices=["major", "minor", "patch"],
        help="Type of version bump"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--skip-tests", 
        action="store_true",
        help="Skip running tests"
    )
    parser.add_argument(
        "--skip-quality", 
        action="store_true",
        help="Skip quality checks"
    )
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent
    manager = ReleaseManager(project_root)
    
    try:
        print_info(f"🚀 Starting release process ({args.version_type})...")
        
        # Get current and new version
        current_version = manager.get_current_version()
        new_version = manager.bump_version(current_version, args.version_type)
        
        print_info(f"Bumping version: {current_version} → {new_version}")
        
        if args.dry_run:
            print_warning("DRY RUN - No changes will be made")
            return
        
        # Run tests
        if not args.skip_tests:
            if not manager.run_tests():
                sys.exit(1)
        
        # Run quality checks
        if not args.skip_quality:
            if not manager.run_quality_checks():
                sys.exit(1)
        
        # Update version files
        manager.update_version_files(new_version)
        manager.update_changelog(new_version)
        
        # Build packages
        if not manager.build_packages():
            sys.exit(1)
        
        # Create git tag
        if not manager.create_git_tag(new_version):
            sys.exit(1)
        
        # Push to remote
        if not manager.push_release():
            sys.exit(1)
        
        print_success(f"🎉 Release v{new_version} completed successfully!")
        print_info("🔗 GitHub Actions will handle the rest of the deployment.")
        
    except KeyboardInterrupt:
        print_error("Release process interrupted!")
        sys.exit(1)
    except Exception as e:
        print_error(f"Release failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()