#!/usr/bin/env python3
"""
Comprehensive installation script for Frai with Python 3.13 compatibility.
Handles PyTorch installation, dependency conflicts, and provides fallback mechanisms.
"""
import os
import platform
import subprocess
import sys
import warnings
from typing import List, Optional, Tuple

# Minimum Python version requirement
MIN_PYTHON_VERSION = (3, 12)
CURRENT_PYTHON_VERSION = sys.version_info[:2]

def check_python_version():
    """Check if Python version meets requirements."""
    if CURRENT_PYTHON_VERSION < MIN_PYTHON_VERSION:
        print(f"Error: Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+ is required.")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    
    if CURRENT_PYTHON_VERSION >= (3, 13):
        print(f"Warning: Python {CURRENT_PYTHON_VERSION[0]}.{CURRENT_PYTHON_VERSION[1]} detected.")
        print("Some dependencies may not have pre-built wheels for this version.")
        print("Installation will continue with compatibility workarounds.\n")

def check_cuda() -> Optional[str]:
    """Check if CUDA is available and return version if found."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "CUDA Version:" in line:
                    return line.split("CUDA Version:")[1].strip()
            return "unknown"
    except FileNotFoundError:
        pass
    return None

def check_rocm() -> bool:
    """Check if ROCm is available."""
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, check=False
        )
        return result.returncode == 0
    except FileNotFoundError:
        pass
    return False

def get_torch_install_command() -> Tuple[List[str], List[str]]:
    """Determine the appropriate PyTorch installation command."""
    cuda_version = check_cuda()
    has_rocm = check_rocm()
    
    packages = ["torch", "torchvision", "torchaudio"]
    
    if cuda_version:
        print(f"CUDA detected (version: {cuda_version})")
        if platform.system() == "Windows":
            extra_args = ["-f", "https://download.pytorch.org/whl/cu121/torch_stable.html"]
        else:
            extra_args = []
    elif has_rocm:
        print("ROCm detected")
        extra_args = ["-f", "https://download.pytorch.org/whl/rocm5.6/torch_stable.html"]
    else:
        print("No GPU acceleration detected, installing CPU-only PyTorch")
        extra_args = ["-f", "https://download.pytorch.org/whl/cpu/torch_stable.html"]
    
    return packages, extra_args

def run_pip_command(packages: List[str], extra_args: List[str] = None, description: str = ""):
    """Run pip install command with error handling."""
    if extra_args is None:
        extra_args = []
    
    cmd = [sys.executable, "-m", "pip", "install"] + packages + extra_args
    print(f"Installing {description}: {' '.join(packages)}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Successfully installed {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {description}")
        print(f"Error: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def install_torch():
    """Install PyTorch with appropriate backend."""
    print("\n" + "="*60)
    print("INSTALLING PYTORCH")
    print("="*60)
    
    packages, extra_args = get_torch_install_command()
    success = run_pip_command(packages, extra_args, "PyTorch")
    
    if not success:
        print("\nTrying fallback CPU-only PyTorch installation...")
        fallback_args = ["-f", "https://download.pytorch.org/whl/cpu/torch_stable.html"]
        success = run_pip_command(packages, fallback_args, "PyTorch (CPU fallback)")
    
    if not success:
        raise RuntimeError("Failed to install PyTorch")

def install_optional_package(package_name: str, alternatives: List[str] = None):
    """Install an optional package with fallback alternatives."""
    if alternatives is None:
        alternatives = []
    
    print(f"\nInstalling optional package: {package_name}")
    
    # Try main package
    if run_pip_command([package_name], description=package_name):
        return True
    
    # Try alternatives
    for alt in alternatives:
        print(f"Trying alternative: {alt}")
        if run_pip_command([alt], description=f"{package_name} (alternative: {alt})"):
            return True
    
    print(f"Warning: Could not install {package_name} or alternatives")
    return False

def install_core_dependencies():
    """Install core dependencies that are required."""
    print("\n" + "="*60)
    print("INSTALLING CORE DEPENDENCIES")
    print("="*60)
    
    core_packages = [
        "accelerate",
        "bcrypt",
        "diffusers",
        "email-validator",
        "fastapi",
        "httptools",
        "inquirerpy",
        "langchain",
        "langchain-community",
        "passlib",
        "pip-chill",
        "protobuf",
        "psycopg2-binary",
        "pydantic[email]",
        "pyjwt",
        "python-dotenv",
        "python-jose",
        "python-multipart",
        "sqlalchemy",
        "transformers",
        "uvicorn",
        "watchfiles",
        "websockets"
    ]
    
    failed_packages = []
    
    for package in core_packages:
        if not run_pip_command([package], description=package):
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\nWarning: Failed to install core packages: {failed_packages}")
        print("The application may not function correctly without these.")
    
    return len(failed_packages) == 0

def install_optional_dependencies():
    """Install optional dependencies with compatibility handling."""
    print("\n" + "="*60)
    print("INSTALLING OPTIONAL DEPENDENCIES")
    print("="*60)
    
    # Dependencies that may have Python 3.13 issues
    optional_deps = {
        "sentence-transformers": {
            "alternatives": [],
            "description": "Sentence transformers for embeddings"
        },
        "chromadb": {
            "alternatives": [],
            "description": "Vector database for embeddings"
        },
        "ctransformers": {
            "alternatives": [],
            "description": "C++ transformer implementations"
        },
        "tf-keras": {
            "alternatives": ["keras"],
            "description": "TensorFlow Keras for ML models"
        }
    }
    
    # For Python 3.13, be more cautious with problematic packages
    if CURRENT_PYTHON_VERSION >= (3, 13):
        print("Python 3.13 detected - using compatibility mode for optional dependencies")
        
        # Try sentencepiece separately first
        print("\nTrying to install sentencepiece (required by sentence-transformers)...")
        sp_success = run_pip_command(["sentencepiece"], description="sentencepiece")
        
        if sp_success:
            install_optional_package("sentence-transformers")
        else:
            print("Skipping sentence-transformers due to sentencepiece build failure")
            print("You can try installing it manually later with system cmake/pkg-config")
    else:
        # For older Python versions, try all optional dependencies
        for package, config in optional_deps.items():
            install_optional_package(package, config["alternatives"])

def install_xformers():
    """Install xformers for GPU acceleration if possible."""
    print("\n" + "="*60)
    print("INSTALLING XFORMERS (GPU ACCELERATION)")
    print("="*60)
    
    cuda_version = check_cuda()
    
    if not cuda_version:
        print("Skipping xformers installation (CUDA not detected)")
        return
    
    print("Installing xformers for GPU acceleration...")
    success = run_pip_command(["xformers"], description="xformers")
    
    if not success:
        print("Warning: xformers installation failed")
        print("This is not critical - the application will work without xformers")
        print("You can try installing it manually later with: pip install xformers")

def install_frai():
    """Install Frai package itself."""
    print("\n" + "="*60)
    print("INSTALLING FRAI PACKAGE")
    print("="*60)
    
    install_type = ".[dev,test]" if "--dev" in sys.argv else "."
    cmd = [sys.executable, "-m", "pip", "install", "-e", install_type]
    
    print(f"Installing Frai in editable mode...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Frai package installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print("✗ Failed to install Frai package")
        print(f"Error: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def create_troubleshooting_guide():
    """Create a troubleshooting guide for common issues."""
    guide = """
TROUBLESHOOTING GUIDE
=====================

1. Python 3.13 Compatibility Issues:
   - Some packages may not have pre-built wheels for Python 3.13
   - Solution: Install build tools or use Python 3.12

2. sentencepiece Build Failures:
   - Requires cmake and pkg-config
   - Ubuntu/Debian: sudo apt install cmake pkg-config
   - macOS: brew install cmake pkg-config
   - Or use pre-built wheel: pip install sentencepiece --find-links https://download.pytorch.org/whl/torch_stable.html

3. xformers Installation Issues:
   - Only supported with CUDA
   - Build from source may take a long time
   - Optional - application works without it

4. General Installation Issues:
   - Update pip: python -m pip install --upgrade pip
   - Clear cache: python -m pip cache purge
   - Try with --no-cache-dir flag

5. Virtual Environment Issues:
   - Create fresh venv: python -m venv .venv
   - Activate: source .venv/bin/activate (Linux/Mac) or .venv\\Scripts\\activate (Windows)
   - Install in venv: python install.py

For more help, visit: https://github.com/Ignemia/Frai/issues
"""
    
    with open("TROUBLESHOOTING.md", "w") as f:
        f.write(guide)
    print("Created TROUBLESHOOTING.md with installation help")

def main():
    """Main installation function."""
    print("FRAI INSTALLATION SCRIPT")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print("=" * 60)
    
    try:
        # Check Python version compatibility
        check_python_version()
        
        # Install components in order
        install_torch()
        
        core_success = install_core_dependencies()
        
        install_optional_dependencies()
        
        install_xformers()
        
        frai_success = install_frai()
        
        # Final summary
        print("\n" + "="*60)
        print("INSTALLATION SUMMARY")
        print("="*60)
        
        if frai_success and core_success:
            print("✓ Frai installation completed successfully!")
            print("\nYou can now run:")
            print("  frai         - Main application")
            print("  frai-chat    - Chat interface")
            print("\nTo verify installation: python main.py --help")
        else:
            print("⚠ Installation completed with some issues")
            print("Some optional features may not be available")
            
        # Create troubleshooting guide
        create_troubleshooting_guide()
        
        print(f"\nInstallation log available in terminal output")
        print("If you encounter issues, check TROUBLESHOOTING.md")
        
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"INSTALLATION FAILED: {e}")
        print("="*60)
        print("\nTroubleshooting steps:")
        print("1. Check your Python version (3.12+ required)")
        print("2. Ensure you have a stable internet connection")
        print("3. Try running: python -m pip install --upgrade pip")
        print("4. For Python 3.13, consider using Python 3.12")
        print("5. Check TROUBLESHOOTING.md for specific issues")
        print("\nFor help, visit: https://github.com/Ignemia/Frai/issues")
        sys.exit(1)

if __name__ == "__main__":
    main()