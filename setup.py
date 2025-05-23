#!/usr/bin/env python
from test_mock_helper import List
"""
Setup script for Personal Chatter.

This setup script uses setuptools to install the Personal Chatter application,
along with all its dependencies. It supports different GPU acceleration options
(CUDA, ROCm, CPU) and can install from official sources when possible.
"""
import os
import platform
import sys
import subprocess
from typing import Dict, Optional, Tuple
from setuptools import setup, find_packages, Command
import setuptools.command.develop
import setuptools.command.install
import pkg_resources

# Package metadata
NAME = "personal-chatter"
DESCRIPTION = "AI chat application with local model support and API interfaces"
URL = "https://github.com/yourusername/personal-chatter"  # Update with your repository URL
AUTHOR = "Andy O"
AUTHOR_EMAIL = "personal-chatter@example.com"  # Update with your email
REQUIRES_PYTHON = ">=3.8.0"
VERSION = "0.2.0"  # Updated version with API and refactored architecture

# Read the README file for the long description
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        LONG_DESCRIPTION = fh.read()
except FileNotFoundError:
    LONG_DESCRIPTION = DESCRIPTION

# Base dependencies that are common for all installations
CORE_DEPENDENCIES = [
    "accelerate",
    "bcrypt",
    "chromadb",
    "ctransformers",  # Added from requirements.txt
    "diffusers",
    "fastapi",
    "httptools",
    "inquirerpy",
    "langchain>=0.1.0",  # Ensure minimum version for compatibility
    "langchain-community>=0.1.0",  # Ensure minimum version for compatibility
    "passlib",
    "pyjwt",
    "python-dotenv",
    "python-jose",
    "python-multipart",
    "sentence-transformers",
    "sentencepiece",
    "tf-keras",
    "transformers>=4.35.0",  # Ensure a minimum version for gemma support
    "uvicorn",
    "watchfiles",
    "websockets",
    "xformers",  # Added from requirements.txt
]

# Optional dependencies
EXTRAS_REQUIRE = {
    "dev": [
        "black",
        "flake8",
        "isort",
        "mypy",
        "pytest",
        "pytest-cov",
        "pytest-asyncio",  # For async API tests
        "pip-chill",
        "ruff",  # Fast Python linter
    ],
    "docs": [
        "sphinx",
        "sphinx-rtd-theme",
        "myst-parser",
        "sphinx-autoapi",  # Auto API documentation
    ],
    "api": [
        "httpx",  # For API client testing
        "websockets",  # For WebSocket client testing
        "pytest-httpx",  # HTTP mocking for tests
    ],
    "gpu": [
        "cupy-cuda11x;platform_system!='Darwin'",  # CUDA support for NumPy operations
        "torch>=2.0.0",  # Base PyTorch requirement
    ],
}

def detect_gpu() -> Tuple[bool, bool, bool, Optional[str]]:
    """
    Detect available GPU hardware to determine which packages to install.
    
    Returns:
        Tuple[bool, bool, bool, Optional[str]]: (has_nvidia, has_amd, has_intel, apple_silicon)
            apple_silicon will be None if not on Apple Silicon, or the chip name if detected
    """
    has_nvidia = False
    has_amd = False
    has_intel = False
    apple_silicon = None
    
    system = platform.system()
    
    if system == "Windows":
        # Check for GPUs on Windows
        try:
            gpu_output = subprocess.check_output(
                "powershell Get-CimInstance -ClassName Win32_VideoController | Select-Object -ExpandProperty Description", 
                shell=True
            ).decode("utf-8").lower()
            
            has_nvidia = any(x in gpu_output for x in ["nvidia", "geforce", "quadro", "tesla"])
            has_amd = any(x in gpu_output for x in ["amd", "radeon", "ati", "firepro"])
            has_intel = any(x in gpu_output for x in ["intel", "iris", "uhd graphics", "arc"])
            
            # Try to get more detailed NVIDIA info if available
            if has_nvidia:
                try:
                    nvidia_smi = subprocess.check_output(
                        "nvidia-smi --query-gpu=gpu_name --format=csv,noheader", 
                        shell=True
                    ).decode("utf-8").strip()
                    print(f"NVIDIA GPU detected: {nvidia_smi}")
                except:
                    pass
        except Exception as e:
            print(f"Warning: Could not detect GPU on Windows: {e}")
            
    elif system == "Linux":
        # Check for GPUs on Linux
        try:
            # Try nvidia-smi first for NVIDIA GPUs
            try:
                nvidia_smi = subprocess.check_output("nvidia-smi -L", shell=True).decode("utf-8").lower()
                has_nvidia = True
                print(f"NVIDIA GPUs detected via nvidia-smi")
            except:
                has_nvidia = False
            
            # Use lspci as fallback or for AMD/Intel
            lspci_output = subprocess.check_output("lspci | grep -i 'vga\\|3d\\|display'", shell=True).decode("utf-8").lower()
            if not has_nvidia:
                has_nvidia = any(x in lspci_output for x in ["nvidia", "geforce", "quadro", "tesla"])
            has_amd = any(x in lspci_output for x in ["amd", "radeon", "ati", "firepro"])
            has_intel = any(x in lspci_output for x in ["intel", "iris", "uhd graphics", "arc"])
            
            # Check for ROCm if AMD is detected
            if has_amd:
                try:
                    rocm_smi = subprocess.check_output("rocm-smi --showproductname", shell=True).decode("utf-8").lower()
                    print("AMD GPU with ROCm support detected")
                except:
                    pass
                    
        except Exception as e:
            print(f"Warning: Could not detect GPU on Linux: {e}")
            
    elif system == "Darwin":  # macOS
        # Check for Apple Silicon
        is_arm64 = platform.machine() == "arm64"
        if is_arm64:
            try:
                # Get specific chip model
                chip_info = subprocess.check_output("sysctl -n machdep.cpu.brand_string", shell=True).decode("utf-8")
                if "Apple" in chip_info:
                    apple_silicon = chip_info.strip()
                    print(f"Apple Silicon detected: {apple_silicon}")
            except:
                apple_silicon = "Apple Silicon (unknown model)"
                print("Apple Silicon detected (unknown model)")
        
        # Check for Intel/AMD GPUs
        try:
            system_profiler_output = subprocess.check_output(
                "system_profiler SPDisplaysDataType", shell=True
            ).decode("utf-8").lower()
            
            has_amd = any(x in system_profiler_output for x in ["amd", "radeon", "firepro"])
            has_intel = any(x in system_profiler_output for x in ["intel", "iris", "uhd graphics"])
            has_nvidia = "nvidia" in system_profiler_output  # Older Macs could have NVIDIA
        except Exception as e:
            print(f"Warning: Could not detect GPU on macOS: {e}")
    
    print(f"Detected GPU hardware: NVIDIA={has_nvidia}, AMD={has_amd}, Intel={has_intel}, Apple Silicon={apple_silicon}")
    return has_nvidia, has_amd, has_intel, apple_silicon

def get_torch_installation_command() -> Optional[str]:
    """
    Determine the appropriate PyTorch installation command based on detected hardware.
    
    Returns:
        Optional[str]: pip installation command for PyTorch, or None if using default
    """
    has_nvidia, has_amd, has_intel, apple_silicon = detect_gpu()
    system = platform.system()
    
    # Check if CUDA toolkit is installed for NVIDIA
    cuda_available = False
    cuda_version = None
    if has_nvidia:
        # Try multiple methods to detect CUDA
        try:
            # Method 1: nvcc --version
            try:
                nvcc_output = subprocess.check_output("nvcc --version", shell=True).decode("utf-8")
                cuda_available = True
                cuda_version = nvcc_output.split("release ")[1].split(",")[0]
                print(f"CUDA Toolkit found via nvcc: version {cuda_version}")
            except:
                # Method 2: nvidia-smi
                try:
                    nvidia_smi = subprocess.check_output("nvidia-smi", shell=True).decode("utf-8")
                    cuda_info = [line for line in nvidia_smi.split("\n") if "CUDA Version:" in line]
                    if cuda_info:
                        cuda_version = cuda_info[0].split("CUDA Version:")[1].strip()
                        cuda_available = True
                        print(f"CUDA found via nvidia-smi: version {cuda_version}")
                except:
                    print("NVIDIA GPU detected, but CUDA could not be detected.")
        except Exception as e:
            print(f"Error detecting CUDA: {e}")
    
    # Check if ROCm is installed for AMD
    rocm_available = False
    rocm_version = None
    if has_amd and system == "Linux":
        try:
            # Try with rocm-smi
            try:
                rocm_output = subprocess.check_output("rocm-smi --version", shell=True).decode("utf-8")
                rocm_available = True
                rocm_version = rocm_output.strip()
                print(f"ROCm found via rocm-smi: {rocm_version}")
            except:
                # Try alternative detection
                try:
                    rocminfo = subprocess.check_output("rocminfo", shell=True).decode("utf-8")
                    rocm_available = True
                    print("ROCm found via rocminfo")
                except:
                    print("AMD GPU detected, but ROCm not found.")
        except Exception as e:
            print(f"Error detecting ROCm: {e}")
    
    # Determine appropriate PyTorch installation
    if cuda_available:
        # For NVIDIA with CUDA, select the right CUDA version
        cuda_major = None
        if cuda_version:
            try:
                cuda_major = float(cuda_version.split(".")[0])
            except:
                pass
        
        if cuda_major and cuda_major >= 12:
            return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        elif cuda_major and cuda_major >= 11.8:
            return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        elif cuda_major and cuda_major >= 11:
            return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117"
        else:
            # Default to latest supported CUDA version
            return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    
    elif rocm_available:
        # For AMD with ROCm on Linux, select appropriate version
        if rocm_version and "5.7" in rocm_version:
            return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7"
        elif rocm_version and "5.6" in rocm_version:
            return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6"
        elif rocm_version and "5.5" in rocm_version:
            return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.5"
        else:
            # Default to latest supported ROCm version
            return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6"
    
    elif apple_silicon:
        # For Apple Silicon (M1/M2/M3)
        return "pip install torch torchvision torchaudio"
    
    elif has_intel and system != "Darwin":
        # For Intel GPUs on Windows/Linux, use Intel's extensions
        try:
            # Try to detect Intel oneAPI
            if system == "Windows":
                oneapi_env = os.environ.get("ONEAPI_ROOT")
                has_oneapi = oneapi_env is not None
            else:
                # Check for Linux oneAPI installation
                has_oneapi = os.path.exists("/opt/intel/oneapi")
                
            if has_oneapi:
                print("Intel oneAPI detected, installing PyTorch with Intel extensions")
                return "pip install torch torchvision torchaudio && pip install intel-extension-for-pytorch"
            else:
                print("Intel GPU detected, but oneAPI not found. Installing CPU version of PyTorch.")
                return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        except Exception as e:
            print(f"Error detecting Intel oneAPI: {e}")
            return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    else:
        # Default to CPU version for other cases
        return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"

def get_huggingface_command() -> str:
    """
    Get the command to install Hugging Face CLI from the official source.
    
    Returns:
        str: pip installation command for Hugging Face
    """
    return "pip install -U huggingface_hub[cli]"

def install_acceleration_dependencies():
    """Install hardware-specific acceleration dependencies."""
    # Install PyTorch with appropriate hardware acceleration
    torch_command = get_torch_installation_command()
    if torch_command:
        print(f"Installing PyTorch with appropriate acceleration: {torch_command}")
        try:
            subprocess.check_call(torch_command, shell=True)
            print("PyTorch installation successful")
        except Exception as e:
            print(f"PyTorch installation failed: {e}")
            print("Trying to install CPU version as fallback...")
            try:
                subprocess.check_call(
                    "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu", 
                    shell=True
                )
                print("PyTorch CPU installation successful")
            except Exception as e2:
                print(f"Fallback installation also failed: {e2}")
    
    # Install Hugging Face CLI
    huggingface_command = get_huggingface_command()
    print(f"Installing Hugging Face CLI: {huggingface_command}")
    try:
        subprocess.check_call(huggingface_command, shell=True)
        print("Hugging Face CLI installation successful")
    except Exception as e:
        print(f"Hugging Face CLI installation failed: {e}")
    
    # Install ctransformers with appropriate hardware acceleration
    has_nvidia, _, _, _ = detect_gpu()
    if has_nvidia:
        try:
            print("Installing ctransformers with CUDA support")
            subprocess.check_call("pip install ctransformers[cuda]", shell=True)
            print("ctransformers with CUDA support installed successfully")
        except Exception as e:
            print(f"ctransformers with CUDA installation failed: {e}")
            print("Installing regular ctransformers as fallback")
            try:
                subprocess.check_call("pip install ctransformers", shell=True)
            except:
                pass
    else:
        # Just make sure ctransformers is installed
        try:
            subprocess.check_call("pip install ctransformers", shell=True)
        except:
            print("ctransformers installation failed")
    
    # Install xformers for enhanced transformer performance when possible
    try:
        print("Installing xformers for enhanced transformer performance")
        subprocess.check_call("pip install xformers", shell=True)
        print("Successfully installed xformers")
    except Exception as e:
        print(f"Could not install xformers: {e}")
        print("This is optional and may not be available for all platforms.")

class CustomDevelopCommand(setuptools.command.develop.develop):
    """Custom installation command for development setups."""
    
    def run(self):
        # Run standard develop first to get dependencies installed
        setuptools.command.develop.develop.run(self)
        
        # Install hardware-specific dependencies
        install_acceleration_dependencies()
        
        # Install development dependencies
        print("Installing development dependencies...")
        try:
            for package in EXTRAS_REQUIRE['dev']:
                subprocess.check_call(f"pip install {package}", shell=True)
            print("Development dependencies installed successfully")
        except Exception as e:
            print(f"Error installing development dependencies: {e}")
        
        print("Development setup complete!")

class CustomInstallCommand(setuptools.command.install.install):
    """Custom installation command for normal installations."""
    
    def run(self):
        # Run standard install first
        setuptools.command.install.install.run(self)
        
        # Install hardware-specific dependencies
        install_acceleration_dependencies()
        
        # Setup model folders if they don't exist
        try:
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                print(f"Created models directory at {model_dir}")
            
            outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
            if not os.path.exists(outputs_dir):
                os.makedirs(outputs_dir)
                print(f"Created outputs directory at {outputs_dir}")
        except Exception as e:
            print(f"Error setting up directories: {e}")
        
        print("Installation complete!")
        
        # Provide instructions for model downloads
        print("\nTo complete setup, you need to download language models.")
        print("Use these commands to download models:")
        print("  python -m huggingface_hub download <model-name> --local-dir models/<model-dir>")
        print("\nExample:")
        print("  python -m huggingface_hub download google/gemma-3-4b-it --local-dir models/gemma-3-4b-it")

if __name__ == "__main__":
    # If running setup.py directly, install acceleration dependencies first
    if len(sys.argv) > 1 and sys.argv[1] in ["install", "develop"]:
        install_acceleration_dependencies()

    # Configure the package setup
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        python_requires=REQUIRES_PYTHON,
        url=URL,
        packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
        install_requires=CORE_DEPENDENCIES,
        extras_require=EXTRAS_REQUIRE,
        include_package_data=True,
        license="MIT",
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
        ],
        cmdclass={
            "develop": CustomDevelopCommand,
            "install": CustomInstallCommand,
        },
        entry_points={
            "console_scripts": [
                "personal-chatter=main:main",
                "pc-chat=main:main",  # Shorter alias
                "pc-api=api.api:start_backend_api_cli",  # API only entry point
            ],
        })
    
    # Print installation guide if this is a direct install
    if len(sys.argv) > 1 and sys.argv[1] == "install":
        print("\n" + "=" * 80)
        print("Personal Chatter Installation Guide".center(80))
        print("=" * 80)
        print("\nInstallation completed successfully!\n")
        
        print("Getting Started:")
        print("---------------")
        print("1. Download required models:")
        print("   python -m huggingface_hub download google/gemma-3-4b-it --local-dir models/gemma-3-4b-it")
        print("")
        print("2. Run the application:")
        print("   - Full application (CLI + API): personal-chatter")
        print("   - Short command: pc-chat")
        print("   - API server only: pc-api")
        print("")
        print("3. Access the API:")
        print("   - API Documentation: http://localhost:8000/docs")
        print("   - OpenAPI Schema: http://localhost:8000/openapi.json")
        print("")
        print("4. Development:")
        print("   - Install dev dependencies: pip install -e .[dev]")
        print("   - Run tests: pytest")
        print("")
        print("For more information, see the README.md file.")
        print("=" * 80)
