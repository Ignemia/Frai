"""
Setup script for Frai that handles PyTorch installation with appropriate backend.
"""
import os
import platform
import subprocess
import sys
from setuptools import setup

def check_cuda():
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

def check_rocm():
    """Check if ROCm is available."""
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, check=False
        )
        return result.returncode == 0
    except FileNotFoundError:
        pass
    return False

def install_torch_packages():
    """Install PyTorch packages with the appropriate backend."""
    cuda_version = check_cuda()
    has_rocm = check_rocm()
    
    # Base packages
    packages = ["torch", "torchvision", "torchaudio"]
    
    if cuda_version:
        print(f"\n*** CUDA detected (version: {cuda_version}) ***")
        if platform.system() == "Windows":
            cmd = [
                sys.executable, "-m", "pip", "install", 
                *packages, "-f", 
                "https://download.pytorch.org/whl/cu121/torch_stable.html"
            ]
        else:
            # Linux/macOS can use the default PyPI index for CUDA packages
            cmd = [sys.executable, "-m", "pip", "install", *packages]
    elif has_rocm:
        print("\n*** ROCm detected ***")
        cmd = [
            sys.executable, "-m", "pip", "install", 
            *packages, "-f",
            "https://download.pytorch.org/whl/rocm5.6/torch_stable.html"
        ]
    else:
        print("\n*** No GPU acceleration detected, installing CPU-only PyTorch ***")
        cmd = [
            sys.executable, "-m", "pip", "install", 
            *packages, "-f",
            "https://download.pytorch.org/whl/cpu/torch_stable.html"
        ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("PyTorch installation completed\n")

# Only run PyTorch installation during pip install, not during other setup.py operations
if "install" in sys.argv and not os.environ.get("SKIP_TORCH_INSTALL"):
    install_torch_packages()

# Use setuptools.setup() for the actual installation
# Configuration is in pyproject.toml
setup()
