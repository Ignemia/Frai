#!/usr/bin/env python3
"""
Installation script for Frai that properly handles PyTorch installation.
"""
import os
import platform
import subprocess
import sys
from typing import List, Optional, Tuple

def check_cuda() -> Optional[str]:
    """Check if CUDA is available and return version if found."""
    # Check for NVIDIA GPU and CUDA
    try:
        # Try to run nvidia-smi to check for CUDA
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            # Extract CUDA version
            for line in result.stdout.split("\n"):
                if "CUDA Version:" in line:
                    return line.split("CUDA Version:")[1].strip()
            # If we found nvidia-smi but couldn't parse version, assume it's available
            return "unknown"
    except FileNotFoundError:
        pass
    
    return None

def check_rocm() -> bool:
    """Check if ROCm is available."""
    # Check for AMD GPU and ROCm
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, check=False
        )
        return result.returncode == 0
    except FileNotFoundError:
        pass
    
    return False

def get_torch_install_command() -> Tuple[List[str], List[str]]:
    """
    Determine the appropriate PyTorch installation command.
    Returns a tuple of (packages to install, pip args)
    """
    cuda_version = check_cuda()
    has_rocm = check_rocm()
    
    # Base packages
    packages = ["torch", "torchvision", "torchaudio"]
    
    # Default to CPU version
    extra_args = ["-f", "https://download.pytorch.org/whl/cpu/torch_stable.html"]
    
    if cuda_version:
        print(f"CUDA detected (version: {cuda_version})")
        # For CUDA, use PyTorch's official index
        if platform.system() == "Windows":
            extra_args = ["-f", "https://download.pytorch.org/whl/cu121/torch_stable.html"]
        else:
            # Linux/macOS can use the default PyPI index for CUDA packages
            extra_args = []
    elif has_rocm:
        print("ROCm detected")
        # For ROCm, use PyTorch's ROCm index
        extra_args = ["-f", "https://download.pytorch.org/whl/rocm5.6/torch_stable.html"]
    else:
        print("No GPU acceleration detected, installing CPU-only PyTorch")
    
    return packages, extra_args

def install_torch():
    """Install PyTorch with appropriate backend."""
    packages, extra_args = get_torch_install_command()
    
    cmd = [sys.executable, "-m", "pip", "install"] + packages + extra_args
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def install_frai():
    """Install Frai and its dependencies."""
    install_type = ".[dev,test]" if "--dev" in sys.argv else "."
    cmd = [sys.executable, "-m", "pip", "install", "-e", install_type]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    """Main installation function."""
    print("Starting Frai installation")
    
    # First install PyTorch packages with appropriate backend
    install_torch()
    
    # Then install Frai
    install_frai()
    
    print("\nFrai installation completed successfully!")
    print("\nYou can now run:")
    print("  frai         - Main application")
    print("  frai-chat    - Chat interface")
    print("  frai-api     - API server")

if __name__ == "__main__":
    main()
