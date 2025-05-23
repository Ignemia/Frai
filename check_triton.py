#!/usr/bin/env python3
"""
Quick script to check Triton availability and identify performance issues.
"""

import sys
import torch
import platform

def check_triton():
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check Triton
    try:
        import triton
        print(f"Triton available: True (version {triton.__version__})")
    except ImportError as e:
        print(f"Triton available: False - {e}")
    
    # Check xformers
    try:
        import xformers
        print(f"xformers available: True (version {xformers.__version__})")
        
        # Test if xformers works
        try:
            from xformers.ops import memory_efficient_attention
            print("xformers memory_efficient_attention: Available")
        except Exception as e:
            print(f"xformers memory_efficient_attention: Not available - {e}")
    except ImportError as e:
        print(f"xformers available: False - {e}")
    
    # Check diffusers optimizations
    try:
        from diffusers import FluxPipeline
        print("FluxPipeline import: Success")
    except ImportError as e:
        print(f"FluxPipeline import: Failed - {e}")

if __name__ == "__main__":
    check_triton()
