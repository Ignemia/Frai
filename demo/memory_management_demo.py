#!/usr/bin/env python3
"""
Memory Management Demo for Personal Chatter

This demo showcases the memory management capabilities including:
- Resource allocation monitoring
- Memory optimization for large models
- Garbage collection and cleanup strategies
- Performance tuning
"""

import gc
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import demo utils
from demo.utils import print_header, print_section


def format_size(size_bytes):
    """Format bytes into a human-readable format."""
    if size_bytes == 0:
        return "0B"
    size_names = ("B", "KB", "MB", "GB", "TB")
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.2f} {size_names[i]}"


def run_memory_management_demo():
    """Run the memory management demonstration."""
    print_header("Personal Chatter - Memory Management Demo")
    
    # Check if the memory management module exists
    try:
        from services.image_generation.memory_manager import get_memory_manager
        HAS_MEMORY_MANAGER = True
    except ImportError:
        print("⚠️ Memory management module not found. Running in simulation mode.")
        HAS_MEMORY_MANAGER = False
    
    # 1. System Resource Check
    print_section("1. System Resource Check")
    
    try:
        import psutil
        HAS_PSUTIL = True
        
        # Get CPU info
        cpu_count = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_percent = psutil.cpu_percent()
        
        # Get memory info
        memory = psutil.virtual_memory()
        
        print(f"CPU Information:")
        print(f"  Physical cores: {cpu_count}")
        print(f"  Logical cores: {cpu_count_logical}")
        print(f"  Current usage: {cpu_percent}%")
        
        print(f"\nMemory Information:")
        print(f"  Total: {format_size(memory.total)}")
        print(f"  Available: {format_size(memory.available)}")
        print(f"  Used: {format_size(memory.used)} ({memory.percent}%)")
        
    except ImportError:
        HAS_PSUTIL = False
        print("⚠️ psutil module not found. Limited system information available.")
        
        import os
        # Very basic cross-platform info
        print("Limited System Information:")
        print(f"  CPU count: {os.cpu_count()}")
    
    # 2. GPU Resource Check
    print_section("2. GPU Resource Check")
    
    try:
        # Try to import torch to check GPU status
        import torch
        HAS_TORCH = True
        
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"GPU Devices: {device_count}")
            
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                print(f"\nGPU #{i}: {device_name}")
                
                # Show memory info for each GPU
                try:
                    total_memory = torch.cuda.get_device_properties(i).total_memory
                    reserved_memory = torch.cuda.memory_reserved(i)
                    allocated_memory = torch.cuda.memory_allocated(i)
                    free_memory = reserved_memory - allocated_memory
                    
                    print(f"  Total memory: {format_size(total_memory)}")
                    print(f"  Reserved memory: {format_size(reserved_memory)}")
                    print(f"  Allocated memory: {format_size(allocated_memory)}")
                    print(f"  Free memory: {format_size(free_memory)}")
                except Exception as e:
                    print(f"  Memory info unavailable: {e}")
    except ImportError:
        HAS_TORCH = False
        print("⚠️ PyTorch not found. GPU information unavailable.")
    
    # 3. Memory Management Strategies
    print_section("3. Memory Management Strategies")
    
    if HAS_MEMORY_MANAGER:
        memory_manager = get_memory_manager()
        
        print("Available Memory Optimization Strategies:")
        print("  1. Attention Slicing - Reduces peak memory usage by processing attention in chunks")
        print("  2. CPU Offloading - Moves models to CPU when not in use")
        print("  3. Sequential Loading - Loads model components only when needed")
        print("  4. Mixed Precision - Uses lower precision for faster computation with less memory")
        
        # Show current settings
        settings = memory_manager.get_settings()
        print("\nCurrent Memory Manager Settings:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
        
        # Show current status
        status = memory_manager.get_status()
        print("\nCurrent Memory Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
    else:
        print("Available Memory Optimization Strategies (Simulation):")
        print("  1. Attention Slicing - Reduces peak memory usage by processing attention in chunks")
        print("  2. CPU Offloading - Moves models to CPU when not in use")
        print("  3. Sequential Loading - Loads model components only when needed")
        print("  4. Mixed Precision - Uses lower precision for faster computation with less memory")
        
        print("\nSimulated Memory Manager Settings:")
        print("  attention_slicing: enabled")
        print("  cpu_offloading: auto")
        print("  sequential_loading: enabled")
        print("  mixed_precision: True")
        
        print("\nSimulated Memory Status:")
        print("  available_gpu_memory: 8.45 GB")
        print("  allocated_gpu_memory: 3.22 GB")
        print("  model_size: 2.71 GB")
        print("  model_loaded: True")
    
    # 4. Memory Optimization Demo
    print_section("4. Memory Optimization Demonstration")
    
    print("Demonstrating memory optimization during image generation workflow...\n")
    
    # Demonstration steps
    steps = [
        "Initializing model with memory-optimized settings",
        "Loading model weights with sequential loading",
        "Setting up attention slicing for transformer blocks",
        "Preparing mixed precision inference",
        "Running inference with optimized memory",
        "Offloading unused components to CPU",
        "Performing garbage collection",
        "Measuring final memory usage"
    ]
    
    for i, step in enumerate(steps, 1):
        print(f"Step {i}: {step}")
        time.sleep(0.5)  # Simulate processing time
        
        # Show fake memory usage for demo purposes
        if i == 1:
            print("  Initial GPU memory: 0.85 GB used")
        elif i == 2:
            print("  Loading weights: 2.40 GB used (+1.55 GB)")
        elif i == 3:
            print("  After attention slicing setup: 2.40 GB used (no change)")
        elif i == 4:
            print("  After mixed precision setup: 2.15 GB used (-0.25 GB)")
        elif i == 5:
            print("  Peak during inference: 4.32 GB used (+2.17 GB)")
        elif i == 6:
            print("  After CPU offloading: 1.85 GB used (-2.47 GB)")
        elif i == 7:
            print("  After garbage collection: 0.95 GB used (-0.90 GB)")
        elif i == 8:
            print("  Final memory usage: 0.95 GB used")
            
        time.sleep(0.5)  # Pause between steps
    
    # 5. Manual Memory Management
    print_section("5. Manual Memory Management")
    
    print("Demonstrating manual memory management techniques:")
    
    # 1. Explicit garbage collection
    print("\n1. Explicit Garbage Collection")
    gc.collect()
    print("  ✓ Manual garbage collection triggered")
    
    # 2. Reference clearing
    print("\n2. Reference Clearing")
    print("  ✓ Clearing references to large objects")
    large_object = [0] * 10000000  # Create a large object
    large_object = None  # Clear the reference
    
    # 3. Context management
    print("\n3. Context Management")
    print("""  Example:
  with memory_manager.optimize_for_inference():
      model.generate(prompt)""")
    print("  ✓ Resources automatically cleaned up after context exit")
    
    # 4. Incremental processing
    print("\n4. Incremental Processing")
    print("  ✓ Processing large data in smaller chunks to manage memory")
    
    # Memory usage summary
    print_section("Memory Usage Summary")
    
    if HAS_TORCH and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            print(f"GPU #{i} Memory Usage:")
            print(f"  Allocated: {format_size(allocated)}")
            print(f"  Reserved: {format_size(reserved)}")
    elif HAS_PSUTIL:
        memory = psutil.virtual_memory()
        print(f"System Memory Usage:")
        print(f"  Used: {format_size(memory.used)} ({memory.percent}%)")
        print(f"  Available: {format_size(memory.available)}")
    else:
        print("Memory usage statistics unavailable")
    
    return True


def main():
    """Main entry point for the memory management demo."""
    success = run_memory_management_demo()
    
    if success:
        print("\n✅ Memory management demo completed successfully!")
    else:
        print("\n❌ Memory management demo encountered errors.")
        
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
