"""
GPU memory checking utilities.

Provides functions to check GPU memory availability and estimate
memory requirements for image generation.
"""
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


def get_gpu_info() -> Dict[str, Any]:
    """
    Get comprehensive GPU information.
    
    Returns:
        Dictionary containing GPU information
    """
    gpu_info = {
        "available": False,
        "device_count": 0,
        "current_device": None,
        "devices": []
    }
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["device_count"] = torch.cuda.device_count()
            gpu_info["current_device"] = torch.cuda.current_device()
            
            # Get information for each device
            for i in range(torch.cuda.device_count()):
                device_info = {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": torch.cuda.get_device_capability(i),
                    "properties": {}
                }
                
                try:
                    props = torch.cuda.get_device_properties(i)
                    device_info["properties"] = {
                        "total_memory": props.total_memory,
                        "multi_processor_count": props.multi_processor_count,
                        "max_threads_per_multi_processor": props.max_threads_per_multi_processor,
                        "max_threads_per_block": props.max_threads_per_block,
                        "warp_size": props.warp_size
                    }
                except Exception as e:
                    logger.warning(f"Could not get properties for GPU {i}: {e}")
                
                gpu_info["devices"].append(device_info)
        
    except ImportError:
        logger.warning("PyTorch not available for GPU detection")
    except Exception as e:
        logger.error(f"Error getting GPU info: {e}")
    
    return gpu_info


def get_memory_info() -> Dict[str, Any]:
    """
    Get current GPU memory information.
    
    Returns:
        Dictionary containing memory information
    """
    memory_info = {
        "available": False,
        "free_bytes": 0,
        "total_bytes": 0,
        "used_bytes": 0,
        "reserved_bytes": 0,
        "allocated_bytes": 0
    }
    
    try:
        import torch
        
        if torch.cuda.is_available():
            free_memory, total_memory = torch.cuda.mem_get_info()
            reserved_memory = torch.cuda.memory_reserved()
            allocated_memory = torch.cuda.memory_allocated()
            
            memory_info.update({
                "available": True,
                "free_bytes": free_memory,
                "total_bytes": total_memory,
                "used_bytes": total_memory - free_memory,
                "reserved_bytes": reserved_memory,
                "allocated_bytes": allocated_memory
            })
            
    except ImportError:
        logger.warning("PyTorch not available for memory checking")
    except Exception as e:
        logger.error(f"Error getting memory info: {e}")
    
    return memory_info


def estimate_memory_requirements(height: int, width: int, steps: int) -> float:
    """
    Estimate memory requirements for image generation.
    
    Args:
        height: Image height in pixels
        width: Image width in pixels
        steps: Number of inference steps
        
    Returns:
        Estimated memory requirement in GB
    """
    # Base memory estimation based on empirical observations
    # This is a rough estimate and may vary based on model and implementation
    
    # Calculate resolution factor relative to 512x512
    resolution_factor = (height * width) / (512 * 512)
    
    # Base memory for 512x512 image (roughly 1GB for Flux.1)
    base_memory_gb = 1.0
    
    # Scale by resolution
    estimated_memory = base_memory_gb * resolution_factor
    
    # Factor in inference steps (more steps = more intermediate tensors)
    step_factor = max(1.0, steps / 20)  # Normalize to 20 steps
    estimated_memory *= step_factor
    
    # Add some overhead for model weights and intermediate computations
    overhead_factor = 1.2
    estimated_memory *= overhead_factor
    
    return estimated_memory


def check_gpu_memory_for_generation(height: int, width: int, steps: int) -> Tuple[bool, str]:
    """
    Check if there's sufficient GPU memory for the requested generation parameters.
    
    Args:
        height: Image height in pixels
        width: Image width in pixels
        steps: Number of inference steps
        
    Returns:
        Tuple of (can_generate: bool, message: str)
    """
    # Small images should always be allowed to attempt generation
    if height <= 256 and width <= 256 and steps <= 15:
        message = f"Small image requested ({height}x{width}, {steps} steps) - bypassing memory check"
        logger.info(message)
        return True, message
    
    try:
        import torch
    except ImportError:
        return True, "PyTorch not available, cannot check GPU memory"
    
    if not torch.cuda.is_available():
        return True, "CUDA not available, will use CPU"
    
    try:
        memory_info = get_memory_info()
        
        if not memory_info["available"]:
            return False, "Could not get GPU memory information"
        
        # Convert to GB for easier calculation
        free_memory_gb = memory_info["free_bytes"] / 1e9
        reserved_memory_gb = memory_info["reserved_bytes"] / 1e9
        allocated_memory_gb = memory_info["allocated_bytes"] / 1e9
        
        # Calculate effective free memory
        free_reserved_memory_gb = max(0, reserved_memory_gb - allocated_memory_gb)
        effective_free_memory = free_memory_gb + free_reserved_memory_gb
        
        logger.debug(f"Memory status: Free={free_memory_gb:.2f}GB, Reserved={reserved_memory_gb:.2f}GB, Allocated={allocated_memory_gb:.2f}GB")
        logger.debug(f"Effective free memory: {effective_free_memory:.2f}GB")
        
        # For medium images, we can proceed even with limited memory
        if height <= 384 and width <= 384 and steps <= 20:
            if effective_free_memory > 0.2:  # Need at least 200MB free
                message = f"Small image generation should fit in available memory: {effective_free_memory:.2f}GB available"
                logger.info(message)
                return True, message
        
        # Estimate memory requirements
        estimated_memory = estimate_memory_requirements(height, width, steps)
        
        logger.debug(f"Memory estimate: {estimated_memory:.2f}GB for {height}x{width}, {steps} steps")
        
        # Check if we can fit the generation
        if effective_free_memory > estimated_memory:
            message = f"Memory check passed: {effective_free_memory:.2f}GB available, need ~{estimated_memory:.2f}GB"
            logger.info(message)
            return True, message
        else:
            # If close enough, let it try anyway
            if effective_free_memory > (estimated_memory * 0.8):
                message = f"Memory might be tight, but we'll try: {effective_free_memory:.2f}GB available, need ~{estimated_memory:.2f}GB"
                logger.warning(message)
                return True, message
            else:
                message = f"Insufficient GPU memory: {effective_free_memory:.2f}GB available, need ~{estimated_memory:.2f}GB"
                logger.warning(message)
                return False, message
        
    except Exception as e:
        logger.warning(f"Could not check GPU memory: {e}")
        # If memory check fails, proceed cautiously for small images
        if height <= 256 and width <= 256 and steps <= 15:
            return True, "Memory check failed, proceeding with small image"
        else:
            return False, f"Memory check failed: {e}"


def get_recommended_settings() -> Dict[str, Any]:
    """
    Get recommended generation settings based on available GPU memory.
    
    Returns:
        Dictionary with recommended settings
    """
    recommendations = {
        "height": 512,
        "width": 512,
        "steps": 20,
        "reason": "default"
    }
    
    try:
        memory_info = get_memory_info()
        
        if memory_info["available"]:
            free_memory_gb = memory_info["free_bytes"] / 1e9
            
            if free_memory_gb < 2:
                # Very limited memory
                recommendations.update({
                    "height": 256,
                    "width": 256,
                    "steps": 10,
                    "reason": "limited_memory"
                })
            elif free_memory_gb < 4:
                # Moderate memory
                recommendations.update({
                    "height": 384,
                    "width": 384,
                    "steps": 15,
                    "reason": "moderate_memory"
                })
            elif free_memory_gb >= 8:
                # Plenty of memory
                recommendations.update({
                    "height": 768,
                    "width": 768,
                    "steps": 30,
                    "reason": "plenty_memory"
                })
            # else: use default settings
        
    except Exception as e:
        logger.warning(f"Could not determine recommended settings: {e}")
    
    return recommendations
