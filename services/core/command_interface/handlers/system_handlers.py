"""
System handlers for the Service Command Interface.

Handles system monitoring, health checks, performance metrics,
and resource utilization operations.
"""
import logging
import time
import gc
import psutil
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..command_system import Command, CommandResult, ExecutionContext

logger = logging.getLogger(__name__)


class SystemHandlers:
    """Handlers for system monitoring and health check operations."""
    
    @staticmethod
    def handle_system_status(command: Command, context: ExecutionContext) -> CommandResult:
        """
        Get comprehensive system status.
        
        Args:
            command: Command object
            context: Execution context
            
        Returns:
            CommandResult with system status information
        """
        try:
            status = {
                "timestamp": time.time(),
                "system": SystemHandlers._get_system_info(),
                "memory": SystemHandlers._get_memory_info(),
                "cpu": SystemHandlers._get_cpu_info(),
                "gpu": SystemHandlers._get_gpu_info(),
                "disk": SystemHandlers._get_disk_info(),
                "process": SystemHandlers._get_process_info()
            }
            
            # Overall health assessment
            health_issues = []
            
            # Check memory usage
            if status["memory"]["percent"] > 90:
                health_issues.append("High memory usage")
            
            # Check CPU usage
            if status["cpu"]["percent"] > 90:
                health_issues.append("High CPU usage")
            
            # Check disk space
            for disk in status["disk"]:
                if disk["percent"] > 90:
                    health_issues.append(f"Low disk space on {disk['mountpoint']}")
            
            status["health"] = {
                "status": "healthy" if not health_issues else "warning",
                "issues": health_issues
            }
            
            return CommandResult(
                success=True,
                message="System status retrieved successfully",
                data=status
            )
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to get system status: {str(e)}",
                error_code="SYSTEM_STATUS_FAILED"
            )
    
    @staticmethod
    def handle_health_check(command: Command, context: ExecutionContext) -> CommandResult:
        """
        Perform comprehensive health check.
        
        Args:
            command: Command with optional 'components' parameter
            context: Execution context
            
        Returns:
            CommandResult with health check results
        """
        try:
            components = command.parameters.get('components', ['all'])
            if 'all' in components:
                components = ['system', 'memory', 'gpu', 'disk', 'services']
            
            health_results = {
                "timestamp": time.time(),
                "overall_status": "healthy",
                "checks": {},
                "warnings": [],
                "errors": []
            }
            
            # System health check
            if 'system' in components:
                health_results["checks"]["system"] = SystemHandlers._check_system_health()
            
            # Memory health check
            if 'memory' in components:
                health_results["checks"]["memory"] = SystemHandlers._check_memory_health()
            
            # GPU health check
            if 'gpu' in components:
                health_results["checks"]["gpu"] = SystemHandlers._check_gpu_health()
            
            # Disk health check
            if 'disk' in components:
                health_results["checks"]["disk"] = SystemHandlers._check_disk_health()
            
            # Service health check
            if 'services' in components:
                health_results["checks"]["services"] = SystemHandlers._check_services_health()
            
            # Aggregate results
            for check_name, check_result in health_results["checks"].items():
                if check_result["status"] == "error":
                    health_results["overall_status"] = "error"
                    health_results["errors"].extend(check_result.get("issues", []))
                elif check_result["status"] == "warning" and health_results["overall_status"] == "healthy":
                    health_results["overall_status"] = "warning"
                    health_results["warnings"].extend(check_result.get("issues", []))
            
            return CommandResult(
                success=True,
                message=f"Health check complete: {health_results['overall_status']}",
                data=health_results
            )
            
        except Exception as e:
            logger.error(f"Error performing health check: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to perform health check: {str(e)}",
                error_code="HEALTH_CHECK_FAILED"
            )
    
    @staticmethod
    def handle_memory_stats(command: Command, context: ExecutionContext) -> CommandResult:
        """
        Get detailed memory statistics.
        
        Args:
            command: Command object
            context: Execution context
            
        Returns:
            CommandResult with memory statistics
        """
        try:
            memory_stats = {
                "timestamp": time.time(),
                "system_memory": SystemHandlers._get_memory_info(),
                "gpu_memory": SystemHandlers._get_gpu_memory_info(),
                "process_memory": SystemHandlers._get_process_memory_info(),
                "memory_management": SystemHandlers._get_memory_management_status()
            }
            
            return CommandResult(
                success=True,
                message="Memory statistics retrieved successfully",
                data=memory_stats
            )
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to get memory stats: {str(e)}",
                error_code="MEMORY_STATS_FAILED"
            )
    
    @staticmethod
    def handle_gpu_status(command: Command, context: ExecutionContext) -> CommandResult:
        """
        Get GPU status and information.
        
        Args:
            command: Command object
            context: Execution context
            
        Returns:
            CommandResult with GPU status
        """
        try:
            gpu_status = {
                "timestamp": time.time(),
                "gpu_info": SystemHandlers._get_gpu_info(),
                "memory_info": SystemHandlers._get_gpu_memory_info(),
                "utilization": SystemHandlers._get_gpu_utilization()
            }
            
            return CommandResult(
                success=True,
                message="GPU status retrieved successfully",
                data=gpu_status
            )
            
        except Exception as e:
            logger.error(f"Error getting GPU status: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to get GPU status: {str(e)}",
                error_code="GPU_STATUS_FAILED"
            )
    
    @staticmethod
    def handle_performance_metrics(command: Command, context: ExecutionContext) -> CommandResult:
        """
        Get system performance metrics.
        
        Args:
            command: Command with optional 'duration' parameter
            context: Execution context
            
        Returns:
            CommandResult with performance metrics
        """
        try:
            duration = command.parameters.get('duration', 5)  # seconds
            
            # Collect initial metrics
            start_time = time.time()
            start_cpu = psutil.cpu_percent()
            start_memory = psutil.virtual_memory()
            
            # Wait for specified duration
            time.sleep(duration)
            
            # Collect final metrics
            end_time = time.time()
            end_cpu = psutil.cpu_percent()
            end_memory = psutil.virtual_memory()
            
            # Calculate metrics
            metrics = {
                "collection_duration": end_time - start_time,
                "timestamp": end_time,
                "cpu": {
                    "average_percent": (start_cpu + end_cpu) / 2,
                    "start_percent": start_cpu,
                    "end_percent": end_cpu,
                    "cores": psutil.cpu_count(logical=False),
                    "logical_cores": psutil.cpu_count(logical=True)
                },
                "memory": {
                    "start_percent": start_memory.percent,
                    "end_percent": end_memory.percent,
                    "average_percent": (start_memory.percent + end_memory.percent) / 2,
                    "start_used_gb": start_memory.used / 1e9,
                    "end_used_gb": end_memory.used / 1e9,
                    "total_gb": end_memory.total / 1e9
                },
                "gpu": SystemHandlers._get_gpu_utilization(),
                "process": SystemHandlers._get_process_metrics()
            }
            
            return CommandResult(
                success=True,
                message=f"Performance metrics collected over {duration}s",
                data=metrics
            )
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to collect performance metrics: {str(e)}",
                error_code="PERFORMANCE_METRICS_FAILED"
            )
    
    @staticmethod
    def handle_cleanup_memory(command: Command, context: ExecutionContext) -> CommandResult:
        """
        Perform memory cleanup operations.
        
        Args:
            command: Command with optional cleanup parameters
            context: Execution context
            
        Returns:
            CommandResult with cleanup results
        """
        try:
            cleanup_types = command.parameters.get('types', ['gc', 'gpu'])
            
            # Get initial memory state
            initial_memory = psutil.virtual_memory()
            initial_gpu_memory = SystemHandlers._get_gpu_memory_info()
            
            cleanup_results = {
                "timestamp": time.time(),
                "operations": [],
                "before": {
                    "system_memory_mb": initial_memory.used / 1e6,
                    "system_memory_percent": initial_memory.percent
                },
                "after": {},
                "freed_memory": {}
            }
            
            # Perform garbage collection
            if 'gc' in cleanup_types:
                collected = gc.collect()
                cleanup_results["operations"].append({
                    "type": "garbage_collection",
                    "objects_collected": collected
                })
            
            # GPU memory cleanup
            if 'gpu' in cleanup_types:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        cleanup_results["operations"].append({
                            "type": "gpu_cache_clear",
                            "status": "completed"
                        })
                except ImportError:
                    cleanup_results["operations"].append({
                        "type": "gpu_cache_clear",
                        "status": "skipped (PyTorch not available)"
                    })
            
            # Memory manager cleanup
            if 'memory_manager' in cleanup_types:
                try:
                    from services.ai.image_generation.memory_manager import get_memory_manager
                    memory_manager = get_memory_manager()
                    memory_manager.clear_gpu_memory()
                    cleanup_results["operations"].append({
                        "type": "memory_manager_cleanup",
                        "status": "completed"
                    })
                except ImportError:
                    cleanup_results["operations"].append({
                        "type": "memory_manager_cleanup",
                        "status": "skipped (memory manager not available)"
                    })
            
            # Get final memory state
            final_memory = psutil.virtual_memory()
            final_gpu_memory = SystemHandlers._get_gpu_memory_info()
            
            cleanup_results["after"] = {
                "system_memory_mb": final_memory.used / 1e6,
                "system_memory_percent": final_memory.percent
            }
            
            cleanup_results["freed_memory"] = {
                "system_memory_mb": (initial_memory.used - final_memory.used) / 1e6,
                "system_memory_percent": initial_memory.percent - final_memory.percent
            }
            
            # Add GPU memory changes if available
            if initial_gpu_memory.get("available") and final_gpu_memory.get("available"):
                cleanup_results["before"]["gpu_memory_gb"] = initial_gpu_memory.get("allocated_gb", 0)
                cleanup_results["after"]["gpu_memory_gb"] = final_gpu_memory.get("allocated_gb", 0)
                cleanup_results["freed_memory"]["gpu_memory_gb"] = (
                    initial_gpu_memory.get("allocated_gb", 0) - 
                    final_gpu_memory.get("allocated_gb", 0)
                )
            
            return CommandResult(
                success=True,
                message=f"Memory cleanup completed ({len(cleanup_results['operations'])} operations)",
                data=cleanup_results
            )
            
        except Exception as e:
            logger.error(f"Error performing memory cleanup: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to perform memory cleanup: {str(e)}",
                error_code="MEMORY_CLEANUP_FAILED"
            )
    
    # Helper methods for system information gathering
    
    @staticmethod
    def _get_system_info() -> Dict[str, Any]:
        """Get basic system information."""
        try:
            import platform
            return {
                "platform": platform.platform(),
                "architecture": platform.architecture(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "uptime": time.time() - psutil.boot_time()
            }
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def _get_memory_info() -> Dict[str, Any]:
        """Get system memory information."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            return {
                "total_gb": memory.total / 1e9,
                "available_gb": memory.available / 1e9,
                "used_gb": memory.used / 1e9,
                "percent": memory.percent,
                "swap_total_gb": swap.total / 1e9,
                "swap_used_gb": swap.used / 1e9,
                "swap_percent": swap.percent
            }
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def _get_cpu_info() -> Dict[str, Any]:
        """Get CPU information."""
        try:
            return {
                "cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "percent": psutil.cpu_percent(interval=1),
                "per_cpu": psutil.cpu_percent(interval=1, percpu=True),
                "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def _get_gpu_info() -> Dict[str, Any]:
        """Get GPU information."""
        try:
            # Try to get GPU info from the image generation utilities
            from services.ai.image_generation.utils.gpu_checker import get_gpu_info
            return get_gpu_info()
        except ImportError:
            # Fallback to basic PyTorch check
            try:
                import torch
                if torch.cuda.is_available():
                    return {
                        "available": True,
                        "device_count": torch.cuda.device_count(),
                        "current_device": torch.cuda.current_device(),
                        "device_name": torch.cuda.get_device_name(0)
                    }
                else:
                    return {"available": False, "reason": "CUDA not available"}
            except ImportError:
                return {"available": False, "reason": "PyTorch not available"}
    
    @staticmethod
    def _get_disk_info() -> List[Dict[str, Any]]:
        """Get disk usage information."""
        try:
            disk_info = []
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_info.append({
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "fstype": partition.fstype,
                        "total_gb": usage.total / 1e9,
                        "used_gb": usage.used / 1e9,
                        "free_gb": usage.free / 1e9,
                        "percent": (usage.used / usage.total) * 100
                    })
                except PermissionError:
                    # Skip inaccessible partitions
                    continue
            return disk_info
        except Exception as e:
            return [{"error": str(e)}]
    
    @staticmethod
    def _get_process_info() -> Dict[str, Any]:
        """Get current process information."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "pid": process.pid,
                "name": process.name(),
                "status": process.status(),
                "cpu_percent": process.cpu_percent(),
                "memory_mb": memory_info.rss / 1e6,
                "memory_percent": process.memory_percent(),
                "threads": process.num_threads(),
                "create_time": process.create_time()
            }
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def _get_gpu_memory_info() -> Dict[str, Any]:
        """Get GPU memory information."""
        try:
            from services.ai.image_generation.utils.gpu_checker import get_memory_info
            return get_memory_info()
        except ImportError:
            try:
                import torch
                if torch.cuda.is_available():
                    free_memory, total_memory = torch.cuda.mem_get_info()
                    allocated_memory = torch.cuda.memory_allocated()
                    reserved_memory = torch.cuda.memory_reserved()
                    
                    return {
                        "available": True,
                        "free_gb": free_memory / 1e9,
                        "total_gb": total_memory / 1e9,
                        "allocated_gb": allocated_memory / 1e9,
                        "reserved_gb": reserved_memory / 1e9,
                        "device_name": torch.cuda.get_device_name(0)
                    }
                else:
                    return {"available": False, "reason": "CUDA not available"}
            except ImportError:
                return {"available": False, "reason": "PyTorch not available"}
    
    @staticmethod
    def _get_process_memory_info() -> Dict[str, Any]:
        """Get detailed process memory information."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_full_info = process.memory_full_info()
            
            return {
                "rss_mb": memory_info.rss / 1e6,  # Resident Set Size
                "vms_mb": memory_info.vms / 1e6,  # Virtual Memory Size
                "shared_mb": getattr(memory_info, 'shared', 0) / 1e6,
                "text_mb": getattr(memory_info, 'text', 0) / 1e6,
                "data_mb": getattr(memory_info, 'data', 0) / 1e6,
                "lib_mb": getattr(memory_info, 'lib', 0) / 1e6,
                "dirty_mb": getattr(memory_info, 'dirty', 0) / 1e6,
                "uss_mb": getattr(memory_full_info, 'uss', 0) / 1e6,  # Unique Set Size
                "pss_mb": getattr(memory_full_info, 'pss', 0) / 1e6,  # Proportional Set Size
                "swap_mb": getattr(memory_full_info, 'swap', 0) / 1e6
            }
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def _get_memory_management_status() -> Dict[str, Any]:
        """Get memory manager status if available."""
        try:
            from services.ai.image_generation.memory_manager import get_memory_manager
            memory_manager = get_memory_manager()
            return memory_manager.get_status()
        except ImportError:
            return {"available": False, "reason": "Memory manager not available"}
    
    @staticmethod
    def _get_gpu_utilization() -> Dict[str, Any]:
        """Get GPU utilization information."""
        try:
            import torch
            if torch.cuda.is_available():
                utilization = {}
                for i in range(torch.cuda.device_count()):
                    utilization[f"gpu_{i}"] = {
                        "name": torch.cuda.get_device_name(i),
                        "memory_allocated": torch.cuda.memory_allocated(i) / 1e9,
                        "memory_reserved": torch.cuda.memory_reserved(i) / 1e9,
                        "memory_cached": torch.cuda.memory_reserved(i) / 1e9
                    }
                return utilization
            else:
                return {"available": False, "reason": "CUDA not available"}
        except ImportError:
            return {"available": False, "reason": "PyTorch not available"}
    
    @staticmethod
    def _get_process_metrics() -> Dict[str, Any]:
        """Get current process performance metrics."""
        try:
            process = psutil.Process()
            return {
                "cpu_times": process.cpu_times()._asdict(),
                "memory_info": process.memory_info()._asdict(),
                "io_counters": process.io_counters()._asdict() if hasattr(process, 'io_counters') else None,
                "num_threads": process.num_threads(),
                "num_fds": process.num_fds() if hasattr(process, 'num_fds') else None,
                "connections": len(process.connections()) if hasattr(process, 'connections') else None
            }
        except Exception as e:
            return {"error": str(e)}
    
    # Health check helper methods
    
    @staticmethod
    def _check_system_health() -> Dict[str, Any]:
        """Check overall system health."""
        try:
            issues = []
            
            # Check uptime
            uptime = time.time() - psutil.boot_time()
            if uptime < 60:  # Less than 1 minute
                issues.append("System recently rebooted")
            
            # Check load average if available
            if hasattr(psutil, 'getloadavg'):
                load_avg = psutil.getloadavg()
                cpu_count = psutil.cpu_count()
                if load_avg[0] > cpu_count * 0.8:
                    issues.append("High system load")
            
            return {
                "status": "warning" if issues else "healthy",
                "issues": issues,
                "uptime": uptime
            }
        except Exception as e:
            return {"status": "error", "issues": [str(e)]}
    
    @staticmethod
    def _check_memory_health() -> Dict[str, Any]:
        """Check memory health."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            issues = []
            
            if memory.percent > 90:
                issues.append("Critical memory usage")
            elif memory.percent > 80:
                issues.append("High memory usage")
            
            if swap.percent > 50:
                issues.append("High swap usage")
            
            status = "error" if memory.percent > 95 else ("warning" if issues else "healthy")
            
            return {
                "status": status,
                "issues": issues,
                "memory_percent": memory.percent,
                "swap_percent": swap.percent
            }
        except Exception as e:
            return {"status": "error", "issues": [str(e)]}
    
    @staticmethod
    def _check_gpu_health() -> Dict[str, Any]:
        """Check GPU health."""
        try:
            gpu_info = SystemHandlers._get_gpu_info()
            gpu_memory = SystemHandlers._get_gpu_memory_info()
            issues = []
            
            if not gpu_info.get("available"):
                issues.append("GPU not available")
                return {"status": "warning", "issues": issues}
            
            if gpu_memory.get("available"):
                total_gb = gpu_memory.get("total_gb", 0)
                allocated_gb = gpu_memory.get("allocated_gb", 0)
                if total_gb > 0:
                    usage_percent = (allocated_gb / total_gb) * 100
                    if usage_percent > 90:
                        issues.append("Critical GPU memory usage")
                    elif usage_percent > 80:
                        issues.append("High GPU memory usage")
            
            status = "error" if any("Critical" in issue for issue in issues) else (
                "warning" if issues else "healthy"
            )
            
            return {
                "status": status,
                "issues": issues,
                "gpu_memory": gpu_memory
            }
        except Exception as e:
            return {"status": "error", "issues": [str(e)]}
    
    @staticmethod
    def _check_disk_health() -> Dict[str, Any]:
        """Check disk health."""
        try:
            disk_info = SystemHandlers._get_disk_info()
            issues = []
            
            for disk in disk_info:
                if "error" not in disk:
                    percent = disk.get("percent", 0)
                    if percent > 95:
                        issues.append(f"Critical disk space on {disk['mountpoint']}")
                    elif percent > 90:
                        issues.append(f"Low disk space on {disk['mountpoint']}")
            
            status = "error" if any("Critical" in issue for issue in issues) else (
                "warning" if issues else "healthy"
            )
            
            return {
                "status": status,
                "issues": issues,
                "disk_info": disk_info
            }
        except Exception as e:
            return {"status": "error", "issues": [str(e)]}
    
    @staticmethod
    def _check_services_health() -> Dict[str, Any]:
        """Check health of key services."""
        try:
            issues = []
            service_status = {}
            
            # Check if memory manager is available
            try:
                from services.ai.image_generation.memory_manager import get_memory_manager
                memory_manager = get_memory_manager()
                service_status["memory_manager"] = "available"
            except ImportError:
                service_status["memory_manager"] = "unavailable"
                issues.append("Memory manager service unavailable")
            
            # Check if image generation is available
            try:
                from services.ai.image_generation.utils.gpu_checker import get_gpu_info
                service_status["image_generation"] = "available"
            except ImportError:
                service_status["image_generation"] = "unavailable"
                issues.append("Image generation service unavailable")
            
            # Check if chat services are available
            try:
                from services.chat.chat_manager import ChatManager
                service_status["chat_manager"] = "available"
            except ImportError:
                service_status["chat_manager"] = "unavailable"
                issues.append("Chat manager service unavailable")
            
            status = "warning" if issues else "healthy"
            
            return {
                "status": status,
                "issues": issues,
                "services": service_status
            }
        except Exception as e:
            return {"status": "error", "issues": [str(e)]}
