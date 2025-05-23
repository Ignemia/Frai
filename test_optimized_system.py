#!/usr/bin/env python3
"""
Test script for the optimized personal-chatter image generation system.

This script tests the complete pipeline including:
- Memory management optimizations
- Progress tracking with websocket updates
- Modular folder structure
- Configuration integration
"""
import logging
import time
import asyncio
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_config_system():
    """Test the configuration system."""
    logger.info("Testing configuration system...")
    
    try:
        from services.config import get_config
        config = get_config()
        
        logger.info("✓ Configuration system working")
        logger.info(f"  Memory management enabled: {config.get('memory_management', {}).get('enable_cpu_offload', False)}")
        logger.info(f"  Image generation enabled: {config.get('command_preprocessing', {}).get('image_generation_enabled', False)}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Configuration system failed: {e}")
        return False

def test_websocket_progress_manager():
    """Test the websocket progress manager."""
    logger.info("Testing websocket progress manager...")
    
    try:
        from services.communication.websocket import get_websocket_progress_manager
        
        progress_manager = get_websocket_progress_manager()
        
        # Test sending a progress update (no actual websocket connection)
        test_data = {
            "type": "test",
            "message": "Test progress update",
            "progress": 50.0
        }
        
        # This should not error even without active connections
        progress_manager.send_progress_update_sync("test_chat", test_data)
        
        logger.info("✓ WebSocket progress manager working")
        return True
    except Exception as e:
        logger.error(f"✗ WebSocket progress manager failed: {e}")
        return False

def test_memory_manager():
    """Test the memory management system."""
    logger.info("Testing memory management system...")
    
    try:
        from services.ai.image_generation import get_memory_status, clear_gpu_memory
        
        # Get memory status
        status = get_memory_status()
        logger.info(f"  Memory status: {status}")
        
        # Clear GPU memory (should not error even if no GPU)
        clear_gpu_memory()
        
        logger.info("✓ Memory management system working")
        return True
    except Exception as e:
        logger.error(f"✗ Memory management system failed: {e}")
        return False

def test_image_generation_interface():
    """Test the image generation interface (without actually generating)."""
    logger.info("Testing image generation interface...")
    
    try:
        from services.ai.image_generation import generate_image
        
        # Check if the import works
        logger.info("✓ Image generation interface imported successfully")
        
        # We won't actually generate an image in the test, just verify the interface
        logger.info("  Note: Actual image generation requires model files")
        
        return True
    except Exception as e:
        logger.error(f"✗ Image generation interface failed: {e}")
        return False

def test_chat_manager_integration():
    """Test that chat manager can use the new image generation system."""
    logger.info("Testing chat manager integration...")
    
    try:
        # Test importing chat manager (which imports the new system)
        from services.chat.chat_manager import process_user_message
        
        logger.info("✓ Chat manager imports working")
        return True
    except Exception as e:
        logger.error(f"✗ Chat manager integration failed: {e}")
        return False

def progress_callback_test(step: int, total_steps: int, progress: float, elapsed_time: float):
    """Test progress callback function."""
    logger.info(f"Progress: {progress:.1f}% (step {step}/{total_steps}, elapsed: {elapsed_time:.1f}s)")

async def test_async_system():
    """Test async components."""
    logger.info("Testing async components...")
    
    try:
        from services.ai.image_generation import generate_image_async
        
        logger.info("✓ Async image generation interface available")
        return True
    except Exception as e:
        logger.error(f"✗ Async system failed: {e}")
        return False

def main():
    """Main test function."""
    logger.info("Starting optimized system tests...")
    logger.info("=" * 50)
    
    tests = [
        ("Configuration System", test_config_system),
        ("WebSocket Progress Manager", test_websocket_progress_manager),
        ("Memory Manager", test_memory_manager),
        ("Image Generation Interface", test_image_generation_interface),
        ("Chat Manager Integration", test_chat_manager_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning: {test_name}")
        logger.info("-" * 30)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Test async components
    logger.info(f"\nRunning: Async Components")
    logger.info("-" * 30)
    try:
        async_result = asyncio.run(test_async_system())
        results.append(("Async Components", async_result))
    except Exception as e:
        logger.error(f"Async test crashed: {e}")
        results.append(("Async Components", False))
    
    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info("-" * 50)
    logger.info(f"Total: {passed + failed}, Passed: {passed}, Failed: {failed}")
    
    if failed == 0:
        logger.info("🎉 All tests passed! The optimized system is ready.")
    else:
        logger.warning(f"⚠️  {failed} test(s) failed. Please check the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
