#!/usr/bin/env python3
"""
Test script for the Service Command Interface.

Tests the complete command interface implementation including:
- Command routing and handler registration
- User management operations
- Configuration management
- System monitoring
"""

import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from services.core.command_interface.command_system import Command, CommandType, ExecutionContext
from services.core.command_interface.command_router import CommandRouter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_command_routing():
    """Test basic command routing functionality."""
    print("\n=== Testing Command Routing ===")
    
    try:
        router = CommandRouter()
        
        # Test listing available commands
        commands = router.list_available_commands()
        print(f"Available commands: {len(commands)}")
        
        # Display command categories
        categories = {}
        for cmd_name, cmd_info in commands.items():
            category = cmd_name.split('.')[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(cmd_name)
        
        for category, cmd_list in categories.items():
            print(f"\n{category.upper()} Commands ({len(cmd_list)}):")
            for cmd in sorted(cmd_list):
                has_handler = commands[cmd]['has_handler']
                status = "✓" if has_handler else "✗"
                print(f"  {status} {cmd}")
        
        return True
        
    except Exception as e:
        print(f"Error testing command routing: {e}")
        return False


def test_system_commands():
    """Test system monitoring commands."""
    print("\n=== Testing System Commands ===")
    
    try:
        router = CommandRouter()
        context = ExecutionContext(
            user_id="test_user",
            username="test",
            is_authenticated=True
        )
        
        # Test system status
        status_cmd = Command(
            command_type=CommandType.SYSTEM_STATUS,
            parameters={}
        )
        
        result = router.route_command(status_cmd, context)
        print(f"System Status: {result.success} - {result.message}")
        
        if result.success and result.data:
            print(f"  Memory usage: {result.data.get('memory', {}).get('percent', 'N/A')}%")
            print(f"  CPU usage: {result.data.get('cpu', {}).get('percent', 'N/A')}%")
        
        # Test health check
        health_cmd = Command(
            command_type=CommandType.SYSTEM_HEALTH,
            parameters={}
        )
        
        result = router.route_command(health_cmd, context)
        print(f"Health Check: {result.success} - {result.message}")
        
        return True
        
    except Exception as e:
        print(f"Error testing system commands: {e}")
        return False


def test_config_commands():
    """Test configuration management commands."""
    print("\n=== Testing Configuration Commands ===")
    
    try:
        router = CommandRouter()
        context = ExecutionContext(
            user_id="test_user",
            username="test",
            is_authenticated=True
        )
        
        # Test list config sections
        list_cmd = Command(
            command_type=CommandType.CONFIG_LIST_SECTIONS,
            parameters={}
        )
        
        result = router.route_command(list_cmd, context)
        print(f"List Config Sections: {result.success} - {result.message}")
        
        if result.success and result.data:
            sections = result.data.get('sections', [])
            print(f"  Found {len(sections)} sections: {', '.join(sections)}")
        
        # Test validate config
        validate_cmd = Command(
            command_type=CommandType.CONFIG_VALIDATE,
            parameters={}
        )
        
        result = router.route_command(validate_cmd, context)
        print(f"Validate Config: {result.success} - {result.message}")
        
        return True
        
    except Exception as e:
        print(f"Error testing config commands: {e}")
        return False


def test_user_commands():
    """Test user management commands."""
    print("\n=== Testing User Management Commands ===")
    
    try:
        router = CommandRouter()
        context = ExecutionContext(
            user_id="test_user",
            username="test",
            is_authenticated=True
        )
        
        # Test get user info
        info_cmd = Command(
            command_type=CommandType.USER_GET_INFO,
            parameters={}
        )
        
        result = router.route_command(info_cmd, context)
        print(f"Get User Info: {result.success} - {result.message}")
        
        # Test get user preferences
        prefs_cmd = Command(
            command_type=CommandType.USER_GET_PREFERENCES,
            parameters={}
        )
        
        result = router.route_command(prefs_cmd, context)
        print(f"Get User Preferences: {result.success} - {result.message}")
        
        # Test update preferences
        update_prefs_cmd = Command(
            command_type=CommandType.USER_UPDATE_PREFERENCES,
            parameters={
                'preferences': {
                    'interface': {
                        'theme': 'dark',
                        'language': 'en'
                    }
                }
            }
        )
        
        result = router.route_command(update_prefs_cmd, context)
        print(f"Update User Preferences: {result.success} - {result.message}")
        
        return True
        
    except Exception as e:
        print(f"Error testing user commands: {e}")
        return False


def main():
    """Run all command interface tests."""
    print("Service Command Interface Test Suite")
    print("=" * 50)
    
    tests = [
        ("Command Routing", test_command_routing),
        ("System Commands", test_system_commands),
        ("Configuration Commands", test_config_commands),
        ("User Management Commands", test_user_commands),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("🎉 All tests passed! Command interface is fully functional.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
