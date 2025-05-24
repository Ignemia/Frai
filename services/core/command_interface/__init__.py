"""
Service Command Interface (SCI) - Core abstraction layer for all user interactions.

This module provides a unified interface for handling all user operations
across different interfaces (CLI, API, Web UI, etc.).
"""

from .command_system import Command, CommandResult, ExecutionContext
from .service_interface import ServiceCommandInterface
from .command_router import CommandRouter
from .response_formatter import ResponseFormatter

__all__ = [
    'Command',
    'CommandResult', 
    'ExecutionContext',
    'ServiceCommandInterface',
    'CommandRouter',
    'ResponseFormatter'
]
