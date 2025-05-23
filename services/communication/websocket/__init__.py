"""
WebSocket communication services.

This module provides WebSocket connection management and
real-time progress broadcasting capabilities.
"""

# Re-export progress broadcasting functionality
from .progress_broadcaster import get_websocket_progress_manager

__all__ = ['get_websocket_progress_manager']
