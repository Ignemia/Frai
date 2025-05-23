"""
Communication services for Personal Chatter.

This package contains communication-related services including:
- WebSocket management
- Notifications
- Progress broadcasting
"""

# Re-export main WebSocket functionality
from .websocket import get_websocket_progress_manager

__all__ = ['get_websocket_progress_manager']
