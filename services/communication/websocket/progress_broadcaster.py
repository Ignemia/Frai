"""
from typing import List
WebSocket manager for real-time progress updates.

This module provides a singleton connection manager that can be used
across the application to send progress updates to connected clients.
"""
import logging
import json
import asyncio
from typing import Dict, Optional
from weakref import WeakSet

logger = logging.getLogger(__name__)

class WebSocketProgressManager:
    """
    Singleton manager for WebSocket progress updates.
    
    This allows services like image generation to send progress updates
    to connected WebSocket clients without direct coupling to the API layer.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WebSocketProgressManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.connection_managers = WeakSet()  # Store references to ConnectionManager instances
            self._initialized = True
    
    def register_connection_manager(self, manager):
        """Register a connection manager instance for progress updates."""
        self.connection_managers.add(manager)
        logger.info("Registered WebSocket connection manager for progress updates")
    
    async def send_progress_update(self, chat_id: str, progress_data: dict):
        """
        Send progress update to all registered connection managers.
        
        Args:
            chat_id: The chat ID to send the update to
            progress_data: Dictionary containing progress information
        """
        if not self.connection_managers:
            logger.debug("No connection managers registered for progress updates")
            return
        
        for manager in list(self.connection_managers):  # Create a copy to avoid modification during iteration
            try:
                if hasattr(manager, 'send_progress_update'):
                    await manager.send_progress_update(chat_id, progress_data)
            except Exception as e:
                logger.error(f"Error sending progress update through manager: {e}")
    
    def send_progress_update_sync(self, chat_id: str, progress_data: dict):
        """
        Synchronous wrapper for sending progress updates.
        
        This creates a new event loop if none exists, or schedules
        the coroutine in the existing loop.
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # Schedule the coroutine to run soon
            asyncio.create_task(self.send_progress_update(chat_id, progress_data))
        except RuntimeError:
            # No event loop running, create one for this operation
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.send_progress_update(chat_id, progress_data))
                loop.close()
            except Exception as e:
                logger.error(f"Error running progress update in new event loop: {e}")
        except Exception as e:
            logger.error(f"Error sending progress update: {e}")

# Global instance
websocket_progress_manager = WebSocketProgressManager()

def get_websocket_progress_manager() -> WebSocketProgressManager:
    """Get the global WebSocket progress manager instance."""
    return websocket_progress_manager
