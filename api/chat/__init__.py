"""
API initialization for the Chat module.

This module provides the default exports and initialization for the chat API.
"""
from .endpoints import chat_router

__all__ = ['chat_router']