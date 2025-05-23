"""
Mock helper module for test integration.

This module provides mock implementations and patching utilities to help with testing.
It ensures imports work properly across modules regardless of import order.
"""
from typing import List, Dict, Any, Tuple, Optional

# Export the types so they can be imported safely
__all__ = ['List', 'Dict', 'Any', 'Tuple', 'Optional']
