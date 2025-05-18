"""
Global state management module.
This module provides functions to get and set global state
without causing circular dependencies.
"""

_global_state = {}

def get_state(key, default=None):
    """Get a value from the global state."""
    return _global_state.get(key, default)

def set_state(key, value):
    """Set a value in the global state."""
    _global_state[key] = value
    return value

def clear_state(key=None):
    """
    Clear a specific key from state, or entire state if no key provided.
    Returns the value that was removed, or None if key didn't exist.
    """
    if key is None:
        _global_state.clear()
        return None
    
    return _global_state.pop(key, None)
