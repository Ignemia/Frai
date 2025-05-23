from test_mock_helper import List

"""
Global state management module.
This module provides functions to get and set global state
without causing circular dependencies.
"""

# Define application states
APP_STATE_HOME = "Home"
APP_STATE_LOGIN = "Login"
APP_STATE_REGISTER = "Register"
APP_STATE_LIST = "List"
APP_STATE_CHAT = "Gemma 3"  # Name of the model for chat state

# Initialize with the home state
_state = {
    "app_state": APP_STATE_HOME
}

def get_state(key, default=None):
    """Get a value from the state"""
    return _state.get(key, default)

def set_state(key, value):
    """Set a value in the state"""
    _state[key] = value
    return value

def clear_state(key=None):
    """
    Clear a specific key from state, or entire state if no key provided.
    Returns the value that was removed, or None if key didn't exist.
    """
    if key is None:
        _state.clear()
        return None
    
    return _state.pop(key, None)

def get_current_app_state():
    """Get the current application state"""
    return _state.get("app_state", APP_STATE_HOME)

def set_app_state(state):
    """Set the current application state"""
    _state["app_state"] = state
    return state
