from typing import Any, Dict, List, Optional
from .observer import Observer

class Observable:
    """
    The Observable class (also known as Subject) manages a list of observers
    and notifies them of state changes.
    """
    def __init__(self):
        self._observers: List[Observer] = []

    def attach(self, observer: Observer) -> None:
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: Observer) -> None:
        try:
            self._observers.remove(observer)
        except ValueError:
            pass  # Observer not found

    def _notify(self, key: str, action: str, value: Any = None, old_value: Any = None) -> None:
        """
        Trigger an update in each subscriber.
        """
        for observer in self._observers:
            observer.update(self, key, action, value, old_value)

class AbstractStateModel(Observable):
    """
    An abstract base model for managing state with observable changes.
    It provides basic CRUD-like operations for key-value pairs.
    """
    def __init__(self):
        super().__init__()
        self._data: Dict[str, Any] = {}

    def add_value(self, key: str, value: Any) -> bool:
        """
        Adds a new value to the state.
        Notifies observers if the key did not exist.
        Returns True if added, False if key already exists (use update_value instead).
        """
        if key in self._data:
            # Optionally, you could choose to overwrite or raise an error.
            # For this implementation, we prevent overwriting via add_value.
            return False
        self._data[key] = value
        self._notify(key, action="add", value=value)
        return True

    def update_value(self, key: str, value: Any) -> bool:
        """
        Updates an existing value in the state.
        Notifies observers with the old and new value if the key exists.
        Returns True if updated, False if key does not exist (use add_value instead).
        """
        if key not in self._data:
            return False
        old_value = self._data[key]
        if old_value == value: # No change, no notification
            return True
        self._data[key] = value
        self._notify(key, action="update", value=value, old_value=old_value)
        return True

    def remove_value(self, key: str) -> bool:
        """
        Removes a value from the state.
        Notifies observers with the removed value if the key existed.
        Returns True if removed, False if key did not exist.
        """
        if key not in self._data:
            return False
        old_value = self._data.pop(key)
        self._notify(key, action="remove", old_value=old_value)
        return True

    def get_value(self, key: str) -> Optional[Any]:
        """
        Retrieves a value from the state.
        Returns the value or None if the key does not exist.
        """
        return self._data.get(key)

    def get_all_values(self) -> Dict[str, Any]:
        """
        Retrieves a copy of all data in the state.
        """
        return self._data.copy()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} data={self._data}>"

