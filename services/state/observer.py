from abc import ABC, abstractmethod

class Observer(ABC):
    """
    The Observer interface declares the update method, used by subjects.
    """
    @abstractmethod
    def update(self, subject, key: str, action: str, value=None, old_value=None) -> None:
        """
        Receive update from subject.
        Action can be 'add', 'update', 'remove'.
        """
        pass

class ConcreteObserver(Observer):
    """
    Concrete Observers react to the updates issued by the Subject they had been
    attached to.
    """
    def update(self, subject, key: str, action: str, value=None, old_value=None) -> None:
        if action == "add":
            print(f"Observer: Value added for key '{key}': {value}")
        elif action == "update":
            print(f"Observer: Value updated for key '{key}'. Old: {old_value}, New: {value}")
        elif action == "remove":
            print(f"Observer: Value removed for key '{key}'. Last known value: {old_value}")
        else:
            print(f"Observer: Unknown action '{action}' for key '{key}'")

