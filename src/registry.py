from typing import Callable, Dict, List


class Registry:
    """Именованный словарь ключ -> класс/фабрика."""

    def __init__(self, name: str):
        self._name = name
        self._entries: Dict[str, Callable] = {}

    def register(self, key: str) -> Callable:
        def decorator(obj):
            if key in self._entries:
                raise KeyError(f"'{key}' уже зарегистрирован в реестре '{self._name}'")
            self._entries[key] = obj
            return obj

        return decorator

    def get(self, key: str):
        if key not in self._entries:
            raise KeyError(
                f"'{key}' не зарегистрирован в реестре '{self._name}'. "
                f"Доступно: {self.available()}"
            )
        return self._entries[key]

    def available(self) -> List[str]:
        return sorted(self._entries)


EXPERIMENTS = Registry("experiments")

ENCODERS = Registry("encoders")

REGULARIZERS = Registry("regularizers")
