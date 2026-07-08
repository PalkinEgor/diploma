from abc import ABC, abstractmethod


class Runnable(ABC):
    """Единый интерфейс запускаемого эксперимента."""

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> "Runnable":
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError
