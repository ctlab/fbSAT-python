from abc import ABC, abstractmethod

__all__ = ['Task']


class Task(ABC):

    @abstractmethod
    def run(self, *args, **kwargs):
        pass
