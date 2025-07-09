from abc import ABC, abstractmethod

class BaseSolver(ABC):
    @abstractmethod
    def configure(self, **kwargs):
        pass

    @abstractmethod
    def solve(self, optimization_vars):
        pass