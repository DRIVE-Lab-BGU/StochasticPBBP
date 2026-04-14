from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class BaseSeeder(ABC):
    def __init__(self, start_seed: int) -> None:
        self.start_seed = int(start_seed)

    def __iter__(self) -> 'BaseSeeder':
        return self

    @abstractmethod
    def __next__(self) -> int:
        """Return the next deterministic seed in the sequence."""


class FibonacciSeeder(BaseSeeder):
    def __init__(self, start_seed: int) -> None:
        super().__init__(start_seed)
        self._previous = 0
        self._current = self.start_seed

    def __next__(self) -> int:
        next_value = self._previous + self._current
        self._previous, self._current = self._current, next_value
        return next_value

    def reset(self, new_start_seed: Optional[int] = None) -> None:
        if new_start_seed is not None:
            self.start_seed = new_start_seed
        self._previous = 0
        self._current = self.start_seed
