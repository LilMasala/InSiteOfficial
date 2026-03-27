"""Writer interface."""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseWriter(ABC):
    @abstractmethod
    def write_patient(self, payload: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> None:
        raise NotImplementedError
