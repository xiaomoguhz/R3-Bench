

from abc import ABC, abstractmethod

from ...protocol import DataProto


__all__ = ["BaseRollout"]


class BaseRollout(ABC):
    @abstractmethod
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences"""
        pass
