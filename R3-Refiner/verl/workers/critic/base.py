
"""
Base class for Critic
"""

from abc import ABC, abstractmethod
from typing import Any

import torch

from ...protocol import DataProto
from .config import CriticConfig


__all__ = ["BasePPOCritic"]


class BasePPOCritic(ABC):
    def __init__(self, config: CriticConfig):
        self.config = config

    @abstractmethod
    def compute_values(self, data: DataProto) -> torch.Tensor:
        """Compute values"""
        pass

    @abstractmethod
    def update_critic(self, data: DataProto) -> dict[str, Any]:
        """Update the critic"""
        pass
