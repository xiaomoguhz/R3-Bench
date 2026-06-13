
"""
The base class for Actor
"""

from abc import ABC, abstractmethod
from typing import Any

import torch

from ...protocol import DataProto
from .config import ActorConfig


__all__ = ["BasePPOActor"]


class BasePPOActor(ABC):
    def __init__(self, config: ActorConfig):
        """The base class for PPO actor

        Args:
            config (ActorConfig): a config passed to the PPOActor.
        """
        self.config = config

    @abstractmethod
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute logits given a batch of data.

        Args:
            data (DataProto): a batch of data represented by DataProto. It must contain key ```input_ids```,
                ```attention_mask``` and ```position_ids```.

        Returns:
            DataProto: a DataProto containing the key ```log_probs```
        """
        pass

    @abstractmethod
    def update_policy(self, data: DataProto) -> dict[str, Any]:
        """Update the policy with an iterator of DataProto

        Args:
            data (DataProto): an iterator over the DataProto that returns by
                ```make_minibatch_iterator```

        Returns:
            Dict: a dictionary contains anything. Typically, it contains the statistics during updating the model
            such as ```loss```, ```grad_norm```, etc,.
        """
        pass
