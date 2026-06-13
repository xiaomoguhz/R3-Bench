
"""
ActorRolloutRef config
"""

from dataclasses import dataclass, field

from .actor import ActorConfig, FSDPConfig, ModelConfig, OptimConfig, RefConfig
from .critic import CriticConfig
from .reward import RewardConfig
from .rollout import RolloutConfig


__all__ = [
    "ActorConfig",
    "CriticConfig",
    "FSDPConfig",
    "ModelConfig",
    "OptimConfig",
    "RefConfig",
    "RewardConfig",
    "RolloutConfig",
    "WorkerConfig",
]


@dataclass
class WorkerConfig:
    hybrid_engine: bool = True
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    ref: RefConfig = field(default_factory=RefConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)

    def post_init(self):
        self.ref.micro_batch_size_per_device_for_experience = self.actor.micro_batch_size_per_device_for_experience
        self.ref.padding_free = self.actor.padding_free
        self.ref.dynamic_batching = self.actor.dynamic_batching
        self.ref.ulysses_size = self.actor.ulysses_size
        self.ref.use_torch_compile = self.actor.use_torch_compile
        # Propagate disable_tqdm to RefConfig to prevent AttributeError.
        # RefConfig does not define disable_tqdm, but dp_actor.py accesses self.config.disable_tqdm.
        if hasattr(self.actor, 'disable_tqdm'):
            self.ref.disable_tqdm = self.actor.disable_tqdm
