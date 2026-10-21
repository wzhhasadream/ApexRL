# Adapted from mttga/purejaxql PQN (Apache-2.0), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/mttga/purejaxql/blob/47af6d7b35c89ddfe633aaf7341bdb8964cb7cce/purejaxql/pqn_atari.py
from __future__ import annotations

import torch
from torch import nn

from ....model.torch import Linear
from ....model.torch.vision import HardMaxCNN, NatureCNN
from ....model.torch.policy import DiscreteQGreedyPolicy
from ..config import PQNConfig


class Critic(nn.Module):
    """Single Q-network used by PQN on stacked Atari observations."""

    def __init__(self, observation_shape: tuple[int, int, int, int], action_dim: int, cfg: PQNConfig) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.policy = DiscreteQGreedyPolicy()
        if cfg.encoder == "hardmax":
            self.encoder = HardMaxCNN(observation_shape, cfg.hidden_dim, cfg.norm_type)
        else:
            self.encoder = NatureCNN(observation_shape, cfg.hidden_dim, cfg.norm_type)
        self.head = Linear(self.encoder.out_dim, action_dim)

    def forward(self, observations: torch.Tensor, training: bool = False) -> torch.Tensor:
        return self.head(self.encoder(observations, training=training)).float()

    def select_action(self, observations: torch.Tensor, training: bool = False) -> torch.Tensor:
        return self.policy.action(self(observations, training=training))


__all__ = ["Critic"]
