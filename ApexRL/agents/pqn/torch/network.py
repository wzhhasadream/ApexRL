from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

from ....model.torch import Linear
from ....model.torch.layer import BatchNorm1d
from ....model.torch.policy import DiscreteQGreedyPolicy
from ..config import PQNConfig


class _Norm(nn.Module):
    """Normalize the channel axis (dim 1) of conv or dense features."""

    def __init__(self, features: int, norm_type: Literal["bn", "ln", None]) -> None:
        super().__init__()
        self.norm_type = norm_type
        if norm_type == "bn":
            self.norm = BatchNorm1d(features)
        elif norm_type == "ln":
            self.norm = nn.LayerNorm(features)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        if self.norm_type == "bn":
            return self.norm(x, training=training)
        if self.norm_type == "ln":
            return self.norm(x.movedim(1, -1)).movedim(-1, 1)
        return x


class Critic(nn.Module):
    """Single Q-network used by PQN on stacked Atari observations."""

    def __init__(self, observation_shape: tuple[int, int, int, int], action_dim: int, cfg: PQNConfig) -> None:
        super().__init__()
        frames, height, width, channels = observation_shape
        in_channels = frames * channels
        self.action_dim = action_dim
        self.policy = DiscreteQGreedyPolicy()

        self.conv1 = nn.Conv2d(in_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.norm1 = _Norm(32, cfg.norm_type)
        self.norm2 = _Norm(64, cfg.norm_type)
        self.norm3 = _Norm(64, cfg.norm_type)
        with torch.no_grad():
            feature_shape = self._encode(torch.zeros(1, in_channels, height, width), training=False).shape
        self.fc = Linear(math.prod(feature_shape[1:]), cfg.hidden_dim)
        self.norm_fc = _Norm(cfg.hidden_dim, cfg.norm_type)
        self.head = Linear(cfg.hidden_dim, action_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        # he_normal for conv/fc, orthogonal head (Linear's default), as in the JAX version.
        for layer in (self.conv1, self.conv2, self.conv3, self.fc):
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def _encode(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        x = F.relu(self.norm1(self.conv1(x), training))
        x = F.relu(self.norm2(self.conv2(x), training))
        x = F.relu(self.norm3(self.conv3(x), training))
        return x.flatten(1)

    def forward(self, observations: torch.Tensor, training: bool = False) -> torch.Tensor:
        # [B, F, H, W, C] uint8 -> [B, F * C, H, W] float in [0, 1]
        x = observations.permute(0, 1, 4, 2, 3).flatten(1, 2).float() / 255.0
        x = F.relu(self.norm_fc(self.fc(self._encode(x, training)), training))
        return self.head(x).float()

    def select_action(self, observations: torch.Tensor, training: bool = False) -> torch.Tensor:
        return self.policy.action(self(observations, training=training))


__all__ = ["Critic"]
