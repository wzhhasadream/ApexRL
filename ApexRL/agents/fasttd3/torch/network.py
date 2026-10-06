from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ....model.torch import CategoricalPolicy, EnsembleLinear, EnsembleMLP, Linear, MLP
from ....model.torch.policy import TanhDeterministicPolicy
from ..config import FastTD3Config


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        cfg: FastTD3Config,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        dims = (cfg.actor_hidden_dim, cfg.actor_hidden_dim // 2, cfg.actor_hidden_dim // 4)
        self.trunk = MLP(obs_dim, dims, layer_norm=cfg.use_layer_norm, activation_fn=F.relu)
        self.head = Linear(dims[-1], action_dim)
        nn.init.normal_(self.head.weight, std=cfg.init_scale)
        nn.init.zeros_(self.head.bias)
        self.policy = TanhDeterministicPolicy(action_low, action_high)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.policy.action(self.head(self.trunk(observations)))


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, cfg: FastTD3Config) -> None:
        super().__init__()
        self.dist = CategoricalPolicy(cfg.num_atoms, cfg.v_min, cfg.v_max)
        dims = (cfg.critic_hidden_dim, cfg.critic_hidden_dim // 2, cfg.critic_hidden_dim // 4)
        self.num_qs = 2
        self.trunk = EnsembleMLP(2, obs_dim + action_dim, dims, layer_norm=cfg.use_layer_norm, activation_fn=F.relu)
        self.head = EnsembleLinear(2, dims[-1], cfg.num_atoms)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((observations.float(), actions), dim=-1).unsqueeze(0).expand(self.num_qs, -1, -1)
        return self.head(self.trunk(inputs)).float()

    def q_values(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.dist.q_values(self(observations, actions))


__all__ = ["Actor", "Critic"]
