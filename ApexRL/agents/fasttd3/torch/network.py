# Adapted from younggyoseo/FastTD3 (MIT), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/younggyoseo/FastTD3/tree/229ed59bbf43ea2f7a2d5d90d1076314839944d7
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from ....model.torch import MLP, CategoricalPolicy, EnsembleLinear, EnsembleMLP
from ....model.torch.policy import TanhDeterministicPolicy
from ..config import FastTD3Config


def torch_default_init_(layer: nn.Module) -> None:
    # Same as nn.Linear default init (kaiming_uniform a=sqrt(5)), applied per ensemble member
    bound = 1.0 / math.sqrt(layer.in_features)
    nn.init.uniform_(layer.weight, -bound, bound)
    nn.init.uniform_(layer.bias, -bound, bound)


class Actor(nn.Module):
    # obs -> h -> h/2 -> h/4 -> tanh, rescaled to [action_low, action_high]
    def __init__(self, obs_dim: int, action_dim: int, cfg: FastTD3Config, action_low: torch.Tensor, action_high: torch.Tensor) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        dims = (cfg.actor_hidden_dim, cfg.actor_hidden_dim // 2, cfg.actor_hidden_dim // 4)
        self.trunk = MLP(obs_dim, dims, layer_norm=cfg.use_layer_norm, activation_fn=F.relu)
        self.head = nn.Linear(dims[-1], action_dim)
        self.policy = TanhDeterministicPolicy(action_low, action_high)
        for layer in self.trunk.layers:
            torch_default_init_(layer)
        nn.init.normal_(self.head.weight, 0.0, cfg.init_scale)
        nn.init.zeros_(self.head.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.policy.action(self.head(self.trunk(observations)).float())


class Critic(nn.Module):
    # Two distributional Q heads computed in one batched einsum: [2, B, num_atoms]
    def __init__(self, obs_dim: int, action_dim: int, cfg: FastTD3Config) -> None:
        super().__init__()
        self.dist = CategoricalPolicy(cfg.num_atoms, cfg.v_min, cfg.v_max)
        dims = (cfg.critic_hidden_dim, cfg.critic_hidden_dim // 2, cfg.critic_hidden_dim // 4)
        self.trunk = EnsembleMLP(2, obs_dim + action_dim, dims, layer_norm=cfg.use_layer_norm, activation_fn=F.relu)
        self.head = EnsembleLinear(2, dims[-1], cfg.num_atoms)
        for layer in [*self.trunk.layers, self.head]:
            torch_default_init_(layer)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(torch.cat((observations, actions), dim=-1))).float()

    def q_values(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.dist.q_values(self(observations, actions))


__all__ = ["Actor", "Critic", "torch_default_init_"]
