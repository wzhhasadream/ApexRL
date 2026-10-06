# Adapted from amazon-far/holosoma FastSAC (Apache-2.0).
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ....model.torch import CategoricalPolicy, EnsembleLinear, EnsembleMLP, MLP
from ....model.torch.policy import GaussianPolicy, SquashedTanhGaussianPolicy
from ...fasttd3.torch.network import torch_default_init_
from ..config import FastSACConfig


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, cfg: FastSACConfig, action_low: torch.Tensor, action_high: torch.Tensor) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.use_tanh = cfg.use_tanh
        dims = (cfg.actor_hidden_dim, cfg.actor_hidden_dim // 2, cfg.actor_hidden_dim // 4)
        self.trunk = MLP(obs_dim, dims, layer_norm=cfg.use_layer_norm, activation_fn=F.silu)
        self.mean = nn.Linear(dims[-1], action_dim)
        self.log_std = nn.Linear(dims[-1], action_dim)
        for layer in self.trunk.layers:
            torch_default_init_(layer)
        for layer in (self.mean, self.log_std):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        # log_std is tanh-squashed into [log_std_min, log_std_max] in both cases (upstream)
        self.policy = SquashedTanhGaussianPolicy(action_low, action_high, cfg.log_std_min, cfg.log_std_max) if cfg.use_tanh else GaussianPolicy(cfg.log_std_min, cfg.log_std_max, squash_log_std=True)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(observations)
        return self.mean(z).float(), self.log_std(z).float()

    def get_action(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy.sample_and_log_prob(*self(observations))

    def get_mean_action(self, observations: torch.Tensor) -> torch.Tensor:
        mean, _ = self(observations)
        return self.policy.mean_action(mean) if self.use_tanh else mean


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, cfg: FastSACConfig) -> None:
        super().__init__()
        self.dist = CategoricalPolicy(cfg.num_atoms, cfg.v_min, cfg.v_max)
        dims = (cfg.critic_hidden_dim, cfg.critic_hidden_dim // 2, cfg.critic_hidden_dim // 4)
        self.trunk = EnsembleMLP(cfg.num_q_networks, obs_dim + action_dim, dims, layer_norm=cfg.use_layer_norm, activation_fn=F.silu)
        self.head = EnsembleLinear(cfg.num_q_networks, dims[-1], cfg.num_atoms)
        for layer in [*self.trunk.layers, self.head]:
            torch_default_init_(layer)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # [B, obs + act] -> [E, B, num_atoms]
        return self.head(self.trunk(torch.cat((observations, actions), dim=-1))).float()

    def q_values(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.dist.q_values(self(observations, actions))


__all__ = ["Actor", "Critic"]
