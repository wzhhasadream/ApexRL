# Adapted from vwxyzjn/cleanrl (MIT) and leggedrobotics/rsl_rl (BSD-3-Clause), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool.py
# https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/ppo.py
import math
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....buffers.on_policy.types import PolicyMetadata
from ....common import flatten_observation_dim
from ....model.torch import Linear, MLP, OnPolicyRMS
from ....model.torch.backbones import NatureCNN
from ....model.torch.policy import GaussianPolicy, MaskedCategoricalPolicy
from ..config import PPOConfig


class Encoder(nn.Module):
    """Flat obs: running mean/std + MLP. Image obs [F, H, W, C] uint8: Nature CNN."""

    def __init__(self, obs_shape: tuple[int, ...], hidden_dims: Sequence[int], activation: Callable, cnn_hidden_dim: int) -> None:
        super().__init__()
        self.image = len(obs_shape) > 1
        if self.image:
            self.cnn = NatureCNN(obs_shape, cnn_hidden_dim)
            self.out_dim = self.cnn.out_dim
        else:
            self.obs_norm = OnPolicyRMS(flatten_observation_dim(obs_shape))
            self.mlp = MLP(flatten_observation_dim(obs_shape), hidden_dims, layer_norm=True, activation_fn=activation)
            self.out_dim = hidden_dims[-1]

    def forward(self, obs: torch.Tensor, update_rms: bool = False) -> torch.Tensor:
        if self.image:
            return self.cnn(obs)
        return self.mlp(self.obs_norm.normalize(obs, update_rms))

    def sync_rms(self) -> None:
        if not self.image:
            self.obs_norm.sync()


class Actor(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...], action_dim: int, discrete: bool, cfg: PPOConfig) -> None:
        super().__init__()
        self.obs_dim = flatten_observation_dim(obs_shape)
        self.discrete = discrete
        self.encoder = Encoder(obs_shape, cfg.actor_hidden_dims, getattr(F, cfg.activation), cfg.cnn_hidden_dim)
        self.head = Linear(self.encoder.out_dim, action_dim)
        if discrete:
            nn.init.orthogonal_(self.head.weight, 0.01)    # near-uniform initial policy (CleanRL)
            self.policy = MaskedCategoricalPolicy()
        else:
            self.log_std = nn.Parameter(torch.ones(action_dim) * math.log(cfg.init_std))
            self.policy = GaussianPolicy()

    def forward(self, obs: torch.Tensor, update_rms: bool = False) -> torch.Tensor:
        return self.head(self.encoder(obs, update_rms)).float()

    def get_action(self, obs: torch.Tensor, update_rms: bool = True, actions: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, PolicyMetadata]:
        out = self(obs, update_rms)
        if self.discrete:
            dist, metadata = self.policy.dist(out), PolicyMetadata(action_logits=out)
        else:
            dist = self.policy.dist(out, self.log_std.expand_as(out))
            metadata = PolicyMetadata(actions_mean=out, actions_std=dist.base_dist.scale)
        if actions is None:
            actions = dist.sample()
        elif self.discrete:
            actions = actions.reshape(-1).long()    # buffer stores discrete actions as [B, 1]
        return actions, dist.log_prob(actions).reshape(-1, 1), dist.entropy().reshape(-1, 1), metadata

    def get_mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        out = self(obs)
        return out.argmax(-1) if self.discrete else out


class Critic(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...], cfg: PPOConfig) -> None:
        super().__init__()
        self.encoder = Encoder(obs_shape, cfg.critic_hidden_dims, getattr(F, cfg.activation), cfg.cnn_hidden_dim)
        self.value_head = Linear(self.encoder.out_dim, 1)

    def forward(self, obs: torch.Tensor, update_rms: bool = False) -> torch.Tensor:
        return self.value_head(self.encoder(obs, update_rms)).float()


class ActorCritic(nn.Module):
    def __init__(self, actor: Actor, critic: Critic) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor.get_mean_action(obs)

    def sync_rms(self) -> None:
        self.actor.encoder.sync_rms()
        self.critic.encoder.sync_rms()


__all__ = ["Actor", "ActorCritic", "Critic", "Encoder"]
