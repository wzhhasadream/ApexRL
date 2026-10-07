# Adapted from vwxyzjn/cleanrl (MIT), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool.py
import torch
import torch.nn as nn

from ....buffers.on_policy.types import PolicyMetadata
from ....common import flatten_observation_dim
from ....model.torch import Linear
from ....model.torch.backbones import NatureCNN
from ....model.torch.policy import MaskedCategoricalPolicy
from ..config import PPOConfig


class Encoder(nn.Module):
    """Stacked uint8 frames [F, H, W, C], scaled by NatureCNN without RMS."""

    def __init__(self, obs_shape: tuple[int, ...], hidden_dim: int) -> None:
        super().__init__()
        self.cnn = NatureCNN(obs_shape, hidden_dim)
        self.out_dim = self.cnn.out_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.cnn(obs)


class Actor(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...], action_dim: int, encoder: Encoder) -> None:
        super().__init__()
        self.obs_dim = flatten_observation_dim(obs_shape)
        self.discrete = True
        self.encoder = encoder
        self.head = Linear(encoder.out_dim, action_dim)
        nn.init.orthogonal_(self.head.weight, 0.01)
        self.policy = MaskedCategoricalPolicy()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(obs)).float()

    def get_mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self(obs).argmax(-1)


class Critic(nn.Module):
    def __init__(self, encoder: Encoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.value_head = Linear(encoder.out_dim, 1)

    def forward(self, obs: torch.Tensor, update_rms: bool = False) -> torch.Tensor:
        return self.value_head(self.encoder(obs)).float()


class ActorCritic(nn.Module):
    """One shared encoder, with categorical policy and value heads."""

    def __init__(self, obs_shape: tuple[int, ...], action_dim: int, cfg: PPOConfig) -> None:
        super().__init__()
        encoder = Encoder(obs_shape, cfg.cnn_hidden_dim)
        self.actor = Actor(obs_shape, action_dim, encoder)
        self.critic = Critic(encoder)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor.get_mean_action(obs)

    def get_action_and_value(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor, update_rms: bool = True, actions: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, PolicyMetadata, torch.Tensor]:
        features = self.actor.encoder(actor_obs)
        logits = self.actor.head(features).float()
        values = self.critic.value_head(features).float()
        dist = self.actor.policy.dist(logits)
        actions = dist.sample() if actions is None else actions.reshape(-1).long()
        return actions, dist.log_prob(actions).reshape(-1, 1), dist.entropy().reshape(-1, 1), PolicyMetadata(action_logits=logits), values

    def sync_rms(self) -> None:
        # Keep the common PPO update interface; Atari has no running statistics.
        pass


__all__ = ["Actor", "ActorCritic", "Critic", "Encoder"]
