# Adapted from vwxyzjn/cleanrl (MIT), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool.py
import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype

from ....buffers.on_policy.types import PolicyMetadata
from ....common import flatten_observation_dim
from ....model.jax.backbones import NatureCNN
from ....model.jax.layer import orthogonal
from ....model.jax.policy import MaskedCategoricalPolicy
from ..config import PPOConfig


class Encoder(nnx.Module):
    """Stacked uint8 frames [F, H, W, C], scaled by NatureCNN without RMS."""

    def __init__(self, obs_shape: tuple[int, ...], rngs: nnx.Rngs, hidden_dim: int, compute_type: Dtype) -> None:
        self.cnn = NatureCNN(obs_shape, rngs, hidden_dim, compute_type)
        self.out_dim = self.cnn.out_dim

    def __call__(self, obs: jax.Array) -> jax.Array:
        return self.cnn(obs)


class Actor(nnx.Module):
    def __init__(self, obs_shape: tuple[int, ...], action_dim: int, encoder: Encoder, rngs: nnx.Rngs, compute_type: Dtype) -> None:
        self.obs_dim = flatten_observation_dim(obs_shape)
        self.discrete = True
        self.encoder = encoder
        self.head = nnx.Linear(encoder.out_dim, action_dim, rngs=rngs, kernel_init=orthogonal(0.01), dtype=compute_type)
        self.policy = MaskedCategoricalPolicy()

    def __call__(self, obs: jax.Array) -> jax.Array:
        return self.head(self.encoder(obs)).astype(jnp.float32)

    def get_mean_action(self, obs: jax.Array) -> jax.Array:
        return self(obs).argmax(-1)


class Critic(nnx.Module):
    def __init__(self, encoder: Encoder, rngs: nnx.Rngs, compute_type: Dtype) -> None:
        self.encoder = encoder
        self.value_head = nnx.Linear(encoder.out_dim, 1, rngs=rngs, kernel_init=orthogonal(1.0), dtype=compute_type)

    def __call__(self, obs: jax.Array, update_rms: bool = False) -> jax.Array:
        return self.value_head(self.encoder(obs)).astype(jnp.float32)


class ActorCritic(nnx.Module):
    """One shared encoder, with categorical policy and value heads."""

    def __init__(self, obs_shape: tuple[int, ...], action_dim: int, rngs: nnx.Rngs, cfg: PPOConfig) -> None:
        compute_type = getattr(jnp, cfg.compute_type)
        encoder = Encoder(obs_shape, rngs, cfg.cnn_hidden_dim, compute_type)
        self.actor = Actor(obs_shape, action_dim, encoder, rngs, compute_type)
        self.critic = Critic(encoder, rngs, compute_type)

    def get_mean_action(self, obs: jax.Array) -> jax.Array:
        return self.actor.get_mean_action(obs)

    def get_action_and_value(self, actor_obs: jax.Array, critic_obs: jax.Array, key: jax.Array | None = None, update_rms: bool = True, actions: jax.Array | None = None) -> tuple[jax.Array, jax.Array, jax.Array, PolicyMetadata, jax.Array]:
        features = self.actor.encoder(actor_obs)
        logits = self.actor.head(features).astype(jnp.float32)
        values = self.critic.value_head(features).astype(jnp.float32)
        dist = self.actor.policy.dist(logits)
        actions = dist.sample(seed=key) if actions is None else actions.reshape(-1).astype(jnp.int32)
        return actions, dist.log_prob(actions).reshape(-1, 1), dist.entropy().reshape(-1, 1), PolicyMetadata(action_logits=logits), values

    def sync_rms(self) -> None:
        # Keep the common PPO update interface; Atari has no running statistics.
        pass


__all__ = ["Actor", "ActorCritic", "Critic", "Encoder"]
