# Adapted from vwxyzjn/cleanrl (MIT) and leggedrobotics/rsl_rl (BSD-3-Clause), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool.py
# https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/ppo.py
import math
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype

from ....buffers.on_policy.types import PolicyMetadata
from ....common import flatten_observation_dim
from ....model.jax import MLP, OnPolicyRMS
from ....model.jax.layer import orthogonal
from ....model.jax.policy import GaussianPolicy, MaskedCategoricalPolicy
from ..config import PPOConfig


def activation_fn(name: str) -> Callable:
    # jax.nn.elu uses expm1, which jax2onnx cannot export; exp(x) - 1 is equivalent here
    if name == "elu":
        return lambda x: jnp.where(x > 0, x, jnp.exp(jnp.minimum(x, 0.0)) - 1.0)
    return getattr(jax.nn, name)


class Encoder(nnx.Module):
    """State observations: frozen rollout statistics followed by an MLP."""

    def __init__(self, obs_shape: tuple[int, ...], hidden_dims: Sequence[int], rngs: nnx.Rngs, activation: Callable, compute_type: Dtype) -> None:
        self.obs_norm = OnPolicyRMS(flatten_observation_dim(obs_shape))
        self.mlp = MLP(flatten_observation_dim(obs_shape), hidden_dims, rngs, activation_fn=activation, compute_type=compute_type)
        self.out_dim = hidden_dims[-1]

    def __call__(self, obs: jax.Array, update_rms: bool = False) -> jax.Array:
        return self.mlp(self.obs_norm.normalize(obs, update_rms))

    def sync_rms(self) -> None:
        self.obs_norm.sync()


class Actor(nnx.Module):
    def __init__(self, obs_shape: tuple[int, ...], action_dim: int, discrete: bool, rngs: nnx.Rngs, cfg: PPOConfig) -> None:
        compute_type = getattr(jnp, cfg.compute_type)
        self.obs_dim = flatten_observation_dim(obs_shape)
        self.discrete = discrete
        self.encoder = Encoder(obs_shape, cfg.actor_hidden_dims, rngs, activation_fn(cfg.activation), compute_type)
        # Near-uniform initial categorical policy (CleanRL)
        self.head = nnx.Linear(self.encoder.out_dim, action_dim, rngs=rngs, kernel_init=orthogonal(0.01 if discrete else 1.0), dtype=compute_type)
        if discrete:
            self.policy = MaskedCategoricalPolicy()
        else:
            self.log_std = nnx.Param(jnp.ones((action_dim,), jnp.float32) * math.log(cfg.init_std))
            self.policy = GaussianPolicy()

    def __call__(self, obs: jax.Array, update_rms: bool = False) -> jax.Array:
        return self.head(self.encoder(obs, update_rms)).astype(jnp.float32)

    def get_action(self, obs: jax.Array, key: jax.Array | None = None, update_rms: bool = True, actions: jax.Array | None = None) -> tuple[jax.Array, jax.Array, jax.Array, PolicyMetadata]:
        out = self(obs, update_rms)
        if self.discrete:
            dist, metadata = self.policy.dist(out), PolicyMetadata(action_logits=out)
        else:
            log_std = jnp.broadcast_to(self.log_std.value, out.shape)
            dist = self.policy.dist(out, log_std)
            metadata = PolicyMetadata(actions_mean=out, actions_std=jnp.exp(self.policy.transform_log_std(log_std)))
        if actions is None:
            actions = dist.sample(seed=key)
        elif self.discrete:
            actions = actions.reshape(-1).astype(jnp.int32)    # buffer stores discrete actions as [B, 1]
        return actions, dist.log_prob(actions).reshape(-1, 1), dist.entropy().reshape(-1, 1), metadata

    def get_mean_action(self, obs: jax.Array) -> jax.Array:
        out = self(obs)
        return out.argmax(-1) if self.discrete else out


class Critic(nnx.Module):
    def __init__(self, obs_shape: tuple[int, ...], rngs: nnx.Rngs, cfg: PPOConfig) -> None:
        compute_type = getattr(jnp, cfg.compute_type)
        self.encoder = Encoder(obs_shape, cfg.critic_hidden_dims, rngs, activation_fn(cfg.activation), compute_type)
        self.value_head = nnx.Linear(self.encoder.out_dim, 1, rngs=rngs, kernel_init=orthogonal(1.0), dtype=compute_type)

    def __call__(self, obs: jax.Array, update_rms: bool = False) -> jax.Array:
        return self.value_head(self.encoder(obs, update_rms)).astype(jnp.float32)


class ActorCritic(nnx.Module):
    def __init__(self, actor: Actor, critic: Critic) -> None:
        self.actor = actor
        self.critic = critic

    def get_mean_action(self, obs: jax.Array) -> jax.Array:
        return self.actor.get_mean_action(obs)

    def get_action_and_value(self, actor_obs: jax.Array, critic_obs: jax.Array, key: jax.Array | None = None, update_rms: bool = True, actions: jax.Array | None = None) -> tuple[jax.Array, jax.Array, jax.Array, PolicyMetadata, jax.Array]:
        actions, log_probs, entropy, metadata = self.actor.get_action(actor_obs, key, update_rms, actions)
        values = self.critic(critic_obs, update_rms)
        return actions, log_probs, entropy, metadata, values

    def sync_rms(self) -> None:
        self.actor.encoder.sync_rms()
        self.critic.encoder.sync_rms()


__all__ = ["Actor", "ActorCritic", "Critic", "Encoder"]
