# Adapted from amazon-far/holosoma FastSAC (Apache-2.0).
# https://github.com/amazon-far/holosoma/tree/d18d6cc50f872c15e904a22ceac22313cec955c8/src/holosoma/holosoma/agents/fast_sac
import jax
import jax.numpy as jnp
from flax import nnx

from ....model.jax import CategoricalPolicy, MLP
from ....model.jax.policy import GaussianPolicy, SquashedTanhGaussianPolicy
from ...fasttd3.jax.network import torch_default_init_
from ..config import FastSACConfig


class Actor(nnx.Module):
    def __init__(self, obs_dim: int, action_dim: int, rngs: nnx.Rngs, cfg: FastSACConfig, action_low: jax.Array, action_high: jax.Array) -> None:
        self.obs_dim = obs_dim
        self.use_tanh = cfg.use_tanh
        dtype = getattr(jnp, cfg.compute_type)
        dims = (cfg.actor_hidden_dim, cfg.actor_hidden_dim // 2, cfg.actor_hidden_dim // 4)
        self.trunk = MLP(obs_dim, dims, rngs, layer_norm=cfg.use_layer_norm, activation_fn=jax.nn.silu, compute_type=dtype)
        self.mean = nnx.Linear(dims[-1], action_dim, rngs=rngs, kernel_init=nnx.initializers.zeros, dtype=dtype)
        self.log_std = nnx.Linear(dims[-1], action_dim, rngs=rngs, kernel_init=nnx.initializers.zeros, dtype=dtype)
        for layer in self.trunk.layers:
            torch_default_init_(layer, rngs)
        # log_std is tanh-squashed into [log_std_min, log_std_max] in both cases (upstream)
        self.policy = SquashedTanhGaussianPolicy(action_low, action_high, cfg.log_std_min, cfg.log_std_max) if cfg.use_tanh else GaussianPolicy(cfg.log_std_min, cfg.log_std_max, squash_log_std=True)

    def __call__(self, observations: jax.Array) -> tuple[jax.Array, jax.Array]:
        z = self.trunk(observations)
        return self.mean(z).astype(jnp.float32), self.log_std(z).astype(jnp.float32)

    def get_action(self, observations: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        return self.policy.sample_and_log_prob(*self(observations), key)

    def get_mean_action(self, observations: jax.Array) -> jax.Array:
        mean, _ = self(observations)
        return self.policy.mean_action(mean) if self.use_tanh else mean


class CriticHead(nnx.Module):
    def __init__(self, obs_dim: int, action_dim: int, rngs: nnx.Rngs, cfg: FastSACConfig) -> None:
        dtype = getattr(jnp, cfg.compute_type)
        dims = (cfg.critic_hidden_dim, cfg.critic_hidden_dim // 2, cfg.critic_hidden_dim // 4)
        self.trunk = MLP(obs_dim + action_dim, dims, rngs, layer_norm=cfg.use_layer_norm, activation_fn=jax.nn.silu, compute_type=dtype)
        self.head = nnx.Linear(dims[-1], cfg.num_atoms, rngs=rngs, dtype=dtype)
        for layer in [*self.trunk.layers, self.head]:
            torch_default_init_(layer, rngs)

    def __call__(self, observations: jax.Array, actions: jax.Array) -> jax.Array:
        return self.head(self.trunk(jnp.concatenate((observations, actions), axis=-1))).astype(jnp.float32)


class Critic(nnx.Module):
    def __init__(self, obs_dim: int, action_dim: int, rngs: nnx.Rngs, cfg: FastSACConfig) -> None:
        self.dist = CategoricalPolicy(cfg.num_atoms, cfg.v_min, cfg.v_max)
        make_heads = nnx.vmap(lambda head_rngs: CriticHead(obs_dim, action_dim, head_rngs, cfg), in_axes=0, out_axes=0)
        self.heads = make_heads(rngs.fork(split=cfg.num_q_networks))

    def __call__(self, observations: jax.Array, actions: jax.Array) -> jax.Array:
        # [B, obs], [B, act] -> [E, B, num_atoms]
        return nnx.vmap(lambda head, obs, act: head(obs, act), in_axes=(0, None, None))(self.heads, observations, actions)

    def q_values(self, observations: jax.Array, actions: jax.Array) -> jax.Array:
        return self.dist.q_values(self(observations, actions))


__all__ = ["Actor", "Critic"]
