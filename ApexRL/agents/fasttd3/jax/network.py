# Adapted from younggyoseo/FastTD3 (MIT), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/younggyoseo/FastTD3/tree/229ed59bbf43ea2f7a2d5d90d1076314839944d7
import math

import jax
import jax.numpy as jnp
from flax import nnx

from ....model.jax import CategoricalPolicy, MLP
from ....model.jax.policy import TanhDeterministicPolicy
from ..config import FastTD3Config


def torch_default_init_(layer: nnx.Linear, rngs: nnx.Rngs) -> None:
    # Upstream uses torch nn.Linear default init: uniform(+-1/sqrt(fan_in)) for weight and bias
    bound = 1.0 / math.sqrt(layer.in_features)
    layer.kernel.value = jax.random.uniform(rngs.params(), layer.kernel.value.shape, minval=-bound, maxval=bound)
    layer.bias.value = jax.random.uniform(rngs.params(), layer.bias.value.shape, minval=-bound, maxval=bound)


class Actor(nnx.Module):
    def __init__(self, obs_dim: int, action_dim: int, rngs: nnx.Rngs, cfg: FastTD3Config, action_low: jax.Array, action_high: jax.Array) -> None:
        self.obs_dim = obs_dim
        dtype = getattr(jnp, cfg.compute_type)
        dims = (cfg.actor_hidden_dim, cfg.actor_hidden_dim // 2, cfg.actor_hidden_dim // 4)
        self.trunk = MLP(obs_dim, dims, rngs, layer_norm=cfg.use_layer_norm, activation_fn=jax.nn.relu, compute_type=dtype)
        self.head = nnx.Linear(dims[-1], action_dim, rngs=rngs, kernel_init=nnx.initializers.normal(cfg.init_scale), bias_init=nnx.initializers.zeros, dtype=dtype)
        self.policy = TanhDeterministicPolicy(action_low, action_high)
        for layer in self.trunk.layers:
            torch_default_init_(layer, rngs)

    def __call__(self, observations: jax.Array) -> jax.Array:
        return self.policy.action(self.head(self.trunk(observations)).astype(jnp.float32))


class CriticHead(nnx.Module):
    def __init__(self, obs_dim: int, action_dim: int, rngs: nnx.Rngs, cfg: FastTD3Config) -> None:
        dtype = getattr(jnp, cfg.compute_type)
        dims = (cfg.critic_hidden_dim, cfg.critic_hidden_dim // 2, cfg.critic_hidden_dim // 4)
        self.trunk = MLP(obs_dim + action_dim, dims, rngs, layer_norm=cfg.use_layer_norm, activation_fn=jax.nn.relu, compute_type=dtype)
        self.head = nnx.Linear(dims[-1], cfg.num_atoms, rngs=rngs, dtype=dtype)
        for layer in [*self.trunk.layers, self.head]:
            torch_default_init_(layer, rngs)

    def __call__(self, observations: jax.Array, actions: jax.Array) -> jax.Array:
        return self.head(self.trunk(jnp.concatenate((observations, actions), axis=-1))).astype(jnp.float32)


class Critic(nnx.Module):
    def __init__(self, obs_dim: int, action_dim: int, rngs: nnx.Rngs, cfg: FastTD3Config) -> None:
        self.dist = CategoricalPolicy(cfg.num_atoms, cfg.v_min, cfg.v_max)
        make_heads = nnx.vmap(lambda head_rngs: CriticHead(obs_dim, action_dim, head_rngs, cfg), in_axes=0, out_axes=0)
        self.heads = make_heads(rngs.fork(split=2))

    def __call__(self, observations: jax.Array, actions: jax.Array) -> jax.Array:
        # [B, obs], [B, act] -> [2, B, num_atoms]
        return nnx.vmap(lambda head, obs, act: head(obs, act), in_axes=(0, None, None))(self.heads, observations, actions)

    def q_values(self, observations: jax.Array, actions: jax.Array) -> jax.Array:
        return self.dist.q_values(self(observations, actions))


__all__ = ["Actor", "Critic", "torch_default_init_"]
