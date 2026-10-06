import jax
import jax.numpy as jnp
from flax import nnx

from ....model.jax import CategoricalPolicy, MLP
from ....model.jax.policy import TanhDeterministicPolicy
from ..config import FastTD3Config


class Actor(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        rngs: nnx.Rngs,
        cfg: FastTD3Config,
        action_low: jax.Array,
        action_high: jax.Array,
    ) -> None:
        self.obs_dim = obs_dim
        dtype = getattr(jnp, cfg.compute_type)
        dims = (cfg.actor_hidden_dim, cfg.actor_hidden_dim // 2, cfg.actor_hidden_dim // 4)
        self.trunk = MLP(obs_dim, dims, rngs, layer_norm=cfg.use_layer_norm, activation_fn=jax.nn.relu, compute_type=dtype)
        self.head = nnx.Linear(
            dims[-1], action_dim, rngs=rngs,
            kernel_init=nnx.initializers.normal(cfg.init_scale),
            bias_init=nnx.initializers.zeros,
            dtype=dtype,
        )
        self.policy = TanhDeterministicPolicy(action_low, action_high)

    def __call__(self, observations: jax.Array) -> jax.Array:
        return self.policy.action(self.head(self.trunk(observations)))


class CriticHead(nnx.Module):
    def __init__(self, obs_dim: int, action_dim: int, rngs: nnx.Rngs, cfg: FastTD3Config) -> None:
        dtype = getattr(jnp, cfg.compute_type)
        dims = (cfg.critic_hidden_dim, cfg.critic_hidden_dim // 2, cfg.critic_hidden_dim // 4)
        self.trunk = MLP(obs_dim + action_dim, dims, rngs, layer_norm=cfg.use_layer_norm, activation_fn=jax.nn.relu, compute_type=dtype)
        self.head = nnx.Linear(dims[-1], cfg.num_atoms, rngs=rngs, dtype=dtype)

    def __call__(self, observations: jax.Array, actions: jax.Array) -> jax.Array:
        inputs = jnp.concatenate((observations, actions), axis=-1)
        return self.head(self.trunk(inputs)).astype(jnp.float32)


class Critic(nnx.Module):
    def __init__(self, obs_dim: int, action_dim: int, rngs: nnx.Rngs, cfg: FastTD3Config) -> None:
        self.dist = CategoricalPolicy(cfg.num_atoms, cfg.v_min, cfg.v_max)

        @nnx.vmap(in_axes=0, out_axes=0)
        def make_head(head_rngs: nnx.Rngs) -> CriticHead:
            return CriticHead(obs_dim, action_dim, head_rngs, cfg)

        self.heads = make_head(rngs.fork(split=2))

    def __call__(self, observations: jax.Array, actions: jax.Array) -> jax.Array:
        observations = observations.astype(jnp.float32)
        return nnx.vmap(
            lambda head: head(observations, actions), in_axes=0, out_axes=0
        )(self.heads)

    def q_values(self, observations: jax.Array, actions: jax.Array) -> jax.Array:
        return self.dist.q_values(self(observations, actions))
