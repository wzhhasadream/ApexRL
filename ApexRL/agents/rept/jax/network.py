from flax import nnx
import jax
import jax.numpy as jnp
from flax.typing import Dtype
from typing import Literal

from ....model.jax.policy import SquashedTanhGaussianPolicy
from ....model.jax import CategoricalPolicy, Alpha
from ....model.jax.backbones import (
    FlashSACEncoder,
    DRQVisionEncoder,
    DRQV2VisionEncoder,
    VSimbaVisionEncoder,
)
from ....model.jax.layer import orthogonal

CRITIC_STATE_AXES = nnx.StateAxes({
    nnx.PathContains("dist"): None,
    nnx.Param: 0,
    nnx.BatchStat: 0,
    nnx.Variable: 0,
})


class Encoder(nnx.Module):
    """Shared representations with next-state and reward prediction heads."""

    def __init__(self, obs_shape: tuple[int, ...], action_dim: int, rngs: nnx.Rngs, num_blocks: int = 1,
                 z_s_dim: int = 256, z_sa_dim: int = 256, use_bias: bool = True, compute_type: Dtype = jnp.float32,
                 vision_encoder: Literal["drq_dot", "drqv2", "vsimba"] = "drqv2") -> None:
        self.cnn = None
        input_dim = obs_shape[0]
        if len(obs_shape) == 4:
            cnn_cls = {"drq_dot": DRQVisionEncoder, "drqv2": DRQV2VisionEncoder, "vsimba": VSimbaVisionEncoder}[vision_encoder]
            self.cnn = cnn_cls(obs_shape, rngs=rngs, compute_type=compute_type)
            input_dim = self.cnn.get_output_dim()
        self.s_encoder = FlashSACEncoder(input_dim, num_blocks, z_s_dim, rngs, use_bias, compute_type)
        self.sa_encoder = FlashSACEncoder(z_s_dim + action_dim, num_blocks, z_sa_dim, rngs, use_bias, compute_type)
        self.reward_head = nnx.Linear(z_sa_dim, 1, use_bias=use_bias, dtype=compute_type, rngs=rngs)
        self.p_w = nnx.Linear(z_sa_dim, z_s_dim, use_bias=use_bias, dtype=compute_type, rngs=rngs)
        self.nce_alpha = Alpha()

    def embed_s(self, observations: jax.Array, training: bool = False) -> jax.Array:
        observations = self.cnn(observations) if self.cnn is not None else observations
        return self.s_encoder(observations, training=training).astype(jnp.float32)

    def embed_sa(self, z_s: jax.Array, actions: jax.Array, training: bool = False) -> jax.Array:
        return self.sa_encoder(jnp.concatenate([z_s, actions], axis=-1), training=training).astype(jnp.float32)

    def get_alpha(self) -> jax.Array:
        return jnp.clip(self.nce_alpha(), 0, 100)

    def pred_reward(self, z_sa: jax.Array) -> jax.Array:
        return self.reward_head(z_sa).astype(jnp.float32)

    def pred_z_s(self, z_sa: jax.Array) -> jax.Array:
        return self.p_w(z_sa).astype(jnp.float32)

class Critic(nnx.Module):
    """Independent categorical Q networks consuming shared [B, D] state-action features."""

    @nnx.vmap(in_axes=(CRITIC_STATE_AXES, None, None, 0, None, None, None, None))
    def __init__(self, z_sa_dim: int, hidden_dim: int, rngs: nnx.Rngs, num_blocks: int = 1, use_bias: bool = True,
                 compute_type: Dtype = jnp.float32, num_head: int = 101) -> None:
        self.trunk = FlashSACEncoder(z_sa_dim, num_blocks, hidden_dim, rngs, use_bias, compute_type)
        self.value_head = nnx.Linear(hidden_dim, num_head, dtype=compute_type, rngs=rngs)
        self.dist = CategoricalPolicy(num_head, -5.0, 5.0)

    @nnx.vmap(in_axes=(CRITIC_STATE_AXES, None, None), out_axes=0)
    def __call__(self, z_sa: jax.Array, training: bool = True) -> jax.Array:
        x = self.trunk(z_sa, training)
        logits = self.value_head(x)
        return logits.astype(jnp.float32)

    def q_values(self, z_sa: jax.Array, training: bool = True) -> jax.Array:
        return self.dist.q_values(self(z_sa, training=training))


class Actor(nnx.Module):
    """Continuous policy consuming the shared state representation."""

    def __init__(self, z_s_dim: int, action_dim: int, rngs: nnx.Rngs, hidden_dim: int = 256, num_blocks: int = 2,
                 action_low: jax.Array = -1.0, action_high: jax.Array = 1.0, use_bias: bool = True,
                 compute_type: Dtype = jnp.float32) -> None:
        self.trunk = FlashSACEncoder(z_s_dim, num_blocks, hidden_dim, rngs, use_bias, compute_type)
        self.fc_mean = nnx.Linear(hidden_dim, action_dim, rngs=rngs, kernel_init=orthogonal(1), dtype=compute_type)
        self.fc_log_std = nnx.Linear(hidden_dim, action_dim, rngs=rngs, kernel_init=orthogonal(1), dtype=compute_type)
        self.policy = SquashedTanhGaussianPolicy(action_low=action_low, action_high=action_high)

    def __call__(self, z_s: jax.Array, training: bool = True) -> tuple[jax.Array, jax.Array]:
        x = self.trunk(z_s, training=training)
        return self.fc_mean(x).astype(jnp.float32), self.fc_log_std(x).astype(jnp.float32)

    def get_action(self, z_s: jax.Array, *, key: jax.Array, training: bool = True) -> tuple[jax.Array, jax.Array]:
        mean, log_std = self(z_s, training=training)
        return self.policy.sample_and_log_prob(mean, log_std, key)

    def get_deterministic_action(self, z_s: jax.Array) -> jax.Array:
        mean, _ = self(z_s, training=False)
        return self.policy.mean_action(mean).astype(jnp.float32)
