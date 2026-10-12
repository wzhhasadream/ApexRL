from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype

from ....model.jax.backbones import HardMaxCNN, NatureCNN
from ....model.jax.dist_head import CategoricalPolicy
from ....model.jax.layer import orthogonal
from ....model.jax.policy import DiscreteQGreedyPolicy


class Critic(nnx.Module):
    """Single categorical Q-network on stacked Atari observations."""

    def __init__(
        self,
        observation_shape: tuple[int, int, int, int],
        action_dim: int,
        rngs: nnx.Rngs,
        hidden_dim: int = 512,
        compute_type: Dtype = jnp.float32,
        norm_type: Literal["bn", "ln", None] = "bn",
        num_bins: int = 101,
        v_min: float = -10.0,
        v_max: float = 10.0,
        encoder: Literal["nature", "hardmax"] = "nature",
    ):
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.dist = CategoricalPolicy(num_bins, v_min, v_max)
        self.policy = DiscreteQGreedyPolicy()
        if encoder == "hardmax":
            self.encoder = HardMaxCNN(observation_shape, rngs, norm_type, compute_type)
        else:
            self.encoder = NatureCNN(observation_shape, rngs, hidden_dim, compute_type, norm_type)
        self.head = nnx.Linear(
            self.encoder.out_dim, action_dim * num_bins, kernel_init=orthogonal(1.0),
            rngs=rngs, dtype=compute_type,
        )

    def __call__(self, observations: jax.Array, training: bool = False) -> jax.Array:
        return self.head(self.encoder(observations, training=training)).reshape((-1, self.action_dim, self.num_bins)).astype(jnp.float32)

    def q_values(self, observations: jax.Array, training: bool = False) -> jax.Array:
        return self.dist.q_values(self(observations, training=training))[..., 0]

    def select_action(self, observations: jax.Array, training: bool = False) -> jax.Array:
        return self.policy.action(self.q_values(observations, training=training))
