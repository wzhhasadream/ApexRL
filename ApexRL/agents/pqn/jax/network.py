# Adapted from mttga/purejaxql PQN (Apache-2.0), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/mttga/purejaxql/blob/47af6d7b35c89ddfe633aaf7341bdb8964cb7cce/purejaxql/pqn_atari.py
from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype

from ....model.jax.backbones import HardMaxCNN, NatureCNN
from ....model.jax.layer import orthogonal
from ....model.jax.policy import DiscreteQGreedyPolicy


class Critic(nnx.Module):
    """Single Q-network used by PQN on stacked Atari observations."""

    def __init__(
        self,
        observation_shape: tuple[int, int, int, int],
        action_dim: int,
        rngs: nnx.Rngs,
        hidden_dim: int = 512,
        compute_type: Dtype = jnp.float32,
        norm_type: Literal["bn", "brn", "ln", "rms"] | None = "ln",
        encoder: Literal["nature", "hardmax"] = "nature",
    ):
        self.action_dim = action_dim
        self.policy = DiscreteQGreedyPolicy()
        if encoder == "hardmax":
            self.encoder = HardMaxCNN(observation_shape, rngs, hidden_dim, compute_type, norm_type)
        else:
            self.encoder = NatureCNN(observation_shape, rngs, hidden_dim, compute_type, norm_type)
        self.head = nnx.Linear(
            self.encoder.out_dim, action_dim, kernel_init=orthogonal(1.0),
            rngs=rngs, dtype=compute_type,
        )

    def __call__(self, observations: jax.Array, training: bool = False) -> jax.Array:
        return self.head(self.encoder(observations, training=training))

    def select_action(self, observations: jax.Array, training: bool = False) -> jax.Array:
        return self.policy.action(self(observations, training=training))
