# Adapted from mttga/purejaxql PQN (Apache-2.0), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/mttga/purejaxql/blob/47af6d7b35c89ddfe633aaf7341bdb8964cb7cce/purejaxql/pqn_atari.py
import math
from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype

from ....model.jax.layer import orthogonal
from ....model.jax.policy import DiscreteQGreedyPolicy


class _Norm(nnx.Module):
    """Normalize the last feature axis of convolutional or dense outputs."""

    def __init__(self, features: int, norm_type: Literal["bn", "ln", None], rngs: nnx.Rngs, dtype: Dtype):
        self.norm_type = norm_type
        if norm_type == "bn":
            self.norm = nnx.BatchNorm(features, rngs=rngs, dtype=dtype)
        elif norm_type == "ln":
            self.norm = nnx.LayerNorm(features, rngs=rngs, dtype=dtype)
        else:
            self.norm = None

    def __call__(self, x: jax.Array, training: bool) -> jax.Array:
        if self.norm_type == "bn":
            return self.norm(x, use_running_average=not training)
        if self.norm_type == "ln":
            return self.norm(x)
        return x


class Critic(nnx.Module):
    """Single Q-network used by PQN on stacked Atari observations."""

    def __init__(
        self,
        observation_shape: tuple[int, int, int, int],
        action_dim: int,
        rngs: nnx.Rngs,
        hidden_dim: int = 512,
        compute_type: Dtype = jnp.float32,
        norm_type: Literal["bn", "ln", None] = "ln",
    ):
        frames, height, width, channels = observation_shape
        in_channels = frames * channels
        self.action_dim = action_dim
        self.compute_type = compute_type
        self.policy = DiscreteQGreedyPolicy()

        self.norm_type = norm_type
        kernel_init = nnx.initializers.he_normal()
        self.conv1 = nnx.Conv(
            in_channels, 32, (8, 8), strides=(4, 4), padding="VALID",
            kernel_init=kernel_init, rngs=rngs, dtype=compute_type,
        )
        self.conv2 = nnx.Conv(
            32, 64, (4, 4), strides=(2, 2), padding="VALID",
            kernel_init=kernel_init, rngs=rngs, dtype=compute_type,
        )
        self.conv3 = nnx.Conv(
            64, 64, (3, 3), strides=(1, 1), padding="VALID",
            kernel_init=kernel_init, rngs=rngs, dtype=compute_type,
        )

        dummy = jnp.zeros((1, height, width, in_channels), dtype=compute_type)
        dummy = self.conv1(dummy)
        self.norm1 = _Norm(dummy.shape[-1], norm_type, rngs, compute_type)
        dummy = jax.nn.relu(self.norm1(dummy, training=False))
        dummy = self.conv2(dummy)
        self.norm2 = _Norm(dummy.shape[-1], norm_type, rngs, compute_type)
        dummy = jax.nn.relu(self.norm2(dummy, training=False))
        dummy = self.conv3(dummy)
        self.norm3 = _Norm(dummy.shape[-1], norm_type, rngs, compute_type)
        dummy = jax.nn.relu(self.norm3(dummy, training=False))

        self.fc = nnx.Linear(
            math.prod(dummy.shape[1:]), hidden_dim,
            kernel_init=kernel_init, rngs=rngs, dtype=compute_type,
        )
        self.norm_fc = _Norm(hidden_dim, norm_type, rngs, compute_type)
        self.head = nnx.Linear(
            hidden_dim, action_dim, kernel_init=orthogonal(1.0),
            rngs=rngs, dtype=compute_type,
        )

    def __call__(self, observations: jax.Array, training: bool = False) -> jax.Array:
        x = observations.transpose(0, 2, 3, 1, 4)
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2], -1))
        x = x.astype(self.compute_type) / 255.0
        x = jax.nn.relu(self.norm1(self.conv1(x), training))
        x = jax.nn.relu(self.norm2(self.conv2(x), training))
        x = jax.nn.relu(self.norm3(self.conv3(x), training))
        x = x.reshape((x.shape[0], -1))
        x = jax.nn.relu(self.norm_fc(self.fc(x), training))
        return self.head(x)


    def select_action(self, observations: jax.Array, training: bool = False) -> jax.Array:
        return self.policy.action(self(observations, training=training))
