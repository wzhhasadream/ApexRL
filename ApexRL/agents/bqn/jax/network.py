import math

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype

from ....model.jax import CategoricalPolicy
from ....model.jax.policy import DiscreteQGreedyPolicy


class _Norm(nnx.Module):
    def __init__(self, features: int | tuple[int, ...], rngs: nnx.Rngs, use_bn: bool, dtype: Dtype):
        self.use_bn = use_bn
        self.features = features
        self.bn = nnx.BatchNorm(
            math.prod(features) if isinstance(features, tuple) else features,
            rngs=rngs,
            dtype=dtype,
        ) if use_bn else None

    def __call__(self, x: jax.Array, training: bool) -> jax.Array:
        if not self.use_bn:
            return x
        if isinstance(self.features, tuple):
            shape = x.shape
            x = x.reshape((shape[0], -1))
            x = self.bn(x, use_running_average=not training)
            return x.reshape(shape)
        return self.bn(x, use_running_average=not training)


class Critic(nnx.Module):
    """CrossQ-style single C101 critic for stacked Atari frames."""

    def __init__(self, observation_shape, action_dim, rngs, hidden_dim=512, num_bins=101, use_bn=True, compute_type=jnp.float32, min_value=-5.0, max_value=5.0):
        frames, height, width, channels = observation_shape
        in_channels = frames * channels
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.compute_type = compute_type
        self.dist = CategoricalPolicy(num_bins, min_value, max_value)
        self.policy = DiscreteQGreedyPolicy()
        self.conv1 = nnx.Conv(in_channels, 32, (8, 8), strides=(4, 4), padding="VALID", rngs=rngs, dtype=compute_type)
        self.conv2 = nnx.Conv(32, 64, (4, 4), strides=(2, 2), padding="VALID", rngs=rngs, dtype=compute_type)
        self.conv3 = nnx.Conv(64, 64, (3, 3), strides=(1, 1), padding="VALID", rngs=rngs, dtype=compute_type)
        dummy = jnp.zeros((1, height, width, in_channels), dtype=compute_type)
        dummy1 = self.conv1(dummy)
        dummy2 = self.conv2(dummy1)
        dummy3 = self.conv3(dummy2)
        self.norm1 = _Norm(tuple(dummy1.shape[1:]), rngs, use_bn, compute_type)
        self.norm2 = _Norm(tuple(dummy2.shape[1:]), rngs, use_bn, compute_type)
        self.norm3 = _Norm(tuple(dummy3.shape[1:]), rngs, use_bn, compute_type)
        self.fc = nnx.Linear(math.prod(dummy3.shape[1:]), hidden_dim, rngs=rngs, dtype=compute_type)
        self.norm_fc = _Norm(hidden_dim, rngs, use_bn, compute_type)
        self.head = nnx.Linear(hidden_dim, action_dim * num_bins, rngs=rngs, dtype=compute_type)

    def __call__(self, observations: jax.Array, training: bool = True) -> jax.Array:
        x = observations.transpose(0, 2, 3, 1, 4)
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2], -1)).astype(self.compute_type) / 255.0
        for conv, norm in ((self.conv1, self.norm1), (self.conv2, self.norm2), (self.conv3, self.norm3)):
            x = jax.nn.relu(norm(conv(x), training))
        x = x.reshape((x.shape[0], -1))
        x = jax.nn.relu(self.norm_fc(self.fc(x), training))
        return self.head(x).reshape((x.shape[0], self.action_dim, self.num_bins))


    def q_values(self, logits: jax.Array) -> jax.Array:
        return self.dist.q_values(logits).squeeze(-1)

    def select_action(self, observations: jax.Array, training: bool = False) -> jax.Array:
        return self.policy.action(self.q_values(self(observations, training)))
