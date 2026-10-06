from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from gymnasium import spaces
import orbax.checkpoint as ocp

from .base_buffer import get_action_dim, get_obs_shape
from .types import SequenceBatch, Transition
from ...common.jax.device import resolve_device


def _dtype(dtype: Any) -> jnp.dtype:
    dtype = jnp.dtype(dtype)
    return jnp.float32 if dtype == jnp.float64 and not jax.config.jax_enable_x64 else dtype


@struct.dataclass
class JaxBuffer:
    observations: jax.Array
    next_observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    terminations: jax.Array
    truncations: jax.Array
    timestamps: jax.Array
    ptr: jax.Array
    steps: jax.Array
    size: jax.Array
    current_time: jax.Array

    max_size: int = struct.field(pytree_node=False)
    capacity: int = struct.field(pytree_node=False)
    num_envs: int = struct.field(pytree_node=False)
    obs_shape: tuple[int, ...] = struct.field(pytree_node=False)
    action_shape: tuple[int, ...] = struct.field(pytree_node=False)
    obs_dtype: jnp.dtype = struct.field(pytree_node=False)
    action_dtype: jnp.dtype = struct.field(pytree_node=False)
    linear_decay_step: int = struct.field(pytree_node=False, default=0)
    min_weight: float = struct.field(pytree_node=False, default=0.1)
    use_approximate_sampling: bool = struct.field(pytree_node=False, default=True)
    num_buckets: int = struct.field(pytree_node=False, default=2000)
    device: jax.Device | None = struct.field(pytree_node=False, default=None)

    @classmethod
    def create(
        cls,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        max_size: int = int(1e6),
        linear_decay_step: int = 0,
        min_weight: float = 0.1,
        num_envs: int = 1,
        use_approximate_sampling: bool = True,
        num_buckets: int = 2000,
        device: str | jax.Device | None = "cuda:0",
    ) -> "JaxBuffer":
        if num_envs < 1 or max_size < num_envs:
            raise ValueError("num_envs must be positive and max_size must be at least num_envs")
        num_envs = int(num_envs)
        max_size = max_size // num_envs * num_envs
        if linear_decay_step and not 0 <= min_weight <= 1:
            raise ValueError("min_weight must be in [0, 1]")
        if num_buckets < 1:
            raise ValueError("num_buckets must be positive")
        obs_shape = get_obs_shape(observation_space)
        if isinstance(obs_shape, dict):
            raise NotImplementedError("JaxBuffer does not support Dict observation spaces")
        device = resolve_device(device)
        capacity = max_size // num_envs
        obs_shape, action_shape = tuple(obs_shape), (get_action_dim(action_space),)
        return cls(
            jnp.empty((capacity, num_envs, *obs_shape), _dtype(observation_space.dtype), device=device),
            jnp.empty((capacity, num_envs, *obs_shape), _dtype(observation_space.dtype), device=device),
            jnp.empty((capacity, num_envs, *action_shape), _dtype(action_space.dtype), device=device),
            jnp.empty((capacity, num_envs), jnp.float32, device=device),
            jnp.empty((capacity, num_envs), jnp.float32, device=device),
            jnp.empty((capacity, num_envs), jnp.float32, device=device),
            jnp.zeros((capacity,), jnp.int32, device=device),
            jnp.array(0, jnp.int32, device=device), jnp.array(0, jnp.int32, device=device),
            jnp.array(0, jnp.int32, device=device), jnp.array(0, jnp.int32, device=device),
            max_size, capacity, num_envs, obs_shape, action_shape,
            _dtype(observation_space.dtype), _dtype(action_space.dtype),
            int(linear_decay_step), float(min_weight), bool(use_approximate_sampling), int(num_buckets), device,
        )

    @partial(jax.jit, donate_argnums=(0,))
    def add(self, transition: Transition) -> "JaxBuffer":
        observations = jnp.asarray(transition.observations, self.obs_dtype, device=self.device).reshape((self.num_envs, *self.obs_shape))
        actions = jnp.asarray(transition.actions, self.action_dtype, device=self.device).reshape((self.num_envs, *self.action_shape))
        rewards = jnp.asarray(transition.rewards, jnp.float32, device=self.device).reshape((self.num_envs,))
        terminations = jnp.asarray(transition.terminations, jnp.float32, device=self.device).reshape((self.num_envs,))
        truncations = jnp.asarray(transition.truncations, jnp.float32, device=self.device).reshape((self.num_envs,))
        next_observations = jnp.asarray(transition.next_observations, self.obs_dtype, device=self.device).reshape((self.num_envs, *self.obs_shape))
        steps = jnp.minimum(self.steps + 1, self.capacity)
        return self.replace(
            observations=self.observations.at[self.ptr].set(observations),
            next_observations=self.next_observations.at[self.ptr].set(next_observations),
            actions=self.actions.at[self.ptr].set(actions), rewards=self.rewards.at[self.ptr].set(rewards),
            terminations=self.terminations.at[self.ptr].set(terminations), truncations=self.truncations.at[self.ptr].set(truncations),
            timestamps=self.timestamps.at[self.ptr].set(self.current_time),
            ptr=(self.ptr + 1) % self.capacity, steps=steps, size=steps * self.num_envs,
            current_time=self.current_time + 1,
        )

    def can_sample(self, n_step: int = 1) -> jax.Array:
        return self.steps >= n_step

    @partial(jax.jit, static_argnames=("batch_size", "n_step"))
    def sample(self, key: jax.Array, batch_size: int, n_step: int = 1) -> SequenceBatch:
        if n_step < 1:
            raise ValueError("n_step must be positive")
        key_start, key_env = jax.random.split(key)
        num_starts = self.steps - n_step + 1
        oldest = jnp.where(self.steps == self.capacity, self.ptr, 0)
        valid = jnp.arange(self.capacity) < num_starts
        physical = (oldest + jnp.arange(self.capacity)) % self.capacity
        if self.linear_decay_step == 0:
            logical = jax.random.randint(key_start, (batch_size,), 0, num_starts)
        else:
            age = (self.current_time - self.timestamps[physical]).astype(jnp.float32)
            if self.linear_decay_step > 0:
                weights = jnp.maximum(self.min_weight, 1.0 - age / self.linear_decay_step)
            else:
                weights = jnp.minimum(1.0, self.min_weight + age / abs(self.linear_decay_step))
            weights = jnp.where(valid, weights, 0.0)
            if self.use_approximate_sampling:
                num_buckets = min(self.num_buckets, self.capacity)
                bucket_size = (num_starts + num_buckets - 1) // num_buckets
                bucket_starts = jnp.arange(num_buckets) * bucket_size
                bucket_ends = jnp.minimum(bucket_starts + bucket_size, num_starts)
                non_empty = bucket_starts < num_starts
                midpoint = (bucket_starts + bucket_ends - 1) // 2
                bucket_weights = weights[midpoint] * (bucket_ends - bucket_starts)
                bucket_weights = jnp.where(non_empty, bucket_weights, 0.0)
                key_bucket, key_offset = jax.random.split(key_start)
                total = bucket_weights.sum()
                safe_total = jnp.where(total > 0, total, 1.0)
                probabilities = jnp.where(total > 0, bucket_weights / safe_total, non_empty / non_empty.sum())
                bucket = jax.random.choice(key_bucket, num_buckets, (batch_size,), p=probabilities)
                offset = (jax.random.uniform(key_offset, (batch_size,)) * (bucket_ends[bucket] - bucket_starts[bucket])).astype(jnp.int32)
                logical = bucket_starts[bucket] + offset
            else:
                total = weights.sum()
                safe_total = jnp.where(total > 0, total, 1.0)
                probabilities = jnp.where(total > 0, weights / safe_total, valid / valid.sum())
                logical = jax.random.choice(key_start, self.capacity, (batch_size,), p=probabilities)
        starts = physical[logical]
        envs = jax.random.randint(key_env, (batch_size,), 0, self.num_envs)
        indices = (starts[None, :] + jnp.arange(n_step)[:, None]) % self.capacity
        return SequenceBatch(
            self.observations[indices, envs[None, :]],
            self.actions[indices, envs[None, :]],
            self.rewards[indices, envs[None, :], None],
            self.terminations[indices, envs[None, :], None],
            self.truncations[indices, envs[None, :], None],
            self.next_observations[indices, envs[None, :]],
        )

    @partial(jax.jit, static_argnames=("batch_size", "n_step"))
    def sample_multiple_batch(self, keys: jax.Array, batch_size: int, n_step: int = 1) -> SequenceBatch:
        return jax.vmap(lambda key: self.sample(key, batch_size, n_step))(keys)

    def reset(self) -> "JaxBuffer":
        return self.replace(ptr=jnp.array(0, jnp.int32, device=self.device), steps=jnp.array(0, jnp.int32, device=self.device),
                            size=jnp.array(0, jnp.int32, device=self.device), current_time=jnp.array(0, jnp.int32, device=self.device))

    def save(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        with ocp.StandardCheckpointer() as checkpointer:
            checkpointer.save(checkpoint_dir, {"buffer": self})
            checkpointer.wait_until_finished()

    def load(self, checkpoint_dir: str | Path) -> "JaxBuffer":
        with ocp.StandardCheckpointer() as checkpointer:
            return checkpointer.restore(Path(checkpoint_dir), {"buffer": self})["buffer"]

    def __len__(self) -> int:
        return int(self.size)
