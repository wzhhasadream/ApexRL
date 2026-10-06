from __future__ import annotations

from pathlib import Path

import numpy as np
from gymnasium import spaces

from .base_buffer import BaseBuffer
from .types import SequenceBatch, Transition


class NumpyBuffer(BaseBuffer):
    """Simple per-environment ring buffer for raw transitions."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        max_size: int = int(1e6),
        linear_decay_step: int = 0,
        min_weight: float = 0.1,
        num_envs: int = 1,
        use_approximate_sampling: bool = True,
        num_buckets: int = 2000,
    ) -> None:
        if num_envs < 1 or max_size < num_envs:
            raise ValueError("num_envs must be positive and max_size must be at least num_envs")
        num_envs = int(num_envs)
        max_size = max_size // num_envs * num_envs
        super().__init__(observation_space, action_space, max_size)
        if linear_decay_step and not 0 <= min_weight <= 1:
            raise ValueError("min_weight must be in [0, 1]")
        if num_buckets < 1:
            raise ValueError("num_buckets must be positive")

        self.num_envs = num_envs
        self.capacity = max_size // self.num_envs
        self.linear_decay_step = int(linear_decay_step)
        self.min_weight = float(min_weight)
        self.use_approximate_sampling = use_approximate_sampling
        self.num_buckets = num_buckets
        self.observations = np.empty((self.capacity, self.num_envs, *self.observation_shape), dtype=observation_space.dtype)
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty((self.capacity, self.num_envs, *self.action_shape), dtype=action_space.dtype)
        self.rewards = np.empty((self.capacity, self.num_envs), dtype=np.float32)
        self.terminations = np.empty_like(self.rewards)
        self.truncations = np.empty_like(self.rewards)
        self.timestamps = np.empty(self.capacity, dtype=np.int64)
        self.reset()

    def _transition(self, transition: Transition) -> Transition:
        shape = (self.num_envs,)
        return Transition(
            np.asarray(transition.observations, self.observations.dtype).reshape((self.num_envs, *self.observation_shape)),
            np.asarray(transition.actions, self.actions.dtype).reshape((self.num_envs, *self.action_shape)),
            np.asarray(transition.rewards, np.float32).reshape(shape),
            np.asarray(transition.truncations, np.float32).reshape(shape),
            np.asarray(transition.terminations, np.float32).reshape(shape),
            np.asarray(transition.next_observations, self.next_observations.dtype).reshape((self.num_envs, *self.observation_shape)),
        )

    def add(self, transition: Transition) -> None:
        transition = self._transition(transition)
        index = self.ptr
        self.observations[index] = transition.observations
        self.actions[index] = transition.actions
        self.rewards[index] = transition.rewards
        self.truncations[index] = transition.truncations
        self.terminations[index] = transition.terminations
        self.next_observations[index] = transition.next_observations
        self.timestamps[index] = self.current_time
        self.ptr = (index + 1) % self.capacity
        self.steps = min(self.steps + 1, self.capacity)
        self.size = self.steps * self.num_envs
        self.current_time += 1

    def _valid_starts(self, n_step: int) -> np.ndarray:
        if n_step < 1:
            raise ValueError("n_step must be positive")
        if self.steps < n_step:
            raise ValueError(f"need {n_step} steps, but only {self.steps} are stored")
        oldest = self.ptr if self.steps == self.capacity else 0
        count = self.steps - n_step + 1
        return (oldest + np.arange(count)) % self.capacity

    def _sample_starts(self, batch_size: int, n_step: int) -> np.ndarray:
        starts = self._valid_starts(n_step)
        if self.linear_decay_step == 0:
            return np.random.choice(starts, batch_size, replace=True)
        if self.use_approximate_sampling:
            bucket_size = max((starts.size + self.num_buckets - 1) // self.num_buckets, 1)
            bucket_starts = np.arange(0, starts.size, bucket_size)
            bucket_ends = np.minimum(bucket_starts + bucket_size, starts.size)
            midpoint = (bucket_starts + bucket_ends - 1) // 2
            age = self.current_time - self.timestamps[starts[midpoint]]
            if self.linear_decay_step > 0:
                weights = np.maximum(self.min_weight, 1.0 - age / self.linear_decay_step)
            else:
                weights = np.minimum(1.0, self.min_weight + age / abs(self.linear_decay_step))
            weights *= bucket_ends - bucket_starts
            total = weights.sum()
            probabilities = weights / total if total > 0 else np.full_like(weights, 1 / weights.size)
            bucket = np.random.choice(bucket_starts.size, batch_size, p=probabilities)
            offset = (np.random.random(batch_size) * (bucket_ends[bucket] - bucket_starts[bucket])).astype(np.int64)
            return starts[bucket_starts[bucket] + offset]
        age = self.current_time - self.timestamps[starts]
        if self.linear_decay_step > 0:
            weights = np.maximum(self.min_weight, 1.0 - age / self.linear_decay_step)
        else:
            weights = np.minimum(1.0, self.min_weight + age / abs(self.linear_decay_step))
        total = weights.sum()
        probabilities = weights / total if total > 0 else np.full_like(weights, 1 / weights.size)
        return np.random.choice(starts, batch_size, p=probabilities)

    def can_sample(self, n_step: int = 1) -> bool:
        return self.steps >= n_step

    def sample(self, batch_size: int, n_step: int = 1) -> SequenceBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        starts = self._sample_starts(batch_size, n_step)
        envs = np.random.randint(self.num_envs, size=batch_size)
        indices = (starts[None, :] + np.arange(n_step)[:, None]) % self.capacity
        return SequenceBatch(
            self.observations[indices, envs[None, :]],
            self.actions[indices, envs[None, :]],
            self.rewards[indices, envs[None, :], None],
            self.terminations[indices, envs[None, :], None],
            self.truncations[indices, envs[None, :], None],
            self.next_observations[indices, envs[None, :]],
        )

    def reset(self) -> None:
        self.ptr = 0
        self.steps = 0
        self.size = 0
        self.current_time = 0

    def save(self, path: str | Path) -> None:
        np.savez(path, observations=self.observations, next_observations=self.next_observations, actions=self.actions,
                 rewards=self.rewards, terminations=self.terminations, truncations=self.truncations,
                 timestamps=self.timestamps, state=np.array([self.ptr, self.steps, self.current_time]))

    def load(self, path: str | Path) -> None:
        with np.load(path) as data:
            for name in ("observations", "next_observations", "actions", "rewards", "terminations", "truncations", "timestamps"):
                getattr(self, name)[...] = data[name]
            self.ptr, self.steps, self.current_time = map(int, data["state"])
            self.size = self.steps * self.num_envs

    def __len__(self) -> int:
        return self.size
