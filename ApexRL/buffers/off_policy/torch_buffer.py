from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces

from .base_buffer import BaseBuffer
from .types import SequenceBatch, Transition


def _torch_dtype(dtype: Any) -> torch.dtype:
    dtype = np.dtype(dtype)
    return {np.dtype(np.float64): torch.float32, np.dtype(np.float32): torch.float32,
            np.dtype(np.float16): torch.float16, np.dtype(np.int64): torch.int64,
            np.dtype(np.int32): torch.int32, np.dtype(np.uint8): torch.uint8,
            np.dtype(np.bool_): torch.bool}.get(dtype, torch.float32)


class TorchBuffer(BaseBuffer):
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
        device: str | torch.device = "cuda:0",
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

        self.device = torch.device(device)
        self.num_envs = num_envs
        self.capacity = max_size // self.num_envs
        self.linear_decay_step = int(linear_decay_step)
        self.min_weight = float(min_weight)
        self.use_approximate_sampling = use_approximate_sampling
        self.num_buckets = num_buckets
        obs_dtype, action_dtype = _torch_dtype(observation_space.dtype), _torch_dtype(action_space.dtype)
        self.observations = torch.empty((self.capacity, self.num_envs, *self.observation_shape), dtype=obs_dtype, device=self.device)
        self.next_observations = torch.empty_like(self.observations)
        self.actions = torch.empty((self.capacity, self.num_envs, *self.action_shape), dtype=action_dtype, device=self.device)
        self.rewards = torch.empty((self.capacity, self.num_envs), dtype=torch.float32, device=self.device)
        self.terminations = torch.empty_like(self.rewards)
        self.truncations = torch.empty_like(self.rewards)
        self.timestamps = torch.empty(self.capacity, dtype=torch.int64, device=self.device)
        self.reset()

    def _tensor(self, value: Any, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=self.device).reshape(shape)

    def _transition(self, transition: Transition) -> Transition:
        return Transition(
            self._tensor(transition.observations, self.observations.dtype, (self.num_envs, *self.observation_shape)),
            self._tensor(transition.actions, self.actions.dtype, (self.num_envs, *self.action_shape)),
            self._tensor(transition.rewards, torch.float32, (self.num_envs,)),
            self._tensor(transition.truncations, torch.float32, (self.num_envs,)),
            self._tensor(transition.terminations, torch.float32, (self.num_envs,)),
            self._tensor(transition.next_observations, self.next_observations.dtype, (self.num_envs, *self.observation_shape)),
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

    def _valid_starts(self, n_step: int) -> torch.Tensor:
        if n_step < 1:
            raise ValueError("n_step must be positive")
        if self.steps < n_step:
            raise ValueError(f"need {n_step} steps, but only {self.steps} are stored")
        oldest = self.ptr if self.steps == self.capacity else 0
        return (oldest + torch.arange(self.steps - n_step + 1, device=self.device)) % self.capacity

    def _sample_starts(self, batch_size: int, n_step: int) -> torch.Tensor:
        starts = self._valid_starts(n_step)
        if self.linear_decay_step == 0:
            return starts[torch.randint(starts.numel(), (batch_size,), device=self.device)]
        if self.use_approximate_sampling:
            bucket_size = max((starts.numel() + self.num_buckets - 1) // self.num_buckets, 1)
            bucket_starts = torch.arange(0, starts.numel(), bucket_size, device=self.device)
            bucket_ends = torch.clamp(bucket_starts + bucket_size, max=starts.numel())
            midpoint = (bucket_starts + bucket_ends - 1) // 2
            age = self.current_time - self.timestamps[starts[midpoint]]
            if self.linear_decay_step > 0:
                weights = torch.clamp(1.0 - age.float() / self.linear_decay_step, min=self.min_weight)
            else:
                weights = torch.clamp(self.min_weight + age.float() / abs(self.linear_decay_step), max=1.0)
            weights = weights * (bucket_ends - bucket_starts).float()
            total = weights.sum()
            probabilities = weights / total if total > 0 else torch.full_like(weights, 1 / weights.numel())
            bucket = torch.multinomial(probabilities, batch_size, replacement=True)
            offset = (torch.rand(batch_size, device=self.device) * (bucket_ends[bucket] - bucket_starts[bucket]).float()).long()
            return starts[bucket_starts[bucket] + offset]
        age = self.current_time - self.timestamps[starts]
        if self.linear_decay_step > 0:
            weights = torch.clamp(1.0 - age.float() / self.linear_decay_step, min=self.min_weight)
        else:
            weights = torch.clamp(self.min_weight + age.float() / abs(self.linear_decay_step), max=1.0)
        total = weights.sum()
        probabilities = weights / total if total > 0 else torch.full_like(weights, 1 / weights.numel())
        return starts[torch.multinomial(probabilities, batch_size, replacement=True)]

    def can_sample(self, n_step: int = 1) -> bool:
        return self.steps >= n_step

    def sample(self, batch_size: int, n_step: int = 1) -> SequenceBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        starts = self._sample_starts(batch_size, n_step)
        envs = torch.randint(self.num_envs, (batch_size,), device=self.device)
        indices = (starts[None, :] + torch.arange(n_step, device=self.device)[:, None]) % self.capacity
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
        torch.save({name: getattr(self, name) for name in Transition._fields} | {
            "timestamps": self.timestamps, "state": (self.ptr, self.steps, self.current_time)}, path)

    def load(self, path: str | Path) -> None:
        data = torch.load(path, map_location=self.device, weights_only=True)
        for name in Transition._fields:
            getattr(self, name).copy_(data[name])
        self.timestamps.copy_(data["timestamps"])
        self.ptr, self.steps, self.current_time = map(int, data["state"])
        self.size = self.steps * self.num_envs

    def __len__(self) -> int:
        return self.size
