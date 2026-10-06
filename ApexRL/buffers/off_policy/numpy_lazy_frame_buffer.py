"""Memory-efficient replay buffer for frame-first pixel observations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from gymnasium import spaces
from numpy.lib.stride_tricks import sliding_window_view

from .base_buffer import BaseBuffer
from .types import SequenceBatch, Transition


class NumpyLazyFrameBuffer(BaseBuffer):
    """Store frames in ``[time, num_envs, ...]`` and rebuild stacks on sample."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        max_size: int = int(1e6),
        linear_decay_step: int = 0,
        min_weight: float = 0.1,
        max_n_step: int = 1,
        num_envs: int = 1,
        use_approximate_sampling: bool = True,
        num_buckets: int = 2000,
    ) -> None:
        if not isinstance(observation_space, spaces.Box) or len(observation_space.shape) != 4:
            raise TypeError("NumpyLazyFrameBuffer requires a [F, H, W, C] Box observation space")
        if num_envs < 1 or max_size < num_envs:
            raise ValueError("num_envs must be positive and max_size must be at least num_envs")
        num_envs = int(num_envs)
        max_size = max_size // num_envs * num_envs
        super().__init__(observation_space, action_space, max_size)
        if max_n_step < 1:
            raise ValueError("max_n_step must be positive")
        if linear_decay_step and not 0 <= min_weight <= 1:
            raise ValueError("min_weight must be in [0, 1]")
        if num_buckets < 1:
            raise ValueError("num_buckets must be positive")

        self.num_envs = num_envs
        self.capacity = max_size // self.num_envs
        self.max_n_step = int(max_n_step)
        self.frame_stack = observation_space.shape[0]
        if self.capacity < max(self.frame_stack, self.max_n_step):
            raise ValueError("max_size per environment is too small for frame_stack/max_n_step")
        self.linear_decay_step = int(linear_decay_step)
        self.min_weight = float(min_weight)
        self.use_approximate_sampling = bool(use_approximate_sampling)
        self.num_buckets = int(num_buckets)

        frame_shape = observation_space.shape[1:]
        obs_dtype = np.float32 if observation_space.dtype == np.float64 else observation_space.dtype
        action_dtype = np.float32 if action_space.dtype == np.float64 else action_space.dtype
        length = self.capacity + self.max_n_step + self.frame_stack - 1
        self._frames = np.empty((length, self.num_envs, *frame_shape), dtype=obs_dtype)
        self._actions = np.empty((length, self.num_envs, *self.action_shape), dtype=action_dtype)
        self._rewards = np.empty((length, self.num_envs), dtype=np.float32)
        self._terminations = np.empty_like(self._rewards)
        self._truncations = np.empty_like(self._rewards)
        self._valid = np.zeros((length, self.num_envs), dtype=bool)
        self._timestamps = np.empty((length, self.num_envs), dtype=np.int64) if self.linear_decay_step else None
        self._ordered_valid = np.empty((self.num_envs, self.capacity + 1), dtype=np.int64) if self.linear_decay_step else None

        window = self.frame_stack + self.max_n_step
        self._frame_windows = sliding_window_view(self._frames, window, axis=0).transpose(0, 5, 1, 2, 3, 4)
        self._reward_windows = sliding_window_view(self._rewards, self.max_n_step, axis=0).transpose(0, 2, 1)
        self._termination_windows = sliding_window_view(self._terminations, self.max_n_step, axis=0).transpose(0, 2, 1)
        self._truncation_windows = sliding_window_view(self._truncations, self.max_n_step, axis=0).transpose(0, 2, 1)
        self.reset()

    def _transition(self, transition: Transition) -> Transition:
        return Transition(
            np.asarray(transition.observations, self._frames.dtype).reshape((self.num_envs, *self.observation_shape)),
            np.asarray(transition.actions, self._actions.dtype).reshape((self.num_envs, *self.action_shape)),
            np.asarray(transition.rewards, np.float32).reshape(self.num_envs),
            np.asarray(transition.truncations, np.float32).reshape(self.num_envs),
            np.asarray(transition.terminations, np.float32).reshape(self.num_envs),
            np.asarray(transition.next_observations, self._frames.dtype).reshape((self.num_envs, *self.observation_shape)),
        )

    def _index(self, index: int) -> int:
        index %= self.capacity
        return index + self.capacity if index < self.frame_stack else index

    def _push_valid(self, env: int, index: int) -> None:
        if self._ordered_valid is None:
            return
        tail = self._order_tail[env]
        if tail == self._ordered_valid.shape[1]:
            active = self._ordered_valid[env, self._order_head[env]:tail].copy()
            self._ordered_valid[env, :active.size] = active
            self._order_head[env], tail = 0, active.size
        self._ordered_valid[env, tail] = index
        self._order_tail[env] = tail + 1

    def _pop_valid(self, env: int) -> None:
        if self._ordered_valid is not None:
            self._order_head[env] += 1

    def _mark(self, env: int, index: int, valid: bool) -> None:
        index = self._index(index)
        was_valid = self._valid[index, env]
        if valid:
            self._num_valid[env] += int(not was_valid)
            self._valid[index, env] = True
            if self._timestamps is not None:
                self._timestamps[index, env] = self._current_time[env]
            if not was_valid:
                self._push_valid(env, index)
        else:
            self._num_valid[env] -= int(was_valid)
            self._valid[index, env] = False
            if was_valid:
                self._pop_valid(env)

    def _add_frame(self, env: int, frame: np.ndarray, action: np.ndarray, reward: np.float32,
                   termination: np.float32, truncation: np.float32) -> None:
        index = int(self._ptr[env])
        targets = (index, self.capacity + index) if index < self.max_n_step + self.frame_stack - 1 else (index,)
        for target in targets:
            self._frames[target, env] = frame
            self._actions[target, env] = action
            self._rewards[target, env] = reward
            self._terminations[target, env] = termination
            self._truncations[target, env] = truncation
            if self._timestamps is not None:
                self._timestamps[target, env] = self._current_time[env]

        if self._trajectory_length[env] >= self.max_n_step:
            self._mark(env, index - self.max_n_step + 1, True)
        self._mark(env, index + self.frame_stack, False)
        self._ptr[env] = (index + 1) % self.capacity

    def add(self, transition: Transition) -> None:
        transition = self._transition(transition)
        for env in range(self.num_envs):
            if self._trajectory_length[env] == 0:
                for frame in transition.observations[env]:
                    self._add_frame(env, frame, transition.actions[env], transition.rewards[env],
                                    transition.terminations[env], transition.truncations[env])
                self._trajectory_length[env] = 1
            self._add_frame(env, transition.next_observations[env, -1], transition.actions[env],
                            transition.rewards[env], transition.terminations[env], transition.truncations[env])
            done = bool(transition.terminations[env]) or bool(transition.truncations[env])
            self._trajectory_length[env] = 0 if done else self._trajectory_length[env] + 1
            self._current_time[env] += 1
        self.ptr = self._ptr.copy()
        self.size = int(self._num_valid.sum())

    def _weights(self, timestamps: np.ndarray, env: int) -> np.ndarray:
        age = (self._current_time[env] - timestamps).astype(np.float32)
        if self.linear_decay_step > 0:
            return np.maximum(self.min_weight, 1.0 - age / self.linear_decay_step)
        return np.minimum(1.0, self.min_weight + age / abs(self.linear_decay_step))

    def _sample_indices(self, env: int, batch_size: int) -> np.ndarray:
        if self.linear_decay_step == 0:
            return np.random.choice(np.flatnonzero(self._valid[:, env]), batch_size, replace=True)
        valid = self._ordered_valid[env, self._order_head[env]:self._order_tail[env]]
        if not self.use_approximate_sampling:
            weights = self._weights(self._timestamps[valid, env], env)
            total = weights.sum()
            probabilities = weights / total if total > 0 else np.full_like(weights, 1 / weights.size)
            return np.random.choice(valid, batch_size, p=probabilities)

        bucket_size = max((valid.size + self.num_buckets - 1) // self.num_buckets, 1)
        starts = np.arange(0, valid.size, bucket_size)
        ends = np.minimum(starts + bucket_size, valid.size)
        midpoints = (starts + ends - 1) // 2
        weights = self._weights(self._timestamps[valid[midpoints], env], env) * (ends - starts)
        total = weights.sum()
        probabilities = weights / total if total > 0 else np.full_like(weights, 1 / weights.size)
        buckets = np.random.choice(len(starts), batch_size, p=probabilities)
        offsets = (np.random.random(batch_size) * (ends[buckets] - starts[buckets])).astype(np.int64)
        return valid[starts[buckets] + offsets]

    def can_sample(self, n_step: int = 1) -> bool:
        if n_step < 1 or n_step > self.max_n_step:
            raise ValueError(f"n_step must be in [1, {self.max_n_step}]")
        return bool(np.any(self._num_valid > 0))

    def sample(self, batch_size: int, n_step: int = 1) -> SequenceBatch:
        """Sample raw consecutive transitions with shape ``[n_step, B, ...]``."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if n_step < 1 or n_step > self.max_n_step:
            raise ValueError(f"n_step must be in [1, {self.max_n_step}]")
        if not self.can_sample(n_step):
            raise AssertionError("Cannot sample from an empty buffer")

        eligible = np.flatnonzero(self._num_valid > 0)
        envs = np.random.choice(eligible, batch_size, replace=True)
        starts = np.empty(batch_size, dtype=np.int64)
        for env in eligible:
            positions = np.flatnonzero(envs == env)
            if positions.size:
                starts[positions] = self._sample_indices(env, positions.size)

        frames = self._frame_windows[starts - self.frame_stack, :, envs][:, :self.frame_stack + n_step].swapaxes(0, 1)
        states = np.stack([frames[offset:offset + self.frame_stack] for offset in range(n_step + 1)]).swapaxes(1, 2)
        indices = starts[None, :] + np.arange(n_step)[:, None]
        time_offsets = np.arange(n_step)[:, None]
        reward_windows = self._reward_windows[starts[None, :], time_offsets, envs[None, :]]
        termination_windows = self._termination_windows[starts[None, :], time_offsets, envs[None, :]]
        truncation_windows = self._truncation_windows[starts[None, :], time_offsets, envs[None, :]]
        return SequenceBatch(
            observations=states[:-1],
            actions=self._actions[indices, envs[None, :]],
            rewards=reward_windows[..., None],
            terminations=termination_windows[..., None],
            truncations=truncation_windows[..., None],
            next_observations=states[1:],
        )

    def reset(self) -> None:
        self._ptr = np.zeros(self.num_envs, dtype=np.int64)
        self._num_valid = np.zeros(self.num_envs, dtype=np.int64)
        self._trajectory_length = np.zeros(self.num_envs, dtype=np.int64)
        self._current_time = np.zeros(self.num_envs, dtype=np.int64)
        self.ptr = self._ptr.copy()
        self.size = 0
        self._valid.fill(False)
        if self._ordered_valid is not None:
            self._order_head = np.zeros(self.num_envs, dtype=np.int64)
            self._order_tail = np.zeros(self.num_envs, dtype=np.int64)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix != ".npz":
            path.mkdir(parents=True, exist_ok=True)
            path = path / "replay.npz"
        np.savez(path, frames=self._frames, actions=self._actions, rewards=self._rewards,
                 terminations=self._terminations, truncations=self._truncations, valid=self._valid,
                 timestamps=self._timestamps if self._timestamps is not None else np.empty(0, np.int64),
                 state=np.stack((self._ptr, self._num_valid, self._trajectory_length, self._current_time)))

    def load(self, path: str | Path) -> None:
        path = Path(path)
        npz_path = path if path.suffix == ".npz" else path / "replay.npz"
        with np.load(npz_path) as data:
            for name in ("frames", "actions", "rewards", "terminations", "truncations", "valid"):
                getattr(self, f"_{name}")[...] = data[name]
            if self._timestamps is not None and data["timestamps"].size:
                self._timestamps[...] = data["timestamps"]
            self._ptr, self._num_valid, self._trajectory_length, self._current_time = data["state"]
        self._ptr = self._ptr.astype(np.int64)
        self._num_valid = self._num_valid.astype(np.int64)
        self._trajectory_length = self._trajectory_length.astype(np.int64)
        self._current_time = self._current_time.astype(np.int64)
        self.ptr, self.size = self._ptr.copy(), int(self._num_valid.sum())
        if self._ordered_valid is not None:
            self._order_head.fill(0)
            self._order_tail.fill(0)
            for env in range(self.num_envs):
                valid = np.flatnonzero(self._valid[:, env])
                valid = valid[np.argsort(self._timestamps[valid, env], kind="stable")]
                self._ordered_valid[env, :valid.size] = valid
                self._order_tail[env] = valid.size

    def __len__(self) -> int:
        return self.size


NumpyPixelBuffer = NumpyLazyFrameBuffer
