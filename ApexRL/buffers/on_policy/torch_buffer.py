"""A minimal fixed-horizon on-policy rollout buffer for PyTorch."""

from __future__ import annotations

from collections.abc import Generator
import numpy as np
import torch
from typing import Any
from .types import PolicyMetadata, RolloutBatch, RolloutTransition, Trajectory
from gymnasium import spaces
from .base_buffer import BaseBuffer
from pathlib import Path

_NP_TO_TORCH_DTYPE: dict[np.dtype[Any], torch.dtype] = {
    np.dtype(np.float64): torch.float32,
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float16): torch.float16,
    np.dtype(np.int64): torch.int64,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.uint8): torch.uint8,
    np.dtype(np.bool_): torch.bool,
}

def _to_torch_dtype(dtype: Any) -> torch.dtype:
    dtype = np.dtype(dtype)
    return _NP_TO_TORCH_DTYPE.get(dtype, torch.float32)

class TorchBuffer(BaseBuffer):
    """Store one current-policy rollout and yield shuffled mini-batches.

    This buffer supports vectorized, feedforward environments only. It stores
    no data across policy updates, so call :meth:`reset` after each update.
    ``log_probs`` and old policy parameters are optional: continuous PPO stores
    Gaussian statistics and discrete PPO stores categorical logits.
    PPO uses compute_returns_and_advantages(); PQN uses compute_returns()
    with policy data disabled.
    """

    def __init__(
        self,
        rollout_steps: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        num_envs: int = 1,
        *,
        store_log_probs: bool = True,
        store_metadata: bool = True,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__(observation_space, action_space, rollout_steps, num_envs)

        self.device = torch.device(device)
        self.store_log_probs = bool(store_log_probs)
        self.store_metadata = bool(store_metadata)
        self.observation_dtype = _to_torch_dtype(observation_space.dtype)
        self.action_dtype = _to_torch_dtype(action_space.dtype)
        logits_dim = action_space.n if isinstance(action_space, spaces.Discrete) else self.action_shape[0]

        self.observations = torch.empty(
            (rollout_steps, num_envs, *self.observation_shape),
            dtype=self.observation_dtype,
            device=self.device,
        )
        self.actions = torch.empty(
            (rollout_steps, num_envs, *self.action_shape),
            dtype=self.action_dtype,
            device=self.device,
        )
        self.actions_mean = torch.empty(
            (rollout_steps, num_envs, *self.action_shape),
            dtype=self.action_dtype,
            device=self.device,
        ) if self.store_metadata and isinstance(action_space, spaces.Box) else None
        self.actions_std = torch.empty(
            (rollout_steps, num_envs, *self.action_shape),
            dtype=self.action_dtype,
            device=self.device,
        ) if self.store_metadata and isinstance(action_space, spaces.Box) else None
        self.action_logits = torch.empty(
            (rollout_steps, num_envs, logits_dim), dtype=torch.float32, device=self.device
        ) if self.store_metadata and isinstance(action_space, spaces.Discrete) else None
        self.rewards = torch.empty((rollout_steps, num_envs), dtype=torch.float32, device=self.device)
        self.dones = torch.empty((rollout_steps, num_envs), dtype=torch.float32, device=self.device)
        self.values = torch.empty((rollout_steps + 1, num_envs), dtype=torch.float32, device=self.device)
        self.log_probs = (
            torch.empty((rollout_steps, num_envs), dtype=torch.float32, device=self.device)
            if self.store_log_probs else None
        )
        self.advantages = (
            torch.empty((rollout_steps, num_envs), dtype=torch.float32, device=self.device)
            if store_log_probs else None
        )
        self.returns = torch.empty((rollout_steps, num_envs), dtype=torch.float32, device=self.device)
        self.step = 0
        self.returns_ready = False

    @property
    def full(self) -> bool:
        return self.step == self.rollout_steps

    @property
    def num_samples(self) -> int:
        return self.step * self.num_envs


    def _tensor(self, value: torch.Tensor, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=self.device).reshape(shape)

    def add(self, transition: RolloutTransition) -> None:
        """Append one vectorized environment step to the rollout."""
        if self.full:
            raise OverflowError("Rollout buffer is full; call reset() before collecting another rollout.")

        index = self.step
        self.observations[index].copy_(
            self._tensor(transition.observations, self.observation_dtype, (self.num_envs, *self.observation_shape))
        )
        self.actions[index].copy_(
            self._tensor(transition.actions, self.action_dtype, (self.num_envs, *self.action_shape))
        )
        if self.store_metadata:
            if self.actions_mean is not None:
                if transition.metadata is None or transition.metadata.actions_mean is None or transition.metadata.actions_std is None:
                    raise ValueError("Gaussian policy metadata is required when store_metadata=True")
                self.actions_mean[index].copy_(
                    self._tensor(transition.metadata.actions_mean, self.action_dtype, (self.num_envs, *self.action_shape))
                )
                self.actions_std[index].copy_(
                    self._tensor(transition.metadata.actions_std, self.action_dtype, (self.num_envs, *self.action_shape))
                )
            if self.action_logits is not None:
                if transition.metadata is None or transition.metadata.action_logits is None:
                    raise ValueError("Categorical policy metadata is required when store_metadata=True")
                self.action_logits[index].copy_(
                    self._tensor(transition.metadata.action_logits, torch.float32, (self.num_envs, self.action_logits.shape[-1]))
                )
        terminated = self._tensor(transition.terminated, torch.bool, (self.num_envs,))
        truncated = self._tensor(transition.truncated, torch.bool, (self.num_envs,))
        self.rewards[index].copy_(self._tensor(transition.rewards, torch.float32, (self.num_envs,)))
        self.dones[index].copy_(torch.logical_or(terminated, truncated))
        self.values[index].copy_(self._tensor(transition.values, torch.float32, (self.num_envs,)))
        if self.store_log_probs:
            if transition.log_probs is None:
                raise ValueError("log_probs is required when store_log_probs=True")
            self.log_probs[index].copy_(self._tensor(transition.log_probs, torch.float32, (self.num_envs,)))
        self.step += 1
        self.returns_ready = False

    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,
        gamma: float,
        gae_lambda: float,
        reward_scale: float | None = None
    ) -> None:
        """Bootstrap the rollout and compute GAE-Lambda targets."""
        if not self.full:
            raise RuntimeError("A complete rollout is required before computing advantages.")
        if not self.store_log_probs:
            raise RuntimeError("store_log_probs=True is required for PPO return computation.")

        self.values[-1].copy_(self._tensor(last_values, torch.float32, (self.num_envs,)))
        rewards = self.rewards
        if reward_scale is not None:
            rewards = rewards * reward_scale
        advantage = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for step in reversed(range(self.rollout_steps)):
            mask = 1.0 - self.dones[step]
            delta = rewards[step] + gamma * self.values[step + 1] * mask - self.values[step]
            advantage = delta + gamma * gae_lambda * mask * advantage
            self.advantages[step].copy_(advantage)
        self.returns.copy_(self.advantages + self.values[:-1])
        self.returns_ready = True

    def compute_returns(
        self,
        last_values: torch.Tensor,
        gamma: float,
        q_lambda: float,
    ) -> None:
        """Compute PQN lambda returns from the current rollout.

        ``values`` stores the bootstrap Q value selected during collection,
        while ``last_values`` is the bootstrap value at the rollout boundary.
        This method writes PQN targets to ``returns`` and does not compute GAE.
        Set store_log_probs=False to keep advantages unallocated as None.
        """
        if not self.full:
            raise RuntimeError("A complete rollout is required before computing returns.")
        if not 0.0 <= q_lambda <= 1.0:
            raise ValueError("q_lambda must be in [0, 1]")

        self.values[-1].copy_(self._tensor(last_values, torch.float32, (self.num_envs,)))
        next_return = self.values[-1].clone()
        for step in reversed(range(self.rollout_steps)):
            mask = 1.0 - self.dones[step]
            next_return = self.rewards[step] + gamma * (
                q_lambda * next_return + (1.0 - q_lambda) * self.values[step + 1]
            ) * mask
            self.returns[step].copy_(next_return)
        if self.advantages is not None:
            self.advantages.zero_()
        self.returns_ready = True

    def normalize_advantages(self) -> None:
        if self.advantages is None:
            raise RuntimeError("store_log_probs=True is required for advantage normalization.")
        mean = self.advantages.mean()
        std = self.advantages.std() + 1e-8
        self.advantages.copy_((self.advantages - mean) / std)

    def can_sample(self) -> bool:
        return self.full and self.returns_ready

    def trajectory(self) -> Trajectory:
        """Return the currently collected rollout without flattening it."""
        return Trajectory(
            observations=self.observations[: self.step],
            actions=self.actions[: self.step],
            metadata=PolicyMetadata(
                actions_mean=self.actions_mean[: self.step] if self.actions_mean is not None else None,
                actions_std=self.actions_std[: self.step] if self.actions_std is not None else None,
                action_logits=self.action_logits[: self.step] if self.action_logits is not None else None,
            ),
            log_probs=self.log_probs[: self.step] if self.log_probs is not None else None,
            values=self.values[: self.step + 1],
            dones=self.dones[: self.step],
            rewards=self.rewards[: self.step],
        )

    def _flat_batch(self) -> RolloutBatch:
        if not self.full or not self.returns_ready:
            raise RuntimeError("Compute returns before requesting mini-batches.")
        return RolloutBatch(
            observations=self.observations.flatten(0, 1),
            actions=self.actions.flatten(0, 1),
            actions_mean=self.actions_mean.flatten(0, 1) if self.actions_mean is not None else None,
            actions_std=self.actions_std.flatten(0, 1) if self.actions_std is not None else None,
            action_logits=self.action_logits.flatten(0, 1) if self.action_logits is not None else None,
            values=self.values[:-1].reshape(-1, 1),
            advantages=self.advantages.reshape(-1, 1) if self.advantages is not None else None,
            returns=self.returns.reshape(-1, 1),
            old_log_probs=self.log_probs.reshape(-1, 1) if self.log_probs is not None else None,
        )

    def sample(
        self,
        num_mini_batches: int,
        num_epochs: int
    ) -> Generator[RolloutBatch, None, None]:
        """Yield independently shuffled, equal-sized current-rollout batches."""
        batch = self._flat_batch()
        total = self.rollout_steps * self.num_envs
        if num_mini_batches <= 0 or num_epochs <= 0:
            raise ValueError("num_mini_batches and num_epochs must both be positive")
        if total % num_mini_batches:
            raise ValueError("rollout_steps * num_envs must be divisible by num_mini_batches")

        mini_batch_size = total // num_mini_batches
        for _ in range(num_epochs):
            for indices in torch.randperm(total, device=self.device).reshape(num_mini_batches, mini_batch_size):
                yield RolloutBatch(*(field[indices] if field is not None else None for field in batch))

    def reset(self) -> None:
        """Reset the write cursor while keeping the allocated storage."""
        self.step = 0
        self.returns_ready = False


    def save(self, file_path: str | Path) -> None:
        traj = {
            "step" : self.step,
            "observations" : self.observations,
            "actions" : self.actions,
            "values" : self.values,
            "dones" : self.dones,
            "rewards" : self.rewards,
        }
        if self.actions_mean is not None:
            traj["actions_mean"] = self.actions_mean
            traj["actions_std"] = self.actions_std
        if self.action_logits is not None:
            traj["action_logits"] = self.action_logits
        if self.log_probs is not None:
            traj["log_probs"] = self.log_probs
        out_path = Path(file_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(traj, out_path)


    def load(self, file_path: str | Path) -> None:
        traj = torch.load(file_path)
        self.step = int(traj["step"])
        for key in ("observations", "actions", "values", "dones", "rewards"):
            getattr(self, key).copy_(traj[key])
        if self.actions_mean is not None:
            self.actions_mean.copy_(traj["actions_mean"])
            self.actions_std.copy_(traj["actions_std"])
        if self.action_logits is not None:
            self.action_logits.copy_(traj["action_logits"])
        if self.log_probs is not None:
            self.log_probs.copy_(traj["log_probs"])


    def __len__(self) -> int:
        return self.num_samples
