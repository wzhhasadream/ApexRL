"""Shared data containers for feedforward on-policy rollouts."""

from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import jax
    import torch

    Tensor: TypeAlias = jax.Array | torch.Tensor


class RolloutTransition(NamedTuple):
    """One vectorized environment step collected by the current policy.

    Policy metadata contains the old distribution parameters retained by PPO.
    PQN leaves it unset because it only needs values, rewards, and actions.
    """

    observations: Tensor  # [num_envs, *observation_shape]
    actions: Tensor  # [num_envs, *action_shape]
    rewards: Tensor  # [num_envs]
    terminated: Tensor  # [num_envs], true MDP terminations
    truncated: Tensor  # [num_envs], time-limit truncations
    values: Tensor  # [num_envs] or [num_envs, 1], PPO V(s_t) or PQN \max_a Q(s_t, a)
    log_probs: Tensor | None = None  # [num_envs] or [num_envs, 1], log pi(a_t | s_t)
    metadata: "PolicyMetadata | None" = None


class PolicyMetadata(NamedTuple):
    """Distribution parameters of the old policy used during rollout.

    Discrete policies store categorical logits; continuous policies store the
    Gaussian mean and standard deviation. PPO uses these values to estimate
    policy KL divergence and to adapt the learning rate when configured.
    """

    actions_mean: Tensor | None = None  # Old Gaussian policy mean.
    actions_std: Tensor | None = None  # Old Gaussian policy standard deviation.
    action_logits: Tensor | None = None  # Old categorical policy logits.


class Trajectory(NamedTuple):
    """A complete rollout stored with time as the leading dimension."""

    observations: Tensor  # [num_steps, num_envs, *observation_shape]
    actions: Tensor  # [num_steps, num_envs, *action_shape]
    metadata: PolicyMetadata | None
    log_probs: Tensor | None  # [num_steps, num_envs]
    values: Tensor  # [num_steps + 1, num_envs]
    dones: Tensor  # [num_steps, num_envs]
    rewards: Tensor  # [num_steps, num_envs]


class RolloutBatch(NamedTuple):
    """A shuffled current-rollout batch, optionally stacked over update steps.

    PPO consumes returns and advantages; PQN only regresses onto returns.
    Disabled storage fields, including PQN advantages, remain None in batches.
    """

    observations: Tensor  # [batch_size, *observation_shape] or [num_batches, batch_size, *observation_shape]
    actions: Tensor  # [batch_size, *action_shape] or [num_batches, batch_size, *action_shape]
    actions_mean: Tensor | None  # [batch_size, *action_shape] or [num_batches, batch_size, *action_shape]
    actions_std: Tensor | None  # [batch_size, *action_shape] or [num_batches, batch_size, *action_shape]
    action_logits: Tensor | None  # [batch_size, action_dim] or [num_batches, batch_size, action_dim]
    values: Tensor  # [batch_size, 1] or [num_batches, batch_size, 1]
    advantages: Tensor | None  # [batch_size, 1] or [num_batches, batch_size, 1]
    returns: Tensor  # [batch_size, 1] or [num_batches, batch_size, 1]
    old_log_probs: Tensor | None  # [batch_size, 1] or [num_batches, batch_size, 1]

__all__ = ["PolicyMetadata", "RolloutBatch", "RolloutTransition", "Trajectory"]
