# Adapted from mttga/purejaxql PQN (Apache-2.0), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/mttga/purejaxql/blob/47af6d7b35c89ddfe633aaf7341bdb8964cb7cce/purejaxql/pqn_atari.py
from __future__ import annotations

import torch

from ....model.torch import Network
from .network import Critic

# Acting uses training=False: BatchNorm reads running statistics and rollout
# batches never update them (matches the JAX version).


@torch.no_grad()
@torch.compile(fullgraph=True, mode="max-autotune")
def get_eval_action(critic: Network[Critic], observations: torch.Tensor) -> torch.Tensor:
    return critic.model.select_action(observations)


@torch.no_grad()
@torch.compile(fullgraph=True, mode="max-autotune")
def get_value(critic: Network[Critic], observations: torch.Tensor) -> torch.Tensor:
    return critic.model(observations).amax(dim=-1, keepdim=True)


@torch.no_grad()
@torch.compile(fullgraph=True, mode="max-autotune")
def sample_action_and_value(
    critic: Network[Critic], observations: torch.Tensor, epsilon: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Epsilon-greedy action plus its bootstrap value max_a Q(s, a)."""
    q_values = critic.model(observations)
    greedy_actions = critic.model.policy.action(q_values)
    random_actions = torch.randint_like(greedy_actions, critic.model.action_dim)
    explore = torch.rand(greedy_actions.shape, device=greedy_actions.device) < epsilon
    actions = torch.where(explore, random_actions, greedy_actions)
    return actions, q_values.amax(dim=-1, keepdim=True)


__all__ = ["get_eval_action", "get_value", "sample_action_and_value"]
