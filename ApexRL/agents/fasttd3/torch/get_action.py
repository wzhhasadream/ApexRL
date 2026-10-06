from __future__ import annotations

import torch

from ....model.torch import Network, RMS
from .network import Actor


def _select_actor(observations: torch.Tensor, asymmetric_obs: bool, actor_obs_dim: int) -> torch.Tensor:
    return observations[..., :actor_obs_dim] if asymmetric_obs else observations


@torch.no_grad()
@torch.compile(fullgraph=True, mode="max-autotune")
def get_eval_action(
    actor: Network[Actor],
    observation_rms: Network[RMS] | None,
    observations: torch.Tensor,
    asymmetric_obs: bool,
    actor_obs_dim: int,
) -> torch.Tensor:
    if observation_rms is not None:
        observations = observation_rms.model(observations.float())
    return actor.model(_select_actor(observations, asymmetric_obs, actor_obs_dim))


@torch.no_grad()
@torch.compile(fullgraph=True, mode="max-autotune")
def get_exploration_action(
    actor: Network[Actor],
    observation_rms: Network[RMS] | None,
    observations: torch.Tensor,
    asymmetric_obs: bool,
    actor_obs_dim: int,
    std_min: float,
    std_max: float,
) -> torch.Tensor:
    if observation_rms is not None:
        observations = observation_rms.model(observations.float())
    actions = actor.model(_select_actor(observations, asymmetric_obs, actor_obs_dim))
    scale = torch.rand((actions.shape[0], 1), device=actions.device) * (std_max - std_min) + std_min
    actions = actions + torch.randn_like(actions) * scale
    return actions.clamp(actor.model.policy.action_low, actor.model.policy.action_high)


__all__ = ["get_eval_action", "get_exploration_action"]
