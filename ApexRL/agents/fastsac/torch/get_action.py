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
    return actor.model.get_mean_action(_select_actor(observations, asymmetric_obs, actor_obs_dim))


@torch.no_grad()
@torch.compile(fullgraph=True, mode="max-autotune")
def get_exploration_action(
    actor: Network[Actor],
    observation_rms: Network[RMS] | None,
    observations: torch.Tensor,
    asymmetric_obs: bool,
    actor_obs_dim: int,
) -> torch.Tensor:
    if observation_rms is not None:
        observations = observation_rms.model(observations.float())
    return actor.model.get_action(_select_actor(observations, asymmetric_obs, actor_obs_dim))[0]


__all__ = ["get_eval_action", "get_exploration_action"]
