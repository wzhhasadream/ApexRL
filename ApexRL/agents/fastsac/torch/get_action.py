from __future__ import annotations

import torch

from ....model.torch import Network, RMS
from .network import Actor


@torch.no_grad()
@torch.compile
def get_eval_action(actor: Network[Actor], observation_rms: Network[RMS] | None, observations: torch.Tensor) -> torch.Tensor:
    if observation_rms is not None:
        observations = observation_rms.model(observations)
    return actor.model.get_mean_action(observations[..., :actor.model.obs_dim])


@torch.no_grad()
@torch.compile
def get_exploration_action(actor: Network[Actor], observation_rms: Network[RMS] | None, observations: torch.Tensor) -> torch.Tensor:
    if observation_rms is not None:
        observations = observation_rms.model(observations)
    return actor.model.get_action(observations[..., :actor.model.obs_dim])[0]


__all__ = ["get_eval_action", "get_exploration_action"]
