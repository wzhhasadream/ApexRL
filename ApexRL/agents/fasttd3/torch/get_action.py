# Adapted from younggyoseo/FastTD3 (MIT), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/younggyoseo/FastTD3/tree/229ed59bbf43ea2f7a2d5d90d1076314839944d7
from __future__ import annotations

import torch

from ....model.torch import Network, RMS
from .network import Actor


@torch.no_grad()
@torch.compile(mode="max-autotune", fullgraph=True)
def get_eval_action(actor: Network[Actor], observation_rms: Network[RMS] | None, observations: torch.Tensor) -> torch.Tensor:
    if observation_rms is not None:
        observations = observation_rms.model(observations)
    return actor.model(observations[..., :actor.model.obs_dim])


@torch.no_grad()
@torch.compile(mode="max-autotune", fullgraph=True)
def get_exploration_action(actor: Network[Actor], observation_rms: Network[RMS] | None, observations: torch.Tensor, noise_scales: torch.Tensor) -> torch.Tensor:
    # noise_scales: [num_envs, 1], fixed per episode as in FastTD3
    if observation_rms is not None:
        observations = observation_rms.model(observations)
    actions = actor.model(observations[..., :actor.model.obs_dim])
    actions = actions + torch.randn_like(actions) * noise_scales
    return actions.clamp(actor.model.policy.action_low, actor.model.policy.action_high)


__all__ = ["get_eval_action", "get_exploration_action"]
