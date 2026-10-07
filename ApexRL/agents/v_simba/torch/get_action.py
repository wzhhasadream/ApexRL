# Adapted from DAVIAN-Robotics/V-Simba (Apache-2.0), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/DAVIAN-Robotics/V-Simba/tree/be811e968bc02589fbb32f3be79f9a7d9a8fa86d/scale_rl/agents/vsimba
import torch

from ....model.torch import Network


@torch.no_grad()
@torch.compile(fullgraph=True, mode="max-autotune")
def get_eval_action(
    encoder: Network,
    actor: Network,
    observations: torch.Tensor,
) -> torch.Tensor:
    return actor.model.get_mean_action(encoder.model(observations))


@torch.no_grad()
@torch.compile(fullgraph=True, mode="max-autotune")
def get_exploration_action(
    encoder: Network,
    actor: Network,
    observations: torch.Tensor,
) -> torch.Tensor:
    actions, _ = actor.model.get_action(encoder.model(observations))
    return actions


__all__ = ["get_eval_action", "get_exploration_action"]
