# Adapted from mttga/purejaxql PQN (Apache-2.0), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/mttga/purejaxql/blob/47af6d7b35c89ddfe633aaf7341bdb8964cb7cce/purejaxql/pqn_atari.py
from __future__ import annotations

import torch

from ....model.torch import Network
from ..config import PQNConfig
from .network import Critic


@torch.compile(mode="max-autotune")
def update_critic(
    critic: Network[Critic], observations: torch.Tensor, actions: torch.Tensor,
    targets: torch.Tensor, cfg: PQNConfig,
) -> dict[str, torch.Tensor]:
    """Apply one Q regression step to one rollout mini-batch."""
    device_type = observations.device.type
    with torch.autocast(device_type, dtype=torch.bfloat16, enabled=cfg.compute_type == "bfloat16"):
        q_values = critic.model(observations, training=True)
    selected_q = q_values.gather(1, actions.long().reshape(-1, 1)).squeeze(1)
    td_error = selected_q - targets.reshape(-1)
    loss = 0.5 * td_error.square().mean()

    critic.opt.zero_grad(set_to_none=True)
    loss.backward()
    critic.grad_step(cfg.max_grad_norm)
    return {
        "training/q_loss": loss.detach(),
        "training/q_values": selected_q.detach().mean(),
        "training/targets": targets.mean(),
        "training/td_abs": td_error.detach().abs().mean(),
    }


__all__ = ["update_critic"]
