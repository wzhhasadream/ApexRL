# Adapted from vwxyzjn/cleanrl (MIT) and leggedrobotics/rsl_rl (BSD-3-Clause), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool.py
# https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/ppo.py
import torch

from ....buffers.on_policy.torch_buffer import TorchBuffer
from ....buffers.on_policy.types import RolloutBatch
from ....common import select_actor_observations
from ....model.torch import Network
from ..config import PPOConfig
from .network_atari import ActorCritic as AtariActorCritic
from .network_state import ActorCritic
from ....common.torch import adapt_lr, categorical_kl, diagonal_gaussian_kl
from .get_action import get_value
from torch.utils._pytree import tree_map

def ppo_loss(agent: Network[ActorCritic | AtariActorCritic], batch: RolloutBatch, cfg: PPOConfig) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    actor = agent.model.actor
    actor_obs = select_actor_observations(batch.observations, cfg.asymmetric_obs, actor.obs_dim)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.compute_type == "bfloat16"):
        _, new_log_probs, entropy, metadata, values = agent.model.get_action_and_value(
            actor_obs, batch.observations, update_rms=False, actions=batch.actions,
        )
    ratio = torch.exp(new_log_probs - batch.old_log_probs)

    if cfg.algo == "ppo":
        pg_loss = -torch.minimum(ratio * batch.advantages, ratio.clamp(1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * batch.advantages).mean()
    else:  # spo: Simple Policy Optimization, Xie et al., ICML 2025 (arXiv:2401.16025)
        pg_loss = -(batch.advantages * ratio - batch.advantages.abs() * (ratio - 1.0).square() / (2.0 * cfg.clip_coef)).mean()

    if cfg.clip_value:
        clipped_values = batch.values + (values - batch.values).clamp(-cfg.clip_coef, cfg.clip_coef)
        value_loss = 0.5 * torch.maximum((values - batch.returns).square(), (clipped_values - batch.returns).square()).mean()
    else:
        value_loss = 0.5 * (values - batch.returns).square().mean()

    entropy_mean = entropy.mean()
    loss = pg_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_mean
    if actor.discrete:
        kl = categorical_kl(metadata.action_logits, batch.action_logits).mean()
    else:
        kl = diagonal_gaussian_kl(metadata.actions_mean, metadata.actions_std, batch.actions_mean, batch.actions_std).mean()
    info = {
        "training/loss": loss.detach(),
        "training/pg_loss": pg_loss.detach(),
        "training/value_loss": value_loss.detach(),
        "training/entropy": entropy_mean.detach(),
        "training/kl": kl.detach(),
        "training/clipfrac": (ratio.sub(1.0).abs() > cfg.clip_coef).float().mean().detach(),
    }
    return loss, info

@torch.compile(mode="max-autotune")
def update_ppo_minibatch(agent: Network[ActorCritic | AtariActorCritic], batch: RolloutBatch, cfg: PPOConfig) -> dict[str, torch.Tensor]:
    loss, info = ppo_loss(agent, batch, cfg)
    if cfg.desired_kl is not None:
        for group in agent.opt.param_groups:
            adapt_lr(group["lr"], info["training/kl"], cfg.desired_kl)
    agent.opt.zero_grad(set_to_none=True)
    loss.backward()
    agent.grad_step(cfg.max_grad_norm)
    return info


def update_ppo(agent: Network[ActorCritic | AtariActorCritic], buffer: TorchBuffer, last_observations: torch.Tensor, cfg: PPOConfig) -> dict[str, torch.Tensor]:
    last_values = get_value(agent, last_observations).clone()
    buffer.compute_returns_and_advantages(last_values, cfg.gamma, cfg.gae_lambda)
    if cfg.normalize_advantages:
        buffer.normalize_advantages()
    infos = [tree_map(torch.clone, update_ppo_minibatch(agent, batch, cfg)) for batch in buffer.sample(cfg.num_mini_batches, cfg.num_epochs)]
    # Freeze the rollout's observation statistics for the next rollout
    agent.model.sync_rms()
    return tree_map(lambda *values: torch.stack(values).mean(), *infos)


__all__ = ["ppo_loss", "update_ppo", "update_ppo_minibatch"]
