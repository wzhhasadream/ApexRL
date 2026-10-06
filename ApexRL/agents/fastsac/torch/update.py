from __future__ import annotations

from collections.abc import Callable

import torch

from ....buffers.off_policy import Batch, SequenceBatch, compress_n_step
from ....model.torch import Alpha, Network, RMS
from ..config import FastSACConfig
from .network import Actor, Critic


def _select_actor(observations: torch.Tensor, asymmetric_obs: bool, actor_obs_dim: int) -> torch.Tensor:
    return observations[..., :actor_obs_dim] if asymmetric_obs else observations


@torch.no_grad()
@torch.compile(fullgraph=True, mode="max-autotune")
def update_rms(rms: Network[RMS], observations: torch.Tensor) -> None:
    rms.model.update(observations)


def update_critic(
    critic: Network[Critic], target_critic: Network[Critic], actor: Network[Actor], alpha: Network[Alpha],
    batch: Batch, cfg: FastSACConfig,
) -> dict[str, torch.Tensor]:
    alpha_value = alpha.model().detach()
    with torch.no_grad():
        next_observations = _select_actor(batch.next_observations, cfg.asymmetric_obs, actor.model.obs_dim)
        next_actions, next_log_probs = actor.model.get_action(next_observations)
        target_logits = target_critic.model(batch.next_observations, next_actions)
        target_values = batch.rewards + batch.discounts * (1.0 - batch.dones) * (
            critic.model.dist.bins.to(target_logits) - alpha_value * next_log_probs
        )
        target_probs = torch.vmap(critic.model.dist.target_probs, in_dims=(0, None))(target_logits, target_values)

    logits = critic.model(batch.observations, batch.actions)
    losses = torch.vmap(critic.model.dist._loss_one, in_dims=(0, 0))(logits, target_probs)
    loss = losses.mean()
    mean_q = critic.model.dist.q_values(logits).mean().detach().clone()
    critic.opt.zero_grad(set_to_none=True)
    loss.backward()
    critic.grad_step(cfg.max_grad_norm)
    return {"critic/loss": loss.detach().clone(), "critic/mean_q": mean_q}


def update_alpha(alpha: Network[Alpha], entropy: torch.Tensor, target_entropy: float) -> dict[str, torch.Tensor]:
    value = alpha.model()
    loss = value * (entropy.detach() - target_entropy)
    info = {"alpha/loss": loss.detach().clone(), "alpha/value": value.detach().clone()}
    alpha.opt.zero_grad(set_to_none=True)
    loss.backward()
    alpha.grad_step()
    return info


def update_actor(
    actor: Network[Actor], critic: Network[Critic], alpha: Network[Alpha], batch: Batch, cfg: FastSACConfig,
) -> dict[str, torch.Tensor]:
    alpha_value = alpha.model().detach()
    actor_observations = _select_actor(batch.observations, cfg.asymmetric_obs, actor.model.obs_dim)
    critic.model.requires_grad_(False)
    try:
        actions, log_probs = actor.model.get_action(actor_observations)
        q_value = critic.model.q_values(batch.observations, actions).mean(dim=0)
        loss = (alpha_value * log_probs - q_value).mean()
        info = {"actor/loss": loss.detach().clone(), "actor/entropy": -log_probs.mean().detach().clone()}
        actor.opt.zero_grad(set_to_none=True)
        loss.backward()
        actor.grad_step(cfg.max_grad_norm)
    finally:
        critic.model.requires_grad_(True)
    return info


def make_update(cfg: FastSACConfig) -> Callable[..., dict[str, torch.Tensor]]:
    @torch.compile(mode="max-autotune")
    def update(
        critic: Network[Critic], target_critic: Network[Critic], actor: Network[Actor], alpha: Network[Alpha],
        observation_rms: Network[RMS] | None, sequence: SequenceBatch, do_actor: bool,
    ) -> dict[str, torch.Tensor]:
        torch.compiler.cudagraph_mark_step_begin()
        if observation_rms is not None:
            sequence = SequenceBatch(
                observation_rms.model(sequence.observations.float()), sequence.actions, sequence.rewards,
                sequence.terminations, sequence.truncations,
                observation_rms.model(sequence.next_observations.float()),
            )
        batch = compress_n_step(sequence, cfg.gamma)
        critic_info = update_critic(critic, target_critic, actor, alpha, batch, cfg)
        actor_observations = _select_actor(batch.observations, cfg.asymmetric_obs, actor.model.obs_dim)
        _, log_probs = actor.model.get_action(actor_observations)
        alpha_info = update_alpha(alpha, -log_probs.mean(), cfg.target_entropy) if cfg.use_autotune else {"alpha/value": alpha.model().detach().clone()}
        actor_info = update_actor(actor, critic, alpha, batch, cfg) if do_actor else {}
        return {**critic_info, **alpha_info, **actor_info}

    return update


__all__ = ["make_update", "update_actor", "update_alpha", "update_critic", "update_rms"]
