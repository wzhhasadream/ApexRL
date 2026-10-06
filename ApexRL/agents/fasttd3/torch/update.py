from __future__ import annotations

from collections.abc import Callable

import torch

from ....buffers.off_policy import Batch, SequenceBatch, compress_n_step
from ....model.torch import Network, RMS
from ..config import FastTD3Config
from .network import Actor, Critic


def _select_actor(observations: torch.Tensor, asymmetric_obs: bool, actor_obs_dim: int) -> torch.Tensor:
    return observations[..., :actor_obs_dim] if asymmetric_obs else observations


@torch.no_grad()
@torch.compile(fullgraph=True, mode="max-autotune")
def update_rms(rms: Network[RMS], observations: torch.Tensor) -> None:
    rms.model.update(observations)


def update_critic(
    critic: Network[Critic], target_critic: Network[Critic], actor: Network[Actor],
    batch: Batch, cfg: FastTD3Config,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        next_observations = _select_actor(batch.next_observations, cfg.asymmetric_obs, actor.model.obs_dim)
        noise = torch.randn_like(batch.actions) * cfg.policy_noise
        noise = noise.clamp(-cfg.noise_clip, cfg.noise_clip)
        next_actions = (actor.model(next_observations) + noise).clamp(
            actor.model.policy.action_low, actor.model.policy.action_high
        )
        target_logits = target_critic.model(batch.next_observations, next_actions)
        target_values = batch.rewards + batch.discounts * (1.0 - batch.dones) * critic.model.dist.bins.to(target_logits)
        if cfg.use_cdq:
            target_logits = critic.model.dist.select_min_logits(target_logits)
        target_probs = critic.model.dist.target_probs(target_logits, target_values)

    logits = critic.model(batch.observations, batch.actions)
    target_axes = None if cfg.use_cdq else 0
    losses = torch.vmap(critic.model.dist._loss_one, in_dims=(0, target_axes))(logits, target_probs)
    loss = losses.mean()
    mean_q = critic.model.dist.q_values(logits).mean().detach().clone()
    critic.opt.zero_grad(set_to_none=True)
    loss.backward()
    critic.grad_step(cfg.max_grad_norm)
    return {"critic/loss": loss.detach().clone(), "critic/mean_q": mean_q}


def update_actor(
    actor: Network[Actor], critic: Network[Critic], batch: Batch, cfg: FastTD3Config,
) -> dict[str, torch.Tensor]:
    actor_observations = _select_actor(batch.observations, cfg.asymmetric_obs, actor.model.obs_dim)
    critic.model.requires_grad_(False)
    try:
        actions = actor.model(actor_observations)
        values = critic.model.q_values(batch.observations, actions)
        value = values.amin(dim=0) if cfg.use_cdq else values.mean(dim=0)
        loss = -value.mean()
        info = {"actor/loss": loss.detach().clone(), "actor/mean_q": value.mean().detach().clone()}
        actor.opt.zero_grad(set_to_none=True)
        loss.backward()
        actor.grad_step(cfg.max_grad_norm)
    finally:
        critic.model.requires_grad_(True)
    return info


def make_update(cfg: FastTD3Config) -> Callable[..., dict[str, torch.Tensor]]:
    @torch.compile(mode="max-autotune")
    def update(
        critic: Network[Critic], target_critic: Network[Critic], actor: Network[Actor],
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
        critic_info = update_critic(critic, target_critic, actor, batch, cfg)
        actor_info = update_actor(actor, critic, batch, cfg) if do_actor else {}
        return {**critic_info, **actor_info}

    return update


__all__ = ["make_update", "update_actor", "update_critic", "update_rms"]
