# Adapted from amazon-far/holosoma FastSAC (Apache-2.0), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/amazon-far/holosoma/tree/d18d6cc50f872c15e904a22ceac22313cec955c8/src/holosoma/holosoma/agents/fast_sac
from __future__ import annotations

from collections.abc import Callable

import torch

from ....buffers.off_policy import Batch, SequenceBatch, compress_n_step
from ....model.torch import Alpha, Network, RMS
from ..config import FastSACConfig
from .network import Actor, Critic


@torch.no_grad()
@torch.compile(mode="max-autotune", fullgraph=True)
def update_rms(rms: Network[RMS], observations: torch.Tensor) -> None:
    rms.model.update(observations)


def update_critic(critic: Network[Critic], target_critic: Network[Critic], actor: Network[Actor], alpha: Network[Alpha], batch: Batch, cfg: FastSACConfig) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    amp = cfg.compute_type == "bfloat16"
    alpha_value = alpha.model().detach()
    # Soft C51 target: each Q head is projected against its own target head (no CDQ)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        next_actions, next_log_probs = actor.model.get_action(batch.next_observations[..., :actor.model.obs_dim])
        target_logits = target_critic.model(batch.next_observations, next_actions)                 # [E, B, A]
        target_values = batch.rewards + batch.discounts * (1.0 - batch.dones) * (critic.model.dist.bins - alpha_value * next_log_probs)
        target_probs = torch.vmap(critic.model.dist.target_probs, in_dims=(0, None))(target_logits, target_values)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        logits = critic.model(batch.observations, batch.actions)
    # Holosoma sums over Q heads after averaging each head over the batch.
    loss = -(target_probs * logits.log_softmax(-1)).sum(-1).mean(-1).sum()
    critic.opt.zero_grad(set_to_none=True)
    loss.backward()
    critic.grad_step(cfg.max_grad_norm)
    # Holosoma reuses next_log_probs in update_alpha.
    return {"critic/loss": loss.detach(), "critic/mean_q": critic.model.dist.q_values(logits.detach()).mean()}, next_log_probs


def update_alpha(alpha: Network[Alpha], log_probs: torch.Tensor, target_entropy: float) -> dict[str, torch.Tensor]:
    value = alpha.model()
    loss = (-value * (log_probs.detach() + target_entropy)).mean()
    alpha.opt.zero_grad(set_to_none=True)
    loss.backward()
    alpha.grad_step()
    return {"alpha/loss": loss.detach(), "alpha/value": value.detach()}


def update_actor(actor: Network[Actor], critic: Network[Critic], alpha: Network[Alpha], batch: Batch, cfg: FastSACConfig) -> dict[str, torch.Tensor]:
    alpha_value = alpha.model().detach()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.compute_type == "bfloat16"):
        actions, log_probs = actor.model.get_action(batch.observations[..., :actor.model.obs_dim])
        q = critic.model.q_values(batch.observations, actions).mean(0)                              # [B, 1]
    loss = (alpha_value * log_probs - q).mean()
    actor.opt.zero_grad(set_to_none=True)
    loss.backward(inputs=list(actor.model.parameters()))    # skip critic grads
    actor.grad_step(cfg.max_grad_norm)
    return {"actor/loss": loss.detach(), "actor/entropy": -log_probs.detach().mean()}


def make_update(cfg: FastSACConfig) -> Callable[..., dict[str, torch.Tensor]]:
    @torch.compile(mode="max-autotune")
    def update(critic: Network[Critic], target_critic: Network[Critic], actor: Network[Actor], alpha: Network[Alpha], observation_rms: Network[RMS] | None, sequence: SequenceBatch, do_actor: bool) -> dict[str, torch.Tensor]:
        # n-step compress first, then normalize only the two observation tensors that are used
        batch = compress_n_step(sequence, cfg.gamma)
        obs, next_obs = batch.observations.float(), batch.next_observations.float()
        if observation_rms is not None:
            obs, next_obs = observation_rms.model(obs), observation_rms.model(next_obs)
        # Build Batch explicitly (not ._replace) so dynamo can reconstruct it across the backward() graph break
        batch = Batch(obs, batch.actions, batch.rewards, batch.dones, next_obs, batch.discounts)
        critic_info, next_log_probs = update_critic(critic, target_critic, actor, alpha, batch, cfg)
        alpha_info = update_alpha(alpha, next_log_probs, cfg.target_entropy) if cfg.use_autotune else {"alpha/value": alpha.model().detach()}
        actor_info = update_actor(actor, critic, alpha, batch, cfg) if do_actor else {}
        return {**critic_info, **alpha_info, **actor_info}

    return update


__all__ = ["make_update", "update_actor", "update_alpha", "update_critic", "update_rms"]
