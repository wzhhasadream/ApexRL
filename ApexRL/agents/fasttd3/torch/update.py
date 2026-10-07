# Adapted from younggyoseo/FastTD3 (MIT), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/younggyoseo/FastTD3/tree/229ed59bbf43ea2f7a2d5d90d1076314839944d7
from __future__ import annotations

from collections.abc import Callable

import torch

from ....buffers.off_policy import Batch, SequenceBatch, compress_n_step
from ....model.torch import Network, RMS
from ..config import FastTD3Config
from .network import Actor, Critic


@torch.no_grad()
@torch.compile(mode="max-autotune", fullgraph=True)
def update_rms(rms: Network[RMS], observations: torch.Tensor) -> None:
    rms.model.update(observations)


def update_critic(critic: Network[Critic], target_critic: Network[Critic], actor: Network[Actor], batch: Batch, cfg: FastTD3Config) -> dict[str, torch.Tensor]:
    amp = cfg.compute_type == "bfloat16"
    bins = critic.model.dist.bins
    # Target policy smoothing + C51 projection of both target heads
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        noise = (torch.randn_like(batch.actions) * cfg.policy_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
        next_actions = (actor.model(batch.next_observations[..., :actor.model.obs_dim]) + noise).clamp(actor.model.policy.action_low, actor.model.policy.action_high)
        target_logits = target_critic.model(batch.next_observations, next_actions)              # [2, B, A]
        target_values = batch.rewards + batch.discounts * (1.0 - batch.dones) * bins            # [B, A]
        target_probs = torch.vmap(critic.model.dist.target_probs, in_dims=(0, None))(target_logits, target_values)
        if cfg.use_cdq:
            # Upstream: project both heads, then keep the distribution with the smaller mean
            projected_q = (target_probs * bins).sum(-1, keepdim=True)                          # [2, B, 1]
            target_probs = torch.where(projected_q[0] < projected_q[1], target_probs[0], target_probs[1]).expand(2, -1, -1)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        logits = critic.model(batch.observations, batch.actions)
    # Sum over the two heads (qf1_loss + qf2_loss), mean over batch
    loss = -(target_probs * logits.log_softmax(-1)).sum(-1).mean(-1).sum()
    critic.opt.zero_grad(set_to_none=True)
    loss.backward()
    critic.grad_step(cfg.max_grad_norm)
    return {"critic/loss": loss.detach(), "critic/mean_q": critic.model.dist.q_values(logits.detach()).mean()}


def update_actor(actor: Network[Actor], critic: Network[Critic], batch: Batch, cfg: FastTD3Config) -> dict[str, torch.Tensor]:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.compute_type == "bfloat16"):
        q = critic.model.q_values(batch.observations, actor.model(batch.observations[..., :actor.model.obs_dim]))   # [2, B, 1]
    q = q.amin(0) if cfg.use_cdq else q.mean(0)
    loss = -q.mean()
    actor.opt.zero_grad(set_to_none=True)
    loss.backward(inputs=list(actor.model.parameters()))    # skip critic grads
    actor.grad_step(cfg.max_grad_norm)
    return {"actor/loss": loss.detach(), "actor/mean_q": q.detach().mean()}


def make_update(cfg: FastTD3Config) -> Callable[..., dict[str, torch.Tensor]]:
    @torch.compile(mode="max-autotune")
    def update(critic: Network[Critic], target_critic: Network[Critic], actor: Network[Actor], observation_rms: Network[RMS] | None, sequence: SequenceBatch, do_actor: bool) -> dict[str, torch.Tensor]:
        # n-step compress first, then normalize only the two observation tensors that are used
        batch = compress_n_step(sequence, cfg.gamma)
        obs, next_obs = batch.observations.float(), batch.next_observations.float()
        if observation_rms is not None:
            obs, next_obs = observation_rms.model(obs), observation_rms.model(next_obs)
        # Build Batch explicitly (not ._replace) so dynamo can reconstruct it across the backward() graph break
        batch = Batch(obs, batch.actions, batch.rewards, batch.dones, next_obs, batch.discounts)
        critic_info = update_critic(critic, target_critic, actor, batch, cfg)
        actor_info = update_actor(actor, critic, batch, cfg) if do_actor else {}
        return {**critic_info, **actor_info}

    return update


__all__ = ["make_update", "update_actor", "update_critic", "update_rms"]
