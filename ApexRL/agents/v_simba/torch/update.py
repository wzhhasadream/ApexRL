import torch
import torch.nn.functional as F

from ....buffers import compress_n_step
from ....buffers.off_policy.types import Batch, SequenceBatch
from ....common.torch import augment_sequencebatch
from ....model.torch import Alpha, Network, update_alpha
from ....model.torch.backbones import VSimbaVisionEncoder
from ..config import VSimbaConfig
from .network import Actor, Critic


def _autocast_config(
    encoder: Network[VSimbaVisionEncoder],
    cfg: VSimbaConfig,
) -> tuple[str, torch.dtype, bool]:
    device_type = next(encoder.model.parameters()).device.type
    enabled = cfg.compute_type == "bfloat16"
    dtype = torch.bfloat16 if enabled else torch.float32
    return device_type, dtype, enabled


def update_actor(
    encoder: Network[VSimbaVisionEncoder],
    actor: Network[Actor],
    critic: Network[Critic],
    alpha: Network[Alpha],
    batch: Batch,
    cfg: VSimbaConfig,
) -> dict[str, torch.Tensor]:
    """Update the actor while keeping critic parameters out of the gradient."""
    device_type, amp_dtype, amp_enabled = _autocast_config(encoder, cfg)
    critic.model.requires_grad_(False)

    try:
        with torch.no_grad():
            vision_z = encoder.model(batch.observations)
            alpha_value = alpha.model()

        with torch.autocast(
            device_type=device_type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            actions, log_probs = actor.model.get_action(vision_z)
            q_values = critic.model.q_values(vision_z, actions).amin(dim=0)
            loss = (alpha_value * log_probs - q_values).mean()
            info = {
                "actor/loss": loss.detach(),
                "actor/entropy": -log_probs.detach().mean(),
            }

        actor.opt.zero_grad(set_to_none=True)
        loss.backward()
        actor.grad_step()
    finally:
        critic.model.requires_grad_(True)

    return info


def update_critic(
    encoder: Network[VSimbaVisionEncoder],
    actor: Network[Actor],
    critic: Network[Critic],
    target_critic: Network[Critic],
    alpha: Network[Alpha],
    batch: Batch,
    sequencebatch: SequenceBatch,
    cfg: VSimbaConfig,
) -> dict[str, torch.Tensor]:
    """Update the critic and encoder with TD, NCE and reward losses."""
    device_type, amp_dtype, amp_enabled = _autocast_config(encoder, cfg)
    alpha_value = alpha.model().detach()

    with torch.autocast(
        device_type=device_type,
        dtype=amp_dtype,
        enabled=amp_enabled,
    ):
        vision_z = encoder.model(batch.observations)
        next_vision_z = encoder.model(batch.next_observations)
        real_next_vision_z = encoder.model(sequencebatch.next_observations[0])
        with torch.no_grad():
            next_actions, next_log_probs = actor.model.get_action(next_vision_z.detach())
            target_logits = target_critic.model(
                next_vision_z.detach(), next_actions
            )[-1]
            target_logits = critic.model.dist.select_min_logits(target_logits)
            target_values = batch.rewards + batch.discounts * (1.0 - batch.dones) * (
                critic.model.dist.bins - alpha_value * next_log_probs
            )
            target_probs = critic.model.dist.target_probs(target_logits, target_values)

        phi_sa, pred_rewards, logits = critic.model(vision_z, batch.actions)
        td_loss = critic.model.dist.loss(logits, target_probs).sum()
        g_s = critic.model.embed_s(real_next_vision_z)
        reward_loss = (sequencebatch.rewards[0] - pred_rewards).square().mean()

        phi_sa = F.normalize(phi_sa.float(), dim=-1, eps=1e-6)
        g_s = F.normalize(g_s.float(), dim=-1, eps=1e-6)
        nce_logits = torch.einsum("qbi,qdi->qbd", phi_sa, g_s) * critic.model.get_alpha()[:, None, None]
        num_heads, batch_size = nce_logits.shape[:2]
        labels = torch.arange(batch_size, device=nce_logits.device).repeat(num_heads)
        nce_loss = F.cross_entropy(
            nce_logits.reshape(num_heads * batch_size, batch_size), labels
        )
        loss = (
            td_loss
            + cfg.nce_coef * nce_loss
            + cfg.reward_coef * reward_loss
        )

    critic.opt.zero_grad(set_to_none=True)
    encoder.opt.zero_grad(set_to_none=True)
    loss.backward()
    critic.grad_step()
    encoder.grad_step()

    return {
        "critic/td_loss": td_loss.detach(),
        "critic/nce_loss": nce_loss.detach(),
        "critic/reward_loss": reward_loss.detach(),
        "critic/batch_rew_min": batch.rewards.min().detach(),
        "critic/batch_rew_mean": batch.rewards.mean().detach(),
        "critic/batch_rew_max": batch.rewards.max().detach(),
        "critic/alpha": critic.model.get_alpha().mean().detach()
    }


@torch.compile(mode="max-autotune")
def update(
    cfg: VSimbaConfig,
    encoder: Network[VSimbaVisionEncoder],
    actor: Network[Actor],
    critic: Network[Critic],
    target_critic: Network[Critic],
    alpha: Network[Alpha],
    reward_normalizer: Network | None,
    sequencebatch: SequenceBatch,
) -> dict[str, torch.Tensor]:
    """Run one V-Simba update for the supplied networks and batch."""
    sequencebatch = augment_sequencebatch(sequencebatch)
    if reward_normalizer is not None:
        rewards = reward_normalizer.model.normalize(sequencebatch.rewards)
        # Explicit construction avoids PyTorch 2.9's _replace + graph-break bug.
        sequencebatch = SequenceBatch(sequencebatch.observations, sequencebatch.actions, rewards, sequencebatch.terminations, sequencebatch.truncations, sequencebatch.next_observations)
    batch = compress_n_step(sequencebatch, cfg.gamma)

    actor_info = update_actor(encoder, actor, critic, alpha, batch, cfg)
    alpha_info = update_alpha(alpha, actor_info["actor/entropy"], cfg.target_entropy)
    critic_info = update_critic(encoder, actor, critic, target_critic, alpha, batch, sequencebatch, cfg)
    target_critic.soft_update()
    return {**actor_info, **alpha_info, **critic_info}


__all__ = ["update", "update_actor", "update_critic"]
