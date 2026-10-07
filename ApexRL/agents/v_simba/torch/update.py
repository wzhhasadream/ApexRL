# Adapted from DAVIAN-Robotics/V-Simba (Apache-2.0), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/DAVIAN-Robotics/V-Simba/tree/be811e968bc02589fbb32f3be79f9a7d9a8fa86d/scale_rl/agents/vsimba
import torch

from ....buffers.off_policy import Batch, SequenceBatch, compress_n_step
from ....common.torch import augment_observations
from ....model.torch import Alpha, Network
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
    """Update the actor; backward(inputs=...) keeps critic parameters out of the gradient."""
    device_type, amp_dtype, amp_enabled = _autocast_config(encoder, cfg)
    alpha_value = alpha.model().detach()
    with torch.no_grad():
        vision_z = encoder.model(batch.observations)

    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
        actions, log_probs = actor.model.get_action(vision_z)
        q_values = critic.model.q_values(vision_z, actions).amin(dim=0)
    loss = (alpha_value * log_probs - q_values).mean()

    actor.opt.zero_grad(set_to_none=True)
    loss.backward(inputs=list(actor.model.parameters()))
    actor.grad_step()
    return {"actor/loss": loss.detach(), "actor/entropy": -log_probs.detach().mean()}


def update_alpha(alpha: Network[Alpha], entropy: torch.Tensor, target_entropy: float) -> dict[str, torch.Tensor]:
    value = alpha.model()
    loss = value * (entropy.detach() - target_entropy)
    alpha.opt.zero_grad(set_to_none=True)
    loss.backward()
    alpha.grad_step()
    return {"temperature/value": value.detach(), "temperature/loss": loss.detach()}


def update_critic(
    encoder: Network[VSimbaVisionEncoder],
    actor: Network[Actor],
    critic: Network[Critic],
    target_critic: Network[Critic],
    alpha: Network[Alpha],
    batch: Batch,
    cfg: VSimbaConfig,
) -> dict[str, torch.Tensor]:
    """Update the critic and encoder with the categorical TD loss."""
    device_type, amp_dtype, amp_enabled = _autocast_config(encoder, cfg)
    alpha_value = alpha.model().detach()

    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
        # Targets need no encoder gradient, so the next-observation pass stays out of autograd.
        next_vision_z = encoder.model(batch.next_observations)
        next_actions, next_log_probs = actor.model.get_action(next_vision_z)
        target_logits = critic.model.dist.select_min_logits(target_critic.model(next_vision_z, next_actions))
        target_values = batch.rewards + batch.discounts * (1.0 - batch.dones) * (critic.model.dist.bins - alpha_value * next_log_probs)
        target_probs = critic.model.dist.target_probs(target_logits, target_values)

    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
        logits = critic.model(encoder.model(batch.observations), batch.actions)
    # Cross-entropy written out (no torch.vmap) so it compiles cleanly; mean over batch,
    # summed over heads as V-Simba does when clipped double Q is enabled.
    loss = -(target_probs * logits.float().log_softmax(-1)).sum(-1).mean(-1).sum()

    critic.opt.zero_grad(set_to_none=True)
    encoder.opt.zero_grad(set_to_none=True)
    loss.backward()
    critic.grad_step()
    encoder.grad_step()

    return {
        "critic/loss": loss.detach(),
        "critic/batch_rew_min": batch.rewards.min(),
        "critic/batch_rew_mean": batch.rewards.mean(),
        "critic/batch_rew_max": batch.rewards.max(),
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
    sequence: SequenceBatch,
) -> dict[str, torch.Tensor]:
    """Compress n-step transitions and run one V-Simba update."""
    batch = compress_n_step(sequence, cfg.gamma)
    rewards = batch.rewards if reward_normalizer is None else reward_normalizer.model.normalize(batch.rewards)
    # Explicit construction avoids PyTorch 2.9's _replace + graph-break bug.
    batch = Batch(augment_observations(batch.observations), batch.actions, rewards, batch.dones, augment_observations(batch.next_observations), batch.discounts)

    actor_info = update_actor(encoder, actor, critic, alpha, batch, cfg)
    alpha_info = update_alpha(alpha, actor_info["actor/entropy"], cfg.target_entropy)
    critic_info = update_critic(encoder, actor, critic, target_critic, alpha, batch, cfg)
    return {**actor_info, **alpha_info, **critic_info}


__all__ = ["update", "update_actor", "update_alpha", "update_critic"]
