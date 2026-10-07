# Adapted from DAVIAN-Robotics/V-Simba (Apache-2.0), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/DAVIAN-Robotics/V-Simba/tree/be811e968bc02589fbb32f3be79f9a7d9a8fa86d/scale_rl/agents/vsimba
from ..config import VSimbaConfig

import jax
from flax import nnx
import jax.numpy as jnp
from ....buffers.off_policy import Batch, SequenceBatch, compress_n_step
from ....common.jax.augment import augment_observations
from ....model.jax import Alpha, Network, RewardNormalizer, soft_update
from ....model.jax.backbones import VSimbaVisionEncoder
from .network import Actor, Critic


def update_actor(encoder: Network[VSimbaVisionEncoder], actor: Network[Actor], critic: Network[Critic], alpha: Network[Alpha], batch: Batch, key: jax.Array):
    alpha_value = alpha.model()
    vision_z = jax.lax.stop_gradient(encoder.model(batch.observations))

    def actor_loss(actor: Actor, critic: Critic):
        actions, log_probs = actor.get_action(vision_z, key)
        q = critic.q_values(vision_z, actions).min(axis=0)
        loss = (alpha_value * log_probs - q).mean()
        return loss, {"actor/loss": loss, "actor/entropy": -log_probs.mean()}

    (_, info), grads = nnx.value_and_grad(actor_loss, has_aux=True)(actor.model, critic.model)
    actor.grad_step(grads)
    return info


def update_alpha(alpha: Network[Alpha], entropy: jax.Array, target_entropy: float):
    def alpha_loss(model: Alpha):
        value = model()
        loss = value * (entropy - target_entropy)
        return loss, {"temperature/value": value, "temperature/loss": loss}

    (_, info), grads = nnx.value_and_grad(alpha_loss, has_aux=True)(alpha.model)
    alpha.grad_step(grads)
    return info


def update_critic(
    encoder: Network[VSimbaVisionEncoder], actor: Network[Actor], critic: Network[Critic],
    target_critic: Network[Critic], alpha: Network[Alpha], batch: Batch, key: jax.Array,
):
    """Update the critic and encoder with the categorical TD loss."""
    # Targets need no encoder gradient, so they are computed outside the loss function.
    next_vision_z = encoder.model(batch.next_observations)
    next_actions, next_log_probs = actor.model.get_action(next_vision_z, key)
    dist = critic.model.dist
    target_logits = dist.select_min_logits(target_critic.model(next_vision_z, next_actions))
    target_values = batch.rewards + batch.discounts * (1.0 - batch.dones) * (dist.bins - alpha.model() * next_log_probs)
    target_probs = jax.lax.stop_gradient(dist.target_probs(target_logits, target_values))

    def critic_loss(critic: Critic, encoder: VSimbaVisionEncoder):
        logits = critic(encoder(batch.observations), batch.actions)
        # V-Simba sums the per-head losses when clipped double Q is enabled.
        loss = critic.dist.loss(logits, target_probs).sum()
        return loss, {"critic/loss": loss, "critic/batch_rew_min": batch.rewards.min(), "critic/batch_rew_mean": batch.rewards.mean(), "critic/batch_rew_max": batch.rewards.max()}

    (_, info), (critic_grads, encoder_grads) = nnx.value_and_grad(critic_loss, argnums=(0, 1), has_aux=True)(critic.model, encoder.model)
    critic.grad_step(critic_grads)
    encoder.grad_step(encoder_grads)
    return info


def make_update(cfg: VSimbaConfig):
    @nnx.jit
    def update(
        encoder: Network[VSimbaVisionEncoder], actor: Network[Actor], critic: Network[Critic],
        target_critic: Network[Critic], alpha: Network[Alpha], reward_normalizer: Network[RewardNormalizer] | None,
        sequence: SequenceBatch, key: jax.Array,
    ):
        batch = compress_n_step(sequence, cfg.gamma)
        obs_key, next_obs_key, actor_key, critic_key = jax.random.split(key, 4)
        rewards = batch.rewards if reward_normalizer is None else reward_normalizer.model.normalize(batch.rewards)
        batch = batch._replace(
            observations=augment_observations(obs_key, batch.observations), rewards=rewards,
            next_observations=augment_observations(next_obs_key, batch.next_observations),
        )

        actor_info = update_actor(encoder, actor, critic, alpha, batch, actor_key)
        alpha_info = update_alpha(alpha, actor_info["actor/entropy"], cfg.target_entropy)
        critic_info = update_critic(encoder, actor, critic, target_critic, alpha, batch, critic_key)
        soft_update(critic.model.heads, target_critic.model.heads, cfg.target_tau)
        return {**actor_info, **alpha_info, **critic_info}

    return update
