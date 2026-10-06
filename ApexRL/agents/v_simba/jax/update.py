from ..config import VSimbaConfig

import jax
from flax import nnx
import jax.numpy as jnp
from ....buffers import compress_n_step
from ....buffers.off_policy.types import Batch, SequenceBatch
from ....common.jax.augment import augment_sequencebatch
from ....common.jax import l2_norm
from optax import softmax_cross_entropy_with_integer_labels
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
    target_critic: Network[Critic], alpha: Network[Alpha], batch: Batch,
    sequencebatch: SequenceBatch, key: jax.Array, nce_coef: float, reward_coef: float
):
    alpha_value = jax.lax.stop_gradient(alpha.model())

    def critic_loss(critic: Critic, encoder: VSimbaVisionEncoder, actor: Actor, target_critic: Critic):
        vision_z = encoder(batch.observations)
        next_vision_z = encoder(batch.next_observations)
        real_next_vision_z = encoder(sequencebatch.next_observations[0])
        next_actions, next_log_probs = actor.get_action(jax.lax.stop_gradient(next_vision_z), key)
        next_actions = jax.lax.stop_gradient(next_actions)
        next_log_probs = jax.lax.stop_gradient(next_log_probs)
        _, _, target_logits = target_critic(jax.lax.stop_gradient(next_vision_z), next_actions)
        dist = critic.dist
        target_logits = dist.select_min_logits(target_logits)
        target_values = batch.rewards + batch.discounts * (1.0 - batch.dones) * (dist.bins - alpha_value * next_log_probs)
        target_probs = dist.target_probs(target_logits, target_values)

        phi_sa, pred_rewards, logits = critic(vision_z, batch.actions)
        # V-Simba sums the two head losses when clipped double Q is enabled.
        td_loss = critic.dist.loss(logits, target_probs).sum()

        g_s = critic.embed_s(real_next_vision_z)
        reward_loss = jnp.square(sequencebatch.rewards[0] - pred_rewards).mean()

        alpha = critic.get_alpha()

        def nce_loss_fn(phi_sa: jax.Array, g_s: jax.Array, nce_alpha: jax.Array):
            phi_sa, g_s = l2_norm(phi_sa), l2_norm(g_s)
            logits = jnp.einsum("bi,di->bd", phi_sa, g_s) * nce_alpha
            label = jnp.arange(batch.observations.shape[0])
            loss = softmax_cross_entropy_with_integer_labels(logits, label).mean()
            return loss

        nce_loss = jax.vmap(nce_loss_fn)(phi_sa, g_s, alpha).mean()

        loss = td_loss + nce_coef * nce_loss + reward_coef * reward_loss

        return loss, {"critic/td_loss": td_loss, "critic/batch_rew_min": batch.rewards.min(), "critic/batch_rew_mean": batch.rewards.mean(), "critic/batch_rew_max": batch.rewards.max(), "critic/nce_loss": nce_loss, "critic/reward_loss": reward_loss, "critic/alpha": alpha.mean()}

    (_, info), (critic_grads, encoder_grads) = nnx.value_and_grad(
        critic_loss, argnums=(0, 1), has_aux=True
    )(critic.model, encoder.model, actor.model, target_critic.model)
    critic.grad_step(critic_grads)
    encoder.grad_step(encoder_grads)
    return info


def make_update(cfg: VSimbaConfig):
    @nnx.jit
    def update(
        encoder: Network[VSimbaVisionEncoder], actor: Network[Actor], critic: Network[Critic],
        target_critic: Network[Critic], alpha: Network[Alpha], reward_normalizer: Network[RewardNormalizer] | None,
        sequencebatch: SequenceBatch, key: jax.Array,
    ):
        sequence_key, actor_key, critic_key = jax.random.split(key, 3)
        sequencebatch = augment_sequencebatch(sequencebatch, sequence_key)
        if reward_normalizer is not None:
            sequencebatch = sequencebatch._replace(rewards=reward_normalizer.model.normalize(sequencebatch.rewards))
        batch = compress_n_step(sequencebatch, cfg.gamma)

        actor_info = update_actor(encoder, actor, critic, alpha, batch, actor_key)
        alpha_info = update_alpha(alpha, actor_info["actor/entropy"], cfg.target_entropy)
        critic_info = update_critic(encoder, actor, critic, target_critic, alpha, batch, sequencebatch, critic_key, cfg.nce_coef, cfg.reward_coef)
        soft_update(critic.model.heads, target_critic.model.heads, cfg.target_tau)
        return {**actor_info, **alpha_info, **critic_info}

    return update
