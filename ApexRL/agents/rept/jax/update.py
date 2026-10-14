from ....common import add_prefix_to_keys
from ....common.jax.augment import augment_sequencebatch
from ....model.jax import Network, Alpha, RewardNormalizer, update_alpha
from ....buffers.off_policy import Batch, SequenceBatch, compress_n_step
from optax import softmax_cross_entropy_with_integer_labels
from .network import Actor, Critic, Encoder
import jax
import jax.numpy as jnp
from flax import nnx
from ..config import RePTConfig


def l2_norm(feature: jax.Array) -> jax.Array:
    return feature * jax.lax.rsqrt(jnp.maximum(jnp.square(feature).sum(axis=-1, keepdims=True), 1e-12))


def update_encoder(
    encoder: Network[Encoder],
    target_encoder: Network[Encoder],
    sequence: SequenceBatch,
    cfg: RePTConfig,
) -> dict[str, jax.Array]:
    horizon, batch_size = sequence.rewards.shape[:2]
    transitions = jax.tree.map(lambda x: x.reshape((-1, *x.shape[2:])), sequence)
    next_z_s = target_encoder.model.embed_s(transitions.next_observations, training=True)
    next_z_s = l2_norm(jax.lax.stop_gradient(next_z_s)).reshape(horizon, batch_size, -1)

    def encoder_loss(encoder: Encoder):
        z_s = encoder.embed_s(transitions.observations, training=True)
        z_sa = encoder.embed_sa(z_s, transitions.actions, training=True)
        pred_z_s = encoder.pred_z_s(z_sa)
        nce_alpha = encoder.get_alpha()
        # Each time step uses its own batch negatives; every raw transition contributes.
        nce_logits = nce_alpha * jnp.einsum("tbd,tcd->tbc", l2_norm(pred_z_s).reshape(horizon, batch_size, -1), next_z_s)
        labels = jnp.broadcast_to(jnp.arange(batch_size), (horizon, batch_size))
        nce_loss = softmax_cross_entropy_with_integer_labels(nce_logits, labels).mean()
        reward_loss = jnp.square(encoder.pred_reward(z_sa) - transitions.rewards).mean()
        loss = cfg.nce_coef * nce_loss + cfg.reward_coef * reward_loss
        return loss, {"nce_loss": nce_loss, "reward_loss": reward_loss, "nce_alpha": nce_alpha}

    (_loss, info), grads = nnx.value_and_grad(encoder_loss, has_aux=True)(encoder.model)
    encoder.grad_step(grads)
    if cfg.critic_normalize_parameters:
        encoder.project_param()
    return info


def update_critic(
    encoder: Network[Encoder],
    critic: Network[Critic],
    target_critic: Network[Critic],
    actor: Network[Actor],
    alpha: Network[Alpha],
    batch: Batch,
    key: jax.Array,
    cfg: RePTConfig,
) -> dict[str, jax.Array]:
    # Encoder parameters and running statistics stay fixed during TD learning.
    observations = jnp.concatenate([batch.observations, batch.next_observations], axis=0)
    z_s = encoder.model.embed_s(observations, training=False)
    next_z_s = jnp.split(z_s, 2, axis=0)[1]
    next_actions, next_log_prob = actor.model.get_action(next_z_s, key=key, training=False)
    actions = jnp.concatenate([batch.actions, next_actions], axis=0)
    all_z_sa = jax.lax.stop_gradient(encoder.model.embed_sa(z_s, actions, training=False))
    z_sa = jnp.split(all_z_sa, 2, axis=0)[0]
    next_logits = jnp.split(target_critic.model(all_z_sa, training=True), 2, axis=1)[1]
    dist = critic.model.dist
    continuation = batch.discounts * (1.0 - batch.dones)
    target_bins = batch.rewards + continuation * (dist.bins - alpha.model() * next_log_prob)
    target_probs = dist.target_probs(dist.select_min_logits(next_logits), target_bins)

    def critic_loss(critic: Critic):
        logits = critic(z_sa, training=True)
        td_loss = critic.dist.loss(logits, target_probs).mean()
        return td_loss, {"td_loss": td_loss}

    (_loss, info), grads = nnx.value_and_grad(critic_loss, has_aux=True)(critic.model)
    critic.grad_step(grads)
    if cfg.critic_normalize_parameters:
        critic.project_param()

    return info


def update_actor(
    encoder: Network[Encoder],
    critic: Network[Critic],
    actor: Network[Actor],
    alpha: Network[Alpha],
    batch: Batch,
    key: jax.Array,
    cfg: RePTConfig,
) -> dict[str, jax.Array]:
    alpha_value = jax.lax.stop_gradient(alpha.model())
    observations = jnp.concatenate([batch.observations, batch.next_observations], axis=0)
    z_s_all = jax.lax.stop_gradient(encoder.model.embed_s(observations, training=False))
    z_s = jnp.split(z_s_all, 2, axis=0)[0]

    def actor_loss(actor: Actor, encoder: Encoder, critic: Critic):
        actions_all, log_probs_all = actor.get_action(z_s_all, key=key, training=True)
        actions, log_probs = jnp.split(actions_all, 2, axis=0)[0], jnp.split(log_probs_all, 2, axis=0)[0]
        # Keep dQ/da through the frozen encoder and critic; only actor parameters are differentiated.
        z_sa = encoder.embed_sa(z_s, actions, training=False)
        q_values = critic.q_values(z_sa, training=False).min(axis=0)
        loss = (alpha_value * log_probs - q_values).mean()
        return loss, {
            "training/actor_loss": loss,
            "training/entropy": -log_probs.mean(),
        }

    (_loss, info), grads = nnx.value_and_grad(actor_loss, argnums=0, has_aux=True)(actor.model, encoder.model, critic.model)
    actor.grad_step(grads)

    if cfg.actor_normalize_parameters:
        actor.project_param()

    return info


def update_policy(
    encoder: Network[Encoder],
    critic: Network[Critic],
    actor: Network[Actor],
    alpha: Network[Alpha],
    config: RePTConfig,
    batch: Batch,
    key: jax.Array,
) -> dict[str, jax.Array]:
    actor_info = update_actor(encoder, critic, actor, alpha, batch, key, config)
    entropy = actor_info["training/entropy"]
    entropy_info = add_prefix_to_keys(update_alpha(alpha, entropy, config.target_entropy), "alpha")
    return {**actor_info, **entropy_info}


def make_update(cfg: RePTConfig):
    @nnx.jit(static_argnames=("do_policy", "do_target"))
    def update(
        encoder: Network[Encoder],
        critic: Network[Critic],
        target_encoder: Network[Encoder],
        target_critic: Network[Critic],
        actor: Network[Actor],
        alpha: Network[Alpha],
        reward_normalizer: Network[RewardNormalizer] | None,
        sequence: SequenceBatch,
        key: jax.Array,
        do_policy: bool,
        do_target: bool,
    ) -> dict[str, jax.Array]:
        actor_key, critic_key, aug_key = jax.random.split(key, 3)
        if cfg.image_obs:
            sequence = augment_sequencebatch(sequence, aug_key)
        if cfg.normalize_rewards and reward_normalizer is not None:
            sequence = sequence._replace(rewards=reward_normalizer.model.normalize(sequence.rewards))

        encoder_info = update_encoder(encoder, target_encoder, sequence, cfg)
        batch = compress_n_step(sequence, cfg.gamma)
        critic_info = update_critic(
            encoder, critic, target_critic, actor, alpha, batch, critic_key, cfg
        )
        actor_info = {}
        if do_policy:
            actor_info = update_policy(encoder, critic, actor, alpha, cfg, batch, actor_key)
        if do_target:
            target_encoder.soft_update()
            target_critic.soft_update()

        return {**encoder_info, **critic_info, **actor_info}

    return update
