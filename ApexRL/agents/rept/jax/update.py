from resprent_rl.buffers import SequenceBatch, compress_n_step
from resprent_rl.common.jax.augment import augment_sequencebatch
from ....common import select_actor_observations
from resprent_rl.model.jax.backbones import VSimbaVisionEncoder, DRQV2VisionEncoder, DRQVisionEncoder
from ....common import add_prefix_to_keys
from ....model.jax import Network, Alpha, RewardNormalizer, update_alpha
from ....buffers.off_policy.types import Batch, SequenceBatch
from optax import softmax_cross_entropy_with_integer_labels
from .network import Actor, Critic
import jax
import jax.numpy as jnp
from flax import nnx
from ..config import RePTConfig


VisionEncoder = DRQVisionEncoder | VSimbaVisionEncoder | DRQV2VisionEncoder


def l2_norm(feature: jax.Array):
    return feature / jnp.maximum(jnp.linalg.norm(feature, axis=-1, keepdims=True), 1e-6)


def encoder_observation(observation: jax.Array, vision_encoder: VisionEncoder | Network[VisionEncoder] | None) -> jax.Array:
    if vision_encoder is not None:
        if isinstance(vision_encoder, Network):
            return vision_encoder.model(observation)
        return vision_encoder(observation)
    return observation


def update_critic(
    critic: Network[Critic],
    target_critic: Network[Critic],
    vision_encoder: Network[VisionEncoder] | None,
    actor: Network[Actor],
    alpha: Network[Alpha],
    batch: Batch,
    sequence: SequenceBatch,
    key: jax.Array,
    cfg: RePTConfig
):
    alpha_value = alpha.model()


    def critic_loss(critic: Critic, vision_encoder: VisionEncoder | None, actor: Actor, target_critic: Critic):
        all_obs = jnp.concat([batch.observations, batch.next_observations], axis=0)
        all_state = encoder_observation(all_obs, vision_encoder)
        next_actions, log_pi = actor.get_action(select_actor_observations(jnp.split(all_state, 2, axis=0)[1], cfg.asymmetric_obs, actor.obs_dim), key=key, training=False)
        next_actions = jax.lax.stop_gradient(next_actions)
        log_pi = jax.lax.stop_gradient(log_pi)
        all_action = jnp.concat([batch.actions, next_actions], axis=0)
        _, _, all_target_logits = target_critic(jax.lax.stop_gradient(all_state), all_action, training=True)
        target_logits = jnp.split(all_target_logits, 2, 1)[1]
        all_phi_sa, all_rewards, all_logits = critic(all_state, all_action, training=True)

        logits = jnp.split(all_logits, 2, axis=1)[0]
        rewards = jnp.split(all_rewards, 2, axis=1)[0]
        phi_sa = jnp.split(all_phi_sa, 2, axis=1)[0]
        real_next_obs = sequence.next_observations[0]
        state = encoder_observation(real_next_obs, vision_encoder)
        g_s = critic.embed_s(state, training=True)

        continuation = batch.discounts * (1.0 - batch.dones)

        reward_loss = jnp.square((rewards - sequence.rewards[0])).mean()

        nce_alpha = critic.get_alpha()

        def nce_loss_fn(phi_sa: jax.Array, g_s: jax.Array, nce_alpha: jax.Array):
            phi_sa, g_s = l2_norm(phi_sa), l2_norm(g_s)
            logits = jnp.einsum("bi,di->bd", phi_sa, g_s) * nce_alpha
            label = jnp.arange(batch.observations.shape[0])
            loss = softmax_cross_entropy_with_integer_labels(logits, label).mean()
            return loss

        nce_loss = jax.vmap(nce_loss_fn)(phi_sa, g_s, nce_alpha).mean()

        target_bins = batch.rewards + continuation * (
            critic.dist.bins - alpha_value * log_pi
        )

        min_next_logits = critic.dist.select_min_logits(target_logits)
        target_probs = critic.dist.target_probs(
            target_logits=min_next_logits,
            target_values=target_bins,
        )

        td_loss = critic.dist.loss(logits, target_probs).mean()

        return td_loss + cfg.nce_coef * nce_loss + cfg.reward_coef *reward_loss, {
            "td_loss": td_loss,
            "nce_loss": nce_loss,
            "reward_loss": reward_loss,
            "nce_alpha": nce_alpha.mean()
        }

    if vision_encoder is not None:
            (_loss, info), (critic_grads, encoder_grads) = nnx.value_and_grad(
                critic_loss, has_aux=True, argnums=(0, 1)
            )(critic.model, vision_encoder.model, actor.model, target_critic.model)
            critic.grad_step(critic_grads)
            vision_encoder.grad_step(encoder_grads)
    else:
            (_loss, info), critic_grads = nnx.value_and_grad(
                critic_loss, has_aux=True, argnums=0
            )(critic.model, None, actor.model, target_critic.model)
            critic.grad_step(critic_grads)
    if cfg.critic_normalize_parameters:
        critic.project_param()

    return info


def update_actor(
    critic: Network[Critic],
    vision_encoder: Network[VisionEncoder] | None,
    actor: Network[Actor],
    alpha: Network[Alpha],
    batch: Batch,
    key: jax.Array,
    cfg: RePTConfig
):
    alpha_value = alpha.model()
    obs_all = jnp.concat([batch.observations, batch.next_observations], axis=0)
    state_all = encoder_observation(obs_all, vision_encoder)

    def actor_loss(actor: Actor, critic: Critic):
        actions_all, log_probs_all = actor.get_action(
            select_actor_observations(
                state_all, cfg.asymmetric_obs, actor.obs_dim
            ),
            key=key,
            training=True,
        )
        actions = jnp.split(actions_all, 2)[0]
        log_probs = jnp.split(log_probs_all, 2)[0]
        q_values = critic.q_values(
            jnp.split(state_all, 2, axis=0)[0], actions, training=False
        ).min(0)
        loss = -(q_values - alpha_value * log_probs).mean()
        entropy = -log_probs.mean()

        return loss, {
            "training/actor_loss": loss,
            "training/entropy": entropy,
        }

    (_loss, info), grads = nnx.value_and_grad(actor_loss, has_aux=True)(actor.model, critic.model)
    actor.grad_step(grads)

    if cfg.actor_normalize_parameters:
        actor.project_param()

    return info


def update_policy(
    critic: Network[Critic],
    vision_encoder: Network[VisionEncoder] | None,
    actor: Network[Actor],
    alpha: Network[Alpha],
    config: RePTConfig,
    batch: Batch,
    key: jax.Array,
):
    actor_info = update_actor(critic, vision_encoder, actor, alpha, batch, key, config)
    entropy = actor_info["training/entropy"]
    entropy_info = add_prefix_to_keys(update_alpha(alpha, entropy, config.target_entropy), "entropy_coef")
    return {**actor_info, **entropy_info}


def make_update(cfg: RePTConfig):
    @nnx.jit(static_argnames=("do_policy", "do_target"))
    def update(
        critic: Network[Critic],
        target_critic: Network[Critic],
        vision_encoder: Network[VisionEncoder] | None,
        actor: Network[Actor],
        alpha: Network[Alpha],
        reward_normalizer: Network[RewardNormalizer] | None,
        sequence: SequenceBatch,
        key: jax.Array,
        do_policy: bool,
        do_target: bool
    ):
        actor_key, critic_key, aug_key = jax.random.split(key, 3)
        if cfg.image_obs:
            sequence = augment_sequencebatch(sequence, aug_key)
        if cfg.normalize_rewards and reward_normalizer is not None:
            sequence = sequence._replace(rewards=reward_normalizer.model.normalize(sequence.rewards))
        
        batch = compress_n_step(sequence, cfg.gamma)

        actor_info = {}
        if do_policy:
            actor_info = update_policy(critic, vision_encoder, actor, alpha, cfg, batch, actor_key)

        critic_info = update_critic(
            critic, target_critic, vision_encoder, actor, alpha, batch, sequence, critic_key, cfg
        )


        if do_target:
            target_critic.soft_update()

        return {**critic_info, **actor_info}

    return update
