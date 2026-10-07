import jax
import jax.numpy as jnp
from flax import nnx
from ....model.jax import (
    Alpha, 
    Network, 
    CategoricalPolicy, 
    QuantilePolicy, 
    RewardNormalizer,
)
from .network import FlashSACActor, FlashSACDoubleCritic
from ....common import (
    select_actor_observations,
)
from ....buffers.off_policy import Batch, SequenceBatch, compress_n_step
from ..config import WarpSACConfig


def update_critic(
        actor: Network[FlashSACActor],
        critic: Network[FlashSACDoubleCritic],
        alpha: Network[Alpha],
        target_critic: Network[FlashSACDoubleCritic],
        config: WarpSACConfig,
        batch: Batch,
        key: jax.Array):
    actor_next_obs = select_actor_observations(batch.next_observations, config.asymmetric_obs, actor.model.obs_dim)
    next_actions, next_log_pi = actor.model.get_action(actor_next_obs, key=key, training=False)
    next_q = target_critic.model(batch.next_observations, next_actions, training=True)
    alpha_value = jax.lax.stop_gradient(alpha())
    policy = critic.model.critic.policy
    discounts = batch.discounts * (1.0 - batch.dones)
    if isinstance(policy, CategoricalPolicy):
        target_bins = batch.rewards + discounts * (policy.bins - alpha_value * next_log_pi)
        targets = policy.target_probs(policy.select_min_logits(next_q), target_bins)
    else:
        targets = jax.lax.stop_gradient(batch.rewards + discounts * (jnp.min(next_q, axis=0) - alpha_value * next_log_pi))

    def loss_fn(model: FlashSACDoubleCritic):
        q = model(batch.observations, batch.actions, training=True)
        if isinstance(policy, CategoricalPolicy) or isinstance(policy, QuantilePolicy):
            loss = policy.loss(q, targets).mean()
            q_mean = policy.q_values(q).mean()
        else:
            loss = jnp.mean((q - targets) ** 2)
            q_mean = q.mean()
        return loss, {"training/q_loss": loss, "training/q_mean": q_mean}

    (_loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(critic.model)
    critic.grad_step(grads)
    if config.critic_normalize_parameters:
        critic.project_param()
    return info


def update_alpha(
    alpha: Network[Alpha],
    entropy: jax.Array,
    config: WarpSACConfig
) -> dict[str, jax.Array]:
    """Update entropy temperature."""

    def alpha_loss_fn(alpha_model: Alpha):
        alpha_loss = (-alpha_model() * (- entropy +
                      config.target_entropy)).mean()
        return alpha_loss, {"training/alpha_loss": alpha_loss, "training/alpha_value": alpha_model()}

    (_loss, info), grads = nnx.value_and_grad(
        alpha_loss_fn, has_aux=True)(alpha.model)
    alpha.grad_step(grads)
    return info


def update_actor(
    critic: Network[FlashSACDoubleCritic],
    actor: Network[FlashSACActor],
    alpha: Network[Alpha],
    config: WarpSACConfig,
    batch: Batch,
    key: jax.Array,
) -> dict[str, jax.Array]:
    """Update actor parameters."""
    alpha_value = alpha()
    action_key, entropy_key = jax.random.split(key, 2)
    alpha_value = jax.lax.stop_gradient(alpha_value)
    actor_observations = select_actor_observations(
        batch.observations, config.asymmetric_obs, actor.model.obs_dim)
    next_actor_observations = select_actor_observations(
        batch.next_observations, config.asymmetric_obs, actor.model.obs_dim)
    actor_obs_all = jnp.concat(
        [actor_observations, next_actor_observations], axis=0)

    def actor_loss_fn(actor: FlashSACActor, critic: FlashSACDoubleCritic):
        actions_all, log_pi_all = actor.get_action(
            actor_obs_all, key=action_key, training=True)
        actions = actions_all[: config.batch_size, ]
        log_pi = log_pi_all[: config.batch_size, ]
        q = critic.q_values(batch.observations, actions, training=False)
        q = getattr(jnp, config.q_agg)(q, axis=0)
        actor_loss = -jnp.mean(q - alpha_value * log_pi)
        return actor_loss, {"training/actor_loss": actor_loss, "training/entropy": -log_pi.mean()}

    (_loss, info), grads = nnx.value_and_grad(
        actor_loss_fn, argnums=0, has_aux=True
    )(actor.model, critic.model)
    actor.grad_step(grads)
    if config.actor_normalize_parameters:
        actor.project_param()
    return info


def update_policy(
    critic: Network[FlashSACDoubleCritic],
    actor: Network[FlashSACActor],
    alpha: Network[Alpha],
    config: WarpSACConfig,
    batch: Batch,
    key: jax.Array,
):
    actor_info = update_actor(critic, actor, alpha, config, batch, key)
    entropy = actor_info["training/entropy"]
    alpha_info = update_alpha(alpha, entropy, config)
    return {**actor_info, **alpha_info}


def make_update_warpsac(config: WarpSACConfig):
    @nnx.jit(static_argnames=("update_actor", "update_target"))
    def update_warpsac(
        critic: Network[FlashSACDoubleCritic],
        actor: Network[FlashSACActor],
        alpha: Network[Alpha],
        target_critic: Network[FlashSACDoubleCritic],
        reward_normalizer: Network[RewardNormalizer] | None,
        key: jax.Array,
        sequence: SequenceBatch,
        update_actor: bool,
        update_target: bool,
    ) -> dict[str, jax.Array]:
        batch = compress_n_step(sequence, config.gamma)
        if reward_normalizer is not None:
            batch = batch._replace(rewards=reward_normalizer.model.normalize(batch.rewards))
        policy_key, critic_key = jax.random.split(key)
        policy_info = update_policy(critic, actor, alpha, config, batch, policy_key) if update_actor else {}
        critic_info = update_critic(actor, critic, alpha, target_critic, config, batch, critic_key)
        if update_target:
            target_critic.soft_update()
        return {**critic_info, **policy_info}

    return update_warpsac
