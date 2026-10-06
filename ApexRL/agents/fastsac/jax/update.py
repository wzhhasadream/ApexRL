from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from ....buffers.off_policy import Batch, SequenceBatch, compress_n_step
from ....common import select_actor_observations
from ....model.jax import Alpha, Network, RMS
from ..config import FastSACConfig
from .network import Actor, Critic


@nnx.jit
def normalize_rms(network: Network[RMS], observations: jax.Array) -> jax.Array:
    return network.model.normalize(observations, update=False)




@nnx.jit
def update_rms(network: Network[RMS], observations: jax.Array) -> None:
    network.model.update(observations)


def _target_distribution(critic: Critic, logits: jax.Array, target_values: jax.Array) -> jax.Array:
    return jax.vmap(lambda head_logits: critic.dist.target_probs(head_logits, target_values))(logits)


def update_critic(
    critic: Network[Critic], target_critic: Network[Critic], actor: Network[Actor],
    alpha: Network[Alpha], batch: Batch, key: jax.Array, cfg: FastSACConfig,
) -> dict[str, jax.Array]:
    alpha_value = jax.lax.stop_gradient(alpha.model())
    next_actor_observations = select_actor_observations(batch.next_observations, cfg.asymmetric_obs, actor.model.obs_dim)
    next_actions, next_log_probs = actor.model.get_action(next_actor_observations, key)
    next_actions = jax.lax.stop_gradient(next_actions)
    next_log_probs = jax.lax.stop_gradient(next_log_probs)
    target_logits = jax.lax.stop_gradient(target_critic.model(batch.next_observations, next_actions))
    continuation = batch.discounts * (1.0 - batch.dones)
    target_values = batch.rewards + continuation * (critic.model.dist.bins - alpha_value * next_log_probs)
    target_probs = jax.lax.stop_gradient(_target_distribution(critic.model, target_logits, target_values))

    def critic_loss(model: Critic) -> tuple[jax.Array, dict[str, jax.Array]]:
        logits = model(batch.observations, batch.actions)
        loss = jax.vmap(model.dist._loss_one, in_axes=(0, 0))(logits, target_probs).mean()
        return loss, {"critic/loss": loss, "critic/mean_q": model.dist.q_values(logits).mean()}

    (_loss, info), grads = nnx.value_and_grad(critic_loss, has_aux=True)(critic.model)
    critic.grad_step(grads, cfg.max_grad_norm)
    return info


def update_alpha(alpha: Network[Alpha], entropy: jax.Array, target_entropy: float) -> dict[str, jax.Array]:
    def alpha_loss(model: Alpha) -> tuple[jax.Array, dict[str, jax.Array]]:
        value = model()
        loss = value * jax.lax.stop_gradient(entropy - target_entropy)
        return loss, {"alpha/loss": loss, "alpha/value": value}

    (_loss, info), grads = nnx.value_and_grad(alpha_loss, has_aux=True)(alpha.model)
    alpha.grad_step(grads)
    return info


def update_actor(
    actor: Network[Actor], critic: Network[Critic], alpha: Network[Alpha],
    batch: Batch, key: jax.Array, cfg: FastSACConfig,
) -> dict[str, jax.Array]:
    alpha_value = jax.lax.stop_gradient(alpha.model())

    def actor_loss(model: Actor, critic_model: Critic) -> tuple[jax.Array, dict[str, jax.Array]]:
        actor_observations = select_actor_observations(batch.observations, cfg.asymmetric_obs, model.obs_dim)
        actions, log_probs = model.get_action(actor_observations, key)
        q_value = critic_model.q_values(batch.observations, actions).mean(axis=0)
        loss = (alpha_value * log_probs - q_value).mean()
        return loss, {"actor/loss": loss, "actor/entropy": -log_probs.mean()}

    (_loss, info), actor_grads = nnx.value_and_grad(actor_loss, has_aux=True, argnums=0)(actor.model, critic.model)
    actor.grad_step(actor_grads, cfg.max_grad_norm)
    return info


def make_update(cfg: FastSACConfig) -> Callable[..., dict[str, jax.Array]]:
    @nnx.jit(static_argnames=("do_policy",))
    def update(
        critic: Network[Critic], target_critic: Network[Critic], actor: Network[Actor],
        alpha: Network[Alpha], observation_rms: Network[RMS] | None,
        sequence: SequenceBatch, key: jax.Array, do_policy: bool,
    ) -> dict[str, jax.Array]:
        critic_key, alpha_key, actor_key = jax.random.split(key, 3)
        if cfg.obs_normalization:
            sequence = sequence._replace(
                observations=observation_rms.model.normalize(sequence.observations, update=False),
                next_observations=observation_rms.model.normalize(sequence.next_observations, update=False),
            )
        batch = compress_n_step(sequence, cfg.gamma)
        critic_info = update_critic(critic, target_critic, actor, alpha, batch, critic_key, cfg)
        actor_observations = select_actor_observations(batch.observations, cfg.asymmetric_obs, actor.model.obs_dim)
        _, log_probs = actor.model.get_action(actor_observations, alpha_key)
        entropy = -log_probs.mean()
        alpha_info = update_alpha(alpha, entropy, cfg.target_entropy) if cfg.use_autotune else {"alpha/value": alpha.model()}
        actor_info = update_actor(actor, critic, alpha, batch, actor_key, cfg) if do_policy else {}
        target_critic.soft_update()
        return {**critic_info, **alpha_info, **actor_info}

    return update
