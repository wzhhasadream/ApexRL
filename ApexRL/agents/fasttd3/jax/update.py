from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from ....buffers.off_policy import Batch, SequenceBatch, compress_n_step
from ....common import select_actor_observations
from ....model.jax import Network, RMS
from ..config import FastTD3Config
from .network import Actor, Critic


@nnx.jit
def update_rms(network: Network[RMS], observations: jax.Array) -> None:
    network.model.update(observations)


def _clip_action(actor: Actor, actions: jax.Array) -> jax.Array:
    return jnp.clip(actions, actor.policy.action_low.value, actor.policy.action_high.value)


def update_critic(
    critic: Network[Critic], target_critic: Network[Critic], actor: Network[Actor],
    batch: Batch, key: jax.Array, cfg: FastTD3Config,
) -> dict[str, jax.Array]:
    actor_next_observations = select_actor_observations(batch.next_observations, cfg.asymmetric_obs, actor.model.obs_dim)
    noise = jnp.clip(jax.random.normal(key, batch.actions.shape) * cfg.policy_noise, -cfg.noise_clip, cfg.noise_clip)
    next_actions = _clip_action(actor.model, actor.model(actor_next_observations) + noise)
    next_actions = jax.lax.stop_gradient(next_actions)
    target_logits = jax.lax.stop_gradient(target_critic.model(batch.next_observations, next_actions))
    continuation = 1.0 - batch.dones
    target_values = batch.rewards + continuation * batch.discounts * critic.model.dist.bins
    if cfg.use_cdq:
        target_logits = critic.model.dist.select_min_logits(target_logits)
    target_probs = jax.lax.stop_gradient(critic.model.dist.target_probs(target_logits, target_values))

    def critic_loss(model: Critic) -> tuple[jax.Array, dict[str, jax.Array]]:
        logits = model(batch.observations, batch.actions)
        target_axes = None if cfg.use_cdq else 0
        losses = jax.vmap(model.dist._loss_one, in_axes=(0, target_axes))(logits, target_probs)
        loss = losses.mean()
        return loss, {"critic/loss": loss, "critic/mean_q": model.dist.q_values(logits).mean()}

    (_loss, info), grads = nnx.value_and_grad(critic_loss, has_aux=True)(critic.model)
    critic.grad_step(grads, cfg.max_grad_norm)
    return info


def update_actor(
    actor: Network[Actor], critic: Network[Critic], batch: Batch, cfg: FastTD3Config,
) -> dict[str, jax.Array]:
    def actor_loss(model: Actor, critic_model: Critic) -> tuple[jax.Array, dict[str, jax.Array]]:
        actor_observations = select_actor_observations(batch.observations, cfg.asymmetric_obs, model.obs_dim)
        actions = model(actor_observations)
        values = critic_model.q_values(batch.observations, actions)
        value = jnp.min(values, axis=0) if cfg.use_cdq else values.mean(axis=0)
        loss = -value.mean()
        return loss, {"actor/loss": loss, "actor/mean_q": value.mean()}

    (_loss, info), actor_grads = nnx.value_and_grad(actor_loss, has_aux=True, argnums=0)(actor.model, critic.model)
    actor.grad_step(actor_grads, cfg.max_grad_norm)
    return info


def make_update(cfg: FastTD3Config) -> Callable[..., dict[str, jax.Array]]:
    @nnx.jit(static_argnames=("do_actor",))
    def update(
        critic: Network[Critic], target_critic: Network[Critic], actor: Network[Actor],
        observation_rms: Network[RMS] | None,
        sequence: SequenceBatch, key: jax.Array, do_actor: bool,
    ) -> dict[str, jax.Array]:
        batch = compress_n_step(sequence, cfg.gamma)
        if cfg.obs_normalization:
            batch = batch._replace(
                observations=observation_rms.model.normalize(batch.observations, update=False),
                next_observations=observation_rms.model.normalize(batch.next_observations, update=False),
            )
        critic_key, _ = jax.random.split(key)
        critic_info = update_critic(critic, target_critic, actor, batch, critic_key, cfg)
        if do_actor:
            critic_info.update(update_actor(actor, critic, batch, cfg))
        target_critic.soft_update()
        return critic_info

    return update
