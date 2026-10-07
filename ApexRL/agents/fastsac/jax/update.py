# Adapted from amazon-far/holosoma FastSAC (Apache-2.0), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/amazon-far/holosoma/tree/d18d6cc50f872c15e904a22ceac22313cec955c8/src/holosoma/holosoma/agents/fast_sac
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from ....buffers.off_policy import Batch, compress_n_step
from ....buffers.off_policy.jax_buffer import JaxBuffer
from ....model.jax import Alpha, Network, RMS
from ..config import FastSACConfig
from .network import Actor, Critic


@nnx.jit
def update_rms(network: Network[RMS], observations: jax.Array) -> None:
    network.model.update(observations)


def update_critic(critic: Network[Critic], target_critic: Network[Critic], actor: Network[Actor], alpha: Network[Alpha], batch: Batch, key: jax.Array, cfg: FastSACConfig) -> tuple[dict[str, jax.Array], jax.Array]:
    alpha_value = jax.lax.stop_gradient(alpha.model())
    # Soft C51 target: each Q head is projected against its own target head (no CDQ)
    next_actions, next_log_probs = jax.lax.stop_gradient(actor.model.get_action(batch.next_observations[..., :actor.model.obs_dim], key))
    target_logits = target_critic.model(batch.next_observations, next_actions)                    # [E, B, A]
    target_values = batch.rewards + batch.discounts * (1.0 - batch.dones) * (critic.model.dist.bins - alpha_value * next_log_probs)
    target_probs = jax.lax.stop_gradient(jax.vmap(critic.model.dist.target_probs, in_axes=(0, None))(target_logits, target_values))

    def critic_loss(model: Critic) -> tuple[jax.Array, jax.Array]:
        logits = model(batch.observations, batch.actions)
        # Holosoma sums over Q heads after averaging each head over the batch.
        return -(target_probs * jax.nn.log_softmax(logits, -1)).sum(-1).mean(-1).sum(), logits

    (loss, logits), grads = nnx.value_and_grad(critic_loss, has_aux=True)(critic.model)
    critic.grad_step(grads, cfg.max_grad_norm)
    # Holosoma reuses next_log_probs in update_alpha.
    return {"critic/loss": loss, "critic/mean_q": critic.model.dist.q_values(logits).mean()}, next_log_probs


def update_alpha(alpha: Network[Alpha], log_probs: jax.Array, target_entropy: float) -> dict[str, jax.Array]:
    def alpha_loss(model: Alpha) -> tuple[jax.Array, jax.Array]:
        value = model()
        return (-value * jax.lax.stop_gradient(log_probs + target_entropy)).mean(), value

    (loss, value), grads = nnx.value_and_grad(alpha_loss, has_aux=True)(alpha.model)
    alpha.grad_step(grads)
    return {"alpha/loss": loss, "alpha/value": value}


def update_actor(actor: Network[Actor], critic: Network[Critic], alpha: Network[Alpha], batch: Batch, key: jax.Array, cfg: FastSACConfig) -> dict[str, jax.Array]:
    alpha_value = jax.lax.stop_gradient(alpha.model())

    # critic is passed as an argument (not closed over) so nnx can trace it; grads only w.r.t. argnums=0
    def actor_loss(model: Actor, critic_model: Critic) -> tuple[jax.Array, jax.Array]:
        actions, log_probs = model.get_action(batch.observations[..., :model.obs_dim], key)
        q = critic_model.q_values(batch.observations, actions).mean(0)                          # [B, 1]
        return (alpha_value * log_probs - q).mean(), -log_probs.mean()

    (loss, entropy), grads = nnx.value_and_grad(actor_loss, has_aux=True, argnums=0)(actor.model, critic.model)
    actor.grad_step(grads, cfg.max_grad_norm)
    return {"actor/loss": loss, "actor/entropy": entropy}


def make_update(cfg: FastSACConfig) -> Callable[..., dict[str, jax.Array]]:
    @nnx.jit(static_argnames=("do_actor",))
    def update(critic: Network[Critic], target_critic: Network[Critic], actor: Network[Actor], alpha: Network[Alpha], observation_rms: Network[RMS] | None, buffer: JaxBuffer, key: jax.Array, do_actor: bool) -> dict[str, jax.Array]:
        sample_key, critic_key, actor_key = jax.random.split(key, 3)
        # Sample on device, n-step compress, then normalize only the two observation tensors that are used
        batch = compress_n_step(buffer.sample(sample_key, cfg.batch_size, cfg.n_step), cfg.gamma)
        batch = batch._replace(observations=batch.observations.astype(jnp.float32), next_observations=batch.next_observations.astype(jnp.float32))
        if observation_rms is not None:
            batch = batch._replace(observations=observation_rms.model.normalize(batch.observations, update=False), next_observations=observation_rms.model.normalize(batch.next_observations, update=False))
        info, next_log_probs = update_critic(critic, target_critic, actor, alpha, batch, critic_key, cfg)
        info |= update_alpha(alpha, next_log_probs, cfg.target_entropy) if cfg.use_autotune else {"alpha/value": alpha.model()}
        if do_actor:
            info |= update_actor(actor, critic, alpha, batch, actor_key, cfg)
        target_critic.soft_update()
        return info

    return update


__all__ = ["make_update", "update_actor", "update_alpha", "update_critic", "update_rms"]
