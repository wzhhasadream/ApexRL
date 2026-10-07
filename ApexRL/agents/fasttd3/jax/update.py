# Adapted from younggyoseo/FastTD3 (MIT), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/younggyoseo/FastTD3/tree/229ed59bbf43ea2f7a2d5d90d1076314839944d7
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from ....buffers.off_policy import Batch, compress_n_step
from ....buffers.off_policy.jax_buffer import JaxBuffer
from ....model.jax import Network, RMS
from ..config import FastTD3Config
from .network import Actor, Critic


@nnx.jit
def update_rms(network: Network[RMS], observations: jax.Array) -> None:
    network.model.update(observations)


def update_critic(critic: Network[Critic], target_critic: Network[Critic], actor: Network[Actor], batch: Batch, key: jax.Array, cfg: FastTD3Config) -> dict[str, jax.Array]:
    bins = critic.model.dist.bins
    # Target policy smoothing + C51 projection of both target heads
    noise = jnp.clip(jax.random.normal(key, batch.actions.shape) * cfg.policy_noise, -cfg.noise_clip, cfg.noise_clip)
    next_actions = jnp.clip(actor.model(batch.next_observations[..., :actor.model.obs_dim]) + noise, actor.model.policy.action_low.value, actor.model.policy.action_high.value)
    target_logits = target_critic.model(batch.next_observations, next_actions)                   # [2, B, A]
    target_values = batch.rewards + batch.discounts * (1.0 - batch.dones) * bins                 # [B, A]
    target_probs = jax.vmap(critic.model.dist.target_probs, in_axes=(0, None))(target_logits, target_values)
    if cfg.use_cdq:
        # FastTD3 projects both heads, then keeps the distribution with the smaller mean.
        projected_q = (target_probs * bins).sum(-1, keepdims=True)                              # [2, B, 1]
        target_probs = jnp.broadcast_to(jnp.where(projected_q[0] < projected_q[1], target_probs[0], target_probs[1]), target_probs.shape)
    target_probs = jax.lax.stop_gradient(target_probs)

    def critic_loss(model: Critic) -> tuple[jax.Array, jax.Array]:
        logits = model(batch.observations, batch.actions)
        # Sum over the two heads (qf1_loss + qf2_loss), mean over batch
        return -(target_probs * jax.nn.log_softmax(logits, -1)).sum(-1).mean(-1).sum(), logits

    (loss, logits), grads = nnx.value_and_grad(critic_loss, has_aux=True)(critic.model)
    critic.grad_step(grads, cfg.max_grad_norm)
    return {"critic/loss": loss, "critic/mean_q": critic.model.dist.q_values(logits).mean()}


def update_actor(actor: Network[Actor], critic: Network[Critic], batch: Batch, cfg: FastTD3Config) -> dict[str, jax.Array]:
    # critic is passed as an argument (not closed over) so nnx can trace it; grads only w.r.t. argnums=0
    def actor_loss(model: Actor, critic_model: Critic) -> tuple[jax.Array, jax.Array]:
        q = critic_model.q_values(batch.observations, model(batch.observations[..., :model.obs_dim]))   # [2, B, 1]
        q = q.min(0) if cfg.use_cdq else q.mean(0)
        return -q.mean(), q.mean()

    (loss, mean_q), grads = nnx.value_and_grad(actor_loss, has_aux=True, argnums=0)(actor.model, critic.model)
    actor.grad_step(grads, cfg.max_grad_norm)
    return {"actor/loss": loss, "actor/mean_q": mean_q}


def make_update(cfg: FastTD3Config) -> Callable[..., dict[str, jax.Array]]:
    @nnx.jit(static_argnames=("do_actor",))
    def update(critic: Network[Critic], target_critic: Network[Critic], actor: Network[Actor], observation_rms: Network[RMS] | None, buffer: JaxBuffer, key: jax.Array, do_actor: bool) -> dict[str, jax.Array]:
        sample_key, critic_key = jax.random.split(key)
        # Sample on device, n-step compress, then normalize only the two observation tensors that are used
        batch = compress_n_step(buffer.sample(sample_key, cfg.batch_size, cfg.n_step), cfg.gamma)
        batch = batch._replace(observations=batch.observations.astype(jnp.float32), next_observations=batch.next_observations.astype(jnp.float32))
        if observation_rms is not None:
            batch = batch._replace(observations=observation_rms.model.normalize(batch.observations, update=False), next_observations=observation_rms.model.normalize(batch.next_observations, update=False))
        info = update_critic(critic, target_critic, actor, batch, critic_key, cfg)
        if do_actor:
            info |= update_actor(actor, critic, batch, cfg)
        target_critic.soft_update()
        return info

    return update


__all__ = ["make_update", "update_actor", "update_critic", "update_rms"]
