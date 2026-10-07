# Adapted from vwxyzjn/cleanrl (MIT) and leggedrobotics/rsl_rl (BSD-3-Clause), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool.py
# https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/ppo.py
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from ....buffers.on_policy.jax_buffer import JaxBuffer
from ....buffers.on_policy.types import RolloutBatch
from ....common import select_actor_observations
from ....model.jax import Network
from ..config import PPOConfig
from .network_atari import ActorCritic as AtariActorCritic
from .network_state import ActorCritic
from ....common.jax import adapt_lr, categorical_kl, diagonal_gaussian_kl


def update_ppo_minibatch(agent: Network[ActorCritic | AtariActorCritic], batch: RolloutBatch, cfg: PPOConfig) -> tuple[Network[ActorCritic | AtariActorCritic], dict[str, jax.Array]]:
    def ppo_loss(model: ActorCritic | AtariActorCritic) -> tuple[jax.Array, dict[str, jax.Array]]:
        actor = model.actor
        actor_obs = select_actor_observations(batch.observations, cfg.asymmetric_obs, actor.obs_dim)
        _, new_log_probs, entropy, metadata, values = model.get_action_and_value(
            actor_obs, batch.observations, update_rms=False, actions=batch.actions,
        )
        ratio = jnp.exp(new_log_probs - batch.old_log_probs)

        if cfg.algo == "ppo":
            pg_loss = -jnp.mean(jnp.minimum(ratio * batch.advantages, jnp.clip(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * batch.advantages))
        else:  # spo: Simple Policy Optimization, Xie et al., ICML 2025 (arXiv:2401.16025)
            pg_loss = -jnp.mean(batch.advantages * ratio - jnp.abs(batch.advantages) * jnp.square(ratio - 1.0) / (2.0 * cfg.clip_coef))

        if cfg.clip_value:
            clipped_values = batch.values + jnp.clip(values - batch.values, -cfg.clip_coef, cfg.clip_coef)
            value_loss = 0.5 * jnp.maximum(jnp.square(values - batch.returns), jnp.square(clipped_values - batch.returns)).mean()
        else:
            value_loss = 0.5 * jnp.square(values - batch.returns).mean()

        entropy_mean = jnp.mean(entropy)
        loss = pg_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_mean
        if actor.discrete:
            kl = categorical_kl(metadata.action_logits, batch.action_logits).mean()
        else:
            kl = diagonal_gaussian_kl(metadata.actions_mean, metadata.actions_std, batch.actions_mean, batch.actions_std).mean()
        return loss, {
            "training/loss": loss,
            "training/pg_loss": pg_loss,
            "training/value_loss": value_loss,
            "training/entropy": entropy_mean,
            "training/kl": kl,
            "training/clipfrac": jnp.mean(jnp.abs(ratio - 1.0) > cfg.clip_coef),
        }

    (_loss, info), grads = nnx.value_and_grad(ppo_loss, has_aux=True)(agent.model)
    if cfg.desired_kl is not None:
        hyperparams = agent.opt.opt_state.hyperparams
        hyperparams["learning_rate"].value = adapt_lr(hyperparams["learning_rate"].value, info["training/kl"], cfg.desired_kl)
    agent.grad_step(grads, cfg.max_grad_norm)
    return agent, info


def make_update_ppo(cfg: PPOConfig):
    @nnx.jit
    def update_ppo(agent: Network[ActorCritic | AtariActorCritic], buffer: JaxBuffer, last_obs: jax.Array, key: jax.Array) -> dict[str, jax.Array]:
        last_value = agent.model.critic(last_obs, update_rms=False)
        buffer = buffer.compute_returns_and_advantages(last_value, cfg.gamma, cfg.gae_lambda)
        if cfg.normalize_advantages:
            buffer = buffer.normalize_advantages()
        batches = buffer.sample(key, cfg.num_mini_batches, cfg.num_epochs)
        scan_fn = nnx.scan(lambda agent, batch: update_ppo_minibatch(agent, batch, cfg), in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        _, info = scan_fn(agent, batches)
        # Freeze the rollout's observation statistics for the next rollout
        agent.model.sync_rms()
        return jax.tree.map(jnp.mean, info)

    return update_ppo


__all__ = ["make_update_ppo", "update_ppo_minibatch"]
