# Adapted from mttga/purejaxql PQN (Apache-2.0), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/mttga/purejaxql/blob/47af6d7b35c89ddfe633aaf7341bdb8964cb7cce/purejaxql/pqn_atari.py
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from ....buffers.on_policy.types import RolloutBatch
from ....buffers.on_policy.jax_buffer import JaxBuffer
from ....model.jax import Network
from .network import Critic


def update_critic(
    critic: Network[Critic],
    batch: RolloutBatch,
    cfg,
) -> tuple[Network[Critic], dict[str, jax.Array]]:
    """Apply one Q regression step to one rollout mini-batch."""

    actions = batch.actions.astype(jnp.int32)
    targets = jax.lax.stop_gradient(batch.returns.astype(jnp.float32).reshape((-1,)))

    def loss_fn(model: Critic):
        q_values = model(batch.observations, training=True)
        selected_q = jnp.take_along_axis(q_values, actions, axis=1).reshape((-1,))
        td_error = selected_q - targets
        loss = 0.5 * jnp.mean(jnp.square(td_error))
        return loss, {
            "training/q_loss": loss,
            "training/q_values": jnp.mean(selected_q),
            "training/targets": jnp.mean(targets),
            "training/td_abs": jnp.mean(jnp.abs(td_error)),
        }

    (_loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(critic.model)
    critic.grad_step(grads, cfg.max_grad_norm)
    return critic, info


def make_update(cfg):
    @nnx.jit
    def update(
        critic: Network[Critic],
        buffer: JaxBuffer,
        last_observations: jax.Array,
        key: jax.Array
    ) -> dict[str, jax.Array]:
        last_q_values = critic.model(last_observations, training=False).max(axis=-1, keepdims=True)
        buffer = buffer.compute_returns(last_q_values, cfg.gamma, cfg.q_lambda)
        batches = buffer.sample(key, cfg.num_minibatches, cfg.update_epochs)

        def scan_minibatch(
            critic: Network[Critic],
            batch: RolloutBatch,
        ):
            return update_critic(critic, batch, cfg)

        scan_fn = nnx.scan(
            scan_minibatch,
            in_axes=(nnx.Carry, 0),
            out_axes=(nnx.Carry, 0),
        )
        _, info = scan_fn(critic, batches)
        return jax.tree.map(jnp.mean, info)

    return update
