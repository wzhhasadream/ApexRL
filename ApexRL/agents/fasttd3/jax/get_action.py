import jax
import jax.numpy as jnp
from flax import nnx

from ....model.jax import Network, RMS
from .network import Actor


@nnx.jit
def get_eval_action(actor: Network[Actor], observation_rms: Network[RMS] | None, observations: jax.Array) -> jax.Array:
    if observation_rms is not None:
        observations = observation_rms.model.normalize(observations, update=False)
    return actor.model(observations[..., :actor.model.obs_dim])


@nnx.jit
def get_exploration_action(actor: Network[Actor], observation_rms: Network[RMS] | None, observations: jax.Array, noise_scales: jax.Array, key: jax.Array) -> jax.Array:
    # noise_scales: [num_envs, 1], fixed per episode (resampled on done), as in upstream FastTD3
    if observation_rms is not None:
        observations = observation_rms.model.normalize(observations, update=False)
    actions = actor.model(observations[..., :actor.model.obs_dim])
    actions = actions + jax.random.normal(key, actions.shape) * noise_scales
    return jnp.clip(actions, actor.model.policy.action_low.value, actor.model.policy.action_high.value)


__all__ = ["get_eval_action", "get_exploration_action"]
