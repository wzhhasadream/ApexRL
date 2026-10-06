import jax
import jax.numpy as jnp
from flax import nnx

from ....common import select_actor_observations
from ....model.jax import Network, RMS
from .network import Actor


@nnx.jit(static_argnames=("asymmetric_obs", "actor_obs_dim"))
def get_exploration_action(
    actor: Network[Actor], actor_rms: Network[RMS] | None,
    asymmetric_obs: bool, actor_obs_dim: int, observations: jax.Array, key: jax.Array,
    std_min: float, std_max: float,
) -> jax.Array:
    if actor_rms is not None:
        observations = actor_rms.model.normalize(observations, update=False)
    observations = select_actor_observations(observations, asymmetric_obs, actor_obs_dim)
    actions = actor.model(observations)
    scale_key, noise_key = jax.random.split(key)
    scale = jax.random.uniform(scale_key, (actions.shape[0], 1), minval=std_min, maxval=std_max)
    actions = actions + jax.random.normal(noise_key, actions.shape) * scale
    return jnp.clip(actions, actor.model.policy.action_low.value, actor.model.policy.action_high.value)


@nnx.jit(static_argnames=("asymmetric_obs", "actor_obs_dim"))
def get_eval_action(
    actor: Network[Actor], actor_rms: Network[RMS] | None,
    asymmetric_obs: bool, actor_obs_dim: int, observations: jax.Array,
) -> jax.Array:
    if actor_rms is not None:
        observations = actor_rms.model.normalize(observations, update=False)
    observations = select_actor_observations(observations, asymmetric_obs, actor_obs_dim)
    return actor.model(observations)
