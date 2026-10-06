import jax
from flax import nnx

from ....common import select_actor_observations
from ....model.jax import Network, RMS
from .network import Actor


@nnx.jit(static_argnames=("asymmetric_obs", "actor_obs_dim"))
def get_exploration_action(
    actor: Network[Actor], observation_rms: Network[RMS] | None,
    asymmetric_obs: bool, actor_obs_dim: int, observations: jax.Array, key: jax.Array,
) -> jax.Array:
    if observation_rms is not None:
        observations = observation_rms.model.normalize(observations, update=False)
    observations = select_actor_observations(observations, asymmetric_obs, actor_obs_dim)
    return actor.model.get_action(observations, key)[0]


@nnx.jit(static_argnames=("asymmetric_obs", "actor_obs_dim"))
def get_eval_action(
    actor: Network[Actor], observation_rms: Network[RMS] | None,
    asymmetric_obs: bool, actor_obs_dim: int, observations: jax.Array,
) -> jax.Array:
    if observation_rms is not None:
        observations = observation_rms.model.normalize(observations, update=False)
    observations = select_actor_observations(observations, asymmetric_obs, actor_obs_dim)
    return actor.model.get_mean_action(observations)
