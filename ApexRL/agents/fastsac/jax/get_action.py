# Adapted from amazon-far/holosoma FastSAC (Apache-2.0), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/amazon-far/holosoma/tree/d18d6cc50f872c15e904a22ceac22313cec955c8/src/holosoma/holosoma/agents/fast_sac
import jax
from flax import nnx

from ....model.jax import Network, RMS
from .network import Actor


@nnx.jit
def get_eval_action(actor: Network[Actor], observation_rms: Network[RMS] | None, observations: jax.Array) -> jax.Array:
    if observation_rms is not None:
        observations = observation_rms.model.normalize(observations, update=False)
    return actor.model.get_mean_action(observations[..., :actor.model.obs_dim])


@nnx.jit
def get_exploration_action(actor: Network[Actor], observation_rms: Network[RMS] | None, observations: jax.Array, key: jax.Array) -> jax.Array:
    if observation_rms is not None:
        observations = observation_rms.model.normalize(observations, update=False)
    return actor.model.get_action(observations[..., :actor.model.obs_dim], key)[0]


__all__ = ["get_eval_action", "get_exploration_action"]
