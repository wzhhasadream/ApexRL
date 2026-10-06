import jax
from flax import nnx

from ....model.jax import Network
from ....model.jax.backbones import VSimbaVisionEncoder
from .network import Actor


@nnx.jit
def get_eval_action(encoder: Network[VSimbaVisionEncoder], actor: Network[Actor], observations: jax.Array) -> jax.Array:
    return actor.model.get_mean_action(encoder.model(observations))


@nnx.jit
def get_exploration_action(encoder: Network[VSimbaVisionEncoder], actor: Network[Actor], observations: jax.Array, key: jax.Array) -> jax.Array:
    actions, _ = actor.model.get_action(encoder.model(observations), key)
    return actions
