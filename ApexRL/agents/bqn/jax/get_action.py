import jax
import jax.numpy as jnp
from flax import nnx

from ....model.jax import Network
from .network import Critic


@nnx.jit
def get_eval_action(critic: Network[Critic], observations: jax.Array) -> jax.Array:
    return critic.model.select_action(observations, training=False)


@nnx.jit
def get_exploration_action(critic: Network[Critic], observations: jax.Array, key: jax.Array, epsilon: float) -> jax.Array:
    actions = critic.model.select_action(observations, training=False)
    action_key, explore_key = jax.random.split(key)
    random_actions = jax.random.randint(action_key, actions.shape, 0, critic.model.action_dim)
    explore = jax.random.uniform(explore_key, actions.shape) < epsilon
    return jnp.where(explore, random_actions, actions)
