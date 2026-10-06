import jax
import jax.numpy as jnp
from flax import nnx

from ....model.jax import Network
from .network import Critic


@nnx.jit
def get_action(critic: Network[Critic], observations: jax.Array, key: jax.Array, epsilon: float = 0.01) -> jax.Array:
    q_values = critic.model.q_values(critic.model(observations, training=False))
    greedy_actions = critic.model.policy.action(q_values)
    action_key, explore_key = jax.random.split(key)
    random_actions = jax.random.randint(action_key, greedy_actions.shape, 0, critic.model.action_dim)
    return jnp.where(jax.random.uniform(explore_key, greedy_actions.shape) < epsilon, random_actions, greedy_actions)


@nnx.jit
def get_eval_action(critic: Network[Critic], observations: jax.Array) -> jax.Array:
    q_values = critic.model.q_values(critic.model(observations, training=False))
    return critic.model.policy.action(q_values)
