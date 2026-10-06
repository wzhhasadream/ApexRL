import jax
import jax.numpy as jnp
from flax import nnx

from ....model.jax import Network
from .network import Critic


@nnx.jit
def get_eval_action(critic: Network[Critic], observations: jax.Array) -> jax.Array:
    q_values = critic.model(observations, training=False)
    return critic.model.policy.action(q_values)



@nnx.jit
def get_value(
    critic: Network[Critic],
    observations: jax.Array
) -> jax.Array:
    q_values = critic.model(observations, training=False)
    return q_values.max(axis=-1, keepdims=True)


@nnx.jit
def sample_action_and_value(
    critic: Network[Critic],
    observations: jax.Array,
    key: jax.Array,
    epsilon: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """One dispatch per env step: split the key, act, and return the device copy of obs."""
    # training=False: rollout batches must not update BatchNorm statistics.
    q_values = critic.model(observations, training=False)
    greedy_actions = critic.model.policy.action(q_values)
    key, action_key, explore_key = jax.random.split(key, 3)
    random_actions = jax.random.randint(
        action_key, greedy_actions.shape, 0, critic.model.action_dim
    )
    explore = jax.random.uniform(explore_key, greedy_actions.shape) < epsilon
    actions = jnp.where(explore, random_actions, greedy_actions)
    values = q_values.max(axis=-1, keepdims=True)
    return actions, values, key, observations
