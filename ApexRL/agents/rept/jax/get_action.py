from .network import Actor, Encoder
from ....model.jax import Network, update_reward_normalizer
import jax
import jax.numpy as jnp
from flax import nnx
from ....common.jax.zeta_dist import sample_truncated_zeta


@nnx.jit
def get_exploration_action(
    encoder: Network[Encoder],
    actor: Network[Actor],
    observation: jax.Array,
    repeat_n: jax.Array,
    repeat_count: jax.Array,
    cached_key: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    zeta_key, action_key = jax.random.split(key)
    z_s = encoder.model.embed_s(observation, training=False)
    refresh = jnp.logical_or(repeat_count == 0, repeat_count >= repeat_n)
    true_action_key = jnp.where(refresh, action_key, cached_key)
    actions = actor.model.get_action(
        z_s,
        key=true_action_key,
        training=False
    )[0]
    new_repeat_n = jnp.where(
        refresh,
        sample_truncated_zeta(zeta_key),
        repeat_n,
    )
    new_repeat_count = jnp.where(refresh, 1, repeat_count + 1)
    return true_action_key, actions, new_repeat_n, new_repeat_count


@nnx.jit
def get_eval_action(
    encoder: Network[Encoder],
    actor: Network[Actor],
    observation: jax.Array,
) -> jax.Array:
    z_s = encoder.model.embed_s(observation, training=False)
    return actor.model.get_deterministic_action(z_s)
