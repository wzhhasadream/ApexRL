from ....common import select_actor_observations
from .network import Actor
from ....model.jax import Network, update_reward_normalizer
from ....model.jax.backbones import DRQVisionEncoder, DRQV2VisionEncoder, VSimbaVisionEncoder
import jax
import jax.numpy as jnp
from flax import nnx
from ....common.jax.zeta_dist import sample_truncated_zeta
from .update import encoder_observation

VisionEncoder = DRQVisionEncoder | DRQV2VisionEncoder | VSimbaVisionEncoder

@nnx.jit(static_argnames=("asymmetric_obs"))
def get_exploration_action(
    actor: Network[Actor],
    asymmetric_obs: bool,
    vision_encoder: Network[VisionEncoder] | None,
    observation: jax.Array,
    repeat_n: jax.Array,
    repeat_count: jax.Array,
    cached_key: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    zeta_key, action_key = jax.random.split(key)
    actor_obs = select_actor_observations(encoder_observation(observation, vision_encoder), asymmetric_obs, actor.model.obs_dim)
    refresh = jnp.logical_or(repeat_count == 0, repeat_count >= repeat_n)
    true_action_key = jnp.where(refresh, action_key, cached_key)
    actions = actor.model.get_action(
        actor_obs,
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


@nnx.jit(static_argnames=("asymmetric_obs"))
def get_eval_action(
    actor: Network[Actor],
    vision_encoder: Network[VisionEncoder] | None,
    asymmetric_obs: bool,
    observation: jax.Array
):
    actor_obs = select_actor_observations(encoder_observation(observation, vision_encoder), asymmetric_obs, actor.model.obs_dim)

    actions = actor.model.get_deterministic_action(actor_obs)

    return actions
