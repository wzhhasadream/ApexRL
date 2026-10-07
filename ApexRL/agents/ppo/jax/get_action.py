# Adapted from vwxyzjn/cleanrl (MIT) and leggedrobotics/rsl_rl (BSD-3-Clause), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool.py
# https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/ppo.py
import jax
from flax import nnx

from ....common import select_actor_observations
from ....model.jax import Network
from .network_atari import ActorCritic as AtariActorCritic
from .network_state import ActorCritic


@nnx.jit(static_argnames=("asymmetric_obs",))
def get_eval_action(agent: Network[ActorCritic | AtariActorCritic], asymmetric_obs: bool, obs: jax.Array) -> jax.Array:
    return agent.model.get_mean_action(select_actor_observations(obs, asymmetric_obs, agent.model.actor.obs_dim))


@nnx.jit
def get_value(agent: Network[ActorCritic | AtariActorCritic], obs: jax.Array) -> jax.Array:
    return agent.model.critic(obs, update_rms=False)


@nnx.jit(static_argnames=("asymmetric_obs",))
def sample_and_value(agent: Network[ActorCritic | AtariActorCritic], asymmetric_obs: bool, obs: jax.Array, key: jax.Array):
    # Rollout step: also accumulates running obs statistics (frozen until sync_rms after the update)
    actor = agent.model.actor
    actions, log_probs, _, metadata, values = agent.model.get_action_and_value(
        select_actor_observations(obs, asymmetric_obs, actor.obs_dim), obs, key, update_rms=True,
    )
    return actions, values, log_probs, metadata


__all__ = ["get_eval_action", "get_value", "sample_and_value"]
