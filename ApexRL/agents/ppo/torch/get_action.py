# Adapted from vwxyzjn/cleanrl (MIT) and leggedrobotics/rsl_rl (BSD-3-Clause), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool.py
# https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/ppo.py
import torch

from ....common import select_actor_observations
from ....model.torch import Network
from .network import ActorCritic


@torch.no_grad()
@torch.compile(mode="max-autotune")
def get_eval_action(agent: Network[ActorCritic], asymmetric_obs: bool, observations: torch.Tensor) -> torch.Tensor:
    return agent(select_actor_observations(observations, asymmetric_obs, agent.model.actor.obs_dim))


@torch.no_grad()
@torch.compile(mode="max-autotune")
def get_value(agent: Network[ActorCritic], observations: torch.Tensor) -> torch.Tensor:
    return agent.model.critic(observations, update_rms=False)


@torch.no_grad()
@torch.compile(mode="max-autotune")
def sample_and_value(agent: Network[ActorCritic], asymmetric_obs: bool, observations: torch.Tensor):
    # Rollout step: also accumulates running obs statistics (frozen until sync_rms after the update)
    actor = agent.model.actor
    actions, log_probs, _, metadata = actor.get_action(select_actor_observations(observations, asymmetric_obs, actor.obs_dim), update_rms=True)
    values = agent.model.critic(observations, update_rms=True)
    return actions, values, log_probs, metadata


__all__ = ["get_eval_action", "get_value", "sample_and_value"]
