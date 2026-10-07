# Adapted from vwxyzjn/cleanrl (MIT) and leggedrobotics/rsl_rl (BSD-3-Clause), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool.py
# https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/ppo.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from gymnasium.vector import VectorEnv

from ....buffers.on_policy.torch_buffer import TorchBuffer
from ....buffers.on_policy.types import PolicyMetadata, RolloutTransition
from ....common.torch import default_device
from ....model.torch import Network
from ...base_agent import OnPolicyAgent, OnPolicySample
from ..config import PPOConfig
from .get_action import get_eval_action, get_value, sample_and_value
from .network_atari import ActorCritic as AtariActorCritic
from .network_state import Actor, ActorCritic, Critic
from .update import update_ppo


class PPOAgent(OnPolicyAgent):
    """PPO for continuous (Gaussian, flat obs) and discrete (categorical, flat or Atari image obs) actions."""

    def __init__(self, envs: VectorEnv, cfg: PPOConfig) -> None:
        super().__init__(envs, cfg)
        self.device = default_device()
        # TF32 matmuls (Ampere+): large fp32 speedup
        torch.set_float32_matmul_precision("high")
        self._num_rollouts = max(1, cfg.total_timesteps // (self.num_train_env * cfg.rollout_steps))
        self._rollout_idx = 0
        self._last_obs, self._last_obs_device = None, None
        self._init_state()

    def _init_state(self) -> None:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        if self.image_obs:
            if not self.action_is_discrete or self.asymmetric_obs:
                raise ValueError("Atari PPO requires discrete actions and symmetric image observations")
            model = AtariActorCritic(self.observation_shape, self.action_dim, cfg)
        else:
            model = ActorCritic(Actor(self.actor_obs_shape, self.action_dim, self.action_is_discrete, cfg), Critic(self.critic_obs_shape, cfg))
        model = model.to(self.device)
        # Tensor lr: adapted / annealed in place on device without host syncs or recompiles
        optimizer = torch.optim.Adam(model.parameters(), lr=torch.tensor(cfg.lr, device=self.device), fused=True)
        self.agent = Network(model, optimizer)
        self.replay_buffer = TorchBuffer(cfg.rollout_steps, self.observation_space, self.action_space, self.num_train_env, device=self.device)

    def _observations(self, observations: np.ndarray | torch.Tensor) -> torch.Tensor:
        # Images stay uint8 on transfer (4x smaller); the CNN casts on device
        dtype = None if self.image_obs else torch.float32
        return torch.as_tensor(observations, dtype=dtype, device=self.device).reshape(-1, *self.critic_obs_shape)

    def get_action(self, observations: np.ndarray | torch.Tensor) -> np.ndarray:
        return get_eval_action(self.agent, self.asymmetric_obs, self._observations(observations)).cpu().numpy()

    def get_exploration_action(self, observations: np.ndarray | torch.Tensor) -> np.ndarray:
        return self.sample_action_and_value(observations).actions

    def get_value(self, observations: np.ndarray | torch.Tensor) -> np.ndarray:
        return get_value(self.agent, self._observations(observations)).cpu().numpy()

    def sample_action_and_value(self, observations: np.ndarray | torch.Tensor) -> OnPolicySample:
        # Cache the device copy so process_transition does not upload the same observations twice
        self._last_obs, self._last_obs_device = observations, self._observations(observations)
        actions, values, log_probs, metadata = sample_and_value(self.agent, self.asymmetric_obs, self._last_obs_device)
        # Policy outputs stay on device (the buffer is on device too); only actions go back to the env
        metadata = PolicyMetadata(*(None if x is None else x.clone() for x in metadata))
        return OnPolicySample(actions.cpu().numpy(), values.clone(), log_probs.clone(), metadata)

    def process_transition(self, transition: RolloutTransition) -> None:
        if transition.observations is self._last_obs:
            transition = transition._replace(observations=self._last_obs_device)
        self.replay_buffer.add(transition)

    @property
    def can_update(self) -> bool:
        return self.replay_buffer.full

    def update(self, last_observations: np.ndarray | torch.Tensor) -> dict[str, float]:
        if not self.can_update:
            raise RuntimeError("Collect a complete rollout before calling update.")
        if self.cfg.anneal_lr:
            for group in self.agent.opt.param_groups:
                group["lr"].fill_(self.cfg.lr * (1.0 - self._rollout_idx / self._num_rollouts))
        info = update_ppo(self.agent, self.replay_buffer, self._observations(last_observations), self.cfg)
        self.replay_buffer.reset()
        self._rollout_idx += 1
        # One device->host sync for all metrics
        return dict(zip(info.keys(), torch.stack(list(info.values())).tolist()))

    def save(self, checkpoint_dir: str | Path) -> None:
        self.agent.save(Path(checkpoint_dir) / "agent.pt")

    def load(self, checkpoint_dir: str | Path) -> None:
        self.agent.load(Path(checkpoint_dir) / "agent.pt")

    def save_onnx(self, path: str | Path) -> None:
        self.agent.save_onnx(path, [(1, *self.actor_obs_shape)], output_names=("actions",))


__all__ = ["PPOAgent"]
