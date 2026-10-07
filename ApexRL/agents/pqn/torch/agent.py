# Adapted from mttga/purejaxql PQN (Apache-2.0), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/mttga/purejaxql/blob/47af6d7b35c89ddfe633aaf7341bdb8964cb7cce/purejaxql/pqn_atari.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from gymnasium.vector import VectorEnv

from ....buffers import RolloutTransition
from ....buffers.on_policy.torch_buffer import TorchBuffer
from ....common.torch import default_device
from ....model.torch import Network
from ...base_agent import OnPolicyAgent, OnPolicySample
from ..config import PQNConfig
from .get_action import get_eval_action, get_value, sample_action_and_value
from .network import Critic
from .update import update_critic


class PQNAgent(OnPolicyAgent):
    def __init__(self, envs: VectorEnv, cfg: PQNConfig) -> None:
        super().__init__(envs, cfg)
        if len(self.observation_shape) != 4 or not self.action_is_discrete:
            raise ValueError("PQN requires discrete Atari observations with shape [F, H, W, C]")
        self.device = default_device()
        self._current_rollout_idx = 0
        self._init_state()

    def _init_state(self) -> None:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        model = Critic(self.observation_shape, self.action_dim, cfg).to(self.device)
        # Tensor lr/epsilon can be changed in place without retriggering torch.compile.
        optimizer = torch.optim.AdamW(model.parameters(), lr=torch.tensor(cfg.learning_rate, device=self.device), fused=True)
        self.critic = Network(model, optimizer, forward_name="select_action")
        self.replay_buffer = TorchBuffer(
            cfg.rollout_steps, self.observation_space, self.action_space,
            num_envs=self.num_envs, store_log_probs=False, store_metadata=False,
            device=self.device,
        )
        self._num_rollouts = max(1, cfg.total_timesteps // (self.num_envs * cfg.rollout_steps))
        self._epsilon = torch.tensor(cfg.start_e, device=self.device)
        self._pinned_obs: torch.Tensor | None = None
        self._last_obs, self._last_obs_device = None, None

    def _observations(self, observations: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(observations, np.ndarray) and self.device.type == "cuda":
            # Stage through pinned memory so the host-to-device copy is a non-blocking DMA.
            if self._pinned_obs is None or self._pinned_obs.shape != observations.shape:
                self._pinned_obs = torch.empty(observations.shape, dtype=torch.uint8, pin_memory=True)
            self._pinned_obs.numpy()[...] = observations
            observations = self._pinned_obs.to(self.device, non_blocking=True)
        return torch.as_tensor(observations, device=self.device).reshape((-1, *self.observation_shape))

    @property
    def _progress(self) -> float:
        return min(1.0, self._current_rollout_idx / self._num_rollouts)

    def get_action(self, observation: np.ndarray | torch.Tensor) -> np.ndarray:
        return get_eval_action(self.critic, self._observations(observation)).cpu().numpy()

    def get_value(self, observation: np.ndarray | torch.Tensor) -> np.ndarray:
        return get_value(self.critic, self._observations(observation)).cpu().numpy()

    def sample_action_and_value(self, observation: np.ndarray | torch.Tensor) -> OnPolicySample:
        # Cache the device copy so process_transition does not upload the same frames twice.
        self._last_obs, self._last_obs_device = observation, self._observations(observation)
        actions, values = sample_action_and_value(self.critic, self._last_obs_device, self._epsilon)
        return OnPolicySample(actions.cpu().numpy(), values.cpu().numpy())

    def process_transition(self, transition: RolloutTransition) -> None:
        if transition.observations is self._last_obs:
            transition = transition._replace(observations=self._last_obs_device)
        self.replay_buffer.add(transition)

    @property
    def can_update(self) -> bool:
        return self.replay_buffer.full

    def update(self, last_observations: np.ndarray | torch.Tensor) -> dict[str, float]:
        if not self.can_update:
            raise RuntimeError("A complete rollout is required before updating PQN.")
        cfg = self.cfg
        if cfg.anneal_lr:
            for group in self.critic.opt.param_groups:
                group["lr"].fill_(cfg.learning_rate * (1.0 - self._progress))

        last_values = get_value(self.critic, self._observations(last_observations)).reshape(self.num_envs)
        self.replay_buffer.compute_returns(last_values, cfg.gamma, cfg.q_lambda)

        infos = []
        for batch in self.replay_buffer.sample(cfg.num_minibatches, cfg.update_epochs):
            torch.compiler.cudagraph_mark_step_begin()
            info = update_critic(self.critic, batch.observations, batch.actions, batch.returns, cfg)
            # Clone out of the CUDA-graph output pool; reduce once after the loop.
            infos.append(torch.stack(tuple(info.values())).clone())
        means = torch.stack(infos).mean(0).tolist()

        self._current_rollout_idx += 1
        self._epsilon.fill_(cfg.start_e + self._progress * (cfg.end_e - cfg.start_e))
        self.replay_buffer.reset()
        return dict(zip(info.keys(), means))

    def save(self, checkpoint_dir: str | Path) -> None:
        self.critic.save(Path(checkpoint_dir) / "critic.pt")

    def load(self, checkpoint_dir: str | Path) -> None:
        self.critic.load(Path(checkpoint_dir) / "critic.pt")

    def save_buffer(self, checkpoint_dir: str | Path) -> None:
        self.replay_buffer.save(Path(checkpoint_dir) / "rollout_buffer.pt")

    def load_buffer(self, checkpoint_dir: str | Path) -> None:
        self.replay_buffer.load(Path(checkpoint_dir) / "rollout_buffer.pt")

    def save_onnx(self, path: str | Path) -> None:
        self.critic.save_onnx(path, [(1, *self.observation_shape)], output_names=("actions",))


__all__ = ["PQNAgent"]
