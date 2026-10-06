from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from gymnasium.vector import VectorEnv
from torch import nn

from ....buffers.off_policy import Transition
from ....buffers.off_policy.torch_buffer import TorchBuffer
from ....common.torch import default_device
from ....model.torch import Alpha, Network, RMS
from ...base_agent import OffPolicyAgent
from ..config import FastSACConfig
from .get_action import get_eval_action, get_exploration_action
from .network import Actor, Critic
from .update import make_update, update_rms


class FastSACAgent(OffPolicyAgent):
    def __init__(self, envs: VectorEnv, cfg: FastSACConfig) -> None:
        super().__init__(envs, cfg)
        if self.action_is_discrete or len(self.observation_shape) != 1 or len(self.actor_obs_shape) != 1:
            raise ValueError("FastSAC requires continuous actions and flat observations")
        if not np.all(np.isfinite(self.action_space.low)) or not np.all(np.isfinite(self.action_space.high)):
            raise ValueError("FastSAC requires finite action bounds")
        self.cfg.target_entropy = -self.action_dim * cfg.target_entropy_ratio
        self.device = default_device()
        self._update_count = 0
        self._init_state()

    def _init_state(self) -> None:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        self.replay_buffer = TorchBuffer(
            self.observation_space, self.action_space, max_size=cfg.buffer_size,
            num_envs=self.num_envs, device=self.device,
        )
        action_low = torch.as_tensor(self.action_space.low, dtype=torch.float32, device=self.device)
        action_high = torch.as_tensor(self.action_space.high, dtype=torch.float32, device=self.device)
        self.observation_rms = Network(RMS(self.observation_shape[0], device=self.device)) if cfg.obs_normalization else None
        actor_model = Actor(self.actor_obs_shape[0], self.action_dim, cfg, action_low, action_high).to(self.device)
        critic_model = Critic(self.observation_shape[0], self.action_dim, cfg).to(self.device)
        self.actor = Network(actor_model, torch.optim.AdamW(actor_model.parameters(), lr=cfg.actor_learning_rate, weight_decay=cfg.weight_decay, fused=True), forward_name="get_mean_action")
        self.critic = Network(critic_model, torch.optim.AdamW(critic_model.parameters(), lr=cfg.critic_learning_rate, weight_decay=cfg.weight_decay, fused=True))
        self.target_critic = Network(deepcopy(critic_model), source_model=critic_model, tau=cfg.tau)
        self.target_critic.model.requires_grad_(False)
        alpha_model = Alpha(cfg.alpha_init).to(self.device)
        self.alpha = Network(alpha_model, torch.optim.Adam(alpha_model.parameters(), lr=cfg.alpha_learning_rate, fused=True))
        self._get_eval_fn = get_eval_action
        self._get_exploration_fn = get_exploration_action
        self._update_rms_fn = update_rms if self.observation_rms is not None else None
        self._update_fn = make_update(cfg)

    def _observations(self, observations: np.ndarray | torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(observations, dtype=torch.float32, device=self.device).reshape((-1, self.observation_shape[0]))

    def get_action(self, observation: np.ndarray | torch.Tensor) -> np.ndarray:
        self.actor.model.eval()
        return self._get_eval_fn(self.actor, self.observation_rms, self._observations(observation), self.asymmetric_obs, self.actor_obs_shape[0]).cpu().numpy()

    def get_exploration_action(self, observation: np.ndarray | torch.Tensor) -> np.ndarray:
        self.actor.model.eval()
        return self._get_exploration_fn(self.actor, self.observation_rms, self._observations(observation), self.asymmetric_obs, self.actor_obs_shape[0]).cpu().numpy()

    def process_transition(self, transition: Transition) -> None:
        self.replay_buffer.add(transition)
        if self._update_rms_fn is not None:
            self._update_rms_fn(self.observation_rms, torch.as_tensor(transition.observations, dtype=torch.float32, device=self.device))

    @property
    def can_update(self) -> bool:
        return self.replay_buffer.size >= self.cfg.learning_starts and self.replay_buffer.can_sample(self.cfg.n_step)

    def update(self) -> dict[str, float]:
        if not self.can_update:
            raise RuntimeError("Replay buffer is not ready for an update")
        sequence = self.replay_buffer.sample(self.cfg.batch_size, self.cfg.n_step)
        self._update_count += 1
        info = self._update_fn(self.critic, self.target_critic, self.actor, self.alpha, self.observation_rms, sequence, self._update_count % self.cfg.policy_frequency == 0)
        self.target_critic.soft_update()
        return {name: float(value) for name, value in info.items()}

    def save(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.actor.save(checkpoint_dir / "actor.pt")
        self.critic.save(checkpoint_dir / "critic.pt")
        self.target_critic.save(checkpoint_dir / "target_critic.pt")
        self.alpha.save(checkpoint_dir / "alpha.pt")
        if self.observation_rms is not None:
            self.observation_rms.save(checkpoint_dir / "observation_rms.pt")

    def load(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.actor.load(checkpoint_dir / "actor.pt")
        self.critic.load(checkpoint_dir / "critic.pt")
        self.target_critic.load(checkpoint_dir / "target_critic.pt")
        self.alpha.load(checkpoint_dir / "alpha.pt")
        if self.observation_rms is not None:
            self.observation_rms.load(checkpoint_dir / "observation_rms.pt")

    def save_buffer(self, checkpoint_dir: str | Path) -> None:
        self.replay_buffer.save(Path(checkpoint_dir) / "replay_buffer.pt")

    def load_buffer(self, checkpoint_dir: str | Path) -> None:
        self.replay_buffer.load(Path(checkpoint_dir) / "replay_buffer.pt")

    def save_onnx(self, path: str | Path) -> None:
        layers: list[nn.Module] = []
        if self.observation_rms is not None:
            rms = RMS(self.actor_obs_shape[0], self.observation_rms.model.epsilon, device=self.device)
            rms.load_state_dict({"mean": self.observation_rms.model.mean[: self.actor_obs_shape[0]], "var": self.observation_rms.model.var[: self.actor_obs_shape[0]], "count": self.observation_rms.model.count})
            layers.append(rms)
        layers.append(self.actor)
        Network(nn.Sequential(*layers)).save_onnx(path, [(1, self.actor_obs_shape[0])], output_names=("actions",))


__all__ = ["FastSACAgent"]
