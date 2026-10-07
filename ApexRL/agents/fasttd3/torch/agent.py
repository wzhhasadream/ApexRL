# Adapted from younggyoseo/FastTD3 (MIT), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/younggyoseo/FastTD3/tree/229ed59bbf43ea2f7a2d5d90d1076314839944d7
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
from ....model.torch import Network, RMS
from ...base_agent import OffPolicyAgent
from ..config import FastTD3Config
from .get_action import get_eval_action, get_exploration_action
from .network import Actor, Critic
from .update import make_update, update_rms


class FastTD3Agent(OffPolicyAgent):
    def __init__(self, envs: VectorEnv, cfg: FastTD3Config) -> None:
        super().__init__(envs, cfg)
        if self.action_is_discrete or len(self.observation_shape) != 1 or len(self.actor_obs_shape) != 1:
            raise ValueError("FastTD3 requires continuous actions and flat observations")
        self.device = default_device()
        # TF32 matmuls (Ampere+): large fp32 speedup, as in FastTD3
        torch.set_float32_matmul_precision("high")
        self._update_count = 0
        self._obs_cache = (None, None)
        self._init_state()

    def _init_state(self) -> None:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        self.replay_buffer = TorchBuffer(
            self.observation_space, self.action_space, max_size=cfg.buffer_size,
            num_envs=self.num_envs, device=self.device,
        )
        # Bounds come from BaseAgent: env bounds where finite, [-1, 1] otherwise
        action_low = torch.as_tensor(self.action_low, device=self.device)
        action_high = torch.as_tensor(self.action_high, device=self.device)
        # epsilon=1e-4 matches FastTD3's EmpiricalNormalization scale.
        self.observation_rms = Network(RMS(self.observation_shape[0], epsilon=1e-4, device=self.device)) if cfg.obs_normalization else None
        actor_model = Actor(self.actor_obs_shape[0], self.action_dim, cfg, action_low, action_high).to(self.device)
        critic_model = Critic(self.observation_shape[0], self.action_dim, cfg).to(self.device)
        self.actor = Network(actor_model, torch.optim.AdamW(actor_model.parameters(), lr=cfg.actor_learning_rate, weight_decay=cfg.weight_decay, fused=True))
        self.critic = Network(critic_model, torch.optim.AdamW(critic_model.parameters(), lr=cfg.critic_learning_rate, weight_decay=cfg.weight_decay, fused=True))
        self.target_critic = Network(deepcopy(critic_model), source_model=critic_model, tau=cfg.tau)
        self.target_critic.model.requires_grad_(False)
        # FastTD3 resamples each environment's exploration std at episode end.
        self.noise_scales = self._sample_noise_scales()
        self._update_fn = make_update(cfg)

    def _sample_noise_scales(self) -> torch.Tensor:
        return torch.rand((self.num_envs, 1), device=self.device) * (self.cfg.std_max - self.cfg.std_min) + self.cfg.std_min

    def _observations(self, observations: np.ndarray | torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(observations, dtype=torch.float32, device=self.device).reshape((-1, self.observation_shape[0]))

    def get_action(self, observation: np.ndarray | torch.Tensor) -> np.ndarray:
        return get_eval_action(self.actor, self.observation_rms, self._observations(observation)).cpu().numpy()

    def get_exploration_action(self, observation: np.ndarray | torch.Tensor) -> np.ndarray:
        obs = self._observations(observation)
        # Keep the device copy: the runner passes the same array to process_transition right after env.step
        self._obs_cache = (observation, obs)
        return get_exploration_action(self.actor, self.observation_rms, obs, self.noise_scales).cpu().numpy()

    def process_transition(self, transition: Transition) -> None:
        # Reuse the device copy from get_exploration_action instead of two more host->device transfers
        if self._obs_cache[0] is transition.observations:
            transition = transition._replace(observations=self._obs_cache[1])
        self.replay_buffer.add(transition)
        if self.observation_rms is not None:
            update_rms(self.observation_rms, self._observations(transition.observations))
        dones = torch.as_tensor(np.logical_or(transition.terminations, transition.truncations), device=self.device).reshape(-1, 1)
        self.noise_scales = torch.where(dones, self._sample_noise_scales(), self.noise_scales)

    @property
    def can_update(self) -> bool:
        return self.replay_buffer.size >= self.cfg.learning_starts and self.replay_buffer.can_sample(self.cfg.n_step)

    def update(self) -> dict[str, float]:
        if not self.can_update:
            raise RuntimeError("Replay buffer is not ready for an update")
        sequence = self.replay_buffer.sample(self.cfg.batch_size, self.cfg.n_step)
        self._update_count += 1
        # max-autotune uses CUDA graphs: mark a new step so the previous step's output buffers can be reused
        torch.compiler.cudagraph_mark_step_begin()
        info = self._update_fn(self.critic, self.target_critic, self.actor, self.observation_rms, sequence, self._update_count % self.cfg.policy_frequency == 0)
        self.target_critic.soft_update()
        # One device->host sync for all metrics instead of one per metric
        return dict(zip(info.keys(), torch.stack(list(info.values())).tolist()))

    def save(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.actor.save(checkpoint_dir / "actor.pt")
        self.critic.save(checkpoint_dir / "critic.pt")
        self.target_critic.save(checkpoint_dir / "target_critic.pt")
        if self.observation_rms is not None:
            self.observation_rms.save(checkpoint_dir / "observation_rms.pt")

    def load(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.actor.load(checkpoint_dir / "actor.pt")
        self.critic.load(checkpoint_dir / "critic.pt")
        self.target_critic.load(checkpoint_dir / "target_critic.pt")
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


__all__ = ["FastTD3Agent"]
