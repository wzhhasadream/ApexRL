from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from gymnasium.vector import VectorEnv
from torch import nn

from ....buffers.off_policy import Transition
from ....buffers.off_policy.numpy_lazy_frame_buffer import NumpyLazyFrameBuffer
from ....buffers.off_policy.types import SequenceBatch
from ....common.torch import default_device
from ....model.torch import Alpha, Network, RewardNormalizer, update_reward_normalizer
from ....model.torch.backbones import VSimbaVisionEncoder
from ...base_agent import OffPolicyAgent
from ..config import VSimbaConfig
from .get_action import get_eval_action, get_exploration_action
from .network import Actor, Critic
from .update import update


class VSimbaAgent(OffPolicyAgent):
    def __init__(self, envs: VectorEnv, cfg: VSimbaConfig):
        super().__init__(envs, cfg)
        if len(self.observation_shape) != 4 or self.action_is_discrete or self.num_envs != 1:
            raise ValueError("V-Simba requires one pixel environment with continuous actions")
        if not (np.allclose(self.action_space.low, -1) and np.allclose(self.action_space.high, 1)):
            raise ValueError("V-Simba requires actions rescaled to [-1, 1]")

        self.cfg.target_entropy = self.cfg.temp_target_entropy_coef * self.action_dim
        self.device = default_device()
        self._init_state()

    def _init_state(self) -> None:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        self.replay_buffer = NumpyLazyFrameBuffer(
            self.observation_space, self.action_space, max_size=cfg.buffer_size,
            linear_decay_step=cfg.decay_step, max_n_step=cfg.n_step,
        )
        encoder_model = VSimbaVisionEncoder(
            self.observation_shape, num_channels=cfg.encoder_num_channels,
            num_blocks=cfg.encoder_num_blocks, conv_kernel_size=cfg.encoder_conv_kernel_size,
        ).to(self.device)
        input_dim = encoder_model.get_output_dim()
        actor_model = Actor(input_dim, self.action_dim, cfg.actor_num_blocks, cfg.actor_hidden_dim).to(self.device)
        critic_model = Critic(
            input_dim, self.action_dim, num_qs=2 if cfg.critic_use_cdq else 1,
            num_blocks=cfg.critic_num_blocks, hidden_dim=cfg.critic_hidden_dim,
            action_embed_dim=cfg.critic_action_embed_dim, c_shift=cfg.critic_c_shift,
            num_bins=cfg.critic_num_bins, min_v=cfg.critic_min_v, max_v=cfg.critic_max_v,
        ).to(self.device)
        alpha_model = Alpha(cfg.temp_initial_value).to(self.device)

        self.encoder = Network(
            encoder_model, torch.optim.AdamW(encoder_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, fused=True),
        )
        self.actor = Network(
            actor_model, torch.optim.AdamW(actor_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, fused=True),
            forward_name="get_mean_action",
        )
        self.critic = Network(
            critic_model, torch.optim.AdamW(critic_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, fused=True),
        )
        self.target_critic = Network(deepcopy(critic_model), source_model=critic_model, tau=cfg.target_tau)
        self.target_critic.model.requires_grad_(False)
        self.alpha = Network(
            alpha_model, torch.optim.AdamW(alpha_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, fused=True),
        )

        self.reward_normalizer = None
        if cfg.normalize_rewards:
            normalizer = RewardNormalizer(
                self.num_envs, cfg.gamma, cfg.normalized_g_max, device=self.device,
            )
            normalizer.g_rms.count.fill_(1e-4)
            self.reward_normalizer = Network(normalizer)

    def get_action(self, observation: np.ndarray | torch.Tensor) -> np.ndarray:
        observations = torch.as_tensor(observation, device=self.device).reshape((-1, *self.observation_shape))
        return get_eval_action(self.encoder, self.actor, observations).cpu().numpy()

    def get_exploration_action(self, observation: np.ndarray | torch.Tensor) -> np.ndarray:
        observations = torch.as_tensor(observation, device=self.device).reshape((-1, *self.observation_shape))
        return get_exploration_action(self.encoder, self.actor, observations).cpu().numpy()

    def process_transition(self, transition: Transition) -> None:
        if self.reward_normalizer is not None:
            update_reward_normalizer(
                self.reward_normalizer,
                torch.as_tensor(transition.rewards, device=self.device),
                torch.as_tensor(transition.terminations, device=self.device),
                torch.as_tensor(transition.truncations, device=self.device),
            )
        self.replay_buffer.add(transition)

    @property
    def can_update(self) -> bool:
        return self.replay_buffer.size >= self.cfg.learning_starts and self.replay_buffer.can_sample(self.cfg.n_step)

    def update(self) -> dict[str, float]:
        if not self.can_update:
            raise RuntimeError("Replay buffer is not ready for an update")
        sequence = self.replay_buffer.sample(self.cfg.batch_size, self.cfg.n_step)
        sequence = SequenceBatch(*(torch.as_tensor(value, device=self.device) for value in sequence))
        info = update(self.cfg, self.encoder, self.actor, self.critic, self.target_critic, self.alpha, self.reward_normalizer, sequence)
        return {name: float(value) for name, value in info.items()}

    def save(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.encoder.save(checkpoint_dir / "encoder.pt")
        self.actor.save(checkpoint_dir / "actor.pt")
        self.critic.save(checkpoint_dir / "critic.pt")
        self.target_critic.save(checkpoint_dir / "target_critic.pt")
        self.alpha.save(checkpoint_dir / "alpha.pt")
        if self.reward_normalizer is not None:
            self.reward_normalizer.save(checkpoint_dir / "reward_normalizer.pt")

    def load(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.encoder.load(checkpoint_dir / "encoder.pt")
        self.actor.load(checkpoint_dir / "actor.pt")
        self.critic.load(checkpoint_dir / "critic.pt")
        self.target_critic.load(checkpoint_dir / "target_critic.pt")
        self.alpha.load(checkpoint_dir / "alpha.pt")
        if self.reward_normalizer is not None:
            self.reward_normalizer.load(checkpoint_dir / "reward_normalizer.pt")

    def save_encoder(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.encoder.save(checkpoint_dir / "encoder.pt")

    def load_encoder(self, checkpoint_dir: str | Path) -> None:
        self.encoder.load(Path(checkpoint_dir) / "encoder.pt")

    def save_buffer(self, checkpoint_dir: str | Path) -> None:
        self.replay_buffer.save(Path(checkpoint_dir) / "replay_buffer")

    def load_buffer(self, checkpoint_dir: str | Path) -> None:
        self.replay_buffer.load(Path(checkpoint_dir) / "replay_buffer")

    def save_onnx(self, onnx_dir: str | Path) -> None:
        policy = Network(nn.Sequential(self.encoder, self.actor))
        policy.save_onnx(Path(onnx_dir) / "policy.onnx", [(1, *self.observation_shape)])


__all__ = ["VSimbaAgent"]
