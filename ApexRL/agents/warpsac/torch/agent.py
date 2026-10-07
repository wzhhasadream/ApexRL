from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from gymnasium.vector import VectorEnv
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from ....buffers.off_policy import SequenceBatch, Transition
from ....buffers.off_policy.torch_buffer import TorchBuffer
from ..config import WarpSACConfig
from ....model.torch import (
    Alpha,
    Network,
    RewardNormalizer,
)
from .network import FlashSACActor, FlashSACDoubleCritic
from .get_action import (
    get_eval_action,
    get_exploration_action,
    update_reward_normalizer,
)
from .update import update_warpsac
from ...base_agent import OffPolicyAgent

class WarpSACAgent(OffPolicyAgent):
    def __init__(self, envs: VectorEnv, cfg: WarpSACConfig) -> None:
        super().__init__(envs, cfg)
        if self.action_is_discrete or len(self.observation_shape) != 1 or len(self.actor_obs_shape) != 1:
            raise ValueError("WarpSAC requires continuous actions and flat observations")
        self.learner_device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_float32_matmul_precision("high")
        torch.manual_seed(cfg.seed)
        self._init_train_state()
        self._update_fn = update_warpsac
        self.repeat_count = torch.tensor(0, device=self.learner_device)
        self.repeat_n = torch.tensor(0, device=self.learner_device)
        self.cached_noise = torch.randn((self.num_train_env, self.action_dim), device=self.learner_device)
        self.critic_grad_updates = 0
        self._obs_cache = (None, None)

    def _init_train_state(self) -> None:
        num_updates = max(1, int(self.cfg.total_timesteps / self.num_train_env * self.cfg.grad_step_per_interaction_step))
        self.replay_buffer = TorchBuffer(
            observation_space=self.observation_space,
            action_space=self.action_space,
            max_size=self.cfg.buffer_size,
            linear_decay_step=self.cfg.decay_step,
            num_envs=self.num_train_env,
            use_approximate_sampling=self.cfg.buffer_device == "cpu",
            device=self.cfg.buffer_device,
        )
        actor_model = FlashSACActor(
            self.actor_obs_shape,
            self.action_dim,
            hidden_dim=self.cfg.actor_hidden_dim,
            num_blocks=self.cfg.actor_num_blocks,
            action_low=torch.as_tensor(self.action_low, device=self.learner_device),
            action_high=torch.as_tensor(self.action_high, device=self.learner_device),
            use_bias=self.cfg.use_bias,
        ).to(device=self.learner_device)
        critic_model = FlashSACDoubleCritic(
            self.critic_obs_shape,
            self.action_dim,
            num_q=self.cfg.num_q,
            hidden_dim=self.cfg.critic_hidden_dim,
            num_blocks=self.cfg.critic_num_blocks,
            num_head=self.cfg.num_head,
            dist_type=(
                "scalar"
                if self.cfg.num_head == 1
                else self.cfg.dist_type
            ),
            use_bias=self.cfg.use_bias,
        ).to(device=self.learner_device)
        use_fused_adam = torch.cuda.is_available()
        actor_optimizer = Adam(
            actor_model.parameters(), lr=self.cfg.policy_lr, fused=use_fused_adam
        )
        critic_optimizer = Adam(
            critic_model.parameters(), lr=self.cfg.q_lr, fused=use_fused_adam
        )
        alpha_model = Alpha().to(self.learner_device)
        alpha_optimizer = Adam(
            alpha_model.parameters(), lr=self.cfg.policy_lr, fused=use_fused_adam
        )
        self.actor = Network(
            actor_model,
            actor_optimizer,
            scheduler=CosineAnnealingLR(
                actor_optimizer, num_updates, eta_min=self.cfg.end_lr
            ),
            forward_name="get_mean_action",  # for onnx export
        )
        self.critic = Network(
            critic_model,
            critic_optimizer,
            scheduler=CosineAnnealingLR(
                critic_optimizer, num_updates, eta_min=self.cfg.end_lr
            ),
        )
        self.alpha = Network(
            alpha_model,
            alpha_optimizer,
            scheduler=CosineAnnealingLR(
                alpha_optimizer, num_updates, eta_min=self.cfg.end_lr
            ),
        )
        if self.cfg.actor_normalize_parameters:
            self.actor.project_param()
        if self.cfg.critic_normalize_parameters:
            self.critic.project_param()
        target_critic_model = deepcopy(critic_model)
        target_critic_model.requires_grad_(False)
        self.target_critic = Network(
            target_critic_model, source_model=critic_model, tau=self.cfg.tau
        )
        self.reward_normalizer = (
            Network(RewardNormalizer(self.num_train_env, self.cfg.gamma, device=self.learner_device))
            if self.cfg.normalize_rewards
            else None
        )

    def _observations(self, observations: np.ndarray | torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(
            observations, device=self.learner_device, dtype=torch.float32
        ).reshape(-1, *self.critic_obs_shape)

    def get_action(self, observations: np.ndarray | torch.Tensor) -> np.ndarray:
        actions = get_eval_action(
            self.actor, self.asymmetric_obs, self._observations(observations)
        )
        return actions.cpu().numpy()

    def get_exploration_action(
        self, observations: np.ndarray | torch.Tensor
    ) -> np.ndarray:
        obs = self._observations(observations)
        # Reuse this device copy when the runner stores the same observation.
        self._obs_cache = (observations, obs)
        noise = torch.randn((self.num_train_env, self.action_dim), device=self.learner_device)
        cached_noise, actions, repeat_n, repeat_count = get_exploration_action(
            self.actor,
            self.asymmetric_obs,
            obs,
            self.repeat_n,
            self.repeat_count,
            self.cached_noise,
            noise,
        )

        self.cached_noise = cached_noise.clone()
        self.repeat_n = repeat_n.clone()
        self.repeat_count = repeat_count.clone()

        return actions.cpu().numpy()

    def process_transition(self, transition: Transition) -> None:
        if self._obs_cache[0] is transition.observations and self._obs_cache[1].device == self.replay_buffer.observations.device:
            transition = transition._replace(observations=self._obs_cache[1])
        update_reward_normalizer(
            self.reward_normalizer,
            torch.as_tensor(transition.rewards, device=self.learner_device),
            torch.as_tensor(transition.terminations, device=self.learner_device),
            torch.as_tensor(transition.truncations, device=self.learner_device),
        )
        self.replay_buffer.add(transition)
        self._obs_cache = (None, None)

    @property
    def can_update(self) -> bool:
        return self.replay_buffer.size >= self.cfg.learning_starts and self.replay_buffer.can_sample(self.cfg.n_step)

    def update(self) -> dict[str, float]:
        if not self.can_update:
            raise RuntimeError("Replay buffer is not ready for an update")
        sequence = self.replay_buffer.sample(self.cfg.batch_size, self.cfg.n_step)
        if self.cfg.buffer_device != self.learner_device:
            sequence = SequenceBatch(*(value.to(self.learner_device) for value in sequence))
        do_policy = self.critic_grad_updates % self.cfg.policy_frequency == 0
        self.critic_grad_updates += 1
        do_target = self.critic_grad_updates % self.cfg.target_frequency == 0
        torch.compiler.cudagraph_mark_step_begin()
        info = self._update_fn(self.critic, self.actor, self.alpha, self.target_critic, self.reward_normalizer, do_policy, sequence, self.cfg)
        # Keep target mutation outside the compiled backward graphs.
        if do_target:
            self.target_critic.soft_update()
        return dict(zip(info.keys(), torch.stack(list(info.values())).tolist()))

    def save(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.actor.save(checkpoint_dir / "actor.pt")
        self.critic.save(checkpoint_dir / "critic.pt")
        self.target_critic.save(checkpoint_dir / "target_critic.pt")
        self.alpha.save(checkpoint_dir / "alpha.pt")
        if self.reward_normalizer is not None:
            self.reward_normalizer.save(checkpoint_dir / "reward_normalizer.pt")

    def load(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.actor.load(checkpoint_dir / "actor.pt")
        self.critic.load(checkpoint_dir / "critic.pt")
        self.target_critic.load(checkpoint_dir / "target_critic.pt")
        self.alpha.load(checkpoint_dir / "alpha.pt")
        if self.reward_normalizer is not None:
            self.reward_normalizer.load(checkpoint_dir / "reward_normalizer.pt")

    def save_onnx(self, path: str | Path) -> None:
        self.actor.save_onnx(path, [(1, *self.actor_obs_shape)], output_names=("actions",))

    def save_buffer(self, checkpoint_dir: str | Path) -> None:
        self.replay_buffer.save(Path(checkpoint_dir) / "replay_buffer.pt")

    def load_buffer(self, checkpoint_dir: str | Path) -> None:
        self.replay_buffer.load(Path(checkpoint_dir) / "replay_buffer.pt")
