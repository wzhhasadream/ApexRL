from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from gymnasium.vector import VectorEnv

from ....buffers.off_policy import Transition
from ....buffers.off_policy.numpy_lazy_frame_buffer import NumpyLazyFrameBuffer
from ....model.jax import Network, RewardNormalizer, update_reward_normalizer
from ...base_agent import OffPolicyAgent
from .get_action import get_action, get_eval_action
from .network import Critic
from .update import make_update


class BQNAgent(OffPolicyAgent):
    """CrossQ-style categorical Atari agent with optional reward normalization."""

    def __init__(self, envs: VectorEnv, cfg) -> None:
        super().__init__(envs, cfg)
        if len(self.observation_shape) != 4 or not self.action_is_discrete:
            raise ValueError("BQN requires discrete Atari observations with shape [F, H, W, C]")
        cfg.gamma = getattr(cfg, "gamma", 0.99)
        self._action_key, self._update_key = jax.random.split(jax.random.PRNGKey(cfg.seed))
        self._env_steps = 0
        self.epsilon_schedule = optax.linear_schedule(1.0, cfg.epsilon, cfg.epsilon_decay_steps)
        self._init_state()
        self._init_cached_fn()

    def _init_state(self):
        cfg = self.cfg
        compute_type = getattr(jnp, getattr(cfg, "compute_type", "float32"))
        self.replay_buffer = NumpyLazyFrameBuffer(
            self.observation_space, self.action_space, max_size=cfg.buffer_size,
            linear_decay_step=getattr(cfg, "decay_step", 0), max_n_step=cfg.n_step,
            num_envs=self.num_envs,
        )
        critic_model = Critic(
            self.observation_shape, self.action_dim, nnx.Rngs(cfg.seed),
            hidden_dim=getattr(cfg, "critic_hidden_dim", 512),
            num_bins=getattr(cfg, "num_bins", 101),
            use_bn=getattr(cfg, "use_bn", True), compute_type=compute_type,
            min_value=getattr(cfg, "min_value", -5.0), max_value=getattr(cfg, "max_value", 5.0),
        )
        self.critic = Network(critic_model, nnx.Optimizer(critic_model, optax.adam(getattr(cfg, "q_lr", 3e-4)), wrt=nnx.Param))
        self.target_critic = Network(deepcopy(critic_model), source_model=critic_model, tau=getattr(cfg, "tau", 0.005))
        self.reward_normalizer = None
        if getattr(cfg, "normalize_rewards", True):
            normalizer = RewardNormalizer(self.num_envs, cfg.gamma, getattr(cfg, "normalized_g_max", 5.0))
            normalizer.g_rms.count.value = jnp.asarray(1e-4, dtype=jnp.float32)
            self.reward_normalizer = Network(normalizer)

    def _init_cached_fn(self):
        self._get_action_fn = nnx.cached_partial(get_action, self.critic)
        self._get_eval_fn = nnx.cached_partial(get_eval_action, self.critic)
        self._update_fn = nnx.cached_partial(make_update(self.cfg), self.critic, self.target_critic, self.reward_normalizer)
        self._update_reward_normalizer = nnx.cached_partial(update_reward_normalizer, self.reward_normalizer)

    def get_action(self, observation):
        observation = jnp.asarray(observation).reshape((-1, *self.observation_shape))
        return np.asarray(self._get_eval_fn(observation))

    def get_exploration_action(self, observation):
        observation = jnp.asarray(observation).reshape((-1, *self.observation_shape))
        self._action_key, key = jax.random.split(self._action_key)
        epsilon = self.epsilon_schedule(self._env_steps)
        return np.asarray(self._get_action_fn(observation, key, epsilon))

    def process_transition(self, transition: Transition) -> None:
        if self.reward_normalizer is not None:
            self._update_reward_normalizer(transition.rewards, transition.terminations, transition.truncations)
        self.replay_buffer.add(transition)
        self._env_steps += self.num_envs

    @property
    def can_update(self) -> bool:
        return self.replay_buffer.size >= self.cfg.learning_starts and self.replay_buffer.can_sample(self.cfg.n_step)

    def update(self):
        if not self.can_update:
            raise RuntimeError("Replay buffer is not ready for an update")
        sequence = jax.tree.map(jnp.asarray, self.replay_buffer.sample(self.cfg.batch_size, self.cfg.n_step))
        self._update_key, key = jax.random.split(self._update_key)
        info = self._update_fn(sequence)
        return {name: float(value) for name, value in info.items()}

    def save(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.critic.save(checkpoint_dir / "critic.ckpt")
        self.target_critic.save(checkpoint_dir / "target_critic.ckpt")
        if self.reward_normalizer is not None:
            self.reward_normalizer.save(checkpoint_dir / "reward_normalizer.ckpt")

    def load(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.critic.load(checkpoint_dir / "critic.ckpt")
        self.target_critic.load(checkpoint_dir / "target_critic.ckpt")
        normalizer_checkpoint = checkpoint_dir / "reward_normalizer.ckpt"
        if self.reward_normalizer is not None and normalizer_checkpoint.is_dir():
            self.reward_normalizer.load(normalizer_checkpoint)
        self._init_cached_fn()

    def save_buffer(self, checkpoint_dir: str | Path) -> None:
        self.replay_buffer.save(Path(checkpoint_dir) / "replay_buffer")

    def load_buffer(self, checkpoint_dir: str | Path) -> None:
        self.replay_buffer.load(Path(checkpoint_dir) / "replay_buffer")

    def save_onnx(self, path: str | Path) -> None:
        self.critic.save_onnx(path, [(1, *self.observation_shape)], output_names=("q_values",))
