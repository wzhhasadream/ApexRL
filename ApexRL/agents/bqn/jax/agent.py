from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from gymnasium.vector import VectorEnv
from optax import linear_schedule
from ....buffers.off_policy import Transition
from ....buffers.off_policy.numpy_lazy_frame_buffer import NumpyLazyFrameBuffer
from ....common.jax import default_device
from ....model.jax import Network, RewardNormalizer, update_reward_normalizer
from ...base_agent import OffPolicyAgent
from ..config import BQNConfig
from .get_action import get_eval_action, get_exploration_action
from .network import Critic
from .update import make_update

class BQNAgent(OffPolicyAgent):
    """Atari C101 with joint BatchNorm and a train-mode target critic."""

    def __init__(self, envs: VectorEnv, cfg: BQNConfig) -> None:
        super().__init__(envs, cfg)
        if len(self.observation_shape) != 4 or not self.action_is_discrete:
            raise ValueError("BQN requires discrete actions and observations with shape [F, H, W, C]")
        self.learner_device = default_device()
        self._action_key = jax.random.PRNGKey(cfg.seed)
        self._num_steps = 0
        self.critic_grad_updates = 0
        self._init_train_state()
        self._init_cached_fn()
        self.key = jax.random.PRNGKey(cfg.seed)

    def _init_train_state(self) -> None:
        cfg = self.cfg
        model = Critic(self.observation_shape, self.action_dim, nnx.Rngs(cfg.seed), hidden_dim=cfg.hidden_dim,
                       compute_type=getattr(jnp, cfg.compute_type), norm_type=cfg.norm_type,
                       num_bins=cfg.num_bins, v_min=cfg.v_min, v_max=cfg.v_max, encoder=cfg.encoder)
        self.critic = Network(model, nnx.Optimizer(model, optax.adam(cfg.learning_rate), wrt=nnx.Param), forward_name="select_action")
        self.target_critic = Network(deepcopy(model), source_model=model, tau=cfg.tau)
        self.reward_normalizer = Network(RewardNormalizer(self.num_train_env, cfg.gamma)) if cfg.normalize_rewards else None
        state = nnx.state((self.critic, self.target_critic, self.reward_normalizer))
        nnx.update((self.critic, self.target_critic, self.reward_normalizer), jax.device_put(state, self.learner_device))
        self.replay_buffer = NumpyLazyFrameBuffer(self.observation_space, self.action_space, max_size=cfg.buffer_size,
                                                linear_decay_step=cfg.decay_step, max_n_step=cfg.n_step, num_envs=self.num_train_env)
        self.epsilon_schedule = linear_schedule(self.cfg.start_e, self.cfg.end_e, self.cfg.total_timesteps * self.cfg.exploration_frac)

    def _init_cached_fn(self) -> None:
        self._get_eval_fn = nnx.cached_partial(get_eval_action, self.critic)
        self._get_exploration_fn = nnx.cached_partial(get_exploration_action, self.critic)
        self._update_fn = nnx.cached_partial(make_update(self.cfg), self.critic, self.target_critic, self.reward_normalizer)
        self._update_reward_normalizer = nnx.cached_partial(update_reward_normalizer, self.reward_normalizer)

    @property
    def epsilon(self) -> float: 
        return self.epsilon_schedule(self._num_steps * self.num_train_env)


    @property
    def next_key(self) -> jax.Array:
        self.key, next_key = jax.random.split(self.key, 2)
        return next_key


    def get_action(self, observations: jax.Array | np.ndarray) -> np.ndarray:
        observations = jax.device_put(observations, self.learner_device).reshape((-1, *self.observation_shape))
        return np.asarray(self._get_eval_fn(observations))

    def get_exploration_action(self, observations: jax.Array | np.ndarray) -> np.ndarray:
        obs_device = jax.device_put(observations, self.learner_device).reshape((-1, *self.observation_shape))
        self._action_key, key = jax.random.split(self._action_key)
        return np.asarray(self._get_exploration_fn(obs_device, key, self.epsilon))

    def process_transition(self, transition: Transition) -> None:
        self._update_reward_normalizer(transition.rewards, transition.terminations, transition.truncations)
        self.replay_buffer.add(transition)
        self._num_steps += 1

    @property
    def can_update(self) -> bool:
        return self._num_steps * self.num_train_env >= self.cfg.learning_starts and self.replay_buffer.can_sample(self.cfg.n_step)

    def update(self) -> dict[str, float]:
        if not self.can_update:
            raise RuntimeError("Replay buffer is not ready for an update")
        sequence = jax.device_put(self.replay_buffer.sample(self.cfg.batch_size, self.cfg.n_step), self.learner_device)
        self.critic_grad_updates += 1
        info = self._update_fn(sequence, update_target=self.critic_grad_updates % self.cfg.target_frequency == 0, key=self.next_key)
        return {**{name: float(value) for name, value in jax.device_get(info).items()}, "training/epsilon": self.epsilon}

    def save(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.critic.save(checkpoint_dir / "critic.ckpt")
        self.target_critic.save(checkpoint_dir / "target_critic.ckpt")
        if self.reward_normalizer is not None:
            self.reward_normalizer.save(checkpoint_dir / "reward_normalizer.ckpt")
        np.savez(checkpoint_dir / "training_state.npz", num_steps=self._num_steps,
                 critic_grad_updates=self.critic_grad_updates, action_key=np.asarray(self._action_key))

    def load(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.critic.load(checkpoint_dir / "critic.ckpt")
        self.target_critic.load(checkpoint_dir / "target_critic.ckpt")
        if self.reward_normalizer is not None:
            self.reward_normalizer.load(checkpoint_dir / "reward_normalizer.ckpt")
        with np.load(checkpoint_dir / "training_state.npz") as state:
            self._num_steps = int(state["num_steps"])
            self.critic_grad_updates = int(state["critic_grad_updates"])
            self._action_key = jax.device_put(state["action_key"], self.learner_device)
        self._init_cached_fn()

    def save_buffer(self, checkpoint_dir: str | Path) -> None:
        self.replay_buffer.save(Path(checkpoint_dir) / "replay_buffer")

    def load_buffer(self, checkpoint_dir: str | Path) -> None:
        self.replay_buffer.load(Path(checkpoint_dir) / "replay_buffer")
        self._num_steps = int(self.replay_buffer._current_time.sum()) // self.num_train_env

    def save_onnx(self, path: str | Path) -> None:
        self.critic.save_onnx(path, [(1, *self.actor_obs_shape)])
