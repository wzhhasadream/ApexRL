from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from gymnasium.vector import VectorEnv

from ....buffers.off_policy import Transition
from ....buffers.off_policy.numpy_buffer import NumpyBuffer
from ....model.jax import Alpha, Network, RMS
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
        self._action_key, self._update_key = jax.random.split(jax.random.PRNGKey(cfg.seed))
        self._update_count = 0
        self._init_state()
        self._init_cached_fn()

    def _init_state(self) -> None:
        cfg = self.cfg
        self.replay_buffer = NumpyBuffer(self.observation_space, self.action_space, max_size=cfg.buffer_size, num_envs=self.num_envs)
        rngs = nnx.Rngs(cfg.seed)
        self.observation_rms = Network(RMS(self.observation_shape[0])) if cfg.obs_normalization else None
        action_low = jnp.asarray(self.action_space.low, dtype=jnp.float32)
        action_high = jnp.asarray(self.action_space.high, dtype=jnp.float32)
        actor_model = Actor(self.actor_obs_shape[0], self.action_dim, rngs.fork(), cfg, action_low, action_high)
        critic_model = Critic(self.observation_shape[0], self.action_dim, rngs.fork(), cfg)
        actor_tx = optax.adamw(cfg.actor_learning_rate, weight_decay=cfg.weight_decay)
        critic_tx = optax.adamw(cfg.critic_learning_rate, weight_decay=cfg.weight_decay)
        alpha_model = Alpha(cfg.alpha_init)
        alpha_tx = optax.adam(cfg.alpha_learning_rate)
        self.actor = Network(actor_model, nnx.Optimizer(actor_model, actor_tx, wrt=nnx.Param), forward_name="get_mean_action")
        self.critic = Network(critic_model, nnx.Optimizer(critic_model, critic_tx, wrt=nnx.Param))
        self.target_critic = Network(deepcopy(critic_model), source_model=critic_model, tau=cfg.tau)
        self.alpha = Network(alpha_model, nnx.Optimizer(alpha_model, alpha_tx, wrt=nnx.Param))

    def _init_cached_fn(self) -> None:
        actor_obs_dim = self.actor_obs_shape[0]
        self._get_eval_fn = nnx.cached_partial(
            get_eval_action, self.actor, self.observation_rms, self.asymmetric_obs, actor_obs_dim
        )
        self._get_exploration_fn = nnx.cached_partial(
            get_exploration_action, self.actor, self.observation_rms, self.asymmetric_obs, actor_obs_dim
        )
        self._update_rms_fn = nnx.cached_partial(update_rms, self.observation_rms) if self.observation_rms is not None else None
        self._update_fn = nnx.cached_partial(
            make_update(self.cfg), self.critic, self.target_critic, self.actor, self.alpha, self.observation_rms
        )

    def get_action(self, observation: jax.Array | np.ndarray) -> np.ndarray:
        observation = jnp.asarray(observation).reshape((-1, self.observation_shape[0]))
        return np.asarray(self._get_eval_fn(observation))

    def get_exploration_action(self, observation: jax.Array | np.ndarray) -> np.ndarray:
        observation = jnp.asarray(observation).reshape((-1, self.observation_shape[0]))
        self._action_key, key = jax.random.split(self._action_key)
        return np.asarray(self._get_exploration_fn(observation, key))

    def process_transition(self, transition: Transition) -> None:
        self.replay_buffer.add(transition)
        if self._update_rms_fn is not None:
            self._update_rms_fn(jnp.asarray(transition.observations))

    @property
    def can_update(self) -> bool:
        return self.replay_buffer.size >= self.cfg.learning_starts and self.replay_buffer.can_sample(self.cfg.n_step)

    def update(self) -> dict[str, float]:
        if not self.can_update:
            raise RuntimeError("Replay buffer is not ready for an update")
        sequence = jax.tree.map(jnp.asarray, self.replay_buffer.sample(self.cfg.batch_size, self.cfg.n_step))
        self._update_key, key = jax.random.split(self._update_key)
        self._update_count += 1
        update_actor = self._update_count % self.cfg.policy_frequency == 0
        info = self._update_fn(sequence, key, update_actor)
        return {name: float(value) for name, value in info.items()}

    def save(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.actor.save(checkpoint_dir / "actor.ckpt")
        self.critic.save(checkpoint_dir / "critic.ckpt")
        self.target_critic.save(checkpoint_dir / "target_critic.ckpt")
        self.alpha.save(checkpoint_dir / "alpha.ckpt")
        if self.observation_rms is not None:
            self.observation_rms.save(checkpoint_dir / "observation_rms.ckpt")

    def load(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.actor.load(checkpoint_dir / "actor.ckpt")
        self.critic.load(checkpoint_dir / "critic.ckpt")
        self.target_critic.load(checkpoint_dir / "target_critic.ckpt")
        self.alpha.load(checkpoint_dir / "alpha.ckpt")
        if self.observation_rms is not None:
            self.observation_rms.load(checkpoint_dir / "observation_rms.ckpt")
        self._init_cached_fn()

    def save_buffer(self, checkpoint_dir: str | Path) -> None:
        self.replay_buffer.save(Path(checkpoint_dir) / "replay_buffer")

    def load_buffer(self, checkpoint_dir: str | Path) -> None:
        self.replay_buffer.load(Path(checkpoint_dir) / "replay_buffer")

    def save_onnx(self, path: str | Path) -> None:
        layers = [self.actor]
        if self.observation_rms is not None:
            if self.asymmetric_obs:
                actor_rms = RMS(self.actor_obs_shape[0], self.observation_rms.model.epsilon)
                actor_rms.mean.value = self.observation_rms.model.mean.value[:self.actor_obs_shape[0]]
                actor_rms.var.value = self.observation_rms.model.var.value[:self.actor_obs_shape[0]]
                actor_rms.count.value = self.observation_rms.model.count.value
                layers.insert(0, Network(actor_rms))
            else:
                layers.insert(0, self.observation_rms)
        Network(nnx.Sequential(*layers)).save_onnx(path, [(1, *self.actor_obs_shape)], output_names=("actions",))
