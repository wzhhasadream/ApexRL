# Adapted from amazon-far/holosoma FastSAC (Apache-2.0), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/amazon-far/holosoma/tree/d18d6cc50f872c15e904a22ceac22313cec955c8/src/holosoma/holosoma/agents/fast_sac
from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from gymnasium.vector import VectorEnv

from ....buffers.off_policy import Transition
from ....buffers.off_policy.jax_buffer import JaxBuffer
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
        self.cfg.target_entropy = -self.action_dim * cfg.target_entropy_ratio
        self._key = jax.random.PRNGKey(cfg.seed)
        self._update_count = 0
        self._num_steps = 0
        self._obs_cache = (None, None)
        self._init_state()
        self._init_cached_fn()

    def _next_key(self) -> jax.Array:
        self._key, key = jax.random.split(self._key)
        return key

    def _init_state(self) -> None:
        cfg = self.cfg
        # On-device replay buffer: sampling happens inside the jitted update, no host->device copy
        self.replay_buffer = JaxBuffer.create(self.observation_space, self.action_space, max_size=cfg.buffer_size, num_envs=self.num_envs, device=jax.devices()[0])
        rngs = nnx.Rngs(cfg.seed)
        # epsilon=1e-4 matches Holosoma's EmpiricalNormalization scale.
        self.observation_rms = Network(RMS(self.observation_shape[0], epsilon=1e-4)) if cfg.obs_normalization else None
        # Bounds come from BaseAgent: env bounds where finite, [-1, 1] otherwise
        action_low, action_high = jnp.asarray(self.action_low), jnp.asarray(self.action_high)
        actor_model = Actor(self.actor_obs_shape[0], self.action_dim, rngs.fork(), cfg, action_low, action_high)
        critic_model = Critic(self.observation_shape[0], self.action_dim, rngs.fork(), cfg)
        alpha_model = Alpha(cfg.alpha_init)
        self.actor = Network(actor_model, nnx.Optimizer(actor_model, optax.adamw(cfg.actor_learning_rate, weight_decay=cfg.weight_decay), wrt=nnx.Param), forward_name="get_mean_action")
        self.critic = Network(critic_model, nnx.Optimizer(critic_model, optax.adamw(cfg.critic_learning_rate, weight_decay=cfg.weight_decay), wrt=nnx.Param))
        self.target_critic = Network(deepcopy(critic_model), source_model=critic_model, tau=cfg.tau)
        # Holosoma's alpha optimizer uses AdamW(betas=(0.9, 0.95), weight_decay=0.01).
        self.alpha = Network(alpha_model, nnx.Optimizer(alpha_model, optax.adamw(cfg.alpha_learning_rate, b1=0.9, b2=0.95, weight_decay=0.01), wrt=nnx.Param))

    def _init_cached_fn(self) -> None:
        self._get_eval_fn = nnx.cached_partial(get_eval_action, self.actor, self.observation_rms)
        self._get_exploration_fn = nnx.cached_partial(get_exploration_action, self.actor, self.observation_rms)
        self._update_rms_fn = nnx.cached_partial(update_rms, self.observation_rms) if self.observation_rms is not None else None
        self._update_fn = nnx.cached_partial(make_update(self.cfg), self.critic, self.target_critic, self.actor, self.alpha, self.observation_rms)

    def get_action(self, observation: jax.Array | np.ndarray) -> np.ndarray:
        return np.asarray(self._get_eval_fn(jnp.asarray(observation, jnp.float32).reshape((-1, self.observation_shape[0]))))

    def get_exploration_action(self, observation: jax.Array | np.ndarray) -> np.ndarray:
        obs = jnp.asarray(observation, jnp.float32).reshape((-1, self.observation_shape[0]))
        # Keep the device copy: the runner passes the same array to process_transition right after env.step
        self._obs_cache = (observation, obs)
        return np.asarray(self._get_exploration_fn(obs, self._next_key()))

    def process_transition(self, transition: Transition) -> None:
        # Reuse the device copy from get_exploration_action instead of two more host->device transfers
        if self._obs_cache[0] is transition.observations:
            transition = transition._replace(observations=self._obs_cache[1])
        self.replay_buffer = self.replay_buffer.add(transition)
        self._num_steps += 1
        if self._update_rms_fn is not None:
            self._update_rms_fn(jnp.asarray(transition.observations, jnp.float32))

    @property
    def can_update(self) -> bool:
        # Track size on host to avoid a device sync every step
        return min(self._num_steps, self.replay_buffer.capacity) * self.num_envs >= self.cfg.learning_starts and self._num_steps >= self.cfg.n_step

    def update(self) -> dict[str, float]:
        if not self.can_update:
            raise RuntimeError("Replay buffer is not ready for an update")
        self._update_count += 1
        info = self._update_fn(self.replay_buffer, self._next_key(), self._update_count % self.cfg.policy_frequency == 0)
        # One device->host transfer for all metrics
        return {name: float(value) for name, value in jax.device_get(info).items()}

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
        self.replay_buffer.save(Path(checkpoint_dir).absolute() / "replay_buffer")

    def load_buffer(self, checkpoint_dir: str | Path) -> None:
        self.replay_buffer = self.replay_buffer.load(Path(checkpoint_dir).absolute() / "replay_buffer")
        self._num_steps = int(self.replay_buffer.steps)

    def save_onnx(self, path: str | Path) -> None:
        layers = [self.actor]
        if self.observation_rms is not None:
            # Only the actor slice of the (critic) observation statistics
            actor_rms = RMS(self.actor_obs_shape[0], self.observation_rms.model.epsilon)
            actor_rms.mean.value = self.observation_rms.model.mean.value[:self.actor_obs_shape[0]]
            actor_rms.var.value = self.observation_rms.model.var.value[:self.actor_obs_shape[0]]
            actor_rms.count.value = self.observation_rms.model.count.value
            layers.insert(0, Network(actor_rms))
        Network(nnx.Sequential(*layers)).save_onnx(path, [(1, *self.actor_obs_shape)], output_names=("actions",))


__all__ = ["FastSACAgent"]
