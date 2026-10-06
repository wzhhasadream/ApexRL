from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from gymnasium.vector import VectorEnv

from ....buffers.on_policy.jax_buffer import JaxBuffer
from ....buffers.on_policy.types import RolloutTransition
from ....common.jax import default_device
from ....model.jax import Network
from ...base_agent import OnPolicyAgent, OnPolicySample
from ..config import PQNConfig
from .get_action import get_eval_action, get_value, sample_action_and_value
from .network import Critic
from .update import make_update


class PQNAgent(OnPolicyAgent):
    """JAX/NNX PQN agent with fixed-horizon Atari rollouts."""

    def __init__(self, envs: VectorEnv, cfg: PQNConfig) -> None:
        super().__init__(envs, cfg)
        if len(self.observation_shape) != 4 or not self.action_is_discrete:
            raise ValueError("PQN requires discrete Atari observations with shape [F, H, W, C]")

        self.learner_device = default_device()
        self._action_key, self._update_key = jax.random.split(
            jax.random.PRNGKey(cfg.seed)
        )
        self._current_rollout_idx = 0
        self._init_train_state()
        self._init_cached_fn()

    def _init_train_state(self) -> None:
        cfg = self.cfg
        compute_type = getattr(jnp, cfg.compute_type)
        num_rollouts = max(
            1,
            cfg.total_timesteps // (self.num_envs * cfg.rollout_steps),
        )
        num_updates = num_rollouts * cfg.num_minibatches * cfg.update_epochs
        self.epsilon_schedule = optax.linear_schedule(cfg.start_e, cfg.end_e, num_rollouts)
        optimizer_schedule = (
            optax.linear_schedule(cfg.learning_rate, 0.0, num_updates)
            if cfg.anneal_lr
            else cfg.learning_rate
        )
        model = Critic(
            self.observation_shape,
            self.action_dim,
            nnx.Rngs(cfg.seed),
            hidden_dim=cfg.hidden_dim,
            compute_type=compute_type,
            norm_type=cfg.norm_type,
        )
        self.critic = Network(
            model,
            nnx.Optimizer(model, optax.radam(optimizer_schedule), wrt=nnx.Param),
            forward_name="select_action"
        )
        self.replay_buffer = JaxBuffer.create(
            cfg.rollout_steps,
            self.num_envs,
            self.observation_space,
            self.action_space,
            store_log_probs=False,
            store_metadata=False,
            device=self.learner_device,
        )

    def _init_cached_fn(self) -> None:
        self._get_eval_fn = nnx.cached_partial(get_eval_action, self.critic)
        self._get_value_fn = nnx.cached_partial(get_value, self.critic)
        self._sample_and_value_fn = nnx.cached_partial(
            sample_action_and_value, self.critic
        )
        self._update_fn = nnx.cached_partial(make_update(self.cfg), self.critic)

    def _observations(self, observations: jax.Array | np.ndarray) -> jax.Array:
        return jnp.asarray(observations, dtype=jnp.float32).reshape(
            (-1, *self.observation_shape)
        )

    def _epsilon(self) -> float:
        return float(self.epsilon_schedule(self._current_rollout_idx))

    def get_action(self, observations: jax.Array | np.ndarray) -> np.ndarray:
        return np.asarray(self._get_eval_fn(self._observations(observations)))


    def get_value(self, observations: jax.Array | np.ndarray) -> np.ndarray:
        return np.asarray(
            self._get_value_fn(self._observations(observations))
        )

    def sample_action_and_value(
        self,
        observations: jax.Array | np.ndarray,
    ) -> OnPolicySample:
        self._action_key, action_key = jax.random.split(self._action_key)
        actions, values = self._sample_and_value_fn(
            self._observations(observations), action_key, self._epsilon()
        )
        return OnPolicySample(np.asarray(actions), np.asarray(values))

    def process_transition(self, transition: RolloutTransition) -> None:
        self.replay_buffer = self.replay_buffer.add(transition)
        self._env_steps += self.num_envs

    @property
    def can_update(self) -> bool:
        return bool(self.replay_buffer.full)

    def update(self, last_observations: jax.Array | np.ndarray) -> dict[str, float]:
        if not self.can_update:
            raise RuntimeError("A complete rollout is required before updating PQN.")
        self._update_key, update_key = jax.random.split(self._update_key)
        info = self._update_fn(
            self.replay_buffer,
            self._observations(last_observations),
            update_key
        )
        self.replay_buffer = self.replay_buffer.reset()
        self._current_rollout_idx += 1
        return {name: float(np.asarray(value)) for name, value in info.items()}

    def save(self, checkpoint_dir: str | Path) -> None:
        self.critic.save(Path(checkpoint_dir) / "critic.ckpt")

    def load(self, checkpoint_dir: str | Path) -> None:
        self.critic.load(Path(checkpoint_dir) / "critic.ckpt")
        self._init_cached_fn()

    def save_buffer(self, checkpoint_dir: str | Path) -> None:
        self.replay_buffer.save(Path(checkpoint_dir) / "rollout_buffer")

    def load_buffer(self, checkpoint_dir: str | Path) -> None:
        self.replay_buffer = self.replay_buffer.load(
            Path(checkpoint_dir) / "rollout_buffer"
        )

    def save_onnx(self, path: str | Path) -> None:
        self.critic.save_onnx(
            path,
            [(1, *self.observation_shape)]
        )
