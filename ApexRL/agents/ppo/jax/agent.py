# Adapted from vwxyzjn/cleanrl (MIT) and leggedrobotics/rsl_rl (BSD-3-Clause), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool.py
# https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/ppo.py
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
from ..config import PPOConfig
from .get_action import get_eval_action, get_value, sample_and_value
from .network_atari import ActorCritic as AtariActorCritic
from .network_state import Actor, ActorCritic, Critic
from .update import make_update_ppo


class PPOAgent(OnPolicyAgent):
    """PPO for continuous (Gaussian, flat obs) and discrete (categorical, flat or Atari image obs) actions."""

    def __init__(self, envs: VectorEnv, cfg: PPOConfig) -> None:
        super().__init__(envs, cfg)
        self.device = default_device()
        self._action_key, self._update_key = jax.random.split(jax.random.PRNGKey(cfg.seed))
        self._num_rollouts = max(1, cfg.total_timesteps // (self.num_train_env * cfg.rollout_steps))
        self._rollout_idx = 0
        self._last_obs, self._last_obs_device = None, None
        self._init_state()
        self._init_cached_fn()

    def _init_state(self) -> None:
        cfg = self.cfg
        rngs = nnx.Rngs(cfg.seed)
        if self.image_obs:
            if not self.action_is_discrete or self.asymmetric_obs:
                raise ValueError("Atari PPO requires discrete actions and symmetric image observations")
            model = AtariActorCritic(self.observation_shape, self.action_dim, rngs, cfg)
        else:
            model = ActorCritic(Actor(self.actor_obs_shape, self.action_dim, self.action_is_discrete, rngs, cfg), Critic(self.critic_obs_shape, rngs, cfg))
        # inject_hyperparams: lr lives in the optimizer state, so it can be adapted / annealed without recompiling
        self.agent = Network(model, nnx.Optimizer(model, optax.inject_hyperparams(optax.adam)(learning_rate=cfg.lr), wrt=nnx.Param), forward_name="get_mean_action")
        # Commit params/opt state to the device up front; otherwise every jitted fn compiles twice
        nnx.update(self.agent, jax.device_put(nnx.state(self.agent), self.device))
        self.replay_buffer = JaxBuffer.create(cfg.rollout_steps, self.num_train_env, self.observation_space, self.action_space, device=self.device)

    def _init_cached_fn(self) -> None:
        self._sample_and_value_fn = nnx.cached_partial(sample_and_value, self.agent, self.asymmetric_obs)
        self._get_eval_fn = nnx.cached_partial(get_eval_action, self.agent, self.asymmetric_obs)
        self._get_value_fn = nnx.cached_partial(get_value, self.agent)
        self._update_fn = nnx.cached_partial(make_update_ppo(self.cfg), self.agent)

    def _observations(self, observations: jax.Array | np.ndarray) -> jax.Array:
        # Images stay uint8 on transfer (4x smaller); the CNN casts on device
        dtype = None if self.image_obs else jnp.float32
        return jax.device_put(jnp.asarray(observations, dtype=dtype), self.device).reshape((-1, *self.critic_obs_shape))

    def get_action(self, observations: jax.Array | np.ndarray) -> np.ndarray:
        return np.asarray(self._get_eval_fn(self._observations(observations)))

    def get_exploration_action(self, observations: jax.Array | np.ndarray) -> np.ndarray:
        return self.sample_action_and_value(observations).actions

    def get_value(self, observations: jax.Array | np.ndarray) -> np.ndarray:
        return np.asarray(self._get_value_fn(self._observations(observations)))

    def sample_action_and_value(self, observations: jax.Array | np.ndarray) -> OnPolicySample:
        self._action_key, key = jax.random.split(self._action_key)
        # Cache the device copy so process_transition does not upload the same observations twice
        self._last_obs, self._last_obs_device = observations, self._observations(observations)
        actions, values, log_probs, metadata = self._sample_and_value_fn(self._last_obs_device, key)
        # Policy outputs stay on device (the buffer is on device too); only actions go back to the env
        return OnPolicySample(np.asarray(actions), values, log_probs, metadata)

    def process_transition(self, transition: RolloutTransition) -> None:
        if transition.observations is self._last_obs:
            transition = transition._replace(observations=self._last_obs_device)
        self.replay_buffer = self.replay_buffer.add(transition)

    @property
    def can_update(self) -> bool:
        return bool(self.replay_buffer.full)

    def update(self, last_observations: jax.Array | np.ndarray) -> dict[str, float]:
        if not self.can_update:
            raise RuntimeError("Collect a complete rollout before calling update.")
        if self.cfg.anneal_lr:
            self.agent.opt.opt_state.hyperparams["learning_rate"].value = jnp.asarray(self.cfg.lr * (1.0 - self._rollout_idx / self._num_rollouts), jnp.float32)
        self._update_key, key = jax.random.split(self._update_key)
        info = self._update_fn(self.replay_buffer, self._observations(last_observations), key)
        self.replay_buffer = self.replay_buffer.reset()
        self._rollout_idx += 1
        # One device->host transfer for all metrics
        return {name: float(value) for name, value in jax.device_get(info).items()}

    def save(self, checkpoint_dir: str | Path) -> None:
        self.agent.save(Path(checkpoint_dir) / "agent.ckpt")

    def load(self, checkpoint_dir: str | Path) -> None:
        self.agent.load(Path(checkpoint_dir) / "agent.ckpt")
        self._init_cached_fn()

    def save_onnx(self, path: str | Path) -> None:
        self.agent.save_onnx(path, [(1, *self.actor_obs_shape)], output_names=("actions",))


__all__ = ["PPOAgent"]
