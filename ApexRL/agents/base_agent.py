from __future__ import annotations

from abc import ABC, abstractmethod
import gymnasium.spaces as spaces
from gymnasium.vector import VectorEnv
from typing import Any, NamedTuple, TYPE_CHECKING, TypeAlias
import numpy as np
from pathlib import Path
import warnings

from ..common import is_image_observation
from ..buffers import PolicyMetadata, RolloutTransition, Transition

if TYPE_CHECKING:
    import jax
    import torch

    Tensor: TypeAlias = np.ndarray | jax.Array | torch.Tensor


class BaseAgent(ABC):
    def __init__(
        self,
        envs: VectorEnv,
        cfg: Any
    ) -> None:
        self.cfg = cfg
        self.observation_space = envs.single_observation_space
        self.action_space = envs.single_action_space
        self.num_train_env = envs.num_envs
        self.observation_shape = tuple(self.observation_space.shape)
        self.continuous_action_dim = int(
            np.prod(np.asarray(self.action_space.shape))
        )
        if isinstance(self.action_space, spaces.Discrete):
            self.action_is_discrete = True
            self.discrete_action_n = int(self.action_space.n)
            self.action_dim = self.discrete_action_n
        else:
            self.action_is_discrete = False
            self.discrete_action_n = None
            self.action_dim = self.continuous_action_dim
        self.cfg.action_is_discrete = self.action_is_discrete
        self.cfg.action_dim = self.action_dim
        self.critic_obs_shape = (
            self.observation_shape
            if len(self.observation_shape) > 1
            else (self.observation_shape[0], )
        )
        self.actor_obs_shape = self.critic_obs_shape
        self.asymmetric_obs = getattr(envs, 'asymmetric_obs', False)
        self.cfg.asymmetric_obs = self.asymmetric_obs
        self.cfg.image_obs = is_image_observation(self.observation_shape)
        self.image_obs = self.cfg.image_obs
        if self.asymmetric_obs:
            actor_observation_size = getattr(
                envs, "actor_observation_size", None
            )
            if actor_observation_size is None:
                raise ValueError(
                    "Asymmetric observations require actor_observation_size"
                )
            actor_observation_shape = (
                (actor_observation_size,)
                if isinstance(actor_observation_size, (int, np.integer))
                else tuple(actor_observation_size)
            )
            self.actor_obs_shape = (
                actor_observation_shape
                if len(actor_observation_shape) > 1
                else (actor_observation_shape[0], )
            )

    @property
    def observation_debug_info(self) -> dict[str, int | tuple[int, ...] | bool]:
        return {
            "asymmetric_obs": self.asymmetric_obs,
            "image_obs": self.image_obs,
            "action_is_discrete": self.action_is_discrete,
            "actor_obs_shape": self.actor_obs_shape,
            "critic_obs_shape": self.critic_obs_shape,
        }

    @property
    def can_update(self) -> bool:
        pass

    @abstractmethod
    def get_action(self, observation: Tensor) -> np.ndarray:
        pass


    @abstractmethod
    def save(self, path: str | Path) -> None:
        pass

    @abstractmethod
    def load(self, path: str | Path) -> None:
        pass


    @abstractmethod
    def save_onnx(self, path: str | Path) -> None:
        pass







class OnPolicySample(NamedTuple):
    """Policy output used by the shared on-policy runner."""

    actions: Tensor
    values: Tensor
    log_probs: Tensor | None = None
    metadata: PolicyMetadata | None = None





class OnPolicyAgent(BaseAgent):
    @abstractmethod
    def sample_action_and_value(self, observation: Tensor) -> OnPolicySample:
        pass

    @abstractmethod
    def get_value(self, observation: Tensor) -> Tensor:
        pass

    @abstractmethod
    def process_transition(self, transition: RolloutTransition) -> None:
        pass

    @abstractmethod
    def update(self, last_observation: Tensor) -> dict[str, float]:
        pass




class OffPolicyAgent(BaseAgent):
    def __init__(self, envs: VectorEnv, cfg: Any) -> None:
        super().__init__(envs, cfg)
        if self.action_is_discrete:
            self.action_low = self.action_high = None
            self.cfg.target_entropy = float(np.log(self.discrete_action_n))
        else:
            # Off-policy actors squash actions with tanh, so they need finite bounds:
            # finite env bounds are used as-is, unbounded dimensions fall back to [-1, 1]
            low = np.asarray(self.action_space.low, dtype=np.float32).reshape(-1)
            high = np.asarray(self.action_space.high, dtype=np.float32).reshape(-1)
            if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
                warnings.warn(f"Action space has non-finite bounds (low={low}, high={high}); using [-1, 1] for those dimensions.")
            self.action_low = np.where(np.isfinite(low), low, -1.0).astype(np.float32)
            self.action_high = np.where(np.isfinite(high), high, 1.0).astype(np.float32)
            self.cfg.target_entropy = float(
                0.5
                * self.action_dim
                * np.log(2 * np.pi * np.e * 0.15**2)
            )

    @abstractmethod
    def process_transition(self, transition: Transition) -> None:
        pass

    @abstractmethod
    def update(self) -> dict[str, float]:
        pass
