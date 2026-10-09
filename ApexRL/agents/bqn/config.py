from dataclasses import dataclass, field
from typing import Literal

from ..base_config import BaseConfig


@dataclass
class BQNConfig(BaseConfig):
    """Replay-based Atari C101 with joint BatchNorm and a target critic."""

    seed: int = 1
    num_train_env: int = 128
    total_timesteps: int = 10_000_000
    grad_step_per_interaction_step: float = 1.0  # Updates per vector environment step.
    buffer_size: int = 1_000_000
    learning_starts: int = 100_000
    batch_size: int = 256
    n_step: int = 3
    gamma: float = 0.99
    decay_step: int = 80_000
    learning_rate: float = 2.5e-4
    target_frequency: int = 1  # Target updates are counted in critic gradient steps.
    tau: float = 0.005
    hidden_dim: int = 512
    norm_type: Literal["bn", "ln", None] = "bn"
    num_bins: int = 101
    v_min: float = -5.0
    v_max: float = 5.0
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_steps: int = 1_000_000  # Global transitions, including replay warmup.
    normalize_rewards: bool = True
    compute_type: Literal["float32", "bfloat16"] = "float32"

    action_dim: int = field(init=False, default=0)
    action_is_discrete: bool = field(init=False, default=True)
    image_obs: bool = field(init=False, default=True)
    asymmetric_obs: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if min(self.num_train_env, self.total_timesteps, self.batch_size, self.n_step, self.target_frequency, self.exploration_steps) < 1:
            raise ValueError("Environment, training, batch, n-step, target and exploration counts must be positive")
        if self.buffer_size < self.num_train_env or not 0 <= self.learning_starts <= self.buffer_size:
            raise ValueError("Replay must hold the training environments and learning_starts transitions")
        if self.num_bins < 2 or self.v_min >= self.v_max:
            raise ValueError("Categorical support requires at least two bins and v_min < v_max")
        if not 0 < self.tau <= 1 or not 0 <= self.gamma <= 1:
            raise ValueError("tau must be in (0, 1] and gamma in [0, 1]")
        if not 0 <= self.end_e <= self.start_e <= 1:
            raise ValueError("Exploration requires 0 <= end_e <= start_e <= 1")


__all__ = ["BQNConfig"]
