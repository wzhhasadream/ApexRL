from dataclasses import dataclass, field
from typing import Literal

from ..base_config import BaseConfig


@dataclass
class BQNConfig(BaseConfig):
    """Replay-based Atari C101 with joint BatchNorm and a target critic."""

    seed: int = 1
    num_train_env: int = 1
    total_timesteps: int = 10_000_000
    grad_step_per_interaction_step: float = 0.25  # Updates per vector environment step.
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000
    batch_size: int = 32
    n_step: int = 3
    gamma: float = 0.99
    decay_step: int = 80_000
    learning_rate: float = 3e-4
    target_frequency: int = 4  # Target updates are counted in critic gradient steps.
    tau: float = 0.005
    hidden_dim: int = 512
    norm_type: Literal["bn", "ln", None] = "bn"
    num_bins: int = 101
    v_min: float = -5.0
    v_max: float = 5.0
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_frac: float = 0.1  # Global transitions, including replay warmup.
    normalize_rewards: bool = True
    compute_type: Literal["float32", "bfloat16"] = "float32"

    action_dim: int = field(init=False, default=0)
    action_is_discrete: bool = field(init=False, default=True)
    image_obs: bool = field(init=False, default=True)
    asymmetric_obs: bool = field(init=False, default=False)



__all__ = ["BQNConfig"]
