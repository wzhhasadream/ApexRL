from dataclasses import dataclass, field
from typing import Literal


@dataclass
class FastTD3Config:
    seed: int = 1
    num_train_env: int = 128
    num_learning_iterations: int = 150_000
    # ApexRL counts total transitions; the upstream value counts vector steps.
    total_timesteps: int | None = None
    grad_step_per_interaction_step: float = 2.0
    buffer_size: int | None = None
    learning_starts: int | None = None
    batch_size: int = 32_768
    n_step: int = 1
    gamma: float = 0.99
    tau: float = 0.1
    policy_frequency: int = 2
    policy_noise: float = 0.001
    noise_clip: float = 0.5
    std_min: float = 0.001
    std_max: float = 0.4
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    weight_decay: float = 0.1
    init_scale: float = 0.01
    actor_hidden_dim: int = 512
    critic_hidden_dim: int = 1024
    num_atoms: int = 101
    v_min: float = -250.0
    v_max: float = 250.0
    use_cdq: bool = True
    obs_normalization: bool = True
    use_layer_norm: bool = False
    max_grad_norm: float | None = None
    compute_type: Literal["float32", "bfloat16"] = "float32"

    action_dim: int = field(init=False, default=0)
    action_is_discrete: bool = field(init=False, default=False)
    asymmetric_obs: bool = field(init=False, default=False)
    image_obs: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.num_train_env < 1 or self.num_learning_iterations < 1:
            raise ValueError("num_train_env and num_learning_iterations must be positive")
        if self.total_timesteps is None:
            self.total_timesteps = self.num_learning_iterations * self.num_train_env
        if self.buffer_size is None:
            self.buffer_size = 1024 * 50 * self.num_train_env
        if self.learning_starts is None:
            self.learning_starts = 10 * self.num_train_env
        if self.n_step < 1 or self.policy_frequency < 1:
            raise ValueError("n_step and policy_frequency must be positive")
        if self.num_atoms < 2 or self.v_min >= self.v_max:
            raise ValueError("C51 requires at least two atoms and v_min < v_max")
        if self.std_min < 0 or self.std_max < self.std_min:
            raise ValueError("std_min and std_max must satisfy 0 <= std_min <= std_max")


__all__ = ["FastTD3Config"]
