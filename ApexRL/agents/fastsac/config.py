# Adapted from amazon-far/holosoma FastSAC (Apache-2.0), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/amazon-far/holosoma/tree/d18d6cc50f872c15e904a22ceac22313cec955c8/src/holosoma/holosoma/agents/fast_sac
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class FastSACConfig:
    seed: int = 1
    num_train_env: int = 4096
    num_learning_iterations: int = 50_000
    # These global counts default from num_train_env and can be overridden explicitly.
    total_timesteps: int | None = None
    grad_step_per_interaction_step: float = 8.0
    buffer_size: int | None = None
    learning_starts: int | None = None
    batch_size: int = 8192
    n_step: int = 1
    gamma: float = 0.97
    tau: float = 0.125
    policy_frequency: int = 4
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    weight_decay: float = 0.001
    actor_hidden_dim: int = 512
    critic_hidden_dim: int = 768
    num_q_networks: int = 2
    num_atoms: int = 101
    v_min: float = -20.0
    v_max: float = 20.0
    alpha_init: float = 0.001
    target_entropy_ratio: float = 0.0
    use_autotune: bool = True
    use_tanh: bool = True
    log_std_min: float = -5.0
    log_std_max: float = 0.0
    obs_normalization: bool = True
    use_layer_norm: bool = True
    max_grad_norm: float | None = None
    compute_type: Literal["float32", "bfloat16"] = "float32"

    # Filled from the environment; buffer sizes count total transitions.
    action_dim: int = field(init=False, default=0)
    action_is_discrete: bool = field(init=False, default=False)
    asymmetric_obs: bool = field(init=False, default=False)
    image_obs: bool = field(init=False, default=False)
    target_entropy: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.num_train_env < 1 or self.num_learning_iterations < 1:
            raise ValueError("num_train_env and num_learning_iterations must be positive")
        if self.total_timesteps is None:
            self.total_timesteps = self.num_learning_iterations * self.num_train_env
        if self.buffer_size is None:
            self.buffer_size = 1024 * self.num_train_env
        if self.learning_starts is None:
            self.learning_starts = 10 * self.num_train_env
        if self.n_step < 1 or self.policy_frequency < 1 or self.num_q_networks < 1:
            raise ValueError("n_step, policy_frequency and num_q_networks must be positive")
        if self.num_atoms < 2 or self.v_min >= self.v_max:
            raise ValueError("C51 requires at least two atoms and v_min < v_max")


__all__ = ["FastSACConfig"]
