from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from ..base_config import BaseConfig


@dataclass
class WarpSACConfig(BaseConfig):
    seed: int = 1
    num_train_env: int = 1
    total_timesteps: int = 1_000_000
    grad_step_per_interaction_step: float = 1.0
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000
    batch_size: int = 512
    n_step: int = 1
    gamma: float = 0.99
    decay_step: int = 80_000
    buffer_device: str = "cpu"
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    end_lr: float = 3e-5
    policy_frequency: int = 2
    target_frequency: int = 1
    tau: float = 0.005
    actor_hidden_dim: int = 256
    actor_num_blocks: int = 2
    critic_hidden_dim: int = 256
    critic_num_blocks: int = 2
    num_q: int = 2
    num_head: int = 101
    use_bias: bool = True
    dist_type: Literal["quantile", "ce", "scalar"] = "ce"
    q_agg: Literal["mean", "min"] = "min"
    actor_normalize_parameters: bool = True
    critic_normalize_parameters: bool = True
    normalize_rewards: bool = True
    compute_type: Literal["float32", "bfloat16"] = "float32"

    action_dim: int = field(init=False, default=0)
    action_is_discrete: bool = field(init=False, default=False)
    asymmetric_obs: bool = field(init=False, default=False)
    image_obs: bool = field(init=False, default=False)
    target_entropy: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.num_train_env < 1 or self.total_timesteps < 1 or self.buffer_size < self.num_train_env:
            raise ValueError("num_train_env, total_timesteps and buffer_size must be positive")
        if self.n_step < 1 or self.policy_frequency < 1 or self.target_frequency < 1 or self.num_q < 1:
            raise ValueError("n_step, policy_frequency, target_frequency and num_q must be positive")
        if self.num_head < 1 or (self.dist_type != "scalar" and self.num_head < 2):
            raise ValueError("distributional critics require at least two heads")

    ENV_TYPE_PRESETS: ClassVar[dict[str, dict[str, Any]]] = {
        # cpu_sim uses the dataclass defaults
        "gpu_sim": dict(num_train_env=1024, total_timesteps=50_000_896, buffer_size=10_000_000, learning_starts=100_000, batch_size=2048, grad_step_per_interaction_step=2.0,
                        decay_step=2_000, compute_type="bfloat16", buffer_device="cuda", actor_normalize_parameters=False, critic_normalize_parameters=False),
        "playground": dict(gamma=0.97),
        "maniskill": dict(gamma=0.9),
        "isaaclab": dict(gamma=0.99, n_step=3),
        "mjlab": dict(gamma=0.99, n_step=3),
    }


__all__ = ["WarpSACConfig"]
