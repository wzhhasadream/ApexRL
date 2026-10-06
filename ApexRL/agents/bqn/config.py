from dataclasses import dataclass, field
from typing import Literal


@dataclass
class BQNConfig:
    seed: int = 1
    total_timesteps: int = 2_500_000
    grad_step_per_interaction_step: float = 0.25
    buffer_size: int = 1_000_000
    learning_starts: int = 20_000
    batch_size: int = 256
    n_step: int = 3
    gamma: float = 0.99
    decay_step: int = 80_000
    compute_type: Literal["float32", "bfloat16"] = "float32"
    q_lr: float = 1e-4
    tau: float = 0.005
    critic_hidden_dim: int = 512
    num_bins: int = 101
    min_value: float = -5.0
    max_value: float = 5.0
    use_bn: bool = True
    normalize_rewards: bool = True
    normalized_g_max: float = 5.0
    epsilon: float = 0.01
    epsilon_decay_steps: int = 250_000

    action_dim: int = field(init=False, default=0)
    action_is_discrete: bool = field(init=False, default=True)
    image_obs: bool = field(init=False, default=True)
    asymmetric_obs: bool = field(init=False, default=False)
    target_entropy: float = field(init=False, default=0.0)
