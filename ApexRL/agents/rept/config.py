from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from ..base_config import BaseConfig


@dataclass
class RePTConfig(BaseConfig):
    seed: int = 1
    num_train_env: int = 1
    total_timesteps: int = 1_000_000
    grad_step_per_interaction_step: float = 1.0
    compute_type: Literal["float32", "bfloat16"] = "bfloat16"
    vision_encoder: Literal["drq_dot", "drqv2", "vsimba"] = "drqv2"

    buffer_size: int = 1_000_000
    learning_starts: int = 10_000
    batch_size: int = 512
    decay_step: int = 80_000
    n_step: int = 1
    gamma: float = 0.99
    buffer_device: Literal["cpu", "cuda"] = "cpu"

    policy_frequency: int = 2
    target_frequency: int = 1
    tau: float = 0.01
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    end_lr: float = 1.5e-4
    nce_coef: float = 1.0
    reward_coef: float = 1.0

    z_s_dim: int = 128
    z_sa_dim: int = 256
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 256
    num_encoder_blocks: int = 1
    num_actor_blocks: int = 1
    num_critic_blocks: int = 1
    num_q: int = 2
    num_head: int = 101
    reward_num_bins: int = 101  # Set to 1 for scalar reward regression.
    use_bias: bool = False
    actor_normalize_parameters: bool = False
    critic_normalize_parameters: bool = True
    normalize_rewards: bool = True
    action_repeat: int = 1

    # Filled by the agent from the environment, not exposed as CLI options.
    action_dim: int = field(init=False, default=0)
    action_is_discrete: bool = field(init=False, default=False)
    asymmetric_obs: bool = field(init=False, default=False)
    image_obs: bool = field(init=False, default=False)
    target_entropy: float = field(init=False, default=0.0)

    ENV_TYPE_PRESETS: ClassVar[dict[str, dict[str, Any]]] = {
        "cpu_sim_visual": dict(total_timesteps=500_000, learning_starts=5_000, action_repeat=2,
                               batch_size=256, grad_step_per_interaction_step=0.5, n_step=3),
    }
