from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RePTConfig:
    seed: int = 1
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

    actor_hidden_dim: int = 128
    actor_num_blocks: int = 2
    critic_hidden_dim: int = 256
    critic_num_trunk_blocks: int = 1
    num_encoder_blocks: int = 1
    num_q: int = 2
    num_head: int = 101
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


# Environment fields such as action_repeat are supplied by the entrypoint Args.
REPT_PROFILE_OVERRIDES = {
    "state": {},
    "visual": dict(total_timesteps=500_000, learning_starts=5_000, action_repeat=2, batch_size=256, grad_step_per_interaction_step=0.5, n_step=3),
    "atari": dict(total_timesteps=2500_000, action_repeat=4, batch_size=256, grad_step_per_interaction_step=0.5)
}
