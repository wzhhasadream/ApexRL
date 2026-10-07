# Adapted from vwxyzjn/cleanrl (MIT) and leggedrobotics/rsl_rl (BSD-3-Clause), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool.py
# https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/ppo.py
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PPOConfig:
    """PPO defaults for massively parallel continuous control (rsl_rl style).

    Use ``PPOConfig.atari()`` for the CleanRL Atari setting (discrete actions, Nature CNN).
    """

    seed: int = 1
    total_timesteps: int = 100_000_000
    num_envs: int = 4096
    rollout_steps: int = 24
    lr: float = 1e-3
    anneal_lr: bool = False             # linear decay to 0 over training
    desired_kl: float | None = 0.01     # adaptive lr from policy KL; None keeps lr fixed
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_epochs: int = 5
    num_mini_batches: int = 4
    algo: Literal["ppo", "spo"] = "ppo"
    clip_coef: float = 0.2
    clip_value: bool = True
    value_coef: float = 1.0
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    normalize_advantages: bool = True
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    init_std: float = 1.0
    cnn_hidden_dim: int = 512           # image observations only
    compute_type: Literal["float32", "bfloat16"] = "float32"

    # Filled from the environment
    action_dim: int = field(init=False, default=0)
    action_is_discrete: bool = field(init=False, default=False)
    asymmetric_obs: bool = field(init=False, default=False)
    image_obs: bool = field(init=False, default=False)

    @classmethod
    def atari(cls, **kwargs) -> "PPOConfig":
        # CleanRL ppo_atari_envpool.py
        defaults = dict(total_timesteps=10_000_000, num_envs=8, rollout_steps=128, lr=2.5e-4, anneal_lr=True, desired_kl=None, num_epochs=4, num_mini_batches=4, clip_coef=0.1, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5, activation="relu")
        return cls(**{**defaults, **kwargs})


__all__ = ["PPOConfig"]
