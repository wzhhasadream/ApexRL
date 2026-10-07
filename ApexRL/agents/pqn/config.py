# Adapted from mttga/purejaxql PQN (Apache-2.0), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/mttga/purejaxql/blob/47af6d7b35c89ddfe633aaf7341bdb8964cb7cce/purejaxql/pqn_atari.py
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PQNConfig:
    """PQN defaults for vectorized Atari rollouts."""

    seed: int = 1
    total_timesteps: int = int(5e7)
    num_envs: int = 128
    rollout_steps: int = 32
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    q_lambda: float = 0.65
    num_minibatches: int = 32
    update_epochs: int = 2
    max_grad_norm: float = 10.0
    start_e: float = 1.0
    end_e: float = 0.001
    hidden_dim: int = 512
    compute_type: Literal["float32", "bfloat16"] = "float32"
    norm_type: Literal["bn", "ln", None] = "ln"

    action_dim: int = field(init=False, default=0)
    action_is_discrete: bool = field(init=False, default=True)
    image_obs: bool = field(init=False, default=True)
    asymmetric_obs: bool = field(init=False, default=False)


__all__ = ["PQNConfig"]
