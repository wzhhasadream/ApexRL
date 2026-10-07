# Adapted from DAVIAN-Robotics/V-Simba (Apache-2.0), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/DAVIAN-Robotics/V-Simba/tree/be811e968bc02589fbb32f3be79f9a7d9a8fa86d/scale_rl/agents/vsimba
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from ..base_config import BaseConfig


@dataclass
class VSimbaConfig(BaseConfig):
    seed: int = 1
    num_train_env: int = 1
    total_timesteps: int = 500_000
    grad_step_per_interaction_step: float = 1.0
    buffer_size: int = 1_000_000
    learning_starts: int = 5_000
    batch_size: int = 256
    n_step: int = 3
    max_episode_steps: int = 1000
    action_repeat: int = 2
    gamma: float | None = None
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    compute_type: Literal["float32", "bfloat16"] = "float32"
    encoder_num_blocks: int = 2
    encoder_num_channels: int = 32
    encoder_conv_kernel_size: int = 3
    actor_num_blocks: int = 1
    actor_hidden_dim: int = 128
    critic_use_cdq: bool = False
    critic_num_blocks: int = 2
    critic_hidden_dim: int = 512
    critic_action_embed_dim: int = 128
    critic_c_shift: float = 3.0
    critic_num_bins: int = 101
    critic_min_v: float = -5.0
    critic_max_v: float = 5.0
    target_tau: float = 0.005
    temp_initial_value: float = 0.01
    temp_target_entropy_coef: float = -0.5
    normalize_rewards: bool = True
    normalized_g_max: float = 5.0
    decay_step: int = 80_000

    # Filled by the agent from the environment, not exposed as CLI options.
    action_dim: int = field(init=False, default=0)
    action_is_discrete: bool = field(init=False, default=False)
    asymmetric_obs: bool = field(init=False, default=False)
    image_obs: bool = field(init=False, default=False)
    target_entropy: float = field(init=False, default=0.0)

    ENV_TYPE_PRESETS: ClassVar[dict[str, dict[str, Any]]] = {
    }

    def __post_init__(self):
        if self.gamma is None:
            effective_episode_length = self.max_episode_steps / self.action_repeat
            self.gamma = max(min(1.0 - 5.0 / effective_episode_length, 0.995), 0.95)
