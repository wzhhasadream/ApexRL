from math import prod
from typing import Literal
import torch
import torch.nn.functional as F
from torch import nn

from ....model.torch.mlp import FlashSACEncoder, FlashSACEnsembleEncoder
from ....model.torch.layer import (
    BatchNorm,
    EnsembleBatchNorm,
    EnsembleLinear,
    EnsembleRMSNorm,
    Linear,
    RMSNorm,
)
from ....model.torch.dist_head import CategoricalPolicy, QuantilePolicy
from ....model.torch.policy import SquashedTanhGaussianPolicy


def _flattened_dim(observation_dim: int | tuple[int, ...]) -> int:
    return observation_dim if isinstance(observation_dim, int) else prod(observation_dim)




class FlashSACActor(nn.Module):
    """Continuous actor whose policy distribution owns the action bounds."""

    def __init__(
        self,
        obs_dim: int | tuple[int, ...],
        action_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        action_low: torch.Tensor | float = -1.0,
        action_high: torch.Tensor | float = 1.0,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        squash_log_std: bool = True,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.obs_dim = _flattened_dim(obs_dim)
        self.action_dim = action_dim
        self.encoder = FlashSACEncoder(
            self.obs_dim,
            num_blocks,
            hidden_dim,
            use_bias=use_bias,
        )
        self.fc_mean = Linear(hidden_dim, action_dim)
        self.fc_log_std = Linear(hidden_dim, action_dim)
        self.policy = SquashedTanhGaussianPolicy(
            action_low,
            action_high,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            squash_log_std=squash_log_std,
        )

    def forward(
        self,
        observations: torch.Tensor,
        training: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.ndim != 2 or observations.shape[-1] != self.obs_dim:
            raise ValueError(
                f"expected observations [B, {self.obs_dim}], got {tuple(observations.shape)}"
            )
        x = self.encoder(observations, training=training)
        return self.fc_mean(x), self.fc_log_std(x)

    def get_action(
        self,
        observations: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        training: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(observations, training=training)
        return self.policy.sample_and_log_prob(mean, log_std, noise=noise)

    def get_mean_action(self, observations: torch.Tensor) -> torch.Tensor:
        mean, _ = self(observations, training=False)
        return self.policy.mean_action(mean)

    def get_mean_std(
        self,
        observations: torch.Tensor,
        training: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_log_std = self(observations, training=training)
        return mean, self.policy.transform_log_std(raw_log_std)


class FlashSACQNetwork(nn.Module):
    """Single FlashSAC Q network with optional distributional heads."""

    def __init__(
        self,
        obs_dim: int | tuple[int, ...],
        action_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        num_head: int = 1,
        dist_type: Literal["scalar", "quantile", "ce"] = "scalar",
        v_min: float = -5.0,
        v_max: float = 5.0,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if num_head < 1:
            raise ValueError(f"num_head must be positive, got {num_head}")
        self.obs_dim = _flattened_dim(obs_dim)
        self.action_dim = action_dim
        self.num_head = num_head
        self.encoder = FlashSACEncoder(
            self.obs_dim + self.action_dim,
            num_blocks,
            hidden_dim,
            use_bias=use_bias,
        )
        self.out = Linear(hidden_dim, num_head)
        self.dist = {
            "quantile": QuantilePolicy(num_head),
            "ce": CategoricalPolicy(num_head, v_min, v_max),
        }.get(dist_type)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        training: bool | None = None,
    ) -> torch.Tensor:
        if observations.ndim != 2 or actions.ndim != 2:
            raise ValueError("observations and actions must both have shape [B, D]")
        if observations.shape[0] != actions.shape[0]:
            raise ValueError("observations and actions must have the same batch size")
        x = torch.cat((observations, actions), dim=-1)
        return self.out(self.encoder(x, training=training))

    def q_values(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        training: bool | None = None,
    ) -> torch.Tensor:
        values = self(observations, actions, training=training)
        return values if self.dist is None else self.dist.q_values(values)


class FlashSACDoubleCritic(nn.Module):
    """Critic ensemble with all member parameters stored in leading dimension E."""

    def __init__(
        self,
        obs_dim: int | tuple[int, ...],
        action_dim: int,
        num_q: int = 2,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        num_head: int = 1,
        dist_type: Literal["scalar", "quantile", "ce"] = "scalar",
        v_min: float = -5.0,
        v_max: float = 5.0,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if num_q < 1:
            raise ValueError(f"num_q must be positive, got {num_q}")
        if num_head < 1:
            raise ValueError(f"num_head must be positive, got {num_head}")
        self.obs_dim = _flattened_dim(obs_dim)
        self.action_dim = action_dim
        self.num_q = num_q
        self.num_head = num_head
        self.encoder = FlashSACEnsembleEncoder(
            num_q,
            self.obs_dim + self.action_dim,
            num_blocks,
            hidden_dim,
            use_bias=use_bias,
        )
        self.out = EnsembleLinear(num_q, hidden_dim, num_head)
        self.dist = {
            "quantile": QuantilePolicy(num_head),
            "ce": CategoricalPolicy(num_head, v_min, v_max),
        }.get(dist_type)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        training: bool | None = None,
    ) -> torch.Tensor:
        x = torch.concat([observations, actions], dim=-1)
        x = torch.broadcast_to(x[None, ...], (self.num_q, *x.shape))
        return self.out(self.encoder(x, training=training))

    def q_values(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        training: bool | None = None,
    ) -> torch.Tensor:
        values = self(observations, actions, training=training)
        return values if self.dist is None else self.dist.q_values(values)
