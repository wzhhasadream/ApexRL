# Adapted from DAVIAN-Robotics/V-Simba (Apache-2.0), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/DAVIAN-Robotics/V-Simba/tree/be811e968bc02589fbb32f3be79f9a7d9a8fa86d/scale_rl/agents/vsimba
import torch
from torch import nn
import torch.nn.functional as F

from ....model.torch import CategoricalPolicy, Linear
from ....model.torch.backbones import (
    VSimbaEncoder,
    VSimbaProjector,
    VSimbaVisionEncoder,
)
from ....model.torch.layer import EnsembleLayerNorm, EnsembleLinear
from ....model.torch.policy import SquashedTanhGaussianPolicy


class Actor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        num_blocks: int = 1,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.trunk = VSimbaEncoder(input_dim, num_blocks, hidden_dim)
        self.mean_w = Linear(hidden_dim, action_dim)
        self.log_std_w = Linear(hidden_dim, action_dim)
        self.policy = SquashedTanhGaussianPolicy(
            -1.0, 1.0, log_std_min=-10.0, log_std_max=2.0
        )

    def forward(self, vision_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(vision_z)
        return self.mean_w(z).float(), self.log_std_w(z).float()

    def get_action(self, vision_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(vision_z)
        return self.policy.sample_and_log_prob(mean, log_std)

    def get_mean_action(self, vision_z: torch.Tensor) -> torch.Tensor:
        mean, _ = self(vision_z)
        return self.policy.mean_action(mean)


class EnsembleVSimbaEmbedder(nn.Module):
    """V-Simba feature projection with independent ensemble parameters."""

    def __init__(self, num_ensemble: int, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w = EnsembleLinear(num_ensemble, input_dim, hidden_dim)
        self.feature_norm = EnsembleLayerNorm(num_ensemble, hidden_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_norm(self.w(x))


class EnsembleVSimbaBlock(nn.Module):
    """Pre-normalized residual V-Simba MLP block for all critic heads."""

    def __init__(self, num_ensemble: int, hidden_dim: int, expansion: int = 4) -> None:
        super().__init__()
        self.feature_norm = EnsembleLayerNorm(num_ensemble, hidden_dim, eps=1e-6)
        self.w1 = EnsembleLinear(num_ensemble, hidden_dim, hidden_dim * expansion)
        self.w2 = EnsembleLinear(num_ensemble, hidden_dim * expansion, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.feature_norm(x)
        x = self.w2(F.relu(self.w1(x)))
        return residual + x


class EnsembleVSimbaEncoder(nn.Module):
    """V-Simba MLP encoder with a leading ensemble axis."""

    def __init__(
        self,
        num_ensemble: int,
        input_dim: int,
        num_blocks: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.embedder = EnsembleVSimbaEmbedder(num_ensemble, input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [EnsembleVSimbaBlock(num_ensemble, hidden_dim) for _ in range(num_blocks)]
        )
        self.post_norm = EnsembleLayerNorm(num_ensemble, hidden_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedder(x)
        for block in self.blocks:
            x = block(x)
        return self.post_norm(x)


class Critic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        num_qs: int = 1,
        num_blocks: int = 2,
        hidden_dim: int = 512,
        action_embed_dim: int = 128,
        c_shift: float = 3.0,
        num_bins: int = 101,
        min_v: float = -5.0,
        max_v: float = 5.0,
    ) -> None:
        super().__init__()
        if num_qs < 1:
            raise ValueError(f"num_qs must be positive, got {num_qs}")

        self.num_qs = num_qs
        self.projector = VSimbaProjector(c_shift)
        self.action_embedder = EnsembleVSimbaEmbedder(
            num_qs, action_dim + 1, action_embed_dim
        )
        self.encoder = EnsembleVSimbaEncoder(
            num_qs, input_dim + action_embed_dim, num_blocks, hidden_dim
        )
        self.w = EnsembleLinear(num_qs, hidden_dim, num_bins)
        self.dist = CategoricalPolicy(num_bins, min_v, max_v)

    def forward(self, vision_z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return categorical value logits with shape [num_qs, B, num_bins]."""
        action_z = self.action_embedder(self.projector(actions).unsqueeze(0).expand(self.num_qs, -1, -1))
        vision_z = vision_z.unsqueeze(0).expand(self.num_qs, -1, -1)
        return self.w(self.encoder(torch.cat((vision_z, action_z), dim=-1))).float()

    def q_values(self, vision_z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.dist.q_values(self(vision_z, actions))


__all__ = ["Actor", "Critic", "VSimbaVisionEncoder"]
