import torch
import torch.nn.functional as F
from torch import nn

from .layer import BatchNorm, EnsembleBatchNorm, EnsembleLinear, EnsembleRMSNorm, Linear, RMSNorm


class FlashSACEmbedder(nn.Module):
    """Input BatchNorm followed by an orthogonally initialized linear layer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.norm = BatchNorm(input_dim)
        self.w = Linear(input_dim, hidden_dim, bias=use_bias)

    def forward(
        self,
        x: torch.Tensor,
        training: bool | None = None,
    ) -> torch.Tensor:
        return self.w(self.norm(x, training=training))


class FlashSACBlock(nn.Module):
    """FlashSAC residual MLP block for a single network."""

    def __init__(
        self,
        hidden_dim: int,
        expansion: int = 4,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.w1 = Linear(hidden_dim, hidden_dim * expansion, bias=use_bias)
        self.w2 = Linear(hidden_dim * expansion, hidden_dim, bias=use_bias)
        self.norm1 = BatchNorm(hidden_dim * expansion)
        self.norm2 = BatchNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        training: bool | None = None,
    ) -> torch.Tensor:
        residual = x
        x = F.relu(self.norm1(self.w1(x), training=training))
        x = F.relu(self.norm2(self.w2(x), training=training))
        return x + residual


class FlashSACEncoder(nn.Module):
    """FlashSAC encoder for a single actor or Q network."""

    def __init__(
        self,
        input_dim: int,
        num_blocks: int,
        hidden_dim: int,
        expansion: int = 4,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if num_blocks < 0:
            raise ValueError(f"num_blocks must be non-negative, got {num_blocks}")
        self.embedder = FlashSACEmbedder(input_dim, hidden_dim, use_bias)
        self.blocks = nn.ModuleList(
            [
                FlashSACBlock(hidden_dim, expansion, use_bias)
                for _ in range(num_blocks)
            ]
        )
        self.post_norm = RMSNorm(hidden_dim, epsilon=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        training: bool | None = None,
    ) -> torch.Tensor:
        x = self.embedder(x, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        return self.post_norm(x)


class EnsembleFlashSACEmbedder(nn.Module):
    """Ensemble input BatchNorm followed by an ensemble linear layer."""

    def __init__(
        self,
        num_ensemble: int,
        input_dim: int,
        hidden_dim: int,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_ensemble = num_ensemble
        self.norm = EnsembleBatchNorm(num_ensemble, input_dim)
        self.w = EnsembleLinear(num_ensemble, input_dim, hidden_dim, bias=use_bias)

    def forward(
        self,
        x: torch.Tensor,
        training: bool | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[0] != self.num_ensemble:
            raise ValueError(
                f"expected input shape [{self.num_ensemble}, B, D], got {tuple(x.shape)}"
            )
        return self.w(self.norm(x, training=training))


class EnsembleFlashSACBlock(nn.Module):
    """FlashSAC residual MLP block with independent parameters per member."""

    def __init__(
        self,
        num_ensemble: int,
        hidden_dim: int,
        expansion: int = 4,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_ensemble = num_ensemble
        self.w1 = EnsembleLinear(
            num_ensemble,
            hidden_dim,
            hidden_dim * expansion,
            bias=use_bias,
        )
        self.w2 = EnsembleLinear(
            num_ensemble,
            hidden_dim * expansion,
            hidden_dim,
            bias=use_bias,
        )
        self.norm1 = EnsembleBatchNorm(num_ensemble, hidden_dim * expansion)
        self.norm2 = EnsembleBatchNorm(num_ensemble, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        training: bool | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[0] != self.num_ensemble:
            raise ValueError(
                f"expected input shape [{self.num_ensemble}, B, D], got {tuple(x.shape)}"
            )
        residual = x
        x = F.relu(self.norm1(self.w1(x), training=training))
        x = F.relu(self.norm2(self.w2(x), training=training))
        return x + residual


class FlashSACEnsembleEncoder(nn.Module):
    """FlashSAC encoder with a leading ensemble dimension throughout."""

    def __init__(
        self,
        num_ensemble: int,
        input_dim: int,
        num_blocks: int,
        hidden_dim: int,
        expansion: int = 4,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if num_blocks < 0:
            raise ValueError(f"num_blocks must be non-negative, got {num_blocks}")
        self.num_ensemble = num_ensemble
        self.embedder = EnsembleFlashSACEmbedder(
            num_ensemble,
            input_dim,
            hidden_dim,
            use_bias,
        )
        self.blocks = nn.ModuleList(
            [
                EnsembleFlashSACBlock(
                    num_ensemble,
                    hidden_dim,
                    expansion,
                    use_bias,
                )
                for _ in range(num_blocks)
            ]
        )
        self.post_norm = EnsembleRMSNorm(num_ensemble, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        training: bool | None = None,
    ) -> torch.Tensor:
        x = self.embedder(x, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        return self.post_norm(x)
