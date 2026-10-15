import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import (
    BatchNorm,
    BatchRenorm,
    LayerNorm,
    EnsembleBatchNorm,
    EnsembleLinear,
    EnsembleRMSNorm,
    Linear,
    RMSNorm,
)


# ------------------------------- FlashSAC state components ---------------


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
            raise ValueError(
                f"num_blocks must be non-negative, got {num_blocks}"
            )
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
        self.w = EnsembleLinear(
            num_ensemble, input_dim, hidden_dim, bias=use_bias
        )

    def forward(
        self,
        x: torch.Tensor,
        training: bool | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[0] != self.num_ensemble:
            raise ValueError(
                f"expected input shape [{self.num_ensemble}, B, D], "
                f"got {tuple(x.shape)}"
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
        self.norm1 = EnsembleBatchNorm(
            num_ensemble, hidden_dim * expansion
        )
        self.norm2 = EnsembleBatchNorm(num_ensemble, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        training: bool | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[0] != self.num_ensemble:
            raise ValueError(
                f"expected input shape [{self.num_ensemble}, B, D], "
                f"got {tuple(x.shape)}"
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
            raise ValueError(
                f"num_blocks must be non-negative, got {num_blocks}"
            )
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


# ------------------------------- BatchNorm pixel components --------------




class DRQVisionEncoder(nn.Module):
    """DR.Q-style CNN encoder for frame-stacked pixel observations.

    Input is expected as [N, F, H, W, C]. The encoder concatenates F*C
    along the channel axis and applies three stride-2 convolutions
    followed by one stride-1 convolution.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int, int],
        num_channels: int = 32,
        num_blocks: int = 2,
        conv_kernel_size: int = 3,
        use_bias: bool = True,
        latent_dim: int = 512,
    ) -> None:
        super().__init__()
        frame_count, height, width, channels = input_shape
        in_channels = frame_count * channels
        kernel_size = (conv_kernel_size, conv_kernel_size)

        self.conv1 = nn.Conv2d(
            in_channels,
            num_channels,
            kernel_size=kernel_size,
            stride=2,
            bias=use_bias,
        )
        self.blocks = nn.ModuleList(
            [
                nn.Conv2d(
                    num_channels,
                    num_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=use_bias,
                )
                for _ in range(num_blocks)
            ]
        )
        self.conv_last = nn.Conv2d(
            num_channels,
            num_channels,
            kernel_size=kernel_size,
            stride=1,
            bias=use_bias,
        )
        self.apply(self._init_weights)

        dummy = torch.zeros(1, in_channels, height, width)
        dummy = self._forward_conv_stack(dummy)
        self.feature_dim = int(dummy.numel())
        if self.feature_dim <= 0:
            raise ValueError(
                "DRQVisionEncoder input is too small for the DR.Q CNN: "
                f"got input_shape={input_shape}"
            )

        self.latent_dim = latent_dim
        self.fc = Linear(self.feature_dim, latent_dim, bias=use_bias)
        self.ln = LayerNorm(latent_dim)
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_output_dim(self) -> int:
        return self.latent_dim

    def _forward_conv_stack(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = F.elu(self.conv1(x))
        for block in self.blocks:
            x = F.elu(block(x))
        return F.elu(self.conv_last(x))

    def forward(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        if observations.ndim != 5:
            raise ValueError(
                "DRQVisionEncoder expects [N, F, H, W, C] observations, "
                f"got {observations.shape}"
            )
        # [N, F, H, W, C] -> [N, F*C, H, W] once at the boundary.
        x = observations.permute(0, 1, 4, 2, 3)
        x = x.reshape(x.shape[0], -1, *x.shape[3:])
        x = x / 255.0 - 0.5
        x = self._forward_conv_stack(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.ln(x)
        return F.elu(x)
class DRQV2VisionEncoder(nn.Module):
    """DrQ-v2 style CNN encoder for frame-stacked pixel observations.

    Input is expected as [N, F, H, W, C]. The encoder concatenates F*C
    along the channel axis and applies one stride-2 convolution followed
    by three stride-1 convolutions.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int, int],
        num_channels: int = 32,
        num_blocks: int = 3,
        conv_kernel_size: int = 3,
        use_bias: bool = True,
        latent_dim: int = 50,
    ) -> None:
        super().__init__()
        frame_count, height, width, channels = input_shape
        in_channels = frame_count * channels
        kernel_size = (conv_kernel_size, conv_kernel_size)

        self.conv1 = nn.Conv2d(
            in_channels,
            num_channels,
            kernel_size=kernel_size,
            stride=2,
            bias=use_bias,
        )
        self.blocks = nn.ModuleList(
            [
                nn.Conv2d(
                    num_channels,
                    num_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    bias=use_bias,
                )
                for _ in range(num_blocks)
            ]
        )
        self.apply(self._init_weights)

        dummy = torch.zeros(1, in_channels, height, width)
        dummy = self._forward_conv_stack(dummy)
        self.feature_dim = int(dummy.numel())
        if self.feature_dim <= 0:
            raise ValueError(
                "DRQV2VisionEncoder input is too small for the DrQ-v2 CNN: "
                f"got input_shape={input_shape}"
            )

        self.latent_dim = latent_dim
        self.fc = Linear(self.feature_dim, latent_dim, bias=use_bias)
        self.ln = LayerNorm(latent_dim)
        nn.init.orthogonal_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_output_dim(self) -> int:
        return self.latent_dim

    def _forward_conv_stack(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        for block in self.blocks:
            x = F.relu(block(x))
        return x

    def forward(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        if observations.ndim != 5:
            raise ValueError(
                "DRQV2VisionEncoder expects [N, F, H, W, C] observations, "
                f"got {observations.shape}"
            )
        x = observations.permute(0, 1, 4, 2, 3)
        x = x.reshape(x.shape[0], -1, *x.shape[3:])
        x = x / 255.0 - 0.5
        x = self._forward_conv_stack(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.ln(x)
        return torch.tanh(x).float()


# ------------------------------- V-Simba components -----------------------


class VSimbaMLP(nn.Module):
    """Two orthogonal linear layers with a ReLU between them."""

    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, use_bias: bool = True):
        super().__init__()
        self.w1 = Linear(input_dim, hidden_dim, bias=use_bias)
        self.w2 = Linear(hidden_dim, out_dim, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.relu(self.w1(x)))


class VSimbaProjector(nn.Module):
    """Append a constant coordinate before L2-normalizing the action."""

    def __init__(self, c_shift: float = 3.0):
        super().__init__()
        self.c_shift = c_shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, torch.full_like(x[..., :1], self.c_shift)], dim=-1)
        return F.normalize(x, p=2, dim=-1, eps=1e-8)


class VSimbaEmbedder(nn.Module):
    """Linear projection followed by LayerNorm."""

    def __init__(self, input_dim: int, hidden_dim: int, use_bias: bool = True):
        super().__init__()
        self.w = Linear(input_dim, hidden_dim, bias=use_bias)
        self.feature_norm = LayerNorm(hidden_dim, epsilon=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_norm(self.w(x))


class VSimbaMLPBlock(nn.Module):
    """Pre-LayerNorm residual MLP block."""

    def __init__(self, hidden_dim: int, expansion: int = 4, use_bias: bool = True):
        super().__init__()
        self.mlp = VSimbaMLP(hidden_dim, hidden_dim * expansion, hidden_dim, use_bias)
        self.feature_norm = LayerNorm(hidden_dim, epsilon=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.feature_norm(x))


class VSimbaEncoder(nn.Module):
    """Embedder, residual MLP blocks and final LayerNorm for actor/Q features."""

    def __init__(self, input_dim: int, num_blocks: int, hidden_dim: int, expansion: int = 4, use_bias: bool = True):
        super().__init__()
        self.embedder = VSimbaEmbedder(input_dim, hidden_dim, use_bias)
        self.blocks = nn.ModuleList([VSimbaMLPBlock(hidden_dim, expansion, use_bias) for _ in range(num_blocks)])
        self.post_norm = LayerNorm(hidden_dim, epsilon=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedder(x)
        for block in self.blocks:
            x = block(x)
        return self.post_norm(x)


class VSimbaConvBlock(nn.Module):
    """NHWC residual conv/MLP block with channel LayerNorm and 2x2 pooling."""

    def __init__(self, num_channels: int, kernel_size: int = 3, expansion: int = 4, use_bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size, padding=kernel_size // 2, bias=use_bias)
        nn.init.orthogonal_(self.conv.weight, gain=1)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        self.mlp = VSimbaMLP(num_channels, num_channels * expansion, num_channels, use_bias)
        self.feature_norm = LayerNorm(num_channels, epsilon=1e-6)
        self.ds_feature_norm = LayerNorm(num_channels, epsilon=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = residual + self.mlp(self.feature_norm(x))
        x = F.max_pool2d(x.permute(0, 3, 1, 2), kernel_size=2, stride=2).permute(0, 2, 3, 1)
        return self.ds_feature_norm(x)


class VSimbaVisionEncoder(nn.Module):
    """Original LayerNorm V-Simba CNN: [N, F, H, W, C] -> flattened HWC features."""

    def __init__(
        self, input_shape: tuple[int, int, int, int], num_channels: int = 32,
        num_blocks: int = 2, conv_kernel_size: int = 3, use_bias: bool = True,
    ):
        super().__init__()
        frame_count, height, width, channels = input_shape
        in_channels = frame_count * channels
        self.input_norm = LayerNorm(in_channels, epsilon=1e-6)
        self.stem_conv = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=2, padding=1, bias=use_bias)
        nn.init.orthogonal_(self.stem_conv.weight, gain=1)
        if self.stem_conv.bias is not None:
            nn.init.zeros_(self.stem_conv.bias)
        self.stem_norm = LayerNorm(num_channels, epsilon=1e-6)
        self.blocks = nn.ModuleList([VSimbaConvBlock(num_channels, conv_kernel_size, use_bias=use_bias) for _ in range(num_blocks)])
        height = (height + 1) // 2 // (2 ** (num_blocks + 1))
        width = (width + 1) // 2 // (2 ** (num_blocks + 1))
        self.feature_dim = height * width * num_channels

    def get_output_dim(self) -> int:
        return self.feature_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.permute(0, 2, 3, 1, 4)
        x = x.reshape(*x.shape[:3], -1) / 255.0 - 0.5
        x = self.stem_conv(self.input_norm(x).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = F.max_pool2d(self.stem_norm(x).permute(0, 3, 1, 2), kernel_size=2, stride=2).permute(0, 2, 3, 1)
        for block in self.blocks:
            x = block(x)
        return x.reshape(x.shape[0], -1)


# ------------------------------- Nature CNN (Atari) ---------------


class _Norm(nn.Module):
    """Apply feature-last normalization to the channel axis of NCHW inputs."""

    def __init__(self, features: int, norm_type: Literal["bn", "brn", "ln", "rms"] | None) -> None:
        super().__init__()
        self.norm_type = norm_type
        norm_cls = {"bn": BatchNorm, "brn": BatchRenorm, "ln": LayerNorm, "rms": RMSNorm, None: None}[norm_type]
        self.norm = norm_cls(features) if norm_cls is not None else None

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        if self.norm is None:
            return x
        x = x.movedim(1, -1)
        x = self.norm(x, training=training) if self.norm_type in ("bn", "brn") else self.norm(x)
        return x.movedim(-1, 1)


class NatureCNN(nn.Module):
    """Nature DQN encoder (Mnih et al., 2015) for stacked Atari frames.

    Input [B, F, H, W, C] uint8 frames; output [B, hidden_dim] ReLU features.
    Conv layers use orthogonal(sqrt(2)) init, as in CleanRL; norm_type adds
    bn/brn/ln/rms/None normalization before each activation.
    """

    def __init__(self, observation_shape: tuple[int, int, int, int], hidden_dim: int = 512, norm_type: Literal["bn", "brn", "ln", "rms"] | None = None) -> None:
        super().__init__()
        frames, height, width, channels = observation_shape
        in_channels = frames * channels
        self.conv1 = nn.Conv2d(in_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.norm1 = _Norm(32, norm_type)
        self.norm2 = _Norm(64, norm_type)
        self.norm3 = _Norm(64, norm_type)
        for conv in (self.conv1, self.conv2, self.conv3):
            nn.init.orthogonal_(conv.weight, math.sqrt(2))
            nn.init.zeros_(conv.bias)
        with torch.no_grad():
            flatten_dim = self._convs(torch.zeros(1, in_channels, height, width), training=False).shape[-1]
        self.fc = Linear(flatten_dim, hidden_dim)
        nn.init.orthogonal_(self.fc.weight, math.sqrt(2))
        self.norm_fc = _Norm(hidden_dim, norm_type)
        self.out_dim = hidden_dim

    def _convs(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        x = F.relu(self.norm1(self.conv1(x), training))
        x = F.relu(self.norm2(self.conv2(x), training))
        x = F.relu(self.norm3(self.conv3(x), training))
        return x.flatten(1)

    def forward(self, observations: torch.Tensor, training: bool = False) -> torch.Tensor:
        # [B, F, H, W, C] uint8 -> [B, F * C, H, W] float in [0, 1]
        x = observations.permute(0, 1, 4, 2, 3).flatten(1, 2).float() / 255.0
        return F.relu(self.norm_fc(self.fc(self._convs(x, training)), training))


# ------------------------------- HardMax CNN (Atari) ---------------


class HardMaxBlock(nn.Module):
    """HardMax conv block: two parallel convs fused by a Hadamard product, then max-pooled.

    Both branches apply normalization + GELU before the element-wise product; convs use
    xavier_normal init, as in Kooi et al. (2025).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int, pool_stride: int, pool_padding: int, norm_type: Literal["bn", "brn", "ln", "rms"] | None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
        self.norm1 = _Norm(out_channels, norm_type)
        self.norm2 = _Norm(out_channels, norm_type)
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding
        for conv in (self.conv1, self.conv2):
            nn.init.xavier_normal_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        x1 = F.gelu(self.norm1(self.conv1(x), training))
        x2 = F.gelu(self.norm2(self.conv2(x), training))
        return F.max_pool2d(x1 * x2, self.pool_size, stride=self.pool_stride, padding=self.pool_padding)


class HardMaxCNN(nn.Module):
    """Hadamax pixel encoder (Kooi et al., 2025) for stacked Atari frames.

    Input [B, F, H, W, C] uint8 frames; output [B, hidden_dim] GELU features.
    Convs use xavier_normal init and bn/brn/ln/rms/None normalization + GELU before a
    Hadamard-product fusion; the head fc uses he_normal, as in github.com/Jacobkooi/Hadamax.
    """

    def __init__(self, observation_shape: tuple[int, int, int, int], hidden_dim: int = 512, norm_type: Literal["bn", "brn", "ln", "rms"] | None = "ln") -> None:
        super().__init__()
        frames, height, width, channels = observation_shape
        in_channels = frames * channels
        self.block1 = HardMaxBlock(in_channels, 32, 8, 4, 4, 0, norm_type)
        self.block2 = HardMaxBlock(32, 64, 4, 2, 2, 0, norm_type)
        self.block3 = HardMaxBlock(64, 64, 3, 3, 1, 1, norm_type)
        with torch.no_grad():
            flatten_dim = self._blocks(torch.zeros(1, in_channels, height, width), training=False).shape[-1]
        self.fc = nn.Linear(flatten_dim, hidden_dim)
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.fc.bias)
        self.norm_fc = _Norm(hidden_dim, norm_type)
        self.out_dim = hidden_dim

    def _blocks(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        x = self.block1(x, training)
        x = self.block2(x, training)
        x = self.block3(x, training)
        return x.reshape(x.shape[0], -1)

    def forward(self, observations: torch.Tensor, training: bool = False) -> torch.Tensor:
        # [B, F, H, W, C] uint8 -> [B, F * C, H, W] float in [0, 1]
        x = observations.permute(0, 1, 4, 2, 3).flatten(1, 2).float() / 255.0
        return F.gelu(self.norm_fc(self.fc(self._blocks(x, training)), training))
