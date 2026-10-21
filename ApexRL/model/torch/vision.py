import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import BatchNorm, BatchRenorm, LayerNorm, RMSNorm, Linear


# ------------------------------- MR.Q / DrQ-v2 pixel encoders ---------------


class MRQVisionEncoder(nn.Module):
    """MR.Q vision encoder (Fujimoto et al., 2025) for frame-stacked pixel observations.

    Input [N, F, H, W, C] uint8 frames; output [N, latent_dim] ELU features.
    Four conv layers (three stride-2 then one stride-1, 32 channels) with ELU,
    followed by Linear + LayerNorm, as in facebookresearch/MRQ.
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
                "MRQVisionEncoder input is too small for the MR.Q CNN: "
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
                "MRQVisionEncoder expects [N, F, H, W, C] observations, "
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
    """DrQ-v2 vision encoder (Yarats et al., 2021) for frame-stacked pixel observations.

    Input [N, F, H, W, C] uint8 frames; output [N, latent_dim] tanh features.
    One stride-2 conv then three stride-1 convs (32 channels) with ReLU, followed
    by Linear + LayerNorm; convs/fc use orthogonal(1) init, as in DrQ-v2.
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
