import torch
import torch.nn as nn
import torch.nn.functional as F

from ....model.torch.layer import LayerNorm, Linear


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


