import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype

from ....model.jax.layer import LayerNorm, orthogonal


# ------------------------------- V-Simba components -----------------------


class VSimbaMLP(nnx.Module):
    """Two orthogonal linear layers with a ReLU between them."""

    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, rngs: nnx.Rngs, use_bias: bool = True, compute_type: Dtype = jnp.float32):
        self.w1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs, kernel_init=orthogonal(1), use_bias=use_bias, dtype=compute_type)
        self.w2 = nnx.Linear(hidden_dim, out_dim, rngs=rngs, kernel_init=orthogonal(1), use_bias=use_bias, dtype=compute_type)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.w2(nnx.relu(self.w1(x)))


class VSimbaProjector(nnx.Module):
    """Append a constant coordinate before L2-normalizing the action."""

    def __init__(self, c_shift: float = 3.0):
        self.c_shift = c_shift

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.concatenate([x, jnp.full_like(x[..., :1], self.c_shift)], axis=-1)
        return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


class VSimbaEmbedder(nnx.Module):
    """Linear projection followed by LayerNorm."""

    def __init__(self, input_dim: int, hidden_dim: int, rngs: nnx.Rngs, use_bias: bool = True, compute_type: Dtype = jnp.float32):
        self.w = nnx.Linear(input_dim, hidden_dim, rngs=rngs, kernel_init=orthogonal(1), use_bias=use_bias, dtype=compute_type)
        self.feature_norm = LayerNorm(hidden_dim, epsilon=1e-6, rngs=rngs, dtype=compute_type)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.feature_norm(self.w(x))


class VSimbaMLPBlock(nnx.Module):
    """Pre-LayerNorm residual MLP block."""

    def __init__(self, hidden_dim: int, rngs: nnx.Rngs, expansion: int = 4, use_bias: bool = True, compute_type: Dtype = jnp.float32):
        self.mlp = VSimbaMLP(hidden_dim, hidden_dim * expansion, hidden_dim, rngs, use_bias, compute_type)
        self.feature_norm = LayerNorm(hidden_dim, epsilon=1e-6, rngs=rngs, dtype=compute_type)

    def __call__(self, x: jax.Array) -> jax.Array:
        return x + self.mlp(self.feature_norm(x))


class VSimbaEncoder(nnx.Module):
    """Embedder, residual MLP blocks and final LayerNorm for actor/Q features."""

    def __init__(self, input_dim: int, num_blocks: int, hidden_dim: int, rngs: nnx.Rngs, expansion: int = 4, use_bias: bool = True, compute_type: Dtype = jnp.float32):
        self.embedder = VSimbaEmbedder(input_dim, hidden_dim, rngs, use_bias, compute_type)
        self.blocks = [VSimbaMLPBlock(hidden_dim, rngs, expansion, use_bias, compute_type) for _ in range(num_blocks)]
        self.post_norm = LayerNorm(hidden_dim, epsilon=1e-6, rngs=rngs, dtype=compute_type)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.embedder(x)
        for block in self.blocks:
            x = block(x)
        return self.post_norm(x)


class VSimbaConvBlock(nnx.Module):
    """NHWC residual conv/MLP block with channel LayerNorm and 2x2 pooling."""

    def __init__(self, num_channels: int, rngs: nnx.Rngs, kernel_size: int = 3, expansion: int = 4, use_bias: bool = True, compute_type: Dtype = jnp.float32):
        padding = kernel_size // 2
        self.conv = nnx.Conv(
            num_channels, num_channels, kernel_size=(kernel_size, kernel_size), strides=(1, 1),
            padding=((padding, padding), (padding, padding)), rngs=rngs,
            kernel_init=orthogonal(1), use_bias=use_bias, dtype=compute_type,
        )
        self.mlp = VSimbaMLP(num_channels, num_channels * expansion, num_channels, rngs, use_bias, compute_type)
        self.feature_norm = LayerNorm(num_channels, epsilon=1e-6, rngs=rngs, dtype=compute_type)
        self.ds_feature_norm = LayerNorm(num_channels, epsilon=1e-6, rngs=rngs, dtype=compute_type)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.mlp(self.feature_norm(self.conv(x)))
        x = nnx.max_pool(x, (2, 2), strides=(2, 2), padding="VALID")
        return self.ds_feature_norm(x)


class VSimbaVisionEncoder(nnx.Module):
    """Original LayerNorm V-Simba CNN: [N, F, H, W, C] -> flattened HWC features."""

    def __init__(
        self, input_shape: tuple[int, int, int, int], rngs: nnx.Rngs, num_channels: int = 32,
        num_blocks: int = 2, conv_kernel_size: int = 3, use_bias: bool = True, compute_type: Dtype = jnp.float32,
    ):
        frame_count, height, width, channels = input_shape
        in_channels = frame_count * channels
        self.input_norm = LayerNorm(in_channels, epsilon=1e-6, rngs=rngs, dtype=compute_type)
        self.stem_conv = nnx.Conv(
            in_channels, num_channels, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)),
            rngs=rngs, kernel_init=orthogonal(1), use_bias=use_bias, dtype=compute_type,
        )
        self.stem_norm = LayerNorm(num_channels, epsilon=1e-6, rngs=rngs, dtype=compute_type)
        self.blocks = [VSimbaConvBlock(num_channels, rngs, conv_kernel_size, use_bias=use_bias, compute_type=compute_type) for _ in range(num_blocks)]
        height = (height + 1) // 2 // (2 ** (num_blocks + 1))
        width = (width + 1) // 2 // (2 ** (num_blocks + 1))
        self.feature_dim = height * width * num_channels

    def get_output_dim(self) -> int:
        return self.feature_dim

    def __call__(self, observations: jax.Array) -> jax.Array:
        x = observations.transpose(0, 2, 3, 1, 4)
        x = x.reshape(*x.shape[:3], -1) / 255.0 - 0.5
        x = self.stem_norm(self.stem_conv(self.input_norm(x)))
        x = nnx.max_pool(x, (2, 2), strides=(2, 2), padding="VALID")
        for block in self.blocks:
            x = block(x)
        return x.reshape(x.shape[0], -1).astype(jnp.float32)


