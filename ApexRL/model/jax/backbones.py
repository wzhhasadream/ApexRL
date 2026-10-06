from flax import nnx
import jax
import jax.numpy as jnp
from flax.typing import Dtype

from .layer import orthogonal


# ------------------------------- FlashSAC state components ---------------


class FlashSACEmbedder(nnx.Module):
    """Input projector used by the FlashSAC state backbone."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rngs: nnx.Rngs,
        use_bias: bool = True,
        compute_type: Dtype = jnp.float32,
    ):
        self.norm = nnx.BatchNorm(
            num_features=input_dim,
            rngs=rngs,
            dtype=compute_type,
        )
        self.w = nnx.Linear(
            input_dim,
            hidden_dim,
            rngs=rngs,
            kernel_init=orthogonal(1),
            use_bias=use_bias,
            dtype=compute_type,
        )

    def __call__(self, x: jax.Array, training: bool) -> jax.Array:
        x = self.norm(x, use_running_average=not training)
        return self.w(x)


class FlashSACBlock(nnx.Module):
    """Pre-activation residual MLP block used by the FlashSAC backbone."""

    def __init__(
        self,
        hidden_dim: int,
        rngs: nnx.Rngs,
        expansion: int = 4,
        use_bias: bool = True,
        compute_type: Dtype = jnp.float32,
    ):
        self.w1 = nnx.Linear(
            hidden_dim,
            hidden_dim * expansion,
            rngs=rngs,
            kernel_init=orthogonal(1),
            use_bias=use_bias,
            dtype=compute_type,
        )
        self.w2 = nnx.Linear(
            hidden_dim * expansion,
            hidden_dim,
            rngs=rngs,
            kernel_init=orthogonal(1),
            use_bias=use_bias,
            dtype=compute_type,
        )
        self.norm1 = nnx.BatchNorm(
            num_features=hidden_dim * expansion,
            rngs=rngs,
            dtype=compute_type,
        )
        self.norm2 = nnx.BatchNorm(
            num_features=hidden_dim,
            rngs=rngs,
            dtype=compute_type,
        )

    def __call__(self, x: jax.Array, training: bool) -> jax.Array:
        residual = x
        x = self.w1(x)
        x = self.norm1(x, use_running_average=not training)
        x = nnx.relu(x)
        x = self.w2(x)
        x = self.norm2(x, use_running_average=not training)
        return residual + nnx.relu(x)


class FlashSACEncoder(nnx.Module):
    """State observation encoder made of FlashSAC residual blocks."""

    def __init__(
        self,
        input_dim: int,
        num_blocks: int,
        hidden_dim: int,
        rngs: nnx.Rngs,
        use_bias: bool = True,
        compute_type: Dtype = jnp.float32,
    ):
        self.embed = FlashSACEmbedder(
            input_dim,
            hidden_dim,
            rngs,
            use_bias,
            compute_type,
        )
        self.blocks = [
            FlashSACBlock(
                hidden_dim, rngs, 4, use_bias, compute_type
            )
            for _ in range(num_blocks)
        ]
        self.rms = nnx.RMSNorm(
            hidden_dim, rngs=rngs, dtype=compute_type
        )

    def __call__(self, x: jax.Array, training: bool) -> jax.Array:
        x = self.embed(x, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        return self.rms(x)


# ------------------------------- pixel components ----------------

class DRQVisionEncoder(nnx.Module):
    """DR.Q-style CNN encoder for frame-stacked pixel observations.

    Input is expected as [N, F, H, W, C] (frames first). The encoder
    concatenates F*C along the channel axis and applies three stride-2
    convolutions followed by one stride-1 convolution.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int, int],
        rngs: nnx.Rngs,
        num_channels: int = 32,
        num_blocks: int = 2,
        conv_kernel_size: int = 3,
        use_bias: bool = True,
        compute_type: Dtype = jnp.float32,
        latent_dim: int = 512,
    ):
        frame_count, height, width, channels = input_shape
        in_channels = frame_count * channels
        kernel_size = (conv_kernel_size, conv_kernel_size)
        kernel_init = jax.nn.initializers.xavier_uniform()

        self.conv1 = nnx.Conv(
            in_channels,
            num_channels,
            kernel_size=kernel_size,
            strides=(2, 2),
            padding="VALID",
            rngs=rngs,
            kernel_init=kernel_init,
            use_bias=use_bias,
            dtype=compute_type,
        )
        self.blocks = [
            nnx.Conv(
                num_channels,
                num_channels,
                kernel_size=kernel_size,
                strides=(2, 2),
                padding="VALID",
                rngs=rngs,
                kernel_init=kernel_init,
                use_bias=use_bias,
                dtype=compute_type,
            )
            for _ in range(num_blocks)
        ]
        self.conv_last = nnx.Conv(
            num_channels,
            num_channels,
            kernel_size=kernel_size,
            strides=(1, 1),
            padding="VALID",
            rngs=rngs,
            kernel_init=kernel_init,
            use_bias=use_bias,
            dtype=compute_type,
        )

        dummy = jnp.zeros((1, height, width, in_channels), dtype=compute_type)
        dummy = jax.nn.elu(self.conv1(dummy))
        for conv in self.blocks:
            dummy = jax.nn.elu(conv(dummy))
        dummy = jax.nn.elu(self.conv_last(dummy))
        self.feature_dim = int(dummy.size)
        if self.feature_dim <= 0:
            raise ValueError(
                "DRQVisionEncoder input is too small for the DR.Q CNN: "
                f"got input_shape={input_shape}"
            )

        self.latent_dim = latent_dim
        self.fc = nnx.Linear(
            self.feature_dim,
            latent_dim,
            rngs=rngs,
            kernel_init=kernel_init,
            use_bias=use_bias,
            dtype=compute_type,
        )
        self.ln = nnx.LayerNorm(
            latent_dim,
            epsilon=1e-6,
            rngs=rngs,
            dtype=compute_type,
        )

    def get_output_dim(self):
        return self.latent_dim

    def __call__(self, observations: jax.Array) -> jax.Array:
        if observations.ndim != 5:
            raise ValueError(
                "DRQVisionEncoder expects [N, F, H, W, C] observations, "
                f"got {observations.shape}"
            )
        x = observations.transpose(0, 2, 3, 1, 4)
        x = x.reshape(x.shape[0], *x.shape[1:3], -1)  # [N, H, W, F*C]
        x = x / 255.0 - 0.5
        x = jax.nn.elu(self.conv1(x))
        for conv in self.blocks:
            x = jax.nn.elu(conv(x))
        x = jax.nn.elu(self.conv_last(x))
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.ln(x)
        return jax.nn.elu(x).astype(jnp.float32)



class DRQV2VisionEncoder(nnx.Module):
    """DrQ-v2 style CNN encoder for frame-stacked pixel observations.

    Input is expected as [N, F, H, W, C]. The encoder concatenates F*C
    along the channel axis and applies one stride-2 convolution followed
    by three stride-1 convolutions.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int, int],
        rngs: nnx.Rngs,
        num_channels: int = 32,
        num_blocks: int = 3,
        conv_kernel_size: int = 3,
        use_bias: bool = True,
        compute_type: Dtype = jnp.float32,
        latent_dim: int = 50,
    ):
        frame_count, height, width, channels = input_shape
        in_channels = frame_count * channels
        kernel_size = (conv_kernel_size, conv_kernel_size)

        self.conv1 = nnx.Conv(
            in_channels,
            num_channels,
            kernel_size=kernel_size,
            strides=(2, 2),
            padding="VALID",
            rngs=rngs,
            kernel_init=orthogonal(1),
            use_bias=use_bias,
            dtype=compute_type,
        )
        self.blocks = [
            nnx.Conv(
                num_channels,
                num_channels,
                kernel_size=kernel_size,
                strides=(1, 1),
                padding="VALID",
                rngs=rngs,
                kernel_init=orthogonal(1),
                use_bias=use_bias,
                dtype=compute_type,
            )
            for _ in range(num_blocks)
        ]

        dummy = jnp.zeros((1, height, width, in_channels), dtype=compute_type)
        dummy = jax.nn.relu(self.conv1(dummy))
        for conv in self.blocks:
            dummy = jax.nn.relu(conv(dummy))
        self.feature_dim = int(dummy.size)
        if self.feature_dim <= 0:
            raise ValueError(
                "DRQV2VisionEncoder input is too small for the DrQ-v2 CNN: "
                f"got input_shape={input_shape}"
            )

        self.latent_dim = latent_dim
        self.fc = nnx.Linear(
            self.feature_dim,
            latent_dim,
            rngs=rngs,
            kernel_init=orthogonal(1),
            use_bias=use_bias,
            dtype=compute_type,
        )
        self.ln = nnx.LayerNorm(
            latent_dim,
            epsilon=1e-6,
            rngs=rngs,
            dtype=compute_type,
        )

    def get_output_dim(self) -> int:
        return self.latent_dim

    def __call__(self, observations: jax.Array) -> jax.Array:
        if observations.ndim != 5:
            raise ValueError(
                "DRQV2VisionEncoder expects [N, F, H, W, C] observations, "
                f"got {observations.shape}"
            )
        x = observations.transpose(0, 2, 3, 1, 4)
        x = x.reshape(x.shape[0], *x.shape[1:3], -1)  # [N, H, W, F*C]
        x = x / 255.0 - 0.5
        x = jax.nn.relu(self.conv1(x))
        for conv in self.blocks:
            x = jax.nn.relu(conv(x))
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.ln(x)
        return jnp.tanh(x).astype(jnp.float32)


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
        self.feature_norm = nnx.LayerNorm(hidden_dim, epsilon=1e-6, rngs=rngs, dtype=compute_type)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.feature_norm(self.w(x))


class VSimbaMLPBlock(nnx.Module):
    """Pre-LayerNorm residual MLP block."""

    def __init__(self, hidden_dim: int, rngs: nnx.Rngs, expansion: int = 4, use_bias: bool = True, compute_type: Dtype = jnp.float32):
        self.mlp = VSimbaMLP(hidden_dim, hidden_dim * expansion, hidden_dim, rngs, use_bias, compute_type)
        self.feature_norm = nnx.LayerNorm(hidden_dim, epsilon=1e-6, rngs=rngs, dtype=compute_type)

    def __call__(self, x: jax.Array) -> jax.Array:
        return x + self.mlp(self.feature_norm(x))


class VSimbaEncoder(nnx.Module):
    """Embedder, residual MLP blocks and final LayerNorm for actor/Q features."""

    def __init__(self, input_dim: int, num_blocks: int, hidden_dim: int, rngs: nnx.Rngs, expansion: int = 4, use_bias: bool = True, compute_type: Dtype = jnp.float32):
        self.embedder = VSimbaEmbedder(input_dim, hidden_dim, rngs, use_bias, compute_type)
        self.blocks = [VSimbaMLPBlock(hidden_dim, rngs, expansion, use_bias, compute_type) for _ in range(num_blocks)]
        self.post_norm = nnx.LayerNorm(hidden_dim, epsilon=1e-6, rngs=rngs, dtype=compute_type)

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
        self.feature_norm = nnx.LayerNorm(num_channels, epsilon=1e-6, rngs=rngs, dtype=compute_type)
        self.ds_feature_norm = nnx.LayerNorm(num_channels, epsilon=1e-6, rngs=rngs, dtype=compute_type)

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
        self.input_norm = nnx.LayerNorm(in_channels, epsilon=1e-6, rngs=rngs, dtype=compute_type)
        self.stem_conv = nnx.Conv(
            in_channels, num_channels, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)),
            rngs=rngs, kernel_init=orthogonal(1), use_bias=use_bias, dtype=compute_type,
        )
        self.stem_norm = nnx.LayerNorm(num_channels, epsilon=1e-6, rngs=rngs, dtype=compute_type)
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
