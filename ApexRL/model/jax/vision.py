from typing import Literal

from flax import nnx
import jax
import jax.numpy as jnp
from flax.typing import Dtype

from .layer import BatchNorm, BatchRenorm, LayerNorm, RMSNorm, orthogonal


# ------------------------------- MR.Q / DrQ-v2 pixel encoders ---------------


class MRQVisionEncoder(nnx.Module):
    """MR.Q vision encoder (Fujimoto et al., 2025) for frame-stacked pixel observations.

    Input [N, F, H, W, C] uint8 frames; output [N, latent_dim] ELU features.
    Four conv layers (three stride-2 then one stride-1, 32 channels) with ELU,
    followed by Linear + LayerNorm, as in facebookresearch/MRQ.
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
                "MRQVisionEncoder input is too small for the MR.Q CNN: "
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
        self.ln = LayerNorm(
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
                "MRQVisionEncoder expects [N, F, H, W, C] observations, "
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
    """DrQ-v2 vision encoder (Yarats et al., 2021) for frame-stacked pixel observations.

    Input [N, F, H, W, C] uint8 frames; output [N, latent_dim] tanh features.
    One stride-2 conv then three stride-1 convs (32 channels) with ReLU, followed
    by Linear + LayerNorm; convs/fc use orthogonal(1) init, as in DrQ-v2.
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
        self.ln = LayerNorm(
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

# ------------------------------- Nature CNN (Atari) ---------------


class _Norm(nnx.Module):
    """Feature-last normalization; BN/BRN reduce batch axes, LN/RMSNorm reduce feature axes."""

    def __init__(self, feature_shape: int | tuple[int, ...], norm_type: Literal["bn", "brn", "ln", "rms"] | None, rngs: nnx.Rngs, dtype: Dtype) -> None:
        self.norm_type = norm_type
        norm_cls = {"bn": BatchNorm, "brn": BatchRenorm, "ln": LayerNorm, "rms": RMSNorm, None: None}[norm_type]
        self.norm = norm_cls(feature_shape, rngs=rngs, dtype=dtype) if norm_cls is not None else None

    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        if self.norm is None:
            return x
        return self.norm(x, training=training) if self.norm_type in ("bn", "brn") else self.norm(x)


class NatureCNN(nnx.Module):
    """Nature DQN encoder (Mnih et al., 2015) for stacked Atari frames.

    Input [B, F, H, W, C] uint8 frames; output [B, hidden_dim] ReLU features.
    Conv layers use orthogonal(sqrt(2)) init, as in CleanRL; norm_type adds
    bn/brn/ln/rms/None normalization before each activation.
    """

    def __init__(self, observation_shape: tuple[int, int, int, int], rngs: nnx.Rngs, hidden_dim: int = 512, compute_type: Dtype = jnp.float32, norm_type: Literal["bn", "brn", "ln", "rms"] | None = None) -> None:
        frames, height, width, channels = observation_shape
        self.compute_type = compute_type
        init = orthogonal(jnp.sqrt(2.0))
        self.conv1 = nnx.Conv(frames * channels, 32, (8, 8), strides=(4, 4), padding="VALID", kernel_init=init, rngs=rngs, dtype=compute_type)
        self.conv2 = nnx.Conv(32, 64, (4, 4), strides=(2, 2), padding="VALID", kernel_init=init, rngs=rngs, dtype=compute_type)
        self.conv3 = nnx.Conv(64, 64, (3, 3), strides=(1, 1), padding="VALID", kernel_init=init, rngs=rngs, dtype=compute_type)
        self.norm1 = _Norm(32, norm_type, rngs, compute_type)
        self.norm2 = _Norm(64, norm_type, rngs, compute_type)
        self.norm3 = _Norm(64, norm_type, rngs, compute_type)
        flatten_dim = self._convs(jnp.zeros((1, height, width, frames * channels), compute_type), training=False).shape[-1]
        self.fc = nnx.Linear(flatten_dim, hidden_dim, kernel_init=init, rngs=rngs, dtype=compute_type)
        self.norm_fc = _Norm(hidden_dim, norm_type, rngs, compute_type)
        self.out_dim = hidden_dim

    def _convs(self, x: jax.Array, training: bool) -> jax.Array:
        x = jax.nn.relu(self.norm1(self.conv1(x), training))
        x = jax.nn.relu(self.norm2(self.conv2(x), training))
        x = jax.nn.relu(self.norm3(self.conv3(x), training))
        return x.reshape(x.shape[0], -1)

    def __call__(self, observations: jax.Array, training: bool = False) -> jax.Array:
        # [B, F, H, W, C] uint8 -> channels-last [B, H, W, F * C] float in [0, 1]
        x = jnp.moveaxis(observations, 1, -2)
        x = x.reshape((*x.shape[:3], -1)).astype(self.compute_type) / 255.0
        return jax.nn.relu(self.norm_fc(self.fc(self._convs(x, training)), training))


# ------------------------------- HardMax CNN (Atari) ---------------


class HardMaxBlock(nnx.Module):
    """HardMax conv block: two parallel convs fused by a Hadamard product, then max-pooled.

    Both branches apply normalization + GELU before the element-wise product; convs use
    xavier_normal init, as in Kooi et al. (2025).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int, pool_stride: int, rngs: nnx.Rngs, norm_type: Literal["bn", "brn", "ln", "rms"] | None, compute_type: Dtype = jnp.float32) -> None:
        kernel_init = jax.nn.initializers.xavier_normal()
        self.conv1 = nnx.Conv(in_channels, out_channels, (kernel_size, kernel_size), strides=(1, 1), padding="SAME", kernel_init=kernel_init, rngs=rngs, dtype=compute_type)
        self.conv2 = nnx.Conv(in_channels, out_channels, (kernel_size, kernel_size), strides=(1, 1), padding="SAME", kernel_init=kernel_init, rngs=rngs, dtype=compute_type)
        self.norm1 = _Norm(out_channels, norm_type, rngs, compute_type)
        self.norm2 = _Norm(out_channels, norm_type, rngs, compute_type)
        self.pool_size = pool_size
        self.pool_stride = pool_stride

    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        x1 = jax.nn.gelu(self.norm1(self.conv1(x), training))
        x2 = jax.nn.gelu(self.norm2(self.conv2(x), training))
        return nnx.max_pool(x1 * x2, (self.pool_size, self.pool_size), strides=(self.pool_stride, self.pool_stride), padding="SAME")


class HardMaxCNN(nnx.Module):
    """Hadamax pixel encoder (Kooi et al., 2025) for stacked Atari frames.

    Input [B, F, H, W, C] uint8 frames; output [B, hidden_dim] GELU features.
    Convs use xavier_normal init and bn/brn/ln/rms/None normalization + GELU before a
    Hadamard-product fusion; the head fc uses he_normal, as in github.com/Jacobkooi/Hadamax.
    """

    def __init__(self, observation_shape: tuple[int, int, int, int], rngs: nnx.Rngs, hidden_dim: int = 512, compute_type: Dtype = jnp.float32, norm_type: Literal["bn", "brn", "ln", "rms"] | None = "ln") -> None:
        frames, height, width, channels = observation_shape
        in_channels = frames * channels
        self.compute_type = compute_type
        self.block1 = HardMaxBlock(in_channels, 32, 8, 4, 4, rngs, norm_type, compute_type)
        self.block2 = HardMaxBlock(32, 64, 4, 2, 2, rngs, norm_type, compute_type)
        self.block3 = HardMaxBlock(64, 64, 3, 3, 1, rngs, norm_type, compute_type)
        flatten_dim = self._blocks(jnp.zeros((1, height, width, in_channels), compute_type), training=False).shape[-1]
        self.fc = nnx.Linear(flatten_dim, hidden_dim, kernel_init=nnx.initializers.he_normal(), rngs=rngs, dtype=compute_type)
        self.norm_fc = _Norm(hidden_dim, norm_type, rngs, compute_type)
        self.out_dim = hidden_dim

    def _blocks(self, x: jax.Array, training: bool) -> jax.Array:
        x = self.block1(x, training)
        x = self.block2(x, training)
        x = self.block3(x, training)
        return x.reshape(x.shape[0], -1)

    def __call__(self, observations: jax.Array, training: bool = False) -> jax.Array:
        # [B, F, H, W, C] uint8 -> channels-last [B, H, W, F * C] float in [0, 1]
        x = jnp.moveaxis(observations, 1, -2)
        x = x.reshape((*x.shape[:3], -1)).astype(self.compute_type) / 255.0
        return jax.nn.gelu(self.norm_fc(self.fc(self._blocks(x, training)), training))
