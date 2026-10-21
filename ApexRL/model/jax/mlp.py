import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype

from .layer import BatchNorm, RMSNorm, orthogonal


class FlashSACEmbedder(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rngs: nnx.Rngs,
        use_bias: bool = True,
        compute_type: Dtype = jnp.float32,
    ):
        self.norm = BatchNorm(
            feature_shape=input_dim,
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
        x = self.norm(x, training=training)
        return self.w(x)


class FlashSACBlock(nnx.Module):
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
        self.norm1 = BatchNorm(
            feature_shape=hidden_dim * expansion,
            rngs=rngs,
            dtype=compute_type,
        )
        self.norm2 = BatchNorm(
            feature_shape=hidden_dim,
            rngs=rngs,
            dtype=compute_type,
        )

    def __call__(self, x: jax.Array, training: bool) -> jax.Array:
        residual = x
        x = self.w1(x)
        x = self.norm1(x, training=training)
        x = nnx.relu(x)
        x = self.w2(x)
        x = self.norm2(x, training=training)
        return residual + nnx.relu(x)


class FlashSACEncoder(nnx.Module):
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
            FlashSACBlock(hidden_dim, rngs, 4, use_bias, compute_type)
            for _ in range(num_blocks)
        ]
        self.rms = RMSNorm(hidden_dim, rngs=rngs, dtype=compute_type)

    def __call__(self, x: jax.Array, training: bool) -> jax.Array:
        x = self.embed(x, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        return self.rms(x)
