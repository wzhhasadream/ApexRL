from collections.abc import Callable, Sequence
from typing import Literal

from flax import nnx
import jax
import jax.numpy as jnp
from flax.typing import Dtype


def orthogonal(scale: jax.Array = 1):
    return nnx.initializers.orthogonal(scale)


class BatchNorm(nnx.Module):
    """Normalize (*batch_shape, *feature_shape) over batch axes; momentum weights new statistics."""

    def __init__(self, feature_shape: int | tuple[int, ...], *, rngs: nnx.Rngs, momentum: float = 0.1, epsilon: float = 1e-5,
                 use_scale: bool = True, use_bias: bool = True, dtype: Dtype | None = None) -> None:
        self.feature_shape = (feature_shape,) if isinstance(feature_shape, int) else feature_shape
        self.scale = nnx.Param(nnx.initializers.ones(rngs.params(), self.feature_shape)) if use_scale else None
        self.bias = nnx.Param(nnx.initializers.zeros(rngs.params(), self.feature_shape)) if use_bias else None
        self.mean = nnx.BatchStat(jnp.zeros(self.feature_shape))
        self.var = nnx.BatchStat(jnp.ones(self.feature_shape))
        self.momentum, self.epsilon, self.dtype = momentum, epsilon, dtype
        self.use_running_average = False

    def __call__(self, x: jax.Array, training: bool | None = None) -> jax.Array:
        training = not self.use_running_average if training is None else training
        dtype = x.dtype if self.dtype is None else self.dtype
        x = x.astype(jnp.float32)
        if training:
            axes = tuple(range(x.ndim - len(self.feature_shape)))
            mean, var = jnp.mean(x, axis=axes), jnp.var(x, axis=axes, ddof=0)
            count = x.size // self.mean.value.size
            self.mean.value = jax.lax.stop_gradient((1 - self.momentum) * self.mean.value + self.momentum * mean)
            self.var.value = jax.lax.stop_gradient((1 - self.momentum) * self.var.value + self.momentum * var * count / (count - 1))
        else:
            mean, var = self.mean.value, self.var.value
        output = (x - mean) * jax.lax.rsqrt(var + self.epsilon)
        if self.scale is not None:
            output = output * self.scale.value
        if self.bias is not None:
            output = output + self.bias.value
        return output.astype(dtype)


class BatchRenorm(BatchNorm):
    """Batch Renormalization with EMA population variance: https://arxiv.org/abs/1702.03275."""

    def __init__(self, feature_shape: int | tuple[int, ...], *, rngs: nnx.Rngs, momentum: float = 0.001, epsilon: float = 1e-5,
                 warmup_step: int = 1000, use_scale: bool = True, use_bias: bool = True, dtype: Dtype | None = None) -> None:
        super().__init__(feature_shape, rngs=rngs, momentum=momentum, epsilon=epsilon, use_scale=use_scale, use_bias=use_bias, dtype=dtype)
        self.steps = nnx.BatchStat(jnp.asarray(0, dtype=jnp.int32))
        self.warmup_step = warmup_step

    def __call__(self, x: jax.Array, training: bool | None = None) -> jax.Array:
        training = not self.use_running_average if training is None else training
        dtype = x.dtype if self.dtype is None else self.dtype
        x = x.astype(jnp.float32)
        if training:
            axes = tuple(range(x.ndim - len(self.feature_shape)))
            mean, var = jnp.mean(x, axis=axes), jnp.var(x, axis=axes, ddof=0)
            batch_std = jnp.sqrt(var + self.epsilon)
            running_std = jnp.sqrt(self.var.value + self.epsilon)
            r = jnp.clip(jax.lax.stop_gradient(batch_std / running_std), 1 / 3, 3)
            d = jnp.clip(jax.lax.stop_gradient((mean - self.mean.value) / running_std), -5, 5)
            r = jnp.where(self.steps.value >= self.warmup_step, r, 1.0)
            d = jnp.where(self.steps.value >= self.warmup_step, d, 0.0)
            output = (x - mean) / batch_std * r + d
            self.mean.value = jax.lax.stop_gradient((1 - self.momentum) * self.mean.value + self.momentum * mean)
            self.var.value = jax.lax.stop_gradient((1 - self.momentum) * self.var.value + self.momentum * var)
            self.steps.value += 1
        else:
            output = (x - self.mean.value) * jax.lax.rsqrt(self.var.value + self.epsilon)
        if self.scale is not None:
            output = output * self.scale.value
        if self.bias is not None:
            output = output + self.bias.value
        return output.astype(dtype)


class LayerNorm(nnx.Module):
    """Normalize (*batch_shape, *feature_shape) over the feature dimensions."""

    def __init__(self, feature_shape: int | tuple[int, ...], *, rngs: nnx.Rngs, epsilon: float = 1e-6,
                 use_scale: bool = True, use_bias: bool = True, dtype: Dtype | None = None) -> None:
        self.feature_shape = (feature_shape,) if isinstance(feature_shape, int) else feature_shape
        self.scale = nnx.Param(nnx.initializers.ones(rngs.params(), self.feature_shape)) if use_scale else None
        self.bias = nnx.Param(nnx.initializers.zeros(rngs.params(), self.feature_shape)) if use_bias else None
        self.epsilon, self.dtype = epsilon, dtype

    def __call__(self, x: jax.Array) -> jax.Array:
        dtype = x.dtype if self.dtype is None else self.dtype
        x = x.astype(jnp.float32)
        axes = tuple(range(-len(self.feature_shape), 0))
        mean = jnp.mean(x, axis=axes, keepdims=True)
        var = jnp.var(x, axis=axes, ddof=0, keepdims=True)
        output = (x - mean) * jax.lax.rsqrt(var + self.epsilon)
        if self.scale is not None:
            output = output * self.scale.value
        if self.bias is not None:
            output = output + self.bias.value
        return output.astype(dtype)


class RMSNorm(nnx.Module):
    """Normalize (*batch_shape, *feature_shape) by the feature root mean square."""

    def __init__(self, feature_shape: int | tuple[int, ...], *, rngs: nnx.Rngs, epsilon: float = 1e-6,
                 use_scale: bool = True, dtype: Dtype | None = None) -> None:
        self.feature_shape = (feature_shape,) if isinstance(feature_shape, int) else feature_shape
        self.scale = nnx.Param(nnx.initializers.ones(rngs.params(), self.feature_shape)) if use_scale else None
        self.epsilon, self.dtype = epsilon, dtype

    def __call__(self, x: jax.Array) -> jax.Array:
        dtype = x.dtype if self.dtype is None else self.dtype
        x = x.astype(jnp.float32)
        axes = tuple(range(-len(self.feature_shape), 0))
        output = x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=axes, keepdims=True) + self.epsilon)
        if self.scale is not None:
            output = output * self.scale.value
        return output.astype(dtype)


class MLP(nnx.Module):
    """Linear, optional normalization, and activation on the last feature axis."""

    def __init__(self, in_dim: int, hidden_dims: Sequence[int], rngs: nnx.Rngs, norm_type: Literal["bn", "brn", "ln", "rms"] | None = "ln",
                 activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.relu, use_bias: bool = True, compute_type: Dtype = jnp.float32):
        dims = (in_dim, *hidden_dims)
        self.layers = [nnx.Linear(left, right, rngs=rngs, kernel_init=orthogonal(1), use_bias=use_bias, dtype=compute_type)
                       for left, right in zip(dims[:-1], dims[1:])]
        norm_cls = {"bn": BatchNorm, "brn": BatchRenorm, "ln": LayerNorm, "rms": RMSNorm, None: None}[norm_type]
        self.norms = [norm_cls(dim, rngs=rngs, dtype=compute_type) if norm_cls is not None else None for dim in hidden_dims]
        self.norm_type, self.activation_fn = norm_type, activation_fn

    def __call__(self, x: jax.Array, training: bool | None = None) -> jax.Array:
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x)
            if norm is not None:
                x = norm(x, training=training) if self.norm_type in ("bn", "brn") else norm(x)
            x = self.activation_fn(x)
        return x
