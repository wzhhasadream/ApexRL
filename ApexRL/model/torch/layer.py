from collections.abc import Callable, Sequence
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Linear):
    """Linear layer with orthogonal weights and zero bias."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        nn.init.orthogonal_(self.weight, gain=1)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class BatchNorm(nn.Module):
    """Normalize (*batch_shape, *feature_shape) over batch axes; momentum weights new statistics."""

    def __init__(self, feature_shape: int | tuple[int, ...], *, momentum: float = 0.1, epsilon: float = 1e-5, use_scale: bool = True, use_bias: bool = True) -> None:
        super().__init__()
        self.feature_shape = (feature_shape,) if isinstance(feature_shape, int) else feature_shape
        self.weight = nn.Parameter(torch.ones(self.feature_shape)) if use_scale else None
        self.bias = nn.Parameter(torch.zeros(self.feature_shape)) if use_bias else None
        self.register_buffer("running_mean", torch.zeros(self.feature_shape))
        self.register_buffer("running_var", torch.ones(self.feature_shape))
        self.momentum, self.epsilon = momentum, epsilon

    def forward(self, x: torch.Tensor, training: bool | None = None) -> torch.Tensor:
        training = self.training if training is None else training
        dtype = x.dtype
        x = x.float()
        if training:
            axes = tuple(range(x.ndim - len(self.feature_shape)))
            var, mean = torch.var_mean(x, dim=axes, correction=0)
            count = x.numel() // self.running_mean.numel()
            with torch.no_grad():
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(var * count / (count - 1), self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        output = (x - mean) * torch.rsqrt(var + self.epsilon)
        if self.weight is not None:
            output = output * self.weight
        if self.bias is not None:
            output = output + self.bias
        return output.to(dtype)


class BatchRenorm(BatchNorm):
    """Batch Renormalization with EMA population variance: https://arxiv.org/abs/1702.03275."""

    def __init__(self, feature_shape: int | tuple[int, ...], *, momentum: float = 0.001, epsilon: float = 1e-5, warmup_step: int = 1000,
                 use_scale: bool = True, use_bias: bool = True) -> None:
        super().__init__(feature_shape, momentum=momentum, epsilon=epsilon, use_scale=use_scale, use_bias=use_bias)
        self.register_buffer("steps", torch.tensor(0, dtype=torch.int64))
        self.warmup_step = warmup_step

    def forward(self, x: torch.Tensor, training: bool | None = None) -> torch.Tensor:
        training = self.training if training is None else training
        dtype = x.dtype
        x = x.float()
        if training:
            axes = tuple(range(x.ndim - len(self.feature_shape)))
            var, mean = torch.var_mean(x, dim=axes, correction=0)
            batch_std = (var + self.epsilon).sqrt()
            running_std = (self.running_var + self.epsilon).sqrt()
            r = (batch_std / running_std).detach().clamp(1 / 3, 3)
            d = ((mean - self.running_mean) / running_std).detach().clamp(-5, 5)
            r = torch.where(self.steps >= self.warmup_step, r, 1.0)
            d = torch.where(self.steps >= self.warmup_step, d, 0.0)
            output = (x - mean) / batch_std * r + d
            with torch.no_grad():
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(var, self.momentum)
                self.steps.add_(1)
        else:
            output = (x - self.running_mean) * torch.rsqrt(self.running_var + self.epsilon)
        if self.weight is not None:
            output = output * self.weight
        if self.bias is not None:
            output = output + self.bias
        return output.to(dtype)


class LayerNorm(nn.Module):
    """Normalize (*batch_shape, *feature_shape) over the feature dimensions."""

    def __init__(self, feature_shape: int | tuple[int, ...], *, epsilon: float = 1e-5, use_scale: bool = True, use_bias: bool = True) -> None:
        super().__init__()
        self.feature_shape = (feature_shape,) if isinstance(feature_shape, int) else feature_shape
        self.weight = nn.Parameter(torch.ones(self.feature_shape)) if use_scale else None
        self.bias = nn.Parameter(torch.zeros(self.feature_shape)) if use_bias else None
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x.float(), self.feature_shape, self.weight, self.bias, self.epsilon).to(x.dtype)


class RMSNorm(nn.Module):
    """Normalize (*batch_shape, *feature_shape) by the feature root mean square."""

    def __init__(self, feature_shape: int | tuple[int, ...], *, epsilon: float = 1e-6, use_scale: bool = True) -> None:
        super().__init__()
        self.feature_shape = (feature_shape,) if isinstance(feature_shape, int) else feature_shape
        self.weight = nn.Parameter(torch.ones(self.feature_shape)) if use_scale else None
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        axes = tuple(range(-len(self.feature_shape), 0))
        output = x * torch.rsqrt(x.square().mean(dim=axes, keepdim=True) + self.epsilon)
        if self.weight is not None:
            output = output * self.weight
        return output.to(dtype)


class MLP(nn.Module):
    """Linear, optional normalization, and activation on the last feature axis."""

    def __init__(self, in_dim: int, hidden_dims: Sequence[int], norm_type: Literal["bn", "brn", "ln", "rms"] | None = "ln",
                 activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu, bias: bool = True) -> None:
        super().__init__()
        dims = (in_dim, *hidden_dims)
        self.layers = nn.ModuleList(Linear(left, right, bias=bias) for left, right in zip(dims[:-1], dims[1:]))
        norm_cls = {"bn": BatchNorm, "brn": BatchRenorm, "ln": LayerNorm, "rms": RMSNorm, None: None}[norm_type]
        self.norms = nn.ModuleList(norm_cls(dim) if norm_cls is not None else None for dim in hidden_dims)
        self.norm_type, self.activation_fn = norm_type, activation_fn

    def forward(self, x: torch.Tensor, training: bool | None = None) -> torch.Tensor:
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x)
            if norm is not None:
                x = norm(x, training=training) if self.norm_type in ("bn", "brn") else norm(x)
            x = self.activation_fn(x)
        return x


def _add_ensemble_dim(x: torch.Tensor, num_ensemble: int, feature_ndim: int = 1) -> torch.Tensor:
    """Broadcast a shared [B, *features] input; per-member inputs already start with E."""
    return x.unsqueeze(0).expand(num_ensemble, *x.shape) if x.ndim == feature_ndim + 1 else x


class EnsembleLinear(nn.Module):
    """Independent linear layers for shared [B, I] or per-member [E, B, I] inputs."""

    def __init__(self, num_ensemble: int, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.num_ensemble, self.in_features, self.out_features = num_ensemble, in_features, out_features
        self.weight = nn.Parameter(torch.empty(num_ensemble, out_features, in_features))
        self.bias = nn.Parameter(torch.empty(num_ensemble, out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.weight:
            nn.init.orthogonal_(weight, gain=1)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _add_ensemble_dim(x, self.num_ensemble)
        output = torch.einsum("ebi,eoi->ebo", x, self.weight)
        if self.bias is not None:
            output = output + self.bias[:, None, :]
        return output


class EnsembleBatchNorm(BatchNorm):
    """Independent BatchNorm for [E, *batch_shape, *feature_shape] inputs."""

    def __init__(self, num_ensemble: int, feature_shape: int | tuple[int, ...], *, momentum: float = 0.1, epsilon: float = 1e-5, use_scale: bool = True, use_bias: bool = True) -> None:
        feature_shape = (feature_shape,) if isinstance(feature_shape, int) else feature_shape
        super().__init__((num_ensemble, *feature_shape), momentum=momentum, epsilon=epsilon, use_scale=use_scale, use_bias=use_bias)
        self.num_ensemble = num_ensemble

    def forward(self, x: torch.Tensor, training: bool | None = None) -> torch.Tensor:
        x = _add_ensemble_dim(x, self.num_ensemble, len(self.feature_shape) - 1)
        # Keep E with the feature axes so only batch axes are reduced.
        axis = -len(self.feature_shape)
        return super().forward(x.movedim(0, axis), training=training).movedim(axis, 0)


class EnsembleBatchRenorm(BatchRenorm):
    """Independent BatchRenorm for [E, *batch_shape, *feature_shape] inputs."""

    def __init__(self, num_ensemble: int, feature_shape: int | tuple[int, ...], *, momentum: float = 0.001, epsilon: float = 1e-5, warmup_step: int = 1000,
                 use_scale: bool = True, use_bias: bool = True) -> None:
        feature_shape = (feature_shape,) if isinstance(feature_shape, int) else feature_shape
        super().__init__((num_ensemble, *feature_shape), momentum=momentum, epsilon=epsilon, warmup_step=warmup_step, use_scale=use_scale, use_bias=use_bias)
        self.num_ensemble = num_ensemble

    def forward(self, x: torch.Tensor, training: bool | None = None) -> torch.Tensor:
        x = _add_ensemble_dim(x, self.num_ensemble, len(self.feature_shape) - 1)
        axis = -len(self.feature_shape)
        return super().forward(x.movedim(0, axis), training=training).movedim(axis, 0)


class EnsembleLayerNorm(nn.Module):
    """LayerNorm for [E, *batch_shape, *feature_shape], with independent affine parameters."""

    def __init__(self, num_ensemble: int, feature_shape: int | tuple[int, ...], *, epsilon: float = 1e-5, use_scale: bool = True, use_bias: bool = True) -> None:
        super().__init__()
        self.feature_shape = (feature_shape,) if isinstance(feature_shape, int) else feature_shape
        self.num_ensemble, self.epsilon = num_ensemble, epsilon
        self.weight = nn.Parameter(torch.ones(num_ensemble, *self.feature_shape)) if use_scale else None
        self.bias = nn.Parameter(torch.zeros(num_ensemble, *self.feature_shape)) if use_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _add_ensemble_dim(x, self.num_ensemble, len(self.feature_shape))
        output = F.layer_norm(x.float(), self.feature_shape, eps=self.epsilon)
        affine_shape = (self.num_ensemble,) + (1,) * (x.ndim - len(self.feature_shape) - 1) + self.feature_shape
        if self.weight is not None:
            output = output * self.weight.view(affine_shape)
        if self.bias is not None:
            output = output + self.bias.view(affine_shape)
        return output.to(x.dtype)


class EnsembleRMSNorm(nn.Module):
    """RMSNorm for [E, *batch_shape, *feature_shape], with independent scales."""

    def __init__(self, num_ensemble: int, feature_shape: int | tuple[int, ...], *, epsilon: float = 1e-6, use_scale: bool = True) -> None:
        super().__init__()
        self.feature_shape = (feature_shape,) if isinstance(feature_shape, int) else feature_shape
        self.num_ensemble, self.epsilon = num_ensemble, epsilon
        self.weight = nn.Parameter(torch.ones(num_ensemble, *self.feature_shape)) if use_scale else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _add_ensemble_dim(x, self.num_ensemble, len(self.feature_shape))
        values = x.float()
        axes = tuple(range(-len(self.feature_shape), 0))
        output = values * torch.rsqrt(values.square().mean(dim=axes, keepdim=True) + self.epsilon)
        affine_shape = (self.num_ensemble,) + (1,) * (x.ndim - len(self.feature_shape) - 1) + self.feature_shape
        if self.weight is not None:
            output = output * self.weight.view(affine_shape)
        return output.to(x.dtype)

class EnsembleMLP(nn.Module):
    """Independent MLPs sharing the same feature-last input contract."""

    def __init__(self, num_ensemble: int, in_dim: int, hidden_dims: Sequence[int], norm_type: Literal["bn", "brn", "ln", "rms"] | None = "ln",
                 activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu, bias: bool = True) -> None:
        super().__init__()
        dims = (in_dim, *hidden_dims)
        self.num_ensemble = num_ensemble
        self.layers = nn.ModuleList(EnsembleLinear(num_ensemble, left, right, bias=bias) for left, right in zip(dims[:-1], dims[1:]))
        norm_cls = {"bn": EnsembleBatchNorm, "brn": EnsembleBatchRenorm, "ln": EnsembleLayerNorm, "rms": EnsembleRMSNorm, None: None}[norm_type]
        self.norms = nn.ModuleList(norm_cls(num_ensemble, dim) if norm_cls is not None else None for dim in hidden_dims)
        self.norm_type, self.activation_fn = norm_type, activation_fn

    def forward(self, x: torch.Tensor, training: bool | None = None) -> torch.Tensor:
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x)
            if norm is not None:
                x = norm(x, training=training) if self.norm_type in ("bn", "brn") else norm(x)
            x = self.activation_fn(x)
        return x

