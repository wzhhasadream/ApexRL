from sympy import maximum
from .device import default_device, resolve_device
from .augment import augment_observations, augment_sequencebatch
from .ppo_utils import adapt_lr, categorical_kl, diagonal_gaussian_kl
from .zeta_dist import (
    build_truncated_zeta_cdf,
    sample_integer_from_cdf,
    sample_truncated_zeta,
)
import jax
import jax.numpy as jnp

def l2_norm(feature: jax.Array):
    return feature / jnp.maximum(jnp.linalg.norm(feature, axis=-1, keepdims=True), 1e-6)
