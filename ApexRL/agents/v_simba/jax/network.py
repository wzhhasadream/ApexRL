# Adapted from DAVIAN-Robotics/V-Simba (Apache-2.0), modified for ApexRL; see THIRD_PARTY_NOTICES.md.
# https://github.com/DAVIAN-Robotics/V-Simba/tree/be811e968bc02589fbb32f3be79f9a7d9a8fa86d/scale_rl/agents/vsimba
import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype

from ....model.jax import CategoricalPolicy
from ....model.jax.backbones import VSimbaEmbedder, VSimbaEncoder, VSimbaProjector
from ....model.jax.layer import orthogonal
from ....model.jax.policy import SquashedTanhGaussianPolicy


class Actor(nnx.Module):
    def __init__(self, input_dim: int, action_dim: int, rngs: nnx.Rngs, action_low: jax.Array, action_high: jax.Array, num_blocks: int = 1, hidden_dim: int = 128, compute_type: Dtype = jnp.float32):
        self.action_dim = action_dim
        self.trunk = VSimbaEncoder(input_dim, num_blocks, hidden_dim, rngs, compute_type=compute_type)
        self.mean_w = nnx.Linear(hidden_dim, action_dim, rngs=rngs, kernel_init=orthogonal(1), dtype=compute_type)
        self.log_std_w = nnx.Linear(hidden_dim, action_dim, rngs=rngs, kernel_init=orthogonal(1), dtype=compute_type)
        self.policy = SquashedTanhGaussianPolicy(action_low, action_high, log_std_min=-10.0, log_std_max=2.0)

    def __call__(self, vision_z: jax.Array) -> tuple[jax.Array, jax.Array]:
        z = self.trunk(vision_z)
        return self.mean_w(z), self.log_std_w(z)

    def get_action(self, vision_z: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        mean, log_std = self(vision_z)
        return self.policy.sample_and_log_prob(mean, log_std, key)

    def get_mean_action(self, vision_z: jax.Array) -> jax.Array:
        mean, _ = self(vision_z)
        return self.policy.mean_action(mean)


class CriticHead(nnx.Module):
    def __init__(
        self, input_dim: int, action_dim: int, rngs: nnx.Rngs, num_blocks: int = 2, hidden_dim: int = 512,
        action_embed_dim: int = 128, c_shift: float = 3.0, num_bins: int = 101, compute_type: Dtype = jnp.float32,
    ):
        self.projector = VSimbaProjector(c_shift)
        self.action_embedder = VSimbaEmbedder(action_dim + 1, action_embed_dim, rngs, compute_type=compute_type)
        self.encoder = VSimbaEncoder(input_dim + action_embed_dim, num_blocks, hidden_dim, rngs, compute_type=compute_type)
        self.w = nnx.Linear(hidden_dim, num_bins, rngs=rngs, kernel_init=orthogonal(1), dtype=compute_type)

    def __call__(self, vision_z: jax.Array, actions: jax.Array) -> jax.Array:
        action_z = self.action_embedder(self.projector(actions))
        return self.w(self.encoder(jnp.concatenate([vision_z, action_z], axis=-1))).astype(jnp.float32)


class Critic(nnx.Module):
    def __init__(
        self, input_dim: int, action_dim: int, rngs: nnx.Rngs, num_qs: int = 1, num_blocks: int = 2,
        hidden_dim: int = 512, action_embed_dim: int = 128, c_shift: float = 3.0,
        num_bins: int = 101, min_v: float = -5.0, max_v: float = 5.0, compute_type: Dtype = jnp.float32,
    ):
        @nnx.vmap(in_axes=0, out_axes=0)
        def make_head(rngs):
            return CriticHead(input_dim, action_dim, rngs, num_blocks, hidden_dim, action_embed_dim, c_shift, num_bins, compute_type)

        self.heads = make_head(rngs.fork(split=num_qs))
        self.dist = CategoricalPolicy(num_bins, min_v, max_v)

    def __call__(self, vision_z: jax.Array, actions: jax.Array) -> jax.Array:
        """Return categorical value logits with shape [num_qs, B, num_bins]."""
        return nnx.vmap(lambda head, z, a: head(z, a), in_axes=(0, None, None))(self.heads, vision_z, actions)

    def q_values(self, vision_z: jax.Array, actions: jax.Array) -> jax.Array:
        return self.dist.q_values(self(vision_z, actions))
