import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype

from ....model.jax import CategoricalPolicy, Alpha
from ....model.jax.backbones import VSimbaEmbedder, VSimbaEncoder, VSimbaProjector
from ....model.jax.layer import orthogonal
from ....model.jax.policy import SquashedTanhGaussianPolicy


class Actor(nnx.Module):
    def __init__(self, input_dim: int, action_dim: int, rngs: nnx.Rngs, num_blocks: int = 1, hidden_dim: int = 128, compute_type: Dtype = jnp.float32):
        self.action_dim = action_dim
        self.trunk = VSimbaEncoder(input_dim, num_blocks, hidden_dim, rngs, compute_type=compute_type)
        self.mean_w = nnx.Linear(hidden_dim, action_dim, rngs=rngs, kernel_init=orthogonal(1), dtype=compute_type)
        self.log_std_w = nnx.Linear(hidden_dim, action_dim, rngs=rngs, kernel_init=orthogonal(1), dtype=compute_type)
        self.policy = SquashedTanhGaussianPolicy(-1.0, 1.0, log_std_min=-10.0, log_std_max=2.0)

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

        self.reward_head = nnx.Linear(action_embed_dim + input_dim, 1, rngs=rngs, dtype=compute_type)

        self.p_w = nnx.Linear(input_dim, input_dim + action_embed_dim, rngs=rngs, dtype=compute_type)

        self.nce_alpha = Alpha(100)


    def __call__(self, vision_z: jax.Array, actions: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        action_z = self.action_embedder(self.projector(actions))
        phi_sa = jnp.concatenate([vision_z, action_z], axis=-1)
        z = self.encoder(phi_sa)
        reward = self.reward_head(phi_sa)
        return phi_sa, reward, self.w(z)

    def embed_sa(self, vision_z: jax.Array, actions: jax.Array) -> jax.Array:
        action_z = self.action_embedder(self.projector(actions))
        return jnp.concatenate([vision_z, action_z], axis=-1)

    def embed_s(self, vision_z: jax.Array) -> jax.Array:
        return self.p_w(vision_z)



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
        return nnx.vmap(lambda head, z, a: head(z, a), in_axes=(0, None, None))(self.heads, vision_z, actions)

    def q_values(self, vision_z: jax.Array, actions: jax.Array) -> jax.Array:
        return self.dist.q_values(self(vision_z, actions)[-1])

    def embed_s(self, vision_z: jax.Array) -> jax.Array:
        return nnx.vmap(lambda head, z: head.embed_s(z), in_axes=(0, None))(self.heads, vision_z)

    def get_alpha(self) -> jax.Array:
        return self.heads.nce_alpha()
