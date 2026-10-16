from flax import nnx
import jax
import jax.numpy as jnp
from flax.typing import Dtype
from typing import Literal

from ....model.jax.policy import SquashedTanhGaussianPolicy
from ....model.jax import CategoricalPolicy, TwoHotReward
from ....model.jax.backbones import (
    FlashSACEncoder,
    FlashSACBlock,
    FlashSACEmbedder,
    DRQVisionEncoder,
    DRQV2VisionEncoder,
    VSimbaVisionEncoder,
)
from ....model.jax.layer import RMSNorm, orthogonal


class SAEncoder(nnx.Module):
    """First part of each Q backbone, ending before the final RMSNorm."""

    def __init__(self, input_dim: int, hidden_dim: int, rngs: nnx.Rngs,
                 use_bias: bool, compute_type: Dtype) -> None:
        self.embed = FlashSACEmbedder(input_dim, hidden_dim, rngs, use_bias, compute_type)
        self.block = FlashSACBlock(hidden_dim, rngs, use_bias=use_bias, compute_type=compute_type)

    def __call__(self, inputs: jax.Array, training: bool) -> jax.Array:
        return self.block(self.embed(inputs, training=training), training=training)


class QHead(nnx.Module):
    """Remaining residual blocks, final RMSNorm, and categorical output."""

    def __init__(self, hidden_dim: int, rngs: nnx.Rngs, num_blocks: int,
                 use_bias: bool, compute_type: Dtype, num_head: int) -> None:
        self.blocks = [FlashSACBlock(hidden_dim, rngs, use_bias=use_bias, compute_type=compute_type) for _ in range(num_blocks)]
        self.rms = RMSNorm(hidden_dim, rngs=rngs, dtype=compute_type)
        self.value_head = nnx.Linear(hidden_dim, num_head, dtype=compute_type, rngs=rngs)

    def __call__(self, z_sa: jax.Array, training: bool = True) -> jax.Array:
        for block in self.blocks:
            z_sa = block(z_sa, training=training)
        return self.value_head(self.rms(z_sa)).astype(jnp.float32)


class QNetwork(nnx.Module):
    """Map each state-action encoder together with its Q and auxiliary heads."""

    @nnx.vmap(in_axes=(0, None, None, None, 0, None, None, None, None, None), out_axes=0)
    def __init__(self, input_dim: int, z_s_dim: int, z_sa_dim: int, rngs: nnx.Rngs, num_blocks: int,
                 use_bias: bool, compute_type: Dtype, num_head: int, reward_num_bins: int) -> None:
        self.sa_encoder = SAEncoder(input_dim, z_sa_dim, rngs, use_bias, compute_type)
        self.q_head = QHead(z_sa_dim, rngs, num_blocks, use_bias, compute_type, num_head)
        self.reward_head = nnx.Linear(z_sa_dim, reward_num_bins, use_bias=use_bias, dtype=compute_type, rngs=rngs)
        self.p_w = nnx.Linear(z_sa_dim, z_s_dim, use_bias=use_bias, dtype=compute_type, rngs=rngs)

    @nnx.vmap(in_axes=(0, None, None), out_axes=0)
    def embed_sa(self, inputs: jax.Array, training: bool) -> jax.Array:
        return self.sa_encoder(inputs, training=training).astype(jnp.float32)

    @nnx.vmap(in_axes=(0, 0), out_axes=0)
    def pred_z_s(self, z_sa: jax.Array) -> jax.Array:
        return self.p_w(z_sa).astype(jnp.float32)

    @nnx.vmap(in_axes=(0, 0), out_axes=0)
    def reward_logits(self, z_sa: jax.Array) -> jax.Array:
        return self.reward_head(z_sa).astype(jnp.float32)

    @nnx.vmap(in_axes=(0, 0, None), out_axes=0)
    def __call__(self, z_sa: jax.Array, training: bool = True) -> jax.Array:
        return self.q_head(z_sa, training=training)


class Critic(nnx.Module):
    """One shared state encoder followed by independent state-action Q branches."""

    def __init__(self, obs_shape: tuple[int, ...], action_dim: int, rngs: nnx.Rngs, num_encoder_blocks: int = 1,
                 z_s_dim: int = 256, z_sa_dim: int = 256, use_bias: bool = True, compute_type: Dtype = jnp.float32,
                 vision_encoder: Literal["drq_dot", "drqv2", "vsimba"] = "drqv2",
                 num_blocks: int = 1, num_q: int = 2, num_head: int = 101, reward_num_bins: int = 101) -> None:
        self.cnn = None
        input_dim = obs_shape[0]
        if len(obs_shape) == 4:
            cnn_cls = {"drq_dot": DRQVisionEncoder, "drqv2": DRQV2VisionEncoder, "vsimba": VSimbaVisionEncoder}[vision_encoder]
            self.cnn = cnn_cls(obs_shape, rngs=rngs, compute_type=compute_type)
            input_dim = self.cnn.get_output_dim()
        self.s_encoder = FlashSACEncoder(input_dim, num_encoder_blocks, z_s_dim, rngs, use_bias, compute_type)
        self.reward_dist = TwoHotReward(reward_num_bins) if reward_num_bins > 1 else None
        self.q = QNetwork(z_s_dim + action_dim, z_s_dim, z_sa_dim, rngs.fork(split=num_q), num_blocks,
                          use_bias, compute_type, num_head, reward_num_bins)
        self.dist = CategoricalPolicy(num_head, -5.0, 5.0)

    def embed_s(self, observations: jax.Array, training: bool = False) -> jax.Array:
        observations = self.cnn(observations) if self.cnn is not None else observations
        return self.s_encoder(observations, training=training).astype(jnp.float32)

    def embed_sa(self, z_s: jax.Array, actions: jax.Array, training: bool = False) -> jax.Array:
        return self.q.embed_sa(jnp.concatenate([z_s, actions], axis=-1), training=training)

    def pred_reward(self, z_sa: jax.Array) -> jax.Array:
        logits = self.q.reward_logits(z_sa)
        return self.reward_dist.decode(logits) if self.reward_dist is not None else logits

    def reward_loss(self, z_sa: jax.Array, rewards: jax.Array) -> jax.Array:
        logits = self.q.reward_logits(z_sa)
        if self.reward_dist is not None:
            return jax.vmap(self.reward_dist.loss, in_axes=(0, None))(logits, rewards).mean()
        return jnp.square(logits - rewards[None, ...]).mean()

    def pred_z_s(self, z_sa: jax.Array) -> jax.Array:
        return self.q.pred_z_s(z_sa)

    def __call__(self, z_sa: jax.Array, training: bool = True) -> jax.Array:
        return self.q(z_sa, training=training)

    def q_values(self, z_sa: jax.Array, training: bool = True) -> jax.Array:
        return self.dist.q_values(self(z_sa, training=training))


class Actor(nnx.Module):
    """Continuous policy consuming the shared state representation."""

    def __init__(self, z_s_dim: int, action_dim: int, rngs: nnx.Rngs, hidden_dim: int = 256, num_blocks: int = 2,
                 action_low: jax.Array = -1.0, action_high: jax.Array = 1.0, use_bias: bool = True,
                 compute_type: Dtype = jnp.float32) -> None:
        self.trunk = FlashSACEncoder(z_s_dim, num_blocks, hidden_dim, rngs, use_bias, compute_type)
        self.fc_mean = nnx.Linear(hidden_dim, action_dim, rngs=rngs, kernel_init=orthogonal(1), dtype=compute_type)
        self.fc_log_std = nnx.Linear(hidden_dim, action_dim, rngs=rngs, kernel_init=orthogonal(1), dtype=compute_type)
        self.policy = SquashedTanhGaussianPolicy(action_low=action_low, action_high=action_high)

    def __call__(self, z_s: jax.Array, training: bool = True) -> tuple[jax.Array, jax.Array]:
        x = self.trunk(z_s, training=training)
        return self.fc_mean(x).astype(jnp.float32), self.fc_log_std(x).astype(jnp.float32)

    def get_action(self, z_s: jax.Array, *, key: jax.Array, training: bool = True) -> tuple[jax.Array, jax.Array]:
        mean, log_std = self(z_s, training=training)
        return self.policy.sample_and_log_prob(mean, log_std, key)

    def get_deterministic_action(self, z_s: jax.Array) -> jax.Array:
        mean, _ = self(z_s, training=False)
        return self.policy.mean_action(mean).astype(jnp.float32)
