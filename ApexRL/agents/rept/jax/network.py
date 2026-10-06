from flax import nnx
import jax
import jax.numpy as jnp
from flax.typing import Dtype



from ....model.jax.policy import SquashedTanhGaussianPolicy, MaskedCategoricalPolicy
from ....model.jax import CategoricalPolicy, Alpha
from ....model.jax.backbones import (
    FlashSACBlock,
    FlashSACEmbedder,
)
from ....model.jax.layer import orthogonal

CRITIC_STATE_AXES = nnx.StateAxes({
    nnx.PathContains("dist"): None,
    nnx.Param: 0,
    nnx.BatchStat: 0,
    nnx.Variable: 0,
})


class Phi(nnx.Module):
    """Per-critic state-action features and shared successor-state features."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        rngs: nnx.Rngs,
        num_mlp_blocks: int = 1,
        latent_dim: int = 256,
        use_bias: bool = True,
        compute_type: Dtype = jnp.float32,
    ):
        self.sa_embed = FlashSACEmbedder(
            obs_dim + action_dim,
            latent_dim,
            rngs=rngs,
            use_bias=use_bias,
            compute_type=compute_type,
        )
        self.sa_blocks = [
            FlashSACBlock(latent_dim, rngs, 4, use_bias, compute_type)
            for _ in range(num_mlp_blocks)
        ]
        self.s_embed = FlashSACEmbedder(obs_dim, latent_dim, rngs, use_bias, compute_type)
        self.s_block = FlashSACBlock(latent_dim, rngs, 4, use_bias, compute_type)

        self.reward_head = nnx.Linear(latent_dim, 1, use_bias=use_bias, dtype=compute_type, rngs=rngs)

        self.nce_alpha = Alpha(100)


    def embed_sa(
        self,
        observations: jax.Array,
        actions: jax.Array,
        training: bool,
    ) -> jax.Array:
        x = self.sa_embed(jnp.concatenate([observations, actions], axis=-1), training)
        for block in self.sa_blocks:
            x = block(x, training)
        return x

    def embed_s(
        self,
        observations: jax.Array,
        training: bool
    ):
        return self.s_block(self.s_embed(observations, training), training)


    def get_alpha(self) -> jax.Array:
        return self.nce_alpha()





class Critic(nnx.Module):
    @nnx.vmap(
        in_axes=(
            CRITIC_STATE_AXES,
            None,
            None,
            None,
            0,
            None,
            None,
            None,
            None,
            None,
        )
    )
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        rngs: nnx.Rngs,
        num_mlp_blocks: int = 1,
        num_trunk_blocks: int = 1,
        use_bias: bool = True,
        compute_type: Dtype = jnp.float32,
        num_head: int = 101,
    ):
        self.encoder = Phi(
            obs_dim,
            action_dim,
            rngs,
            num_mlp_blocks=num_mlp_blocks,
            latent_dim=hidden_dim,
            use_bias=use_bias,
            compute_type=compute_type,
        )
        self.blocks = [
            FlashSACBlock(hidden_dim, rngs, 4, use_bias, compute_type)
            for _ in range(num_trunk_blocks)
        ]
        self.final_norm = nnx.RMSNorm(hidden_dim, rngs=rngs, dtype=compute_type)
        self.value_head = nnx.Linear(
            hidden_dim, num_head, dtype=compute_type, rngs=rngs
        )
        self.dist = CategoricalPolicy(num_head, -5.0, 5.0)

    @nnx.vmap(in_axes=(CRITIC_STATE_AXES, None, None, None))
    def __call__(
        self,
        observation: jax.Array,
        action: jax.Array,
        training: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        phi_sa = self.encoder.embed_sa(observation, action, training=training)
        x = phi_sa
        for block in self.blocks:
            x = block(x, training)
        x = self.final_norm(x)
        logits = self.value_head(x)
        reward = self.encoder.reward_head(phi_sa)
        return phi_sa.astype(jnp.float32), reward.astype(jnp.float32), logits.astype(jnp.float32)

    def q_values(
        self,
        observation: jax.Array,
        action: jax.Array,
        training: bool,
    ) -> jax.Array:
        return self.dist.q_values(self(observation, action, training)[-1]).astype(jnp.float32)

    @nnx.vmap(in_axes=(CRITIC_STATE_AXES, None, None))
    def embed_s(self, observation: jax.Array, training: bool) -> jax.Array:
        return self.encoder.embed_s(observation, training).astype(jnp.float32)

    def get_alpha(self):
        return self.encoder.get_alpha()


class ActorEncoder(nnx.Module):
    """State MLP trunk for the continuous actor."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        rngs: nnx.Rngs,
        num_mlp_blocks: int = 1,
        use_bias: bool = True,
        compute_type: Dtype = jnp.float32,
    ):
        self.embed = FlashSACEmbedder(
            obs_dim,
            hidden_dim,
            rngs=rngs,
            use_bias=use_bias,
            compute_type=compute_type,
        )
        self.blocks = [
            FlashSACBlock(hidden_dim, rngs, 4, use_bias, compute_type)
            for _ in range(num_mlp_blocks)
        ]
        self.rms = nnx.RMSNorm(hidden_dim, rngs=rngs, dtype=compute_type)

    def __call__(self, x: jax.Array, training: bool) -> jax.Array:
        x = self.embed(x, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        return self.rms(x)


class Actor(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        rngs: nnx.Rngs,
        hidden_dim: int = 256,
        num_mlp_blocks: int = 2,
        action_low: jax.Array = -1.0,
        action_high: jax.Array = 1.0,
        action_is_discrete: bool = False,
        use_bias: bool = True,
        compute_type: Dtype = jnp.float32,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.compute_type = compute_type
        self.encoder = ActorEncoder(
            obs_dim,
            hidden_dim,
            rngs=rngs,
            num_mlp_blocks=num_mlp_blocks,
            use_bias=use_bias,
            compute_type=compute_type,
        )
        self.action_is_discrete = action_is_discrete
        if self.action_is_discrete:
            self.logits_head = nnx.Linear(hidden_dim, action_dim, rngs=rngs, kernel_init=orthogonal(1), dtype=compute_type)
            self.policy = MaskedCategoricalPolicy()
        
        else:
            self.fc_mean = nnx.Linear(hidden_dim, action_dim, rngs=rngs, kernel_init=orthogonal(1), dtype=compute_type)
            self.fc_log_std = nnx.Linear(hidden_dim, action_dim, rngs=rngs, kernel_init=orthogonal(1), dtype=compute_type)
            self.policy = SquashedTanhGaussianPolicy(
                action_low=action_low,
                action_high=action_high
            )

    def __call__(
        self,
        observations: jax.Array,
        training: bool = True,
    ):
        x = self.encoder(observations, training=training)
        if self.action_is_discrete:
            logits = self.logits_head(x).astype(jnp.float32)
            return logits.astype(jnp.float32)
        else:
            mean = self.fc_mean(x)
            log_std = self.fc_log_std(x)
            return mean.astype(jnp.float32), log_std.astype(jnp.float32)

    def get_action(
        self,
        observations: jax.Array,
        *,
        key: jax.Array,
        training: bool = True
    ) -> tuple[jax.Array, jax.Array]:
        if self.action_is_discrete:
            logits = self(observations, training=training)
            actions, log_probs = self.policy.sample_and_log_prob(logits, key)
            return actions, log_probs

        mean, log_std = self(observations, training=training)
        return self.policy.sample_and_log_prob(mean, log_std, key)

    def get_deterministic_action(self, observations: jax.Array) -> jax.Array:
        if self.action_is_discrete:
            logits = self(observations, training=False)
            return self.policy.greedy_action(logits)
        mean, _ = self(observations, training=False)
        return self.policy.mean_action(mean).astype(jnp.float32)
