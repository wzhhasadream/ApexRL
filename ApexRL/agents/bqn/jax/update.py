import jax
import jax.numpy as jnp
from flax import nnx

from ....buffers.off_policy import Batch, SequenceBatch, compress_n_step
from ....model.jax import Network, RewardNormalizer
from ..config import BQNConfig
from .network import Critic
from ....common.jax import augment_batch 

def update_critic(critic: Network[Critic], target_critic: Network[Critic], batch: Batch, cfg: BQNConfig) -> dict[str, jax.Array]:
    if cfg.norm_type == "bn":
        observations = jnp.concatenate((batch.observations, batch.next_observations), axis=0)
        batch_size = batch.observations.shape[0]
        # Train target BatchNorm on the same joint batch, outside the gradient closure.
        next_logits = jax.lax.stop_gradient(target_critic.model(observations, training=True)[batch_size:])
    
    else:
        next_logits = jax.lax.stop_gradient(target_critic.model(batch.next_observations, training=True))

    next_q = target_critic.model.dist.q_values(next_logits)[..., 0]
    next_actions = target_critic.model.policy.action(next_q)
    next_logits = jnp.take_along_axis(next_logits, next_actions[:, None, None], axis=1)[:, 0]
    target_values = batch.rewards + batch.discounts * (1.0 - batch.dones) * critic.model.dist.bins
    target_probs = critic.model.dist.target_probs(next_logits, target_values)

    def critic_loss(model: Critic):
        # Current and next observations share the online network's batch statistics.
        if cfg.norm_type == "bn":
            logits = model(observations, training=True)[:batch_size]
        else:
            logits = model(batch.observations, training=True)
        actions = batch.actions.reshape((-1, 1, 1)).astype(jnp.int32)
        action_logits = jnp.take_along_axis(logits, actions, axis=1)[:, 0]
        loss = model.dist.loss(action_logits, target_probs)
        return loss, {"training/critic_loss": loss, "training/q_mean": model.dist.q_values(action_logits).mean(),
                      "training/target_q_mean": (target_probs * model.dist.bins).sum(axis=-1).mean(),
                      "training/reward_mean": batch.rewards.mean()}

    (_, info), grads = nnx.value_and_grad(critic_loss, has_aux=True)(critic.model)
    critic.grad_step(grads)
    return info


def make_update(cfg: BQNConfig):
    @nnx.jit(static_argnames=("update_target",))
    def update(critic: Network[Critic], target_critic: Network[Critic], reward_normalizer: Network[RewardNormalizer] | None,
               sequence: SequenceBatch, update_target: bool, key: jax.Array) -> dict[str, jax.Array]:
        batch = compress_n_step(sequence, cfg.gamma)
        if reward_normalizer is not None:
            batch = batch._replace(rewards=reward_normalizer.model.normalize(batch.rewards))
        if cfg.augment_obs:
            batch = augment_batch(batch, key)
        info = update_critic(critic, target_critic, batch, cfg)
        if update_target:
            target_critic.soft_update()
        return info

    return update
