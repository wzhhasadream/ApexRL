import jax
import jax.numpy as jnp
from flax import nnx

from ....buffers.off_policy import Batch, SequenceBatch, compress_n_step
from ....model.jax import Network, RewardNormalizer
from .network import Critic


def update_critic(critic: Network[Critic], target_critic: Network[Critic], batch: Batch, cfg, reward_normalizer: Network[RewardNormalizer] | None = None):
    rewards = batch.rewards.astype(jnp.float32)
    continuation = batch.discounts * (1.0 - batch.dones)
    all_observations = jnp.concatenate((batch.observations, batch.next_observations), axis=0)
    target_logits = jax.lax.stop_gradient(target_critic.model(batch.next_observations, training=True))

    def loss_fn(model: Critic):
        all_logits = model(all_observations, training=True)
        logits, next_logits = jnp.split(all_logits, 2, axis=0)
        online_q = model.q_values(next_logits)
        next_actions = model.policy.greedy_action(online_q)
        selected_target_logits = jnp.take_along_axis(target_logits, next_actions[:, None, None], axis=1)[:, 0]
        target_values = rewards + continuation * model.dist.bins
        target_probs = model.dist.target_probs(selected_target_logits, target_values)
        actions = batch.actions.astype(jnp.int32).reshape((-1,))
        logits = jnp.take_along_axis(logits, actions[:, None, None], axis=1)[:, 0]
        loss = model.dist.loss(logits, target_probs)
        return loss, {"critic_loss": loss, "mean_q": jnp.mean(model.q_values(logits))}

    (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(critic.model)
    critic.grad_step(grads)
    return info


def make_update(cfg):
    @nnx.jit
    def update(critic, target_critic, reward_normalizer, sequence: SequenceBatch):
        if reward_normalizer is not None:
            sequence = sequence._replace(rewards=reward_normalizer.model.normalize(sequence.rewards))
        batch = compress_n_step(sequence, cfg.gamma)
        info = update_critic(critic, target_critic, batch, cfg)
        target_critic.soft_update()
        return info
    return update
