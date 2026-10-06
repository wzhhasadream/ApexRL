from typing import Any, Literal
import gymnasium as gym
from .types import Batch, SequenceBatch, Transition
import numpy as np

BufferType = Literal[
    "numpy",
    "np",
    "numpy_pixel",
    "numpy_frame_stack",
    "jax",
    "torch",
    "pytorch",
]

def create_buffer(
    action_space: gym.spaces.Space,
    observation_space: gym.spaces.Space,
    buffer_type: BufferType = "numpy",
    num_env: int = 1,
    device: Any = "cpu",
    max_size: int = int(1e6),
    linear_decay_step: int = 0,
    min_weight: float = 0.1,
    max_n_step: int = 1,
    use_approximate_sampling: bool = True,
    num_buckets: int = 2000,
) -> Any:
    buffer_type = buffer_type.lower()
    common_kwargs = dict(
        observation_space=observation_space,
        action_space=action_space,
        max_size=max_size,
        linear_decay_step=linear_decay_step,
        min_weight=min_weight,
        num_envs=num_env,
        use_approximate_sampling=use_approximate_sampling,
        num_buckets=num_buckets,
    )

    if buffer_type in ("numpy", "np"):
        from .numpy_buffer import NumpyBuffer

        return NumpyBuffer(**common_kwargs)
    if buffer_type in ("numpy_pixel", "numpy_frame_stack"):
        from .numpy_lazy_frame_buffer import NumpyLazyFrameBuffer

        return NumpyLazyFrameBuffer(**common_kwargs, max_n_step=max_n_step)
    if buffer_type in ("torch", "pytorch"):
        from .torch_buffer import TorchBuffer

        return TorchBuffer(**common_kwargs, device=device)
    if buffer_type == "jax":
        from .jax_buffer import JaxBuffer

        return JaxBuffer.create(
            observation_space=observation_space,
            action_space=action_space,
            max_size=max_size,
            linear_decay_step=linear_decay_step,
            min_weight=min_weight,
            num_envs=num_env,
            use_approximate_sampling=use_approximate_sampling,
            num_buckets=num_buckets,
            device=device,
        )

    raise ValueError(f"Invalid buffer_type: {buffer_type}")




def compress_n_step(sequence: SequenceBatch, gamma: float) -> Batch:
    """Compress a ``[T, B, ...]`` sequence into a ``[B, ...]`` batch."""
    rewards = sequence.rewards
    terminations = sequence.terminations
    truncations = sequence.truncations
    end = ((terminations > 0) | (truncations > 0))[..., 0]

    if isinstance(rewards, np.ndarray):
        keep = np.logical_and.accumulate(np.concatenate((np.ones((1, rewards.shape[1]), bool), ~end[:-1]), axis=0), axis=0)
        lengths = keep.sum(axis=0)
        endpoint = lengths - 1
        batch_index = np.arange(rewards.shape[1])
        powers = np.power(np.float32(gamma), np.arange(rewards.shape[0], dtype=np.float32))[:, None, None]
        compressed_rewards = (rewards * keep[..., None] * powers).sum(axis=0)
        return Batch(
            sequence.observations[0], sequence.actions[0], compressed_rewards,
            terminations[endpoint, batch_index], sequence.next_observations[endpoint, batch_index],
            np.power(np.float32(gamma), lengths)[:, None],
        )

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and torch.is_tensor(rewards):
        keep = torch.cat((torch.ones_like(end[:1]), (~end[:-1]).to(end.dtype)), dim=0).cumprod(dim=0).bool()
        lengths = keep.sum(dim=0)
        endpoint = lengths - 1
        batch_index = torch.arange(rewards.shape[1], device=rewards.device)
        powers = torch.pow(torch.as_tensor(gamma, dtype=torch.float32, device=rewards.device), torch.arange(rewards.shape[0], device=rewards.device, dtype=torch.float32))[:, None, None]
        compressed_rewards = (rewards * keep[..., None] * powers).sum(dim=0)
        return Batch(
            sequence.observations[0], sequence.actions[0], compressed_rewards,
            terminations[endpoint, batch_index], sequence.next_observations[endpoint, batch_index],
            torch.pow(torch.as_tensor(gamma, dtype=torch.float32, device=rewards.device), lengths.float())[:, None],
        )

    import jax.numpy as jnp

    keep = jnp.concatenate((jnp.ones_like(end[:1]), (~end[:-1]).astype(end.dtype)), axis=0).cumprod(axis=0).astype(bool)
    lengths = keep.sum(axis=0)
    endpoint = lengths - 1
    batch_index = jnp.arange(rewards.shape[1])
    powers = jnp.power(jnp.asarray(gamma, dtype=jnp.float32), jnp.arange(rewards.shape[0], dtype=jnp.float32))[:, None, None]
    compressed_rewards = (rewards * keep[..., None] * powers).sum(axis=0)
    return Batch(
        sequence.observations[0], sequence.actions[0], compressed_rewards,
        terminations[endpoint, batch_index], sequence.next_observations[endpoint, batch_index],
        jnp.power(jnp.asarray(gamma, dtype=jnp.float32), lengths)[:, None],
    )




__all__ = ["Batch", "SequenceBatch", "Transition", "compress_n_step", "create_buffer", "BufferType"]
