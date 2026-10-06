import jax
import jax.numpy as jnp

from ...buffers import SequenceBatch


def augment_observations(
    rng: jax.Array,
    observations: jax.Array,
    padding: int = 4,
) -> jax.Array:
    """Apply the integer random shift used by DrQ-v2.

    Observations are channels-last and frame-first: [B, F, H, W, C].
    """
    n, f, h, w, c = observations.shape
    assert h == w

    observations = jnp.pad(
        observations,
        pad_width=((0, 0), (0, 0), (padding, padding), (padding, padding), (0, 0)),
        mode="edge",
    )

    shifts = jax.random.randint(rng, (n, 2), 0, 2 * padding + 1)

    def crop(image, shift):
        return jax.lax.dynamic_slice(
            image,
            (0, shift[0], shift[1], 0),
            (f, h, w, c),
        )

    return jax.vmap(crop)(observations, shifts)


def augment_sequencebatch(sequencebatch: SequenceBatch, key: jax.Array) -> SequenceBatch:
    """Randomly shift each frame stack in a [T, B, F, H, W, C] sequence."""
    shape = sequencebatch.observations.shape
    obs_key, next_obs_key = jax.random.split(key)
    observations = sequencebatch.observations.reshape((-1, *shape[2:]))
    next_observations = sequencebatch.next_observations.reshape((-1, *shape[2:]))
    observations = augment_observations(obs_key, observations).reshape(shape)
    next_observations = augment_observations(next_obs_key, next_observations).reshape(shape)
    return sequencebatch._replace(observations=observations, next_observations=next_observations)
