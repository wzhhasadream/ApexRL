import torch
import torch.nn.functional as F

from ...buffers import SequenceBatch


def augment_observations(
    observations: torch.Tensor,
    padding: int = 4,
) -> torch.Tensor:
    """Random shift with bilinear interpolation, matching DrQ-v2.

    Observations are channels-last and frame-first: [B, F, H, W, C].
    """
    batch_size, frame_stack, height, width, channels = observations.shape
    assert height == width

    images = observations.permute(0, 1, 4, 2, 3).reshape(
        batch_size, frame_stack * channels, height, width
    )
    images = images.float()
    padded = F.pad(
        images,
        (padding,) * 4,
        mode="replicate",
    )

    eps = 1.0 / (height + 2 * padding)
    coordinates = torch.linspace(
        -1.0 + eps,
        1.0 - eps,
        height + 2 * padding,
        device=observations.device,
        dtype=torch.float32,
    )[:height]
    coordinates = coordinates.unsqueeze(0).repeat(height, 1).unsqueeze(2)
    grid = torch.cat((coordinates, coordinates.transpose(0, 1)), dim=2)
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    shift = torch.randint(
        0,
        2 * padding + 1,
        (batch_size, 1, 1, 2),
        device=observations.device,
        dtype=torch.float32,
    )
    shift *= 2.0 / (height + 2 * padding)
    augmented = F.grid_sample(
        padded,
        grid + shift,
        padding_mode="zeros",
        align_corners=False,
    )
    augmented = augmented.reshape(
        batch_size, frame_stack, channels, height, width
    ).permute(0, 1, 3, 4, 2)
    return augmented


def augment_sequencebatch(sequencebatch: SequenceBatch) -> SequenceBatch:
    """Randomly shift each frame stack in a [T, B, F, H, W, C] sequence."""
    shape = sequencebatch.observations.shape
    observations = sequencebatch.observations.reshape((-1, *shape[2:]))
    next_observations = sequencebatch.next_observations.reshape((-1, *shape[2:]))
    observations = augment_observations(observations).reshape(shape)
    next_observations = augment_observations(next_observations).reshape(shape)
    return SequenceBatch(
        observations, sequencebatch.actions, sequencebatch.rewards,
        sequencebatch.terminations, sequencebatch.truncations, next_observations,
    )
