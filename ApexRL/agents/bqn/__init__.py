"""BQN configuration and JAX implementation."""

from typing import Any

from .config import BQNConfig

__all__ = ["BQNConfig", "JaxBQNAgent"]


def __getattr__(name: str) -> Any:
    if name == "JaxBQNAgent":
        from .jax import BQNAgent

        return BQNAgent
    raise AttributeError(name)
