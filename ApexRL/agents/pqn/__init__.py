"""PQN agents and backend implementations."""

from typing import Any

from .config import PQNConfig

__all__ = ["JaxPQNAgent", "TorchPQNAgent", "PQNConfig"]


def __getattr__(name: str) -> Any:
    if name == "JaxPQNAgent":
        from .jax import PQNAgent

        return PQNAgent
    if name == "TorchPQNAgent":
        from .torch import PQNAgent

        return PQNAgent
    raise AttributeError(name)
