from typing import Any

from .config import PPOConfig

__all__ = ["JaxPPOAgent", "TorchPPOAgent", "PPOConfig"]


def __getattr__(name: str) -> Any:
    if name == "JaxPPOAgent":
        from .jax import PPOAgent

        return PPOAgent
    if name == "TorchPPOAgent":
        from .torch import PPOAgent

        return PPOAgent
    raise AttributeError(name)
