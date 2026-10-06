from typing import Any

from .config import FastSACConfig

__all__ = ["JaxFastSACAgent", "TorchFastSACAgent", "FastSACConfig"]


def __getattr__(name: str) -> Any:
    if name == "JaxFastSACAgent":
        from .jax import FastSACAgent

        return FastSACAgent
    if name == "TorchFastSACAgent":
        from .torch import FastSACAgent

        return FastSACAgent
    raise AttributeError(name)
