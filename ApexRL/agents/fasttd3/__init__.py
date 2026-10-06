from typing import Any

from .config import FastTD3Config

__all__ = ["JaxFastTD3Agent", "TorchFastTD3Agent", "FastTD3Config"]


def __getattr__(name: str) -> Any:
    if name == "JaxFastTD3Agent":
        from .jax import FastTD3Agent

        return FastTD3Agent
    if name == "TorchFastTD3Agent":
        from .torch import FastTD3Agent

        return FastTD3Agent
    raise AttributeError(name)
