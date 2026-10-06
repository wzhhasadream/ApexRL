"""V-Simba agents and backend implementations."""

from .config import VSimbaConfig

__all__ = ["JaxVSimbaAgent", "TorchVSimbaAgent", "VSimbaConfig"]


def __getattr__(name: str):
    if name == "JaxVSimbaAgent":
        from .jax import VSimbaAgent

        return VSimbaAgent
    if name == "TorchVSimbaAgent":
        from .torch import VSimbaAgent

        return VSimbaAgent
    raise AttributeError(name)
