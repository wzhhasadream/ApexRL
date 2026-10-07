"""WarpSAC agents and backend implementations."""

from .config import WarpSACConfig

__all__ = ["JaxWarpSACAgent", "TorchWarpSACAgent", "WarpSACConfig"]


def __getattr__(name: str):
    if name == "JaxWarpSACAgent":
        from .jax import WarpSACAgent

        return WarpSACAgent
    if name == "TorchWarpSACAgent":
        from .torch import WarpSACAgent

        return WarpSACAgent
    raise AttributeError(name)
