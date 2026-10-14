"""RePrent agents and backend implementations."""

__all__ = ["JaxRePTAgent"]


def __getattr__(name: str):
    if name == "JaxRePTAgent":
        from .jax import RePTAgent

        return RePTAgent
    raise AttributeError(name)
