from .config import BQNConfig

__all__ = ["BQNConfig", "JaxBQNAgent"]


def __getattr__(name):
    if name == "JaxBQNAgent":
        from .jax import BQNAgent
        return BQNAgent
    raise AttributeError(name)
