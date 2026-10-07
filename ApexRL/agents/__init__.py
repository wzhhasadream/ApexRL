"""Public agent interfaces."""

__all__ = ["JaxVSimbaAgent", "TorchVSimbaAgent", "VSimbaConfig", "JaxPQNAgent", "TorchPQNAgent", "PQNConfig", "JaxFastSACAgent", "TorchFastSACAgent", "FastSACConfig", "JaxFastTD3Agent", "TorchFastTD3Agent", "FastTD3Config", "JaxPPOAgent", "TorchPPOAgent", "PPOConfig", "JaxWarpSACAgent", "TorchWarpSACAgent", "WarpSACConfig"]


def __getattr__(name: str):
    if name == "JaxVSimbaAgent":
        from .v_simba.jax import VSimbaAgent

        return VSimbaAgent
    if name == "TorchVSimbaAgent":
        from .v_simba.torch import VSimbaAgent

        return VSimbaAgent
    if name == "VSimbaConfig":
        from .v_simba.config import VSimbaConfig

        return VSimbaConfig
    if name == "JaxPQNAgent":
        from .pqn.jax import PQNAgent

        return PQNAgent
    if name == "TorchPQNAgent":
        from .pqn.torch import PQNAgent

        return PQNAgent
    if name == "PQNConfig":
        from .pqn.config import PQNConfig

        return PQNConfig
    if name == "JaxFastSACAgent":
        from .fastsac.jax import FastSACAgent

        return FastSACAgent
    if name == "TorchFastSACAgent":
        from .fastsac.torch import FastSACAgent

        return FastSACAgent
    if name == "FastSACConfig":
        from .fastsac.config import FastSACConfig

        return FastSACConfig
    if name == "JaxFastTD3Agent":
        from .fasttd3.jax import FastTD3Agent

        return FastTD3Agent
    if name == "TorchFastTD3Agent":
        from .fasttd3.torch import FastTD3Agent

        return FastTD3Agent
    if name == "FastTD3Config":
        from .fasttd3.config import FastTD3Config

        return FastTD3Config
    if name == "JaxPPOAgent":
        from .ppo.jax import PPOAgent

        return PPOAgent
    if name == "TorchPPOAgent":
        from .ppo.torch import PPOAgent

        return PPOAgent
    if name == "PPOConfig":
        from .ppo.config import PPOConfig

        return PPOConfig
    if name == "JaxWarpSACAgent":
        from .warpsac.jax import WarpSACAgent

        return WarpSACAgent
    if name == "TorchWarpSACAgent":
        from .warpsac.torch import WarpSACAgent

        return WarpSACAgent
    if name == "WarpSACConfig":
        from .warpsac.config import WarpSACConfig

        return WarpSACConfig
    raise AttributeError(name)
