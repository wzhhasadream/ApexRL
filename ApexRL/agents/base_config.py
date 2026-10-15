from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, ClassVar, TypeVar

from ..common import explicit_cli_fields
from ..envs import CPU_SIM, ENVPOOL_SIM, GPU_SIM

C = TypeVar("C", bound="BaseConfig")

# Environment families: env_type -> family name used as a key in ENV_TYPE_PRESETS
ENV_FAMILY = {
    **dict.fromkeys(CPU_SIM, "cpu_sim"),
    **dict.fromkeys(ENVPOOL_SIM, "envpool_sim"),
    **dict.fromkeys(GPU_SIM, "gpu_sim"),
}


@dataclasses.dataclass
class BaseConfig:
    """Layered agent config. Later layers win:

        instance values < family < cpu_sim_visual < env_type < env_name < user / CLI overrides

    ENV_TYPE_PRESETS accepts families ("envpool_sim", "cpu_sim", "gpu_sim") and concrete env_types.
    CPU tasks ending in "-visual" also apply "cpu_sim_visual" before the concrete env_type.
    """

    ENV_TYPE_PRESETS: ClassVar[dict[str, dict[str, Any]]] = {}
    ENV_NAME_PRESETS: ClassVar[dict[str, dict[str, Any]]] = {}
    DERIVED_FIELDS: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        self._auto_fields: set[str] = {name for name in self.DERIVED_FIELDS if getattr(self, name) is None}

    @classmethod
    def presets(cls, env_type: str, env_name: str | None = None) -> dict[str, Any]:
        """Merged preset values for one environment (without user overrides)."""
        if env_type not in ENV_FAMILY:
            raise ValueError(f"Unknown env_type {env_type!r}; expected one of {tuple(ENV_FAMILY)}")
        family = ENV_FAMILY[env_type]
        merged = dict(cls.ENV_TYPE_PRESETS.get(family, {}))
        if family == "cpu_sim" and env_name is not None and env_name.endswith("-visual"):
            merged.update(cls.ENV_TYPE_PRESETS.get("cpu_sim_visual", {}))
        merged.update(cls.ENV_TYPE_PRESETS.get(env_type, {}))
        if env_name is not None:
            merged.update(cls.ENV_NAME_PRESETS.get(env_name, {}))
        return merged

    def from_env(self: C, env_type: str, env_name: str | None = None, **overrides: Any) -> C:
        """Apply environment presets and overrides in place, then return self."""
        values = {**self.presets(env_type, env_name), **overrides}
        unknown = set(values) - {f.name for f in dataclasses.fields(self)}
        if unknown:
            raise ValueError(f"{type(self).__name__} has no fields {sorted(unknown)}")
        # Recompute automatic values after their inputs change; preserve explicit values.
        for name in self._auto_fields - values.keys():
            setattr(self, name, None)
        for name, value in values.items():
            setattr(self, name, value)
        self.__post_init__()
        return self

    def from_cli(self: C, env_type: str, env_name: str | None = None, argv: Sequence[str] | None = None, **overrides: Any) -> C:
        """Apply presets to this parsed CLI instance and return self.

        Only options typed on the command line override the presets; keyword
        overrides beat everything, so they can carry sampled hyperparameters.
        """
        fields = {f.name for f in dataclasses.fields(self)}
        explicit = (explicit_cli_fields(argv) & fields) - {"env_type", "env_name"}
        return self.from_env(env_type, env_name, **{**{name: getattr(self, name) for name in explicit}, **overrides})

    def replace(self: C, **changes: Any) -> C:
        return dataclasses.replace(self, **changes)


__all__ = ["BaseConfig", "ENV_FAMILY"]
