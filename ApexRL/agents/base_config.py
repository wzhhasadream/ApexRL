from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, ClassVar, TypeVar

from ..common import explicit_cli_fields

C = TypeVar("C", bound="BaseConfig")

# Environment families: env_type -> family name used as a key in ENV_TYPE_PRESETS
ENV_FAMILY = {
    "atari": "atari",
    "mujoco": "cpu_sim", "dmc": "cpu_sim", "myosuite": "cpu_sim", "humanoid_bench": "cpu_sim", "metaworld": "cpu_sim",
    "playground": "gpu_sim", "isaaclab": "gpu_sim", "maniskill": "gpu_sim", "mjlab": "gpu_sim",
}


@dataclasses.dataclass
class BaseConfig:
    """Layered agent config. Later layers win:

        dataclass defaults < ENV_TYPE_PRESETS[family or env_type] < ENV_NAME_PRESETS[env_name] < user / CLI overrides

    Subclasses only declare the two preset dicts. A preset key in ENV_TYPE_PRESETS can be a family
    ("atari", "cpu_sim", "gpu_sim") or a concrete env_type ("isaaclab"); a concrete env_type is applied after its family.
    """

    ENV_TYPE_PRESETS: ClassVar[dict[str, dict[str, Any]]] = {}
    ENV_NAME_PRESETS: ClassVar[dict[str, dict[str, Any]]] = {}

    @classmethod
    def presets(cls, env_type: str, env_name: str | None = None) -> dict[str, Any]:
        """Merged preset values for one environment (without user overrides)."""
        if env_type not in ENV_FAMILY:
            raise ValueError(f"Unknown env_type {env_type!r}; expected one of {tuple(ENV_FAMILY)}")
        family = ENV_FAMILY[env_type]
        merged = {**cls.ENV_TYPE_PRESETS.get(family, {}), **cls.ENV_TYPE_PRESETS.get(env_type, {})}
        if env_name is not None:
            merged.update(cls.ENV_NAME_PRESETS.get(env_name, {}))
        return merged

    @classmethod
    def from_env(cls: type[C], env_type: str, env_name: str | None = None, **overrides: Any) -> C:
        """Build a config for one environment; keyword overrides beat every preset."""
        values = {**cls.presets(env_type, env_name), **overrides}
        unknown = set(values) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"{cls.__name__} has no fields {sorted(unknown)}")
        return cls(**values)

    @classmethod
    def from_cli(cls: type[C], args: Any, env_type: str, env_name: str | None = None, argv: Sequence[str] | None = None) -> C:
        """Build a config from parsed CLI args: only options typed on the command line override the presets."""
        fields = {f.name for f in dataclasses.fields(cls)}
        explicit = explicit_cli_fields(argv) & fields
        return cls.from_env(env_type, env_name, **{name: getattr(args, name) for name in explicit})

    def replace(self: C, **changes: Any) -> C:
        return dataclasses.replace(self, **changes)


__all__ = ["BaseConfig", "ENV_FAMILY"]
